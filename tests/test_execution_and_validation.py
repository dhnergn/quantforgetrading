"""Execution safety, statistics, and validation.

The execution tests encode the failure modes that lose real money: duplicate
submissions, unknown order states, and position mismatches.
"""
from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantforge.core.enums import (
    AuditAction, ExecutionMode, OrderSide, OrderStatus, OrderType, TradingState,
)
from quantforge.core.errors import (
    AuthenticationError, AuthorizationError, BrokerTimeoutError,
    ExecutionStateError, LiveTradingNotAuthorized, OrderRejected,
    SandboxViolation,
)


# ==========================================================================
# Order lifecycle
# ==========================================================================


class TestOrderStateMachine:
    def test_happy_path(self, order_book):
        o = order_book.new_order("AAPL", OrderSide.LONG, 100)
        assert o.status == OrderStatus.CREATED
        order_book.transition(o.order_id, OrderStatus.SUBMITTED)
        assert o.status == OrderStatus.SUBMITTED and o.attempts == 1
        order_book.transition(o.order_id, OrderStatus.ACCEPTED)
        order_book.transition(o.order_id, OrderStatus.PARTIALLY_FILLED,
                              filled_quantity=40, fill_price=150.0)
        assert o.filled_quantity == 40
        order_book.transition(o.order_id, OrderStatus.FILLED,
                              filled_quantity=100, fill_price=150.0)
        assert o.status == OrderStatus.FILLED
        assert o.remaining_quantity == 0

    def test_terminal_states_are_final(self, order_book):
        o = order_book.new_order("AAPL", OrderSide.LONG, 10)
        order_book.transition(o.order_id, OrderStatus.SUBMITTED)
        order_book.transition(o.order_id, OrderStatus.REJECTED,
                              reject_reason="no buying power")
        with pytest.raises(ExecutionStateError, match="terminal"):
            order_book.transition(o.order_id, OrderStatus.FILLED)

    def test_unknown_cannot_return_to_submitted(self, order_book):
        """A blind resubmit after a timeout is how a position gets doubled."""
        o = order_book.new_order("MSFT", OrderSide.LONG, 10)
        order_book.transition(o.order_id, OrderStatus.SUBMITTED)
        order_book.transition(o.order_id, OrderStatus.UNKNOWN, reason="timeout")
        assert o.unknown_since is not None
        with pytest.raises(ExecutionStateError, match="illegal transition"):
            order_book.transition(o.order_id, OrderStatus.SUBMITTED)

    def test_unknown_resolves_to_a_known_state(self, order_book):
        o = order_book.new_order("MSFT", OrderSide.LONG, 10)
        order_book.transition(o.order_id, OrderStatus.SUBMITTED)
        order_book.transition(o.order_id, OrderStatus.UNKNOWN)
        order_book.transition(o.order_id, OrderStatus.FILLED,
                              filled_quantity=10, fill_price=300.0)
        assert o.status == OrderStatus.FILLED
        assert o.unknown_since is None

    def test_overfill_is_rejected(self, order_book):
        o = order_book.new_order("AAPL", OrderSide.LONG, 100)
        order_book.transition(o.order_id, OrderStatus.SUBMITTED)
        order_book.transition(o.order_id, OrderStatus.ACCEPTED)
        with pytest.raises(ExecutionStateError, match="exceeds the order quantity"):
            order_book.transition(o.order_id, OrderStatus.PARTIALLY_FILLED,
                                  filled_quantity=200)

    def test_partial_covering_everything_becomes_filled(self, order_book):
        o = order_book.new_order("AAPL", OrderSide.LONG, 100)
        order_book.transition(o.order_id, OrderStatus.SUBMITTED)
        order_book.transition(o.order_id, OrderStatus.ACCEPTED)
        order_book.transition(o.order_id, OrderStatus.PARTIALLY_FILLED,
                              filled_quantity=100)
        assert o.status == OrderStatus.FILLED

    def test_limit_order_requires_a_price(self):
        from quantforge.execution.orders import Order

        with pytest.raises(ExecutionStateError, match="requires limit_price"):
            Order(order_id="x", client_order_id="", instrument_id="A",
                  side=OrderSide.LONG, quantity=10, order_type=OrderType.LIMIT)

    def test_non_positive_quantity_is_rejected(self):
        from quantforge.execution.orders import Order

        with pytest.raises(ExecutionStateError, match="must be positive"):
            Order(order_id="x", client_order_id="", instrument_id="A",
                  side=OrderSide.LONG, quantity=-5)

    def test_decision_chain_violation_is_caught(self):
        from quantforge.execution.orders import Order

        t0 = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        o = Order(order_id="x", client_order_id="", instrument_id="A",
                  side=OrderSide.LONG, quantity=10,
                  information_cutoff=t0,
                  decision_time=t0 - dt.timedelta(seconds=1))
        with pytest.raises(Exception):
            o.validate_decision_chain()


class TestDuplicateProtection:
    def test_idempotency_key_is_deterministic(self):
        from quantforge.execution.orders import make_idempotency_key

        t = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        a = make_idempotency_key("s1", "p1", "AAPL", OrderSide.LONG, 100, t)
        b = make_idempotency_key("s1", "p1", "AAPL", OrderSide.LONG, 100, t)
        assert a == b

    def test_key_differs_for_a_different_quantity(self):
        from quantforge.execution.orders import make_idempotency_key

        t = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        a = make_idempotency_key("s1", "p1", "AAPL", OrderSide.LONG, 100, t)
        b = make_idempotency_key("s1", "p1", "AAPL", OrderSide.LONG, 101, t)
        assert a != b

    def test_duplicate_submission_returns_the_existing_order(self, order_book):
        from quantforge.execution.orders import Order

        def new():
            return order_book.new_order("MSFT", OrderSide.LONG, 50,
                                        strategy_id="s1", portfolio_id="p1")

        first = new()
        order_book.transition(first.order_id, OrderStatus.SUBMITTED)
        dup = Order(order_id="other", client_order_id="", instrument_id="MSFT",
                    side=OrderSide.LONG, quantity=50, strategy_id="s1",
                    portfolio_id="p1", decision_time=first.created_at)
        returned, is_new = order_book.submit(dup)
        assert is_new is False
        assert returned.order_id == first.order_id
        assert order_book.duplicates_blocked == 1

    def test_dedupe_survives_a_restart(self):
        """In-memory dedupe is lost on restart; the database query is durable."""
        from quantforge.execution.orders import Order
        from quantforge.persistence.db import (
            OrderRepository, init_db, session_scope,
        )

        url = f"sqlite:///{tempfile.mkdtemp()}/dedupe.db"
        init_db(url)
        o = Order(order_id="o1", client_order_id="qf-1", instrument_id="A",
                  side=OrderSide.LONG, quantity=10, idempotency_key="k1")
        with session_scope(url) as s:
            OrderRepository(s).upsert(o)
        with session_scope(url) as s:
            found = OrderRepository(s).by_idempotency_key("k1")
            # Read inside the session: a detached ORM instance cannot refresh.
            order_id = found.order_id if found is not None else None
        assert order_id == "o1"


# ==========================================================================
# Paper broker
# ==========================================================================


class TestPaperBroker:
    def test_market_order_fills_and_moves_cash(self, paper_broker, order_book):
        o = order_book.new_order("TEST", OrderSide.LONG, 100, strategy_id="s")
        ack = paper_broker.submit_order(o)
        assert ack.accepted
        assert ack.broker_order_id
        acct = paper_broker.get_account()
        assert acct.cash < 100_000.0
        positions = paper_broker.get_positions()
        assert len(positions) == 1 and positions[0].quantity == 100

    def test_unreachable_limit_does_not_fill(self, paper_broker, order_book, ohlcv):
        o = order_book.new_order("TEST", OrderSide.LONG, 100,
                                 order_type=OrderType.LIMIT,
                                 limit_price=float(ohlcv["low"].min()) * 0.5)
        paper_broker.submit_order(o)
        paper_broker.advance("TEST", ohlcv.index[-1])
        assert paper_broker.get_order(o.client_order_id)["status"] == "accepted"

    def test_reachable_limit_fills(self, paper_broker, order_book, ohlcv):
        # A buy limit above the market fills immediately as a marketable order.
        o = order_book.new_order("TEST", OrderSide.LONG, 100,
                                 order_type=OrderType.LIMIT,
                                 limit_price=float(ohlcv["high"].max()) * 1.05)
        paper_broker.submit_order(o)
        assert paper_broker.get_order(o.client_order_id)["status"] == "filled"

    def test_timeout_leaves_state_unknown(self, order_book, ohlcv):
        from quantforge.execution.brokers.paper import PaperBrokerAdapter

        b = PaperBrokerAdapter(timeout_probability=1.0, seed=1)
        b.set_market("TEST", ohlcv)
        b.connect()
        o = order_book.new_order("TEST", OrderSide.LONG, 10)
        with pytest.raises(BrokerTimeoutError, match="Do not resubmit"):
            b.submit_order(o)
        assert b.get_order(o.client_order_id)["status"] == "unknown"

    def test_unmapped_broker_state_is_escalated_not_assumed(self, order_book, ohlcv):
        """An unrecognised status must not be treated as 'accepted'."""
        from quantforge.execution.brokers.paper import PaperBrokerAdapter
        from quantforge.execution.reconciliation import PositionReconciler

        b = PaperBrokerAdapter(timeout_probability=1.0, seed=1)
        b.set_market("TEST", ohlcv)
        b.connect()
        o = order_book.new_order("TEST", OrderSide.LONG, 10)
        with pytest.raises(BrokerTimeoutError):
            b.submit_order(o)
        order_book.transition(o.order_id, OrderStatus.SUBMITTED)
        order_book.transition(o.order_id, OrderStatus.UNKNOWN)
        report = PositionReconciler().reconcile_unknown_orders(b, order_book)
        assert len(report["unresolved"]) == 1
        assert "escalate" in str(report["unresolved"][0]).lower() or \
               "unmapped" in str(report["unresolved"][0]).lower()
        # The order must still be UNKNOWN, not silently accepted.
        assert order_book.get(o.order_id).status == OrderStatus.UNKNOWN

    def test_disconnected_broker_refuses_submission(self, ohlcv):
        from quantforge.core.errors import BrokerConnectionError
        from quantforge.execution.brokers.paper import PaperBrokerAdapter
        from quantforge.execution.orders import Order

        b = PaperBrokerAdapter()
        b.set_market("TEST", ohlcv)      # not connected
        o = Order(order_id="x", client_order_id="qf-x", instrument_id="TEST",
                  side=OrderSide.LONG, quantity=10)
        with pytest.raises(BrokerConnectionError, match="reconciliation"):
            b.submit_order(o)

    def test_idempotency_key_is_required(self, ohlcv):
        from quantforge.core.errors import BrokerError
        from quantforge.execution.brokers.paper import PaperBrokerAdapter
        from quantforge.execution.orders import Order

        b = PaperBrokerAdapter(require_idempotency_key=True)
        b.set_market("TEST", ohlcv)
        b.connect()
        o = Order(order_id="x", client_order_id="qf-x", instrument_id="TEST",
                  side=OrderSide.LONG, quantity=10)
        o.idempotency_key = None
        with pytest.raises(BrokerError, match="idempotency"):
            b.submit_order(o)

    def test_paper_mode_is_never_live(self, paper_broker):
        assert paper_broker.mode == ExecutionMode.PAPER


# ==========================================================================
# Reconciliation
# ==========================================================================


class TestReconciliation:
    def test_exact_match(self, portfolio):
        from quantforge.execution.brokers.base import BrokerPosition
        from quantforge.execution.reconciliation import PositionReconciler

        portfolio.apply_fill("AAPL", OrderSide.LONG, 100, 100.0)
        r = PositionReconciler().reconcile_portfolio(
            portfolio, [BrokerPosition("AAPL", 100, 100.0)]
        )
        assert r.matched
        assert r.state == TradingState.NORMAL

    def test_unknown_position_at_the_broker_is_critical(self, portfolio):
        from quantforge.execution.brokers.base import BrokerPosition
        from quantforge.execution.reconciliation import PositionReconciler

        portfolio.apply_fill("AAPL", OrderSide.LONG, 100, 100.0)
        r = PositionReconciler().reconcile_portfolio(
            portfolio, [BrokerPosition("AAPL", 100, 100.0),
                        BrokerPosition("TSLA", 10, 200.0)]
        )
        assert not r.matched
        assert r.state == TradingState.REQUIRES_RECONCILIATION
        tsla = [d for d in r.discrepancies if d.instrument_id == "TSLA"][0]
        assert "the platform does not know about" in tsla.explanation

    def test_corrections_are_explicit_not_automatic(self, portfolio):
        from quantforge.execution.brokers.base import BrokerPosition
        from quantforge.execution.reconciliation import PositionReconciler

        portfolio.apply_fill("AAPL", OrderSide.LONG, 100, 100.0)
        broker = [BrokerPosition("AAPL", 50, 100.0)]
        rec = PositionReconciler()
        r = rec.reconcile_portfolio(portfolio, broker)
        assert not r.matched
        # Internal state is untouched until corrections are requested.
        assert portfolio.positions["AAPL"].quantity == 100
        applied = rec.apply_corrections(portfolio, r)
        assert applied and applied[0]["previous_quantity"] == 100
        assert portfolio.positions["AAPL"].quantity == 50
        assert rec.reconcile_portfolio(portfolio, broker).matched

    def test_price_difference_alone_is_a_warning(self, portfolio):
        from quantforge.execution.brokers.base import BrokerPosition
        from quantforge.execution.reconciliation import PositionReconciler

        portfolio.apply_fill("AAPL", OrderSide.LONG, 100, 100.0)
        r = PositionReconciler().reconcile_portfolio(
            portfolio, [BrokerPosition("AAPL", 100, 108.0)]
        )
        # Quantities agree, so trading is not blocked.
        assert r.matched


# ==========================================================================
# Live trading gate
# ==========================================================================


class TestLiveTradingGate:
    @pytest.fixture
    def gate(self):
        from quantforge.execution.live_gate import LiveTradingGate

        return LiveTradingGate(
            auth_path=Path(tempfile.mkdtemp()) / "auth.json", secret=b"a" * 40
        )

    def test_refused_when_no_file_exists(self, gate):
        ok, why = gate.check()
        assert not ok
        assert "no live-trading authorisation" in why

    def test_refused_when_preconditions_missing(self, gate):
        from quantforge.execution.live_gate import build_authorisation

        a = build_authorisation("alice", confirmed=[])
        a.sign(b"a" * 40)
        a.save(gate.auth_path)
        ok, why = gate.check()
        assert not ok and "missing required confirmations" in why

    def test_permitted_with_full_confirmation(self, gate):
        from quantforge.execution.live_gate import (
            AUTHORISATION_CHECKLIST, build_authorisation,
        )

        a = build_authorisation("alice",
                                confirmed=[k for k, _ in AUTHORISATION_CHECKLIST])
        a.sign(b"a" * 40)
        a.save(gate.auth_path)
        ok, why = gate.check()
        assert ok, why

    def test_refused_with_the_wrong_secret(self, gate):
        from quantforge.execution.live_gate import (
            AUTHORISATION_CHECKLIST, LiveTradingGate, build_authorisation,
        )

        a = build_authorisation("alice",
                                confirmed=[k for k, _ in AUTHORISATION_CHECKLIST])
        a.sign(b"a" * 40)
        a.save(gate.auth_path)
        other = LiveTradingGate(auth_path=gate.auth_path, secret=b"b" * 40)
        ok, why = other.check()
        assert not ok and "signature is invalid" in why

    def test_refused_after_tampering(self, gate):
        import json

        from quantforge.execution.live_gate import (
            AUTHORISATION_CHECKLIST, build_authorisation,
        )

        a = build_authorisation("alice", portfolio_id="PF1",
                                confirmed=[k for k, _ in AUTHORISATION_CHECKLIST])
        a.sign(b"a" * 40)
        a.save(gate.auth_path)
        data = json.loads(gate.auth_path.read_text())
        data["portfolio_id"] = "PF_EVIL"
        gate.auth_path.chmod(0o600)
        gate.auth_path.write_text(json.dumps(data))
        ok, why = gate.check()
        assert not ok and "signature is invalid" in why

    def test_refused_when_expired(self, gate):
        from quantforge.execution.live_gate import (
            AUTHORISATION_CHECKLIST, build_authorisation,
        )

        a = build_authorisation("alice",
                                confirmed=[k for k, _ in AUTHORISATION_CHECKLIST])
        a.expires_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
        a.sign(b"a" * 40)
        a.save(gate.auth_path)
        ok, why = gate.check()
        assert not ok and "expired" in why

    def test_refused_when_the_running_config_differs(self, gate):
        from quantforge.execution.live_gate import (
            AUTHORISATION_CHECKLIST, build_authorisation,
        )

        a = build_authorisation("alice", strategy_version="2.0",
                                confirmed=[k for k, _ in AUTHORISATION_CHECKLIST])
        a.sign(b"a" * 40)
        a.save(gate.auth_path)
        assert gate.check(expected_strategy_version="3.0")[0] is False
        assert gate.check(expected_strategy_version="2.0")[0] is True

    def test_require_raises_when_refused(self, gate):
        with pytest.raises(LiveTradingNotAuthorized):
            gate.require()

    def test_file_is_written_read_only(self, gate):
        from quantforge.execution.live_gate import (
            AUTHORISATION_CHECKLIST, build_authorisation,
        )

        a = build_authorisation("alice",
                                confirmed=[k for k, _ in AUTHORISATION_CHECKLIST])
        a.sign(b"a" * 40)
        a.save(gate.auth_path)
        assert gate.auth_path.stat().st_mode & 0o222 == 0


# ==========================================================================
# Security
# ==========================================================================


class TestAuthorization:
    def test_researcher_cannot_submit_live_orders(self, users):
        from quantforge.core.enums import Permission
        from quantforge.security.access import require_permission

        with pytest.raises(AuthorizationError, match="CAN_SUBMIT_LIVE_ORDER"):
            require_permission(users["researcher"], Permission.CAN_SUBMIT_LIVE_ORDER)

    def test_live_trading_requires_mfa(self, users):
        from quantforge.core.enums import Role
        from quantforge.security.access import User, require_live_trading_authority

        no_mfa = User("x", "pm", frozenset({Role.PORTFOLIO_MANAGER}), mfa_enabled=False)
        with pytest.raises(AuthorizationError, match="MFA"):
            require_live_trading_authority(no_mfa)

    def test_permission_check_fails_closed_on_error(self, users):
        from quantforge.core.enums import Permission
        from quantforge.security import access

        class BrokenUser:
            username = "broken"

            def has(self, _permission):
                raise RuntimeError("authorisation store unavailable")

        with pytest.raises(AuthorizationError, match="denying by default"):
            access.require_permission(BrokenUser(), Permission.CAN_RESEARCH)

    def test_disabled_user_has_no_permissions(self):
        from quantforge.core.enums import Permission, Role
        from quantforge.security.access import User

        u = User("x", "admin", frozenset({Role.ADMINISTRATOR}), disabled=True)
        assert not u.has(Permission.CAN_SUBMIT_LIVE_ORDER)

    def test_short_password_is_rejected(self):
        from quantforge.security.access import PasswordHasher

        with pytest.raises(AuthenticationError, match="12 characters"):
            PasswordHasher().hash("short")

    def test_password_hash_verifies_and_rejects(self):
        from quantforge.security.access import PasswordHasher

        ph = PasswordHasher()
        h = ph.hash("correct-horse-battery")
        assert ph.verify("correct-horse-battery", h)
        assert not ph.verify("wrong-password-here", h)

    def test_short_token_secret_is_rejected(self):
        from quantforge.security.access import TokenService

        with pytest.raises(AuthenticationError, match="32 bytes"):
            TokenService(b"tooshort")

    def test_forged_token_is_rejected(self):
        from quantforge.core.enums import Role
        from quantforge.security.access import TokenService, User

        ts = TokenService(b"k" * 40)
        u = User("u1", "alice", frozenset({Role.RESEARCHER}))
        token, payload = ts.issue(u)
        assert ts.verify(token).subject == "u1"
        with pytest.raises(AuthenticationError, match="signature"):
            ts.verify(token[:-4] + "0000")
        ts.revoke(payload.token_id)
        with pytest.raises(AuthenticationError, match="revoked"):
            ts.verify(token)

    def test_rate_limit_and_lockout(self):
        from quantforge.security.access import RateLimiter

        rl = RateLimiter()
        allowed = [rl.check("u1", "order_submit").allowed for _ in range(15)]
        assert sum(allowed) == 13          # 10/min + 3 burst
        for _ in range(6):
            rl.record_auth_failure("evil")
        assert rl.is_locked_out("evil")
        rl.clear_auth_failures("evil")
        assert not rl.is_locked_out("evil")


class TestAuditLog:
    def test_chain_detects_tampering(self):
        from quantforge.security.audit import AuditLog

        log = AuditLog(secret=b"k" * 40)
        log.record("alice", AuditAction.STRATEGY_CREATED, "s/1", {"a": 1})
        log.record("alice", AuditAction.BACKTEST_CREATED, "bt/1", {"sharpe": 1.2})
        assert log.verify_chain()["valid"]
        log._entries[1].detail = {"sharpe": 99.0}
        v = log.verify_chain()
        assert not v["valid"] and v["broken_at_sequence"] == 2

    def test_entries_are_queryable(self):
        from quantforge.security.audit import AuditLog

        log = AuditLog()
        log.record("alice", AuditAction.STRATEGY_CREATED, "s/1")
        log.record("risk", AuditAction.RISK_LIMITS_CHANGED, "l/1")
        assert [e["action"] for e in log.entries(actor="risk")] == \
            ["risk_limits_changed"]

    def test_writes_to_disk(self):
        from quantforge.security.audit import AuditLog

        p = Path(tempfile.mkdtemp()) / "audit.jsonl"
        log = AuditLog(path=p)
        log.record("alice", AuditAction.USER_LOGIN)
        assert p.exists() and p.read_text().strip()


class TestSandbox:
    @pytest.fixture
    def sandbox(self):
        from quantforge.security.audit import StrategySandbox

        return StrategySandbox()

    @pytest.mark.parametrize("source", [
        "import socket\ndef f(): pass",
        "from subprocess import run\ndef f(): pass",
        "import os\ndef f():\n    os.system('ls')",
        "def f():\n    return eval('1+1')",
        "from sqlalchemy import create_engine\ndef f(): pass",
        "import pickle\ndef f(): pass",
    ])
    def test_dangerous_source_is_blocked(self, sandbox, source):
        with pytest.raises(SandboxViolation):
            sandbox.check_source(source)

    def test_legitimate_quant_code_is_allowed(self, sandbox):
        src = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "def handle(frame):\n"
            "    return np.mean(frame['close'])\n"
        )
        assert sandbox.check_source(src)["ok"]

    def test_syntax_error_is_reported(self, sandbox):
        with pytest.raises(SandboxViolation, match="syntax error"):
            sandbox.check_source("def f(:\n  pass")

    def test_describe_discloses_it_is_not_a_boundary(self, sandbox):
        assert "not a security boundary" in sandbox.describe()["boundary_note"]


# ==========================================================================
# Statistics
# ==========================================================================


class TestStatistics:
    def test_constant_series_correlation_is_nan(self):
        from quantforge.quant.statistics import correlation

        a = pd.Series(np.random.default_rng(1).normal(0, 1, 100))
        assert np.isnan(correlation(pd.Series([1.0] * 100), a))

    def test_constant_variance_is_zero_not_nan(self):
        from quantforge.quant.statistics import variance

        assert variance(pd.Series([3.0] * 10)) == 0.0

    def test_too_few_observations_is_nan(self):
        from quantforge.quant.statistics import std, variance

        assert np.isnan(variance(pd.Series([1.0])))
        assert np.isnan(std(pd.Series([1.0])))

    def test_downside_deviation_conventions_differ(self):
        from quantforge.quant.statistics import downside_deviation

        r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 1000))
        sqrt = downside_deviation(r, kind="sqrt")
        simple = downside_deviation(r, kind="simple")
        assert sqrt > simple > 0

    def test_beta_of_a_constant_benchmark_is_nan(self):
        from quantforge.quant.statistics import beta

        a = pd.Series(np.random.default_rng(3).normal(0, 0.01, 100))
        assert np.isnan(beta(a, pd.Series([0.01] * 100)))

    def test_shrinkage_fixes_a_rank_deficient_covariance(self):
        from quantforge.quant.statistics import shrinkage_covariance

        rng = np.random.default_rng(4)
        X = pd.DataFrame(rng.normal(0, 0.01, (250, 100)),
                         columns=[f"A{i}" for i in range(100)])
        cov = shrinkage_covariance(X)
        assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0

    def test_shrinkage_refuses_rank_deficient_input(self):
        from quantforge.quant.statistics import shrinkage_covariance

        rng = np.random.default_rng(5)
        X = pd.DataFrame(rng.normal(0, 0.01, (30, 100)),
                         columns=[f"A{i}" for i in range(100)])
        with pytest.raises(ValueError, match="rank-deficient"):
            shrinkage_covariance(X)

    def test_variance_ratio_test_is_calibrated(self):
        from quantforge.quant.statistics import variance_ratio_test

        rng = np.random.default_rng(6)
        rejects = sum(
            variance_ratio_test(rng.normal(0, 0.01, 1000), 5, n_bootstrap=1000)
            ["p_value"] < 0.05
            for _ in range(20)
        )
        assert rejects <= 4, f"{rejects}/20 false positives on iid data"

    def test_variance_ratio_detects_dependence(self):
        from quantforge.quant.statistics import variance_ratio_test

        rng = np.random.default_rng(7)
        ar = np.zeros(2000)
        for i in range(1, 2000):
            ar[i] = 0.4 * ar[i - 1] + rng.normal(0, 0.01)
        r = variance_ratio_test(pd.Series(ar), 5, n_bootstrap=2000)
        assert r["variance_ratio"] > 1.0        # positive autocorrelation
        assert r["verdict"] == "reject random walk"


# ==========================================================================
# Validation
# ==========================================================================


class TestSplits:
    def test_windows_do_not_overlap(self):
        from quantforge.validation.splits import temporal_split

        idx = pd.date_range("2018-01-01", periods=1200, freq="B", tz="UTC")
        s = temporal_split(idx, 0.6, 0.2)
        assert s.train.end < s.validation.start
        assert s.validation.end < s.test.start

    def test_a_test_window_is_mandatory(self):
        from quantforge.core.errors import ValidationError
        from quantforge.validation.splits import temporal_split

        idx = pd.date_range("2018-01-01", periods=1000, freq="B", tz="UTC")
        with pytest.raises(ValidationError, match="no test period"):
            temporal_split(idx, 0.6, 0.4)

    def test_embargo_creates_a_gap(self):
        from quantforge.validation.splits import temporal_split

        idx = pd.date_range("2018-01-01", periods=1200, freq="B", tz="UTC")
        s = temporal_split(idx, 0.6, 0.2, embargo_bars=5)
        gap = (s.validation.start - s.train.end).days
        assert gap >= 5

    def test_oversized_embargo_is_rejected(self):
        from quantforge.core.errors import ValidationError
        from quantforge.validation.splits import temporal_split

        idx = pd.date_range("2018-01-01", periods=1000, freq="B", tz="UTC")
        with pytest.raises(ValidationError):
            temporal_split(idx, 0.5, 0.2, embargo_bars=400)


class TestWalkForward:
    def test_produces_windows_and_an_aggregate(self):
        from quantforge.validation.splits import walk_forward

        idx = pd.date_range("2018-01-01", periods=1200, freq="B", tz="UTC")
        rng = np.random.default_rng(8)

        def fit(tr, va, params):
            return pd.Series(rng.normal(0.0002, 0.01, len(va)), index=va)

        r = walk_forward(idx, fit, train_bars=400, validation_bars=100,
                         test_bars=100, step=100)
        assert len(r.windows) > 1
        assert r.failed_windows == 0
        assert r.aggregate["n_periods"] > 0

    def test_insufficient_data_raises(self):
        from quantforge.core.errors import ValidationError
        from quantforge.validation.splits import walk_forward

        idx = pd.date_range("2018-01-01", periods=100, freq="B", tz="UTC")
        with pytest.raises(ValidationError, match="at least"):
            walk_forward(idx, lambda a, b, c: None, train_bars=500,
                         validation_bars=100, test_bars=100)


class TestMonteCarlo:
    def test_bootstrap_of_a_zero_drift_series_is_symmetric(self):
        """Zero-mean returns must resample to a near-coin-flip loss rate."""
        from quantforge.validation.splits import monte_carlo_returns

        rng = np.random.default_rng(9)
        r = pd.Series(rng.normal(0.0, 0.01, 1000))
        mc = monte_carlo_returns(r, 2000, "bootstrap", seed=1)
        assert 0.42 < mc.probability_of_loss < 0.58
        assert mc.statistics["median_final_return"] == pytest.approx(0.0, abs=0.15)

    def test_bootstrap_preserves_a_positive_drift(self):
        from quantforge.validation.splits import monte_carlo_returns

        rng = np.random.default_rng(9)
        r = pd.Series(rng.normal(0.0002, 0.01, 1000))
        mc = monte_carlo_returns(r, 2000, "bootstrap", seed=1)
        # A positive per-period drift compounds to a positive total, and the
        # probability of loss must be materially below one half.
        assert mc.statistics["median_final_return"] > 0
        assert mc.probability_of_loss < 0.45

    def test_bootstrap_cannot_invent_a_worse_single_period(self):
        """A documented limitation, asserted so it is not forgotten.

        The iid bootstrap draws from observed returns, so no resampled *period*
        can be worse than the worst one on record. Compounded multi-period
        totals obviously can be, which is why this tests the resampled draws
        rather than the cumulative result.
        """
        from quantforge.validation.splits import monte_carlo_returns

        rng = np.random.default_rng(10)
        r = pd.Series(rng.normal(0, 0.02, 500))
        worst_period = float(r.min())
        mc = monte_carlo_returns(r, 3000, "bootstrap", seed=2, horizon=1)
        # horizon=1: each simulated path is a single resampled period.
        assert mc.statistics["worst_final_return"] >= worst_period - 1e-12
        assert any("cannot generate a loss beyond" in w for w in mc.warnings)

    def test_short_sample_refuses_to_simulate(self):
        from quantforge.validation.splits import monte_carlo_returns

        mc = monte_carlo_returns(pd.Series([0.01, -0.01] * 4), 100)
        assert mc.n_simulations == 0
        assert any("insufficient" in w.lower() or "minimum" in w.lower()
                   for w in mc.warnings)


class TestOverfitting:
    def test_noise_is_not_significant_after_selection_bias(self):
        from quantforge.validation.overfitting import deflated_sharpe_ratio

        rng = np.random.default_rng(11)
        noise = pd.Series(rng.normal(0, 0.01, 1000))
        d = deflated_sharpe_ratio(noise, n_trials=1000)
        assert d["deflated_sharpe"] < 0
        assert not d["significant_at_95"]

    def test_single_trial_applies_no_adjustment(self):
        from quantforge.validation.overfitting import deflated_sharpe_ratio

        rng = np.random.default_rng(12)
        r = pd.Series(rng.normal(0.0008, 0.01, 1000))
        d = deflated_sharpe_ratio(r, n_trials=1)
        assert d["expected_max_sharpe_under_null"] == 0.0
        assert d["deflated_sharpe"] == pytest.approx(d["observed_sharpe"])

    def test_expected_max_is_monotonic_in_trials(self):
        from quantforge.validation.overfitting import _expected_max_sharpe

        vals = [_expected_max_sharpe(n, 1.0) for n in range(1, 40)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_expected_max_scales_with_trial_dispersion(self):
        from quantforge.validation.overfitting import _expected_max_sharpe

        assert _expected_max_sharpe(100, 2.0) > _expected_max_sharpe(100, 1.0)

    def test_flags_fire_on_a_suspicious_result(self):
        from quantforge.validation.overfitting import analyse

        r = analyse(n_parameters=25, n_trades=18,
                    in_sample={"sharpe": 4.2}, out_of_sample={"sharpe": -0.3})
        names = {f.name for f in r.flags}
        assert "few_trades" in names
        assert "parameter_complexity" in names
        assert "extreme_in_sample_sharpe" in names
        assert "oos_degradation" in names

    def test_effective_parameter_count_counts_configurations(self):
        from quantforge.validation.overfitting import parameter_count_effective

        n = parameter_count_effective({"fast": [5, 10, 20, 40, 80],
                                       "slow": [20, 30, 50, 100, 200]})
        assert n == 25      # configurations, not the 2 varied numbers

    def test_report_never_declares_robustness(self):
        from quantforge.validation.overfitting import analyse

        r = analyse(n_parameters=1, n_trades=500, in_sample={"sharpe": 0.5},
                    out_of_sample={"sharpe": 0.5})
        assert "not that the strategy is robust" in r.summary()


class TestSensitivity:
    def test_narrow_peak_is_flagged(self):
        from quantforge.validation.splits import sensitivity_analysis

        def ev(p):
            # A sharp spike only at the centre value.
            score = 2.0 if p["x"] == 10 else 0.1
            return {"sharpe": score, "n_trades": 100}

        _, stability = sensitivity_analysis(
            ev, {"x": 10}, {"x": [1, 5, 10, 15, 20]}, "sharpe", 1
        )
        assert stability["x"]["is_plateau"] is False
        assert "NARROW PEAK" in stability["x"]["verdict"]

    def test_plateau_is_accepted(self):
        from quantforge.validation.splits import sensitivity_analysis

        def ev(p):
            return {"sharpe": 1.0, "n_trades": 100}

        _, stability = sensitivity_analysis(
            ev, {"x": 10}, {"x": [1, 5, 10, 15, 20]}, "sharpe", 1
        )
        assert stability["x"]["is_plateau"] is True


class TestGridSearch:
    def test_min_trades_filters_the_optimum(self):
        """A 2-trade Sharpe of 4 is not a finding."""
        from quantforge.validation.splits import grid_search, parameter_grid

        def ev(p):
            if p["k"] == 1:
                return {"sharpe": 4.0, "n_trades": 2}
            return {"sharpe": 1.0, "n_trades": 200}

        best, table = grid_search(ev, parameter_grid(k=[1, 2]), "sharpe",
                                 min_trades=30)
        assert best["k"] == 2
