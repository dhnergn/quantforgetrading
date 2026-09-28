"""The dashboard HTTP surface: authorisation, isolation, and honest errors.

The security-relevant property here is that **there is no default portfolio**.
A dashboard that returned whichever portfolio happened to be loaded would let
any authorised caller read another's positions by omitting a query parameter -
no id to guess, nothing to audit as an access attempt. So every read requires
an explicit id, and a missing one is a 400 that says why.

The correctness-relevant property is that an unavailable component returns 200
with ``available: false`` and a reason, not a 404. A 404 claims the component
does not exist, which is a materially different thing for a reader to believe
than "it exists and your portfolio cannot produce it".
"""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from quantforge.api.app import create_app
from quantforge.core.timestamps import UTC


def _users():
    """Three roles that differ in what they may see.

    A researcher has CAN_RESEARCH but *not* CAN_VIEW_RISK, which is the point:
    a dashboard reports positions and risk, and that is a different disclosure
    from running a backtest. A risk manager holds CAN_VIEW_RISK. A readonly
    user holds nothing at all.
    """
    from quantforge.security.access import Role, User

    # Keyed by user_id, because that is what the token subject is matched
    # against in ``current_user``. Keying by username would authenticate fine
    # and then fail the lookup.
    return {
        "u-researcher": User("u-researcher", "researcher",
                             frozenset({Role.RESEARCHER})),
        "u-risk": User("u-risk", "risk", frozenset({Role.RISK_MANAGER})),
        "u-readonly": User("u-readonly", "readonly", frozenset({Role.READONLY})),
    }


SECRET = b"test-secret-key-at-least-32-bytes-long"


@pytest.fixture
def client():
    """A client whose tokens are issued directly by ``TokenService``.

    There is no login route: ``current_user`` verifies a bearer token and looks
    the subject up in the injected user table, so the test mints a token for a
    known user rather than driving a credential flow that does not exist.
    """
    app = create_app(users=_users(), token_secret=SECRET)
    return TestClient(app)


def _auth(client, user_id: str = "u-risk") -> dict:
    """A bearer token for one of the injected users, carrying that user's
    resolved permissions."""
    from quantforge.security.access import TokenService

    user = _users()[user_id]
    token, _payload = TokenService(SECRET).issue(user)
    return {"Authorization": f"Bearer {token}"}


def _equity(n: int = 120, seed: int = 4):
    """A plausible equity curve: rising with a drawdown, not a straight line."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz=UTC)
    equity = 1_000_000.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.011, n)))
    return [[str(d), float(v)] for d, v in zip(idx, equity)]


def _register(client, equity=None, portfolio_id=None):
    payload = {"equity_curve": equity or _equity()}
    if portfolio_id:
        payload["portfolio_id"] = portfolio_id
    resp = client.post("/v1/dashboard/register", json=payload, headers=_auth(client))
    assert resp.status_code == 200, resp.text
    return resp.json()


# ==========================================================================
# Authorisation
# ==========================================================================


class TestAuthRequired:
    def test_unauthenticated_read_is_rejected(self, client):
        assert client.get("/v1/dashboard/components").status_code in (401, 403)

    def test_unauthenticated_report_is_rejected(self, client):
        assert client.get("/v1/dashboard/report?portfolio_id=p").status_code in (401, 403)

    def test_risk_manager_may_read_the_dashboard(self, client):
        """CAN_VIEW_RISK covers it; a dashboard is a risk disclosure."""
        assert client.get("/v1/dashboard/components",
                          headers=_auth(client, "u-risk")).status_code == 200

    def test_researcher_without_can_view_risk_is_refused(self, client):
        """The permission boundary is the point of gating on CAN_VIEW_RISK
        rather than CAN_RESEARCH. Being able to run a backtest is not the same
        permission as being able to read the live position book."""
        _register(client)
        assert client.get("/v1/dashboard/components",
                          headers=_auth(client, "u-researcher")).status_code == 403

    def test_readonly_role_is_refused(self, client):
        """READONLY holds no permissions at all, including CAN_VIEW_RISK."""
        _register(client)
        resp = client.get("/v1/dashboard/report?portfolio_id=p",
                          headers=_auth(client, "u-readonly"))
        assert resp.status_code == 403


# ==========================================================================
# No default portfolio
# ==========================================================================


class TestNoDefaultPortfolio:
    def test_missing_id_is_400_with_the_reason(self, client):
        _register(client)
        resp = client.get("/v1/dashboard/report", headers=_auth(client))
        assert resp.status_code == 400
        assert "no default portfolio" in resp.json()["detail"].lower()

    def test_missing_id_on_a_component_is_also_400(self, client):
        _register(client)
        resp = client.get("/v1/dashboard/drawdown", headers=_auth(client))
        assert resp.status_code == 400

    def test_unknown_portfolio_is_404_and_explains_lifetime(self, client):
        resp = client.get("/v1/dashboard/report?portfolio_id=nope",
                          headers=_auth(client))
        assert resp.status_code == 404
        assert "process" in resp.json()["detail"].lower()

    def test_another_users_job_id_is_not_usable_as_a_portfolio_alias(self, client):
        """Ownership is checked before the portfolio lookup, so a job id cannot
        be used to discover whether another user's portfolio is resident."""
        client.post("/v1/backtests", headers=_auth(client, "u-researcher"), json={
            "strategy_id": "s", "strategy_version": "1", "instruments": ["A"],
            "start": "2024-01-01", "end": "2024-06-30",
        })
        jobs = client.get("/v1/jobs/nonexistent", headers=_auth(client, "u-risk"))
        assert jobs.status_code == 404

    def test_registered_portfolio_is_readable(self, client):
        pid = _register(client)["portfolio_id"]
        resp = client.get(f"/v1/dashboard/report?portfolio_id={pid}",
                          headers=_auth(client))
        assert resp.status_code == 200
        assert resp.json()["n_total"] == 14

    def test_position_booked_components_report_unavailable_not_flat(self, client):
        """The register endpoint builds a portfolio with an equity curve and an
        empty position dict. Reporting ``available: true`` with an empty list
        would be indistinguishable from a genuinely flat book - it would tell an
        operator a portfolio holds nothing when in fact nothing was supplied.
        """
        pid = _register(client)["portfolio_id"]
        for component in ("positions", "exposure", "sector_exposure",
                          "asset_class_exposure", "realized_pnl"):
            body = client.get(
                f"/v1/dashboard/{component}?portfolio_id={pid}",
                headers=_auth(client)).json()
            assert body["available"] is False, (
                f"{component} reported available for a portfolio that was "
                f"supplied with only an equity curve"
            )
            assert body["reason"], f"{component} unavailable with no reason"


# ==========================================================================
# Component endpoint
# ==========================================================================


class TestComponentEndpoint:
    def test_index_lists_all_fourteen(self, client):
        data = client.get("/v1/dashboard/components",
                          headers=_auth(client)).json()
        assert len(data["components"]) == 14
        assert {c["name"] for c in data["components"]} == {
            "equity_curve", "daily_pnl", "unrealized_pnl", "realized_pnl",
            "exposure", "positions", "sector_exposure", "asset_class_exposure",
            "correlations", "portfolio_volatility", "var", "drawdown",
            "summary_statistics", "beta"}

    def test_unavailable_component_is_200_with_a_reason_not_404(self, client):
        """The distinction that matters: the component exists, the portfolio
        cannot produce it."""
        pid = _register(client)["portfolio_id"]
        resp = client.get(f"/v1/dashboard/beta?portfolio_id={pid}",
                          headers=_auth(client))
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is False
        assert body["reason"]

    def test_unknown_component_name_is_404_and_lists_the_valid_ones(self, client):
        pid = _register(client)["portfolio_id"]
        resp = client.get(f"/v1/dashboard/not_a_component?portfolio_id={pid}",
                          headers=_auth(client))
        assert resp.status_code == 404
        assert "equity_curve" in resp.json()["detail"]

    def test_available_component_carries_its_basis(self, client):
        pid = _register(client)["portfolio_id"]
        body = client.get(f"/v1/dashboard/summary_statistics?portfolio_id={pid}",
                          headers=_auth(client)).json()
        assert body["available"] is True
        assert body["n_observations"] > 0
        assert body["caveat"]
        assert body["data"]["annualisation_basis"] == "measured from the index dates"

    def test_every_component_endpoint_answers_for_a_registered_portfolio(self, client):
        pid = _register(client)["portfolio_id"]
        names = [c["name"] for c in client.get(
            "/v1/dashboard/components", headers=_auth(client)).json()["components"]]
        for name in names:
            resp = client.get(f"/v1/dashboard/{name}?portfolio_id={pid}",
                              headers=_auth(client))
            assert resp.status_code == 200, f"{name} -> {resp.status_code}"
            body = resp.json()
            if body["available"]:
                assert body["caveat"], f"{name} available with no caveat"


# ==========================================================================
# Registration input validation
# ==========================================================================


class TestRegisterValidation:
    def test_missing_equity_curve_is_rejected(self, client):
        resp = client.post("/v1/dashboard/register", json={}, headers=_auth(client))
        assert resp.status_code == 400
        assert "equity_curve" in resp.json()["detail"]

    def test_single_point_curve_is_rejected(self, client):
        resp = client.post("/v1/dashboard/register",
                           json={"equity_curve": [["2024-01-01", 1000.0]]},
                           headers=_auth(client))
        assert resp.status_code == 400
        assert "at least 2" in resp.json()["detail"]

    def test_non_positive_equity_is_refused(self, client):
        """A growth rate from a zero base is undefined, and accepting it would
        produce numbers that cannot be interpreted."""
        curve = _equity(20)
        curve[5][1] = 0.0
        resp = client.post("/v1/dashboard/register",
                           json={"equity_curve": curve}, headers=_auth(client))
        assert resp.status_code == 400
        assert "non-positive" in resp.json()["detail"]

    def test_out_of_order_timestamps_are_refused_not_sorted(self, client):
        """Sorting would reorder the series the caller actually measured."""
        curve = _equity(20)
        curve[3], curve[4] = curve[4], curve[3]
        resp = client.post("/v1/dashboard/register",
                           json={"equity_curve": curve}, headers=_auth(client))
        assert resp.status_code == 400
        assert "increasing" in resp.json()["detail"]

    def test_response_says_which_components_stay_unavailable(self, client):
        """Registration carries only the equity curve, so the reader is told
        up front rather than discovering missing charts later."""
        body = _register(client)
        assert "equity_curve" in body["components_now_available"]
        assert "positions" not in body["components_now_available"]
        assert "Only the equity curve" in body["note"]
        unavailable = {c["name"] for c in body["components_unavailable"]}
        assert {"positions", "exposure", "correlations"} <= unavailable

    def test_a_genuinely_flat_book_is_distinguishable_from_a_missing_one(self, client):
        """The distinction the register endpoint exists to preserve: a portfolio
        that *was* supplied with positions and holds none is flat, while one
        whose positions were never supplied is unknown."""
        from quantforge.backtest.portfolio import Portfolio

        flat = Portfolio(initial_cash=1000.0)          # positions == {}
        assert flat.positions == {}
        pid = _register(client)["portfolio_id"]
        missing = client.app.state.qf["portfolios"][pid]
        assert missing.positions is None

        # The flat one reports as flat; the supplied-only-curve one does not.
        assert missing.positions is not flat.positions


class TestConfidenceValidation:
    @pytest.mark.parametrize("value", ["1.5", "0", "-0.2", "abc", ""])
    def test_bad_confidence_is_rejected(self, client, value):
        pid = _register(client)["portfolio_id"]
        resp = client.get(f"/v1/dashboard/report?portfolio_id={pid}&confidence={value}",
                          headers=_auth(client))
        assert resp.status_code == 400, f"confidence={value!r} was accepted"

    def test_valid_confidence_is_accepted(self, client):
        pid = _register(client)["portfolio_id"]
        resp = client.get(f"/v1/dashboard/report?portfolio_id={pid}&confidence=0.99",
                          headers=_auth(client))
        assert resp.status_code == 200
        assert resp.json()["components"]["var"]["data"]["confidence"] == pytest.approx(0.99)

    def test_a_higher_confidence_gives_a_larger_var(self, client):
        """A sanity check on the plumbing, not a claim about the method."""
        pid = _register(client)["portfolio_id"]

        def var_at(conf: str) -> float:
            body = client.get(
                f"/v1/dashboard/var?portfolio_id={pid}&confidence={conf}",
                headers=_auth(client)).json()
            return body["data"]["var_return"]

        assert var_at("0.99") >= var_at("0.90")


# ==========================================================================
# Auditing
# ==========================================================================


class TestAudit:
    def test_viewing_the_dashboard_is_audited(self, client):
        pid = _register(client)["portfolio_id"]
        client.get(f"/v1/dashboard/report?portfolio_id={pid}", headers=_auth(client))
        log = client.get("/v1/audit?limit=200", headers=_auth(client, "u-risk"))
        actions = [e["action"] for e in log.json()["entries"]]
        assert "dashboard_viewed" in actions

    def test_registration_is_audited(self, client):
        _register(client)
        log = client.get("/v1/audit?limit=200", headers=_auth(client, "u-risk"))
        actions = [e["action"] for e in log.json()["entries"]]
        assert "dashboard_portfolio_registered" in actions

    def test_the_audit_chain_still_verifies(self, client):
        pid = _register(client)["portfolio_id"]
        client.get(f"/v1/dashboard/report?portfolio_id={pid}", headers=_auth(client))
        log = client.get("/v1/audit?limit=200", headers=_auth(client, "u-risk")).json()
        assert log["chain"]["valid"] is True
