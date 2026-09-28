"""Kill switches and trading-state machine.

The kill switch is the last line of defence and the one component that must
work when everything else has failed. Design constraints:

* **Fails closed.** If the switch cannot determine its state, trading is
  blocked. An unreadable switch is treated as engaged, because the cost of a
  false halt is a missed trade and the cost of a false all-clear is real money.
* **Scope-aware.** Global, portfolio, strategy, and broker scopes, so a single
  failing strategy can be stopped without halting the book.
* **Auditable.** Every engagement and release is recorded with who, when, and
  why.
* **Not auto-releasable.** An automatic-triggered halt stays engaged until a
  human clears it. Silently resuming after a kill switch is how a system
  re-enters a market that just rejected it several times.
"""
from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from quantforge.core.enums import ExecutionMode, KillSwitchScope, TradingState
from quantforge.core.errors import KillSwitchEngaged
from quantforge.core.timestamps import utcnow


@dataclass(slots=True)
class KillSwitchConfig:
    """Automatic engagement thresholds.

    The defaults are conservative by design: they halt sooner than most
    strategies would like. A halt costs a missed trade; a runaway loss costs
    capital.
    """

    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    stale_data_seconds: float = 300.0
    max_execution_latency_ms: float = 2000.0
    max_order_rejections: int = 5
    reconciliation_failures: int = 3
    max_portfolio_volatility: float = 0.60
    auto_engage: bool = True
    #: Auto-engaged switches require manual release.
    require_manual_release: bool = True

    def as_dict(self) -> dict:
        return {
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "stale_data_seconds": self.stale_data_seconds,
            "max_execution_latency_ms": self.max_execution_latency_ms,
            "max_order_rejections": self.max_order_rejections,
            "reconciliation_failures": self.reconciliation_failures,
            "max_portfolio_volatility": self.max_portfolio_volatility,
            "auto_engage": self.auto_engage,
            "require_manual_release": self.require_manual_release,
        }


@dataclass(slots=True)
class KillSwitchEvent:
    scope: KillSwitchScope
    reason: str
    automatic: bool
    detail: dict = field(default_factory=dict)
    engaged_at: dt.datetime = field(default_factory=utcnow)
    engaged_by: str = "system"
    released_at: Optional[dt.datetime] = None
    released_by: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.released_at is None

    def as_dict(self) -> dict:
        return {
            "scope": self.scope.value,
            "reason": self.reason,
            "automatic": self.automatic,
            "detail": self.detail,
            "engaged_at": self.engaged_at.isoformat(),
            "engaged_by": self.engaged_by,
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "released_by": self.released_by,
            "active": self.active,
        }


class KillSwitch:
    """Hierarchical kill switches.

    A global engagement overrides everything. A portfolio engagement blocks
    only that portfolio. Strategy-scoped engagements block one strategy.
    """

    def __init__(self, config: Optional[KillSwitchConfig] = None) -> None:
        self.config = config or KillSwitchConfig()
        self._events: list[KillSwitchEvent] = []
        self._counter = itertools.count(1)
        #: Live counters for the automatic triggers.
        self.order_rejections = 0
        self.reconciliation_failures = 0
        self.max_latency_observed_ms = 0.0
        self.last_market_data_at: Optional[dt.datetime] = None
        self.audit_hook: Optional[Callable[[KillSwitchEvent], None]] = None

    # -- manual -----------------------------------------------------------

    def engage(
        self,
        scope: KillSwitchScope,
        reason: str,
        engaged_by: str = "operator",
        detail: Optional[dict] = None,
        automatic: bool = False,
    ) -> KillSwitchEvent:
        ev = KillSwitchEvent(scope=scope, reason=reason, automatic=automatic,
                             detail=detail or {}, engaged_by=engaged_by)
        self._events.append(ev)
        if self.audit_hook:
            self.audit_hook(ev)
        return ev

    def release(
        self, scope: KillSwitchScope, released_by: str, reason: str = "",
        target: Optional[KillSwitchEvent] = None,
    ) -> bool:
        if target is not None:
            if not target.active:
                return False
            target.released_at = utcnow()
            target.released_by = released_by
            if reason:
                target.detail["release_reason"] = reason
            return True
        released = False
        for ev in self._events:
            if ev.scope == scope and ev.active:
                if self.config.require_manual_release and ev.automatic:
                    # Still allowed, but the manual nature is recorded.
                    ev.detail["released_manually_after_auto"] = True
                ev.released_at = utcnow()
                ev.released_by = released_by
                if reason:
                    ev.detail["release_reason"] = reason
                released = True
        return released

    # -- state ------------------------------------------------------------

    @property
    def active_events(self) -> list[KillSwitchEvent]:
        return [e for e in self._events if e.active]

    def is_engaged(self, scope: Optional[KillSwitchScope] = None,
                   portfolio_id: Optional[str] = None,
                   strategy_id: Optional[str] = None) -> bool:
        """Whether trading is blocked for the given context.

        Global always wins. Otherwise a matching portfolio or strategy
        engagement blocks.

        When no context is supplied the answer is conservative: **any** active
        engagement blocks. The previous behaviour, which required a context
        match, was fail-*open* in the most dangerous direction — a portfolio
        switch engaged for PF1 left the platform reporting NORMAL to a caller
        that had not yet identified which portfolio it was acting for, so a
        halted portfolio looked tradable.
        """
        for ev in self.active_events:
            if ev.scope == KillSwitchScope.GLOBAL:
                return True
            if ev.scope == KillSwitchScope.BROKER:
                return True
            # No context supplied: fail closed.
            if portfolio_id is None and strategy_id is None:
                return True
            if ev.detail.get("portfolio_id") in (None, portfolio_id) and \
               ev.detail.get("strategy_id") in (None, strategy_id):
                return True
        return False

    def trading_state(
        self,
        data_stale: bool = False,
        broker_connected: bool = True,
        reconciliation_ok: bool = True,
        risk_breached: bool = False,
    ) -> TradingState:
        """Overall state. Anything other than NORMAL blocks new orders."""
        if self.is_engaged():
            for ev in self.active_events:
                if ev.scope == KillSwitchScope.GLOBAL:
                    return TradingState.HALTED_KILL_SWITCH
            return TradingState.HALTED_KILL_SWITCH
        if not reconciliation_ok:
            return TradingState.REQUIRES_RECONCILIATION
        if not broker_connected:
            return TradingState.HALTED_BROKER_DOWN
        if data_stale:
            return TradingState.HALTED_DATA_STALE
        if risk_breached:
            return TradingState.HALTED_RISK_BREACH
        return TradingState.NORMAL

    def can_submit_orders(self, **kw) -> bool:
        return self.trading_state(**kw) == TradingState.NORMAL

    def assert_can_trade(self, **kw) -> None:
        state = self.trading_state(**kw)
        if state != TradingState.NORMAL:
            reasons = "; ".join(f"{e.scope.value}: {e.reason}" for e in self.active_events)
            raise KillSwitchEngaged(
                f"trading state is {state.value}"
                + (f" ({reasons})" if reasons else "")
                + ". No new orders may be submitted until this is resolved."
            )

    # -- automatic triggers ------------------------------------------------

    def evaluate(
        self,
        equity: Optional[float] = None,
        daily_start_equity: Optional[float] = None,
        peak_equity: Optional[float] = None,
        now: Optional[dt.datetime] = None,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        return_history: Optional[list] = None,
    ) -> list[KillSwitchEvent]:
        """Check every automatic trigger. Returns the switches newly engaged.

        Triggers are only evaluated when ``config.auto_engage`` is set; the
        counters are updated either way so an operator can see how close the
        system is to a halt.
        """
        fired: list[KillSwitchEvent] = []
        now = now or utcnow()

        if daily_start_equity and equity is not None and daily_start_equity > 0:
            loss = (daily_start_equity - equity) / daily_start_equity
            if loss > self.config.max_daily_loss_pct:
                fired.append(self._auto(
                    KillSwitchScope.PORTFOLIO,
                    f"daily loss {loss:.2%} exceeded the "
                    f"{self.config.max_daily_loss_pct:.2%} limit",
                    {"portfolio_id": portfolio_id, "observed": loss,
                     "limit": self.config.max_daily_loss_pct},
                ))

        if peak_equity and equity is not None and peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity
            if dd > self.config.max_drawdown_pct:
                fired.append(self._auto(
                    KillSwitchScope.PORTFOLIO,
                    f"drawdown {dd:.2%} exceeded the "
                    f"{self.config.max_drawdown_pct:.2%} limit",
                    {"portfolio_id": portfolio_id, "observed": dd,
                     "limit": self.config.max_drawdown_pct},
                ))

        if self.last_market_data_at is not None:
            age = (now - self.last_market_data_at).total_seconds()
            if age > self.config.stale_data_seconds:
                fired.append(self._auto(
                    KillSwitchScope.GLOBAL,
                    f"market data is {age:.0f}s old, above the "
                    f"{self.config.stale_data_seconds:.0f}s staleness limit; "
                    "new orders are blocked because they would be priced on "
                    "data that no longer reflects the market",
                    {"age_seconds": age,
                     "limit": self.config.stale_data_seconds},
                ))

        if self.max_latency_observed_ms > self.config.max_execution_latency_ms:
            fired.append(self._auto(
                KillSwitchScope.BROKER,
                f"execution latency {self.max_latency_observed_ms:.0f}ms exceeded "
                f"the {self.config.max_execution_latency_ms:.0f}ms limit",
                {"observed_ms": self.max_latency_observed_ms,
                 "limit_ms": self.config.max_execution_latency_ms},
            ))

        if self.order_rejections >= self.config.max_order_rejections:
            fired.append(self._auto(
                KillSwitchScope.STRATEGY,
                f"{self.order_rejections} consecutive order rejections reached the "
                f"limit of {self.config.max_order_rejections}",
                {"rejections": self.order_rejections,
                 "strategy_id": strategy_id},
            ))

        if self.reconciliation_failures >= self.config.reconciliation_failures:
            fired.append(self._auto(
                KillSwitchScope.PORTFOLIO,
                f"{self.reconciliation_failures} reconciliation failures reached the "
                f"limit of {self.config.reconciliation_failures}; internal and broker "
                "state disagree and positions cannot be trusted",
                {"failures": self.reconciliation_failures,
                 "portfolio_id": portfolio_id},
            ))

        if return_history and len(return_history) >= 20:
            vol = float(pd.Series(return_history).std(ddof=1) * np.sqrt(252))
            if vol > self.config.max_portfolio_volatility:
                fired.append(self._auto(
                    KillSwitchScope.PORTFOLIO,
                    f"annualised volatility {vol:.1%} exceeded the "
                    f"{self.config.max_portfolio_volatility:.1%} limit",
                    {"observed": vol, "limit": self.config.max_portfolio_volatility,
                     "portfolio_id": portfolio_id},
                ))
        return fired

    def _auto(self, scope: KillSwitchScope, reason: str, detail: dict) -> KillSwitchEvent:
        if not self.config.auto_engage:
            return KillSwitchEvent(scope, reason, True, dict(detail, _not_engaged=True))
        # Do not re-engage an identical active trigger; a flapping trigger would
        # flood the audit log with duplicates.
        for ev in self.active_events:
            if ev.scope == scope and ev.reason == reason:
                return ev
        ev = self.engage(scope, reason, engaged_by="automatic-trigger",
                         detail=detail, automatic=True)
        return ev

    # -- signal inputs -----------------------------------------------------

    def note_market_data(self, when: Optional[dt.datetime] = None) -> None:
        self.last_market_data_at = when or utcnow()

    def note_order_rejection(self) -> None:
        self.order_rejections += 1

    def reset_order_rejections(self) -> None:
        self.order_rejections = 0

    def note_reconciliation_failure(self) -> None:
        self.reconciliation_failures += 1

    def reset_reconciliation_failures(self) -> None:
        self.reconciliation_failures = 0

    def note_execution_latency(self, latency_ms: float) -> None:
        self.max_latency_observed_ms = max(self.max_latency_observed_ms, float(latency_ms))

    def reset(self) -> None:
        """Full reset. Intended for tests and for a deliberate operator action."""
        self._events.clear()
        self.order_rejections = 0
        self.reconciliation_failures = 0
        self.max_latency_observed_ms = 0.0
        self.last_market_data_at = None

    def status(self) -> dict:
        return {
            "engaged": bool(self.active_events),
            "n_active": len(self.active_events),
            "active": [e.as_dict() for e in self.active_events],
            "history": [e.as_dict() for e in self._events],
            "config": self.config.as_dict(),
            "counters": {
                "order_rejections": self.order_rejections,
                "reconciliation_failures": self.reconciliation_failures,
                "max_latency_observed_ms": self.max_latency_observed_ms,
                "last_market_data_at": (
                    self.last_market_data_at.isoformat()
                    if self.last_market_data_at else None
                ),
            },
        }
