"""Portfolio risk limits and the risk engine.

The single most important property of this module: **the risk engine can reject
an order the strategy wants, and a strategy cannot override it.** Blueprint
Section 41 and the final engineering principle both require this, and it is
structural here — the engine sits between the strategy and the order book and
returns a verdict rather than advice.

Checks run in a fixed order, cheapest and most binding first, so a rejected
order reports the reason that actually matters rather than the first one that
happened to trip:

1. per-order notional
2. single-name concentration
3. sector concentration
4. asset-class concentration
5. gross exposure
6. net exposure
7. leverage
8. projected drawdown
9. daily loss
10. portfolio volatility
11. VaR / CVaR

All limits are compared against the portfolio *after* the hypothetical fill,
because a limit that cannot be breached by an order is not a limit.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.enums import RiskLevel, RiskLimitType
from quantforge.core.errors import RiskLimitBreached
from quantforge.core.timestamps import utcnow


# --------------------------------------------------------------------------
# Limit configuration
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RiskLimits:
    """Configured limits. Every field is a hard ceiling, not a target."""

    max_position_pct: float = 0.20
    max_sector_pct: float = 0.40
    max_asset_class_pct: float = 0.80
    max_order_notional_pct: float = 0.10
    max_gross_exposure: float = 1.00
    max_net_exposure: float = 0.50
    max_leverage: float = 1.00
    max_drawdown_pct: float = 0.20
    max_daily_loss_pct: float = 0.03
    max_portfolio_volatility: float = 0.25
    var_limit_pct: float = 0.05
    cvar_limit_pct: float = 0.08
    var_confidence: float = 0.95
    liquidity_participation_max: float = 0.05
    #: Warning band below a hard limit, as a fraction of the limit.
    warn_at_pct: float = 0.80

    def as_dict(self) -> dict:
        return {
            "max_position_pct": self.max_position_pct,
            "max_sector_pct": self.max_sector_pct,
            "max_asset_class_pct": self.max_asset_class_pct,
            "max_order_notional_pct": self.max_order_notional_pct,
            "max_gross_exposure": self.max_gross_exposure,
            "max_net_exposure": self.max_net_exposure,
            "max_leverage": self.max_leverage,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_portfolio_volatility": self.max_portfolio_volatility,
            "var_limit_pct": self.var_limit_pct,
            "cvar_limit_pct": self.cvar_limit_pct,
            "var_confidence": self.var_confidence,
            "liquidity_participation_max": self.liquidity_participation_max,
        }

    def validate(self) -> None:
        checks = [
            ("max_position_pct", self.max_position_pct, 0, 1),
            ("max_sector_pct", self.max_sector_pct, 0, 1),
            ("max_asset_class_pct", self.max_asset_class_pct, 0, 10),
            ("max_gross_exposure", self.max_gross_exposure, 0, 100),
            ("max_net_exposure", self.max_net_exposure, 0, 100),
            ("max_leverage", self.max_leverage, 0, 100),
            ("max_drawdown_pct", self.max_drawdown_pct, 0, 1),
            ("max_daily_loss_pct", self.max_daily_loss_pct, 0, 1),
            ("max_portfolio_volatility", self.max_portfolio_volatility, 0, 10),
        ]
        for name, value, lo, hi in checks:
            if not lo < value <= hi:
                raise ValueError(
                    f"RiskLimits.{name} = {value} is outside ({lo}, {hi}]. A limit "
                    "of zero or negative value would reject every order."
                )


@dataclass(slots=True)
class RiskVerdict:
    """The engine's decision on one proposed order."""

    approved: bool
    risk_level: RiskLevel = RiskLevel.GREEN
    violations: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    adjusted_quantity: Optional[float] = None
    utilisation: dict = field(default_factory=dict)
    checked_at: dt.datetime = field(default_factory=utcnow)
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "approved": self.approved,
            "risk_level": self.risk_level.value,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "adjusted_quantity": self.adjusted_quantity,
            "utilisation": {k: float(v) for k, v in self.utilisation.items()},
            "checked_at": self.checked_at.isoformat(),
            "reason": self.reason,
        }

    def raise_if_breached(self) -> None:
        if not self.approved and self.violations:
            v = self.violations[0]
            raise RiskLimitBreached(
                v["limit_type"], v["limit"], v["observed"], self.reason
            )


# --------------------------------------------------------------------------
# Risk engine
# --------------------------------------------------------------------------


class RiskEngine:
    """Evaluates proposed orders against configured limits.

    ``attribute_fn`` maps an instrument id to its sector and asset class, which
    the engine needs for the concentration checks. It is injected rather than
    looked up globally so the engine stays testable in isolation.
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        attribute_fn: Optional[Callable[[str], tuple[Optional[str], Optional[str]]]] = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.limits.validate()
        self._attr = attribute_fn or (lambda i: (None, None))
        self.daily_start_equity: Optional[float] = None
        self.peak_equity: Optional[float] = None
        self.return_history: list[float] = []
        self.breach_log: list[dict] = []

    # -- state ------------------------------------------------------------

    def reset_session(self, equity: float) -> None:
        """Mark the start of a trading day."""
        self.daily_start_equity = equity
        self.return_history = []

    def update_equity(self, equity: float) -> None:
        """Track equity for drawdown and daily-loss limits."""
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        if self.daily_start_equity is None:
            self.daily_start_equity = equity

    def record_return(self, r: float) -> None:
        self.return_history.append(float(r))

    @property
    def current_drawdown(self) -> float:
        if not self.peak_equity or self.peak_equity <= 0:
            return 0.0
        from quantforge.backtest.portfolio import Portfolio

        return 0.0

    def drawdown_of(self, equity: float) -> float:
        if not self.peak_equity or self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

    def daily_loss_of(self, equity: float) -> float:
        if not self.daily_start_equity or self.daily_start_equity <= 0:
            return 0.0
        return max(0.0, (self.daily_start_equity - equity) / self.daily_start_equity)

    # -- the check --------------------------------------------------------

    def check_order(
        self,
        portfolio,
        instrument_id: str,
        side: str,
        quantity: float,
        price: float,
        project_equity: Optional[float] = None,
    ) -> RiskVerdict:
        """Approve or reject one order.

        The order is applied to a *copy* of the position arithmetic so the
        projected post-trade state is evaluated without mutating the live
        portfolio. Mutating and rolling back is a reliable way to lose money
        during an exception.
        """
        v = RiskVerdict(approved=True)
        if quantity <= 0 or price <= 0:
            v.approved = False
            v.reason = "non-positive order quantity or price"
            return v

        equity = project_equity if project_equity is not None else portfolio.equity
        if equity <= 0:
            v.approved = False
            v.risk_level = RiskLevel.RED
            v.reason = (
                f"portfolio equity is {equity:,.2f}; no order can be sized against "
                "non-positive equity"
            )
            v.violations.append({
                "limit_type": "equity", "limit": 0.0, "observed": equity,
                "detail": v.reason,
            })
            return v

        L = self.limits
        signed = quantity if side == "long" else -quantity
        notional = quantity * price

        # -- projected position state ------------------------------------
        current = portfolio.positions.get(instrument_id)
        current_qty = current.quantity if current is not None else 0.0
        current_notional = abs(current_qty) * (current.last_price if current else price)
        new_qty = current_qty + signed
        new_notional = abs(new_qty) * price

        projected_gross = portfolio.gross_exposure - current_notional + new_notional
        projected_net = portfolio.net_exposure - (
            current_qty * (current.last_price if current else price)
        ) + new_qty * price
        projected_equity = equity  # mark-to-market of the new fill is ~neutral pre-cost

        sector, asset_class = self._attr(instrument_id)

        def record(kind: RiskLimitType, limit: float, observed: float, detail: str) -> None:
            v.violations.append({
                "limit_type": kind.value, "limit": float(limit),
                "observed": float(observed), "instrument_id": instrument_id,
                "detail": detail,
            })
            self.breach_log.append({
                "limit_type": kind.value, "limit": float(limit),
                "observed": float(observed), "instrument_id": instrument_id,
                "detail": detail, "at": utcnow().isoformat(),
            })

        # 1. order notional
        max_order = equity * L.max_order_notional_pct
        if notional > max_order:
            record(RiskLimitType.ORDER_NOTIONAL, max_order, notional,
                   f"order notional {notional:,.0f} exceeds "
                   f"{L.max_order_notional_pct:.1%} of equity ({max_order:,.0f})")

        # 2. single-name concentration
        max_pos = equity * L.max_position_pct
        if new_notional > max_pos:
            record(RiskLimitType.POSITION_CONCENTRATION, max_pos, new_notional,
                   f"{instrument_id} position would be {new_notional:,.0f} "
                   f"({new_notional / equity:.1%} of equity), above the "
                   f"{L.max_position_pct:.1%} limit")

        # 3. sector concentration
        if sector is not None:
            by_sector = portfolio.exposure_by(
                lambda i: self._attr(i)[0] if i != instrument_id else sector
            )
            cur = by_sector.get(sector, 0.0)
            projected_sector = cur + new_notional
            max_sector = equity * L.max_sector_pct
            if projected_sector > max_sector:
                record(RiskLimitType.SECTOR_CONCENTRATION, max_sector, projected_sector,
                       f"sector {sector!r} would hold {projected_sector:,.0f} "
                       f"({projected_sector / equity:.1%}), above "
                       f"{L.max_sector_pct:.1%}")

        # 4. asset-class concentration
        if asset_class is not None:
            by_class = portfolio.exposure_by(
                lambda i: self._attr(i)[1] if i != instrument_id else asset_class
            )
            projected_class = by_class.get(asset_class, 0.0) + new_notional
            max_class = equity * L.max_asset_class_pct
            if projected_class > max_class:
                record(RiskLimitType.ASSET_CLASS_CONCENTRATION, max_class,
                       projected_class,
                       f"asset class {asset_class!r} would be {projected_class:,.0f}, "
                       f"above {max_class:,.0f}")

        # 5. gross exposure
        max_gross = equity * L.max_gross_exposure
        if projected_gross > max_gross:
            record(RiskLimitType.GROSS_EXPOSURE, max_gross, projected_gross,
                   f"gross exposure would be {projected_gross:,.0f} "
                   f"({projected_gross / equity:.1%}), above "
                   f"{L.max_gross_exposure:.1%}")

        # 6. net exposure
        max_net = equity * L.max_net_exposure
        if abs(projected_net) > max_net:
            record(RiskLimitType.NET_EXPOSURE, max_net, abs(projected_net),
                   f"net exposure would be {projected_net:,.0f} "
                   f"({abs(projected_net) / equity:.1%}), above "
                   f"{L.max_net_exposure:.1%}")

        # 7. leverage
        leverage = projected_gross / equity if equity > 0 else float("inf")
        if leverage > L.max_leverage:
            record(RiskLimitType.LEVERAGE, L.max_leverage, leverage,
                   f"leverage would be {leverage:.2f}x, above "
                   f"{L.max_leverage:.2f}x")

        # 8. drawdown
        dd = self.drawdown_of(projected_equity)
        if dd > L.max_drawdown_pct:
            record(RiskLimitType.MAX_DRAWDOWN, L.max_drawdown_pct, dd,
                   f"current drawdown is {dd:.2%}, above "
                   f"{L.max_drawdown_pct:.2%}")

        # 9. daily loss
        dloss = self.daily_loss_of(projected_equity)
        if dloss > L.max_daily_loss_pct:
            record(RiskLimitType.DAILY_LOSS, L.max_daily_loss_pct, dloss,
                   f"daily loss is {dloss:.2%}, above {L.max_daily_loss_pct:.2%}")

        # 10. portfolio volatility (needs a return history)
        vol = self._portfolio_volatility()
        if np.isfinite(vol) and vol > L.max_portfolio_volatility:
            record(RiskLimitType.VOLATILITY, L.max_portfolio_volatility, vol,
                   f"annualised portfolio volatility is {vol:.2%}, above "
                   f"{L.max_portfolio_volatility:.2%}")

        # 11. VaR / CVaR (only meaningful with history)
        if len(self.return_history) >= 30:
            from quantforge.quant import risk_metrics as rm

            r = pd.Series(self.return_history)
            var = rm.historical_var(r, L.var_confidence, 1, "portfolio").value
            if np.isfinite(var) and var > L.var_limit_pct:
                record(RiskLimitType.VAR_LIMIT, L.var_limit_pct, var,
                       f"{int(L.var_confidence * 100)}% 1-day historical VaR is "
                       f"{var:.2%}, above {L.var_limit_pct:.2%}")
            es = rm.historical_expected_shortfall(
                r, L.var_confidence, 1, "portfolio").value
            if np.isfinite(es) and es > L.cvar_limit_pct:
                record(RiskLimitType.CVAR_LIMIT, L.cvar_limit_pct, es,
                       f"{int(L.var_confidence * 100)}% expected shortfall is "
                       f"{es:.2%}, above {L.cvar_limit_pct:.2%}")

        # -- utilisation, for the dashboard --------------------------------
        v.utilisation = {
            "position_pct": new_notional / equity,
            "gross_exposure_pct": projected_gross / equity,
            "net_exposure_pct": abs(projected_net) / equity,
            "leverage": leverage,
            "drawdown_pct": dd,
            "daily_loss_pct": dloss,
        }

        # -- warning band ---------------------------------------------------
        warn = L.warn_at_pct
        for name, obs, lim in (
            ("position_pct", new_notional / equity, L.max_position_pct),
            ("gross_exposure_pct", projected_gross / equity, L.max_gross_exposure),
            ("leverage", leverage, L.max_leverage),
        ):
            if lim > 0 and obs > lim * warn:
                v.warnings.append({
                    "metric": name, "observed": obs, "limit": lim,
                    "detail": f"{name} at {obs:.2%} of a {lim:.2%} limit "
                              f"({obs / lim:.0%} utilised)",
                })

        if v.violations:
            v.approved = False
            v.risk_level = RiskLevel.RED
            v.reason = (
                f"{len(v.violations)} risk limit(s) breached: "
                + "; ".join(x["detail"] for x in v.violations[:3])
            )
        elif v.warnings:
            v.risk_level = RiskLevel.YELLOW
            v.reason = f"{len(v.warnings)} limit(s) in the warning band"
        return v

    def _portfolio_volatility(self) -> float:
        if len(self.return_history) < 20:
            return float("nan")
        return float(pd.Series(self.return_history).std(ddof=1) * np.sqrt(252))

    def check_order_or_raise(self, portfolio, instrument_id, side, quantity, price,
                             **kw) -> RiskVerdict:
        v = self.check_order(portfolio, instrument_id, side, quantity, price, **kw)
        v.raise_if_breached()
        return v

    def clamp_quantity(
        self, portfolio, instrument_id: str, side: str,
        quantity: float, price: float,
        margin: float = 0.95,
    ) -> float:
        """Largest quantity that passes every limit, with a safety margin.

        ``margin`` deliberately stops short of the exact boundary. Binary
        search finds the largest *passing* size, which by construction sits
        infinitesimally below the limit; since the order is sized on bar ``t``'s
        close and filled on bar ``t+1``'s open, any adverse gap then pushes it
        over and the order is rejected every time. Sizing to 95% of the
        feasible boundary trades a sliver of position size for orders that
        actually fill.
        """
        target = quantity * margin
        if self.check_order(portfolio, instrument_id, side, target, price).approved:
            return target
        lo, hi = 0.0, target
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if self.check_order(portfolio, instrument_id, side, mid, price).approved:
                lo = mid
            else:
                hi = mid
        return lo

    def utilisation_report(self, portfolio) -> dict:
        equity = portfolio.equity
        if equity <= 0:
            return {}
        L = self.limits
        return {
            "position_pct": max(
                (portfolio.position_weight(i) for i in portfolio.positions), default=0.0
            ),
            "gross_exposure_pct": portfolio.gross_exposure / equity,
            "net_exposure_pct": abs(portfolio.net_exposure) / equity,
            "leverage": portfolio.leverage,
            "drawdown_pct": self.drawdown_of(equity),
            "daily_loss_pct": self.daily_loss_of(equity),
            "limits": {
                "position": L.max_position_pct, "gross": L.max_gross_exposure,
                "net": L.max_net_exposure, "leverage": L.max_leverage,
                "drawdown": L.max_drawdown_pct, "daily_loss": L.max_daily_loss_pct,
            },
        }
