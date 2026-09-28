"""Transaction cost and slippage models.

Costs are where backtests most often flatter themselves. A strategy whose edge
is 0.3% per trade on a liquid equity can show 20% annualised Sharpe at zero
cost and nothing at all under a realistic one. The models here are therefore
explicit, inspectable, and always reported.

Three cost components, all of which default to non-zero:

``commission``   broker/exchange/regulatory fees
``spread``       the cost of crossing the bid-ask
``market impact`` price concession from consuming liquidity

Slippage is modelled separately from spread because they are conceptually
different: spread is a property of the resting book, impact is a consequence
of the order's own size. Conflating them (as "slippage = 0.1%" does) hides
which one is driving the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from quantforge.core.enums import SlippageModel
from quantforge.core.errors import BacktestError


# --------------------------------------------------------------------------
# Commission
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CommissionSchedule:
    """Tiered or flat commission.

    ``tiers`` is a list of ``(quantity_threshold, rate_per_share, min_charge)``
    evaluated against the order's share count. Real broker schedules are tiered
    this way, and a flat model systematically misprices small-versus-large
    order behaviour.

    All rates are in the instrument's quote currency.
    """

    per_share: float = 0.0
    pct_of_notional: float = 0.0
    per_order: float = 0.0
    min_per_order: float = 0.0
    regulatory_bps: float = 0.0     # basis points of notional, sell-side only
    exchange_fee_per_share: float = 0.0
    tiers: Optional[list] = None    # [(qty, rate_per_share, min_charge), ...]

    def compute(
        self,
        quantity: float,
        price: float,
        side: str = "long",
        instrument_type: str = "equity",
    ) -> float:
        """Total commission for one fill, always >= 0."""
        qty = abs(float(quantity))
        if qty == 0:
            return 0.0
        notional = qty * price

        if self.tiers:
            rate, min_charge = self.tiers[-1][1], self.tiers[-1][2]
            for threshold, t_rate, t_min in self.tiers:
                if qty >= threshold:
                    rate, min_charge = t_rate, t_min
                    break
            broker = qty * rate
        else:
            broker = qty * self.per_share + self.per_order
            min_charge = self.min_per_order

        if instrument_type not in ("equity", "etf"):
            # Per-share fees are an equity-market construct. Charging them on
            # FX or futures produces costs off by orders of magnitude, so a
            # notional-based model is used for non-share instruments.
            broker = self.per_order + notional * self.pct_of_notional
            min_charge = 0.0

        pct_fee = notional * self.pct_of_notional if self.tiers is None else 0.0
        exchange = qty * self.exchange_fee_per_share if instrument_type in ("equity", "etf") else 0.0
        reg = notional * (self.regulatory_bps / 1e4) if side == "short" else 0.0

        total = broker + pct_fee + exchange + reg
        if self.tiers is None and min_charge > 0 and notional > 0:
            total = max(total, min_charge)
        return float(max(0.0, total))

    def as_dict(self) -> dict:
        return {
            "per_share": self.per_share,
            "pct_of_notional": self.pct_of_notional,
            "per_order": self.per_order,
            "min_per_order": self.min_per_order,
            "regulatory_bps": self.regulatory_bps,
            "exchange_fee_per_share": self.exchange_fee_per_share,
            "tiered": bool(self.tiers),
        }


# --------------------------------------------------------------------------
# Slippage
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SlippageModelSpec:
    """Slippage parameters.

    ``half_spread_bps`` is applied on every fill: crossing the spread is
    unavoidable even for a market order, and using the *half* spread on both
    sides is the standard convention for a mid-price reference. A full spread
    per side is common in the literature and doubles the modelled cost, so the
    choice is explicit.
    """

    model: SlippageModel = SlippageModel.PERCENTAGE
    pct: float = 0.0005
    fixed: float = 0.0
    half_spread_bps: float = 2.5
    volatility_scale: float = 1.0
    volume_participation_ref: float = 0.01
    square_root_coefficient: float = 1.0
    max_slippage_pct: float = 0.05

    def compute(
        self,
        quantity: float,
        price: float,
        side: str = "long",
        bar_volume: Optional[float] = None,
        bar_volatility: Optional[float] = None,
        close: Optional[float] = None,
    ) -> float:
        """Slippage in currency units for one fill (always a cost, >= 0)."""
        qty = abs(float(quantity))
        if qty == 0 or price <= 0:
            return 0.0
        notional = qty * price

        # The spread is a baseline cost present in every model.
        spread_cost = notional * (self.half_spread_bps / 1e4)

        if self.model == SlippageModel.NONE:
            return float(spread_cost)

        if self.model == SlippageModel.FIXED:
            extra = qty * self.fixed

        elif self.model == SlippageModel.PERCENTAGE:
            extra = notional * self.pct

        elif self.model == SlippageModel.SPREAD_BASED:
            # Modelled entirely by the baseline spread.
            extra = 0.0

        elif self.model == SlippageModel.VOLATILITY_BASED:
            # Scales with the bar's realised volatility: an illiquid,
            # volatile instrument costs more to cross.
            if bar_volatility is None or not np.isfinite(bar_volatility) or bar_volatility <= 0:
                extra = notional * self.pct
            else:
                extra = notional * self.pct * (1.0 + self.volatility_scale * bar_volatility)

        elif self.model == SlippageModel.VOLUME_BASED:
            # Participation-driven. A linear-in-participation model badly
            # overstates the cost of very small orders, so the square-root form
            # is used, which is the standard empirical shape of the
            # price-impact relationship.
            if bar_volume is None or bar_volume <= 0:
                extra = notional * self.pct
            else:
                participation = qty / bar_volume
                extra = (notional * self.square_root_coefficient
                         * np.sqrt(participation / max(self.volume_participation_ref, 1e-9)))

        elif self.model == SlippageModel.SQUARE_ROOT:
            if bar_volume is None or bar_volume <= 0:
                extra = notional * self.pct
            else:
                extra = notional * self.square_root_coefficient * np.sqrt(
                    qty / bar_volume
                ) / 10.0
        else:
            raise BacktestError(f"unknown slippage model: {self.model}")

        extra = float(max(0.0, extra))
        total = spread_cost + extra
        if self.max_slippage_pct > 0:
            cap = notional * self.max_slippage_pct
            total = min(total, cap)
        return float(total)

    def as_dict(self) -> dict:
        return {
            "model": self.model.value,
            "pct": self.pct,
            "fixed_per_share": self.fixed,
            "half_spread_bps": self.half_spread_bps,
            "volatility_scale": self.volatility_scale,
            "volume_participation_ref": self.volume_participation_ref,
            "square_root_coefficient": self.square_root_coefficient,
            "max_slippage_pct": self.max_slippage_pct,
        }


# --------------------------------------------------------------------------
# Combined cost model
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CostModel:
    """Commission + spread + slippage, with an auditable report."""

    commission: CommissionSchedule = field(default_factory=CommissionSchedule)
    slippage: SlippageModelSpec = field(default_factory=SlippageModelSpec)
    #: True when neither spread nor slippage is modelled. Loudly flagged.
    is_free: bool = False

    def __post_init__(self) -> None:
        self.is_free = (
            self.slippage.model == SlippageModel.NONE
            and self.slippage.half_spread_bps == 0.0
            and self.slippage.pct == 0.0
            and self.commission.per_share == 0.0
            and self.commission.per_order == 0.0
            and self.commission.pct_of_notional == 0.0
        )

    def estimate(
        self,
        quantity: float,
        price: float,
        side: str = "long",
        instrument_type: str = "equity",
        bar_volume: Optional[float] = None,
        bar_volatility: Optional[float] = None,
    ) -> dict:
        """Total cost breakdown for one fill."""
        qty = abs(float(quantity))
        notional = qty * price
        comm = self.commission.compute(qty, price, side, instrument_type)
        slip = self.slippage.compute(
            qty, price, side, bar_volume=bar_volume, bar_volatility=bar_volatility
        )
        return {
            "quantity": qty,
            "price": float(price),
            "notional": float(notional),
            "commission": float(comm),
            "slippage": float(slip),
            "total_cost": float(comm + slip),
            "cost_bps": float(1e4 * (comm + slip) / notional) if notional > 0 else 0.0,
            "side": side,
        }

    def apply(
        self,
        quantity: float,
        price: float,
        side: str = "long",
        instrument_type: str = "equity",
        bar_volume: Optional[float] = None,
        bar_volatility: Optional[float] = None,
    ) -> tuple[float, dict]:
        """Execution price after costs.

        A buy pays up, a sell receives less. The adjustment is per share and
        always moves the price *against* the trader.
        """
        est = self.estimate(quantity, price, side, instrument_type,
                            bar_volume, bar_volatility)
        if est["notional"] == 0:
            return float(price), est
        per_share = est["total_cost"] / est["quantity"]
        fill = float(price) + per_share if side == "long" else float(price) - per_share
        est["fill_price"] = max(0.0, fill)
        est["per_share_cost"] = per_share
        return est["fill_price"], est

    def as_dict(self) -> dict:
        return {
            "commission": self.commission.as_dict(),
            "slippage": self.slippage.as_dict(),
            "is_free": self.is_free,
            "warning": (
                "COSTS ARE ZERO. Any result from this configuration is an upper "
                "bound that no real execution could achieve."
                if self.is_free else None
            ),
        }

    @classmethod
    def from_config(cls, cfg) -> "CostModel":
        """Build from a :class:`~quantforge.core.config.BacktestConfig`."""
        return cls(
            commission=CommissionSchedule(
                per_share=getattr(cfg, "commission_per_share", 0.0),
                pct_of_notional=getattr(cfg, "commission_pct", 0.0),
                min_per_order=getattr(cfg, "min_commission", 0.0),
                regulatory_bps=getattr(cfg, "regulatory_fee_bps", 0.0),
            ),
            slippage=SlippageModelSpec(
                model=SlippageModel(getattr(cfg, "slippage_model", "percentage")),
                pct=getattr(cfg, "slippage_pct", 0.0),
                fixed=getattr(cfg, "slippage_fixed", 0.0),
                half_spread_bps=getattr(cfg, "assume_spread_bps", 5.0) / 2.0,
            ),
        )

    @classmethod
    def conservative_equities(cls) -> "CostModel":
        """A deliberately pessimistic US equity cost model.

        Useful as a robustness baseline: if a strategy only works under the
        cheap configuration, it is not robust.
        """
        return cls(
            commission=CommissionSchedule(
                per_share=0.005, pct_of_notional=0.0, min_per_order=1.0,
                regulatory_bps=0.0002, exchange_fee_per_share=0.0002,
            ),
            slippage=SlippageModelSpec(
                model=SlippageModel.VOLUME_BASED,
                half_spread_bps=5.0,
                volume_participation_ref=0.01,
                square_root_coefficient=0.10,
            ),
        )


# --------------------------------------------------------------------------
# Cost sensitivity analysis
# --------------------------------------------------------------------------


def cost_sensitivity(
    notional_per_trade: float,
    trades_per_year: float,
    cost_models: dict[str, CostModel],
    gross_return_per_year: float,
) -> pd.DataFrame:
    """Net return as a function of assumed costs.

    Answers the question a Sharpe ratio hides: *how much of this edge is
    execution, and how much survives a worse cost assumption?* A strategy
    whose net return is near zero under conservative costs is a strategy
    whose profitability is a modelling artefact.
    """
    rows = []
    for name, model in cost_models.items():
        est = model.estimate(notional_per_trade, 1.0, "long")
        cost_per_trade = est["cost_bps"] / 1e4 * notional_per_trade
        annual_cost = cost_per_trade * trades_per_year
        net = gross_return_per_year - (annual_cost / notional_per_trade
                                       if notional_per_trade else 0.0)
        rows.append({
            "model": name,
            "cost_bps": est["cost_bps"],
            "cost_per_trade": cost_per_trade,
            "annual_cost": annual_cost,
            "gross_return": gross_return_per_year,
            "net_return": net,
            "erodes_edge": net < 0,
        })
    return pd.DataFrame(rows).set_index("model")
