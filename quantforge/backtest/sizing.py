"""Position sizing.

The sizing methods differ in what they assume about the distribution of
outcomes, and those assumptions are stated here because they dominate the
results:

``fixed_fractional``
    No distributional assumption. Risk is capped by construction, so the method
    survives being wrong about the strategy.

``volatility_atr`` / ``volatility_hv``
    Assumes volatility is a reasonable proxy for the dispersion of returns.
    It scales *size* correctly but says nothing about direction, and it fails
    for a strategy whose loss distribution is asymmetric (gap risk).

``kelly``
    Assumes the estimated win probability and payoff ratio are close to their
    true out-of-sample values. They usually are not. Kelly's growth rate is
    maximised at the full fraction and the *drawdown* is maximised there too,
    roughly 2x the drawdown at half-Kelly and 4x at quarter-Kelly.

``risk_based``
    Sizes so that a stop-loss distance corresponds to a fixed fraction of
    capital at risk. It is the only method that connects position size to an
    explicit invalidation point, which is why it is the default for strategies
    with defined stops.

Every method returns a size and is then clamped by the risk engine. Sizing
never has the final word.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from quantforge.core.enums import SizingMethod
from quantforge.core.errors import RiskError


@dataclass(slots=True)
class SizingResult:
    """A size request together with its provenance and its clamps."""

    quantity: float
    method: str
    raw_quantity: float
    notional: float
    equity_fraction: float
    risk_amount: float
    risk_per_share: float
    capped_by: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "method": self.method,
            "raw_quantity": self.raw_quantity,
            "notional": self.notional,
            "equity_fraction": self.equity_fraction,
            "risk_amount": self.risk_amount,
            "risk_per_share": self.risk_per_share,
            "capped_by": self.capped_by,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class SizingConfig:
    method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    fraction: float = 0.10
    atr_multiplier: float = 2.0
    target_volatility: float = 0.15
    risk_fraction: float = 0.01
    kelly_fraction: float = 0.25
    max_kelly_weight: float = 0.10
    min_quantity: float = 0.0
    max_quantity: float = float("inf")
    lot_size: int = 1

    def as_dict(self) -> dict:
        return {
            "method": self.method.value,
            "fraction": self.fraction,
            "atr_multiplier": self.atr_multiplier,
            "target_volatility": self.target_volatility,
            "risk_fraction": self.risk_fraction,
            "kelly_fraction": self.kelly_fraction,
            "max_kelly_weight": self.max_kelly_weight,
            "min_quantity": self.min_quantity,
            "max_quantity": self.max_quantity,
            "lot_size": self.lot_size,
        }


class PositionSizer:
    """Computes target position sizes. Never the final authority."""

    def __init__(self, config: Optional[SizingConfig] = None) -> None:
        self.config = config or SizingConfig()

    # -- individual methods ----------------------------------------------

    def fixed_fractional(
        self, equity: float, price: float, instrument=None
    ) -> SizingResult:
        """Fixed fraction of equity. No edge assumption required."""
        cfg = self.config
        if price <= 0 or equity <= 0:
            return self._zero("fixed_fractional", equity, "non-positive price or equity")
        notional = equity * cfg.fraction
        qty = notional / price
        return SizingResult(
            quantity=qty, method="fixed_fractional", raw_quantity=qty,
            notional=notional, equity_fraction=cfg.fraction, risk_amount=float("nan"),
            risk_per_share=float("nan"),
            notes=[f"risk is capped at {cfg.fraction:.1%} of equity by construction, "
                   "independent of the strategy's actual edge"],
        )

    def volatility_atr(
        self, equity: float, price: float, atr: float, instrument=None
    ) -> SizingResult:
        """Size so a stop at ``atr_multiplier * ATR`` risks ``risk_fraction``.

        The ATR stop defines the risk per share; the risk budget defines the
        total. Where ATR is missing or zero the method degrades to a fixed
        fraction rather than producing an unbounded size.
        """
        cfg = self.config
        if price <= 0 or equity <= 0:
            return self._zero("volatility_atr", equity, "non-positive price or equity")
        notes: list[str] = []
        if not np.isfinite(atr) or atr <= 0:
            fallback = self.fixed_fractional(equity, price, instrument)
            notes = ["ATR unavailable or zero; fell back to fixed fractional. "
                     "A silent fallback would hide that the volatility assumption "
                     "was not applied."]
            fallback.method = "volatility_atr(fallback_fixed_fractional)"
            fallback.notes = notes + fallback.notes
            return fallback
        risk_per_share = cfg.atr_multiplier * atr
        risk_budget = equity * cfg.risk_fraction
        qty = risk_budget / risk_per_share
        notional = qty * price
        if notional > equity * 0.95:
            notes.append(
                f"notional {notional:,.0f} exceeds 95% of equity; the ATR stop is "
                "tighter than the position size implies, which usually means the "
                "ATR input is too small for this instrument"
            )
        return SizingResult(
            quantity=qty, method="volatility_atr", raw_quantity=qty,
            notional=notional, equity_fraction=notional / equity,
            risk_amount=qty * risk_per_share, risk_per_share=risk_per_share,
            notes=notes,
        )

    def volatility_hv(
        self, equity: float, price: float, realized_vol: float,
        instrument=None, periods_per_year: int = 252,
    ) -> SizingResult:
        """Size to a target annualised volatility contribution."""
        cfg = self.config
        if price <= 0 or equity <= 0:
            return self._zero("volatility_hv", equity, "non-positive price or equity")
        if not np.isfinite(realized_vol) or realized_vol <= 1e-9:
            return self.fixed_fractional(equity, price, instrument)
        period_vol = realized_vol / math.sqrt(periods_per_year)
        # Fraction of capital whose volatility contribution hits the target.
        weight = cfg.target_volatility / realized_vol
        if weight > 1.0:
            notes = [f"target volatility {cfg.target_volatility:.1%} exceeds the "
                     f"instrument's own {realized_vol:.1%}, so the full equity is "
                     "used; the target cannot be met by sizing alone"]
        else:
            notes = []
        notional = equity * weight
        qty = notional / price
        return SizingResult(
            quantity=qty, method="volatility_hv", raw_quantity=qty,
            notional=notional, equity_fraction=weight,
            risk_amount=notional * period_vol, risk_per_share=period_vol * price,
            notes=notes,
        )

    def kelly(
        self,
        equity: float,
        price: float,
        win_probability: float,
        payoff_ratio: float,
        instrument=None,
    ) -> SizingResult:
        """Fractional Kelly, hard-bounded.

        ``f* = p - (1 - p) / b`` for a binary payoff. The result is multiplied
        by ``kelly_fraction`` and clamped to ``max_kelly_weight``.

        The estimate of ``p`` carries far more error than the arithmetic
        suggests, and Kelly is extremely sensitive to it: overestimating the
        edge by a small amount is what produces ruin, not a smaller gain. This
        is why the fraction defaults to 0.25 and the cap is 10%.
        """
        cfg = self.config
        if price <= 0 or equity <= 0:
            return self._zero("kelly", equity, "non-positive price or equity")
        p = float(win_probability)
        b = float(payoff_ratio)
        notes: list[str] = []
        if not 0.0 < p < 1.0:
            notes.append(
                f"win probability {p:.3f} is outside (0, 1); Kelly is undefined. "
                "Falling back to fixed fractional rather than extrapolating."
            )
            fb = self.fixed_fractional(equity, price, instrument)
            fb.notes = notes + fb.notes
            return fb
        if b <= 0:
            notes.append("payoff ratio must be positive; falling back to fixed fractional")
            fb = self.fixed_fractional(equity, price, instrument)
            fb.notes = notes + fb.notes
            return fb

        f_star = p - (1.0 - p) / b
        if f_star <= 0:
            notes.append(
                f"Kelly fraction is non-positive ({f_star:.4f}), i.e. the estimated "
                "edge is negative. Taking no position is the correct action."
            )
            return SizingResult(
                quantity=0.0, method="kelly", raw_quantity=0.0, notional=0.0,
                equity_fraction=0.0, risk_amount=0.0, risk_per_share=0.0,
                capped_by="negative_kelly", notes=notes,
            )
        f = f_star * cfg.kelly_fraction
        capped = False
        if f > cfg.max_kelly_weight:
            notes.append(
                f"fractional Kelly {f:.2%} exceeded the {cfg.max_kelly_weight:.2%} cap "
                "and was reduced. Full Kelly maximises expected growth and "
                "maximises drawdown simultaneously."
            )
            f = cfg.max_kelly_weight
            capped = True
        notional = equity * f
        qty = notional / price
        return SizingResult(
            quantity=qty, method="kelly", raw_quantity=qty, notional=notional,
            equity_fraction=f, risk_amount=notional * (1.0 - p),
            risk_per_share=price / b, capped_by="max_kelly_weight" if capped else None,
            notes=notes + [f"full Kelly would be {f_star:.2%}; applied "
                           f"{cfg.kelly_fraction:.0%} of it"],
        )

    def risk_based(
        self, equity: float, price: float, stop_price: float, instrument=None
    ) -> SizingResult:
        """Size so the distance to the stop is a fixed fraction of equity."""
        cfg = self.config
        if price <= 0 or equity <= 0:
            return self._zero("risk_based", equity, "non-positive price or equity")
        risk_per_share = abs(price - stop_price)
        notes: list[str] = []
        if risk_per_share <= 0:
            notes.append("stop equals the entry price, so risk per share is zero; "
                         "size is undefined. Check the stop logic.")
            return self._zero("risk_based", equity, "zero stop distance", notes)
        if risk_per_share > price:
            notes.append(
                f"stop distance {risk_per_share:,.2f} exceeds the price {price:,.2f}; "
                "the stop implies a loss greater than 100% of the position value"
            )
        risk_budget = equity * cfg.risk_fraction
        qty = risk_budget / risk_per_share
        notional = qty * price
        return SizingResult(
            quantity=qty, method="risk_based", raw_quantity=qty, notional=notional,
            equity_fraction=notional / equity, risk_amount=qty * risk_per_share,
            risk_per_share=risk_per_share, notes=notes,
        )

    # -- dispatch ---------------------------------------------------------

    def size(
        self,
        equity: float,
        price: float,
        instrument=None,
        atr: Optional[float] = None,
        realized_vol: Optional[float] = None,
        stop_price: Optional[float] = None,
        win_probability: Optional[float] = None,
        payoff_ratio: Optional[float] = None,
    ) -> SizingResult:
        cfg = self.config
        if cfg.method == SizingMethod.FIXED_FRACTIONAL:
            return self.fixed_fractional(equity, price, instrument)
        if cfg.method == SizingMethod.VOLATILITY_ATR:
            return self.volatility_atr(equity, price, atr or float("nan"), instrument)
        if cfg.method == SizingMethod.VOLATILITY_HV:
            return self.volatility_hv(equity, price, realized_vol or float("nan"),
                                      instrument)
        if cfg.method == SizingMethod.KELLY:
            if win_probability is None or payoff_ratio is None:
                return self.fixed_fractional(equity, price, instrument)
            return self.kelly(equity, price, win_probability, payoff_ratio, instrument)
        if cfg.method == SizingMethod.RISK_BASED:
            if stop_price is None:
                return self.fixed_fractional(equity, price, instrument)
            return self.risk_based(equity, price, stop_price, instrument)
        return self.fixed_fractional(equity, price, instrument)

    # -- final adjustments -------------------------------------------------

    def apply_lot_size(self, qty: float, lot_size: int = 1) -> float:
        """Truncate to whole lots. Rounds *down* so sizing never overshoots."""
        if lot_size < 1:
            return qty
        return float(math.floor(abs(qty) / lot_size) * lot_size) * (1 if qty >= 0 else -1)

    def clamp_to_limits(
        self, result: SizingResult, equity: float,
        max_position_notional: Optional[float] = None,
    ) -> SizingResult:
        """Apply the sizing config's own bounds. Risk limits come separately."""
        cfg = self.config
        qty = result.quantity
        note_added = False
        if max_position_notional is not None and result.notional > max_position_notional:
            qty = max_position_notional / result.notional * qty
            result.capped_by = "max_position_notional"
            result.notes = result.notes + [
                f"reduced from {result.notional:,.0f} to the "
                f"{max_position_notional:,.0f} position limit"
            ]
            note_added = True
        capped = self.apply_lot_size(qty, cfg.lot_size)
        if capped != qty and result.capped_by is None:
            result.capped_by = "lot_size"
        result.quantity = capped
        result.notional = abs(capped) * result.notional / abs(result.quantity) if result.quantity else 0.0
        if not note_added and result.quantity != result.raw_quantity:
            result.equity_fraction = result.notional / equity if equity else 0.0
        return result

    def _zero(self, method: str, equity: float, reason: str,
              notes: Optional[list] = None) -> SizingResult:
        n = [f"no position: {reason}"] + list(notes or [])
        return SizingResult(
            quantity=0.0, method=method, raw_quantity=0.0, notional=0.0,
            equity_fraction=0.0, risk_amount=0.0, risk_per_share=0.0, notes=n,
        )


def equal_weight_quantities(
    instruments: list[str],
    prices: dict[str, float],
    equity: float,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Equal- or signal-weighted allocation, in units.

    Inverse-volatility weighting is deliberately not the default: it
    concentrates capital in low-volatility names, which for a multi-asset
    portfolio is often a hidden factor bet rather than a risk reduction.
    """
    tradable = [i for i in instruments if prices.get(i) and prices[i] > 0]
    if not tradable:
        return {}
    w = weights or {i: 1.0 / len(tradable) for i in tradable}
    total = sum(abs(w.get(i, 0.0)) for i in tradable)
    if total <= 0:
        return {i: 0.0 for i in tradable}
    out = {}
    for i in tradable:
        notional = equity * abs(w.get(i, 0.0)) / total
        out[i] = notional / prices[i]
    return out
