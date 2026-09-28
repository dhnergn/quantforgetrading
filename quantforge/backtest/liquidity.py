"""Liquidity constraints and participation caps.

The working assumption is that an order cannot be filled instantaneously at the
touch regardless of its size. Two mechanisms enforce this:

1. **Participation cap** — order size is limited to a fraction of the bar's
   volume. An order for 40% of a day's volume does not fill at the close.
2. **Fill simulation** — market orders fill across a range of prices, limit
   orders fill only when the bar trades through them.

A backtest that fills a market order at the bar close regardless of size will
show a Sharpe ratio that cannot be reproduced, because the fills it assumes do
not exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from quantforge.core.errors import ExecutionSimulationError


@dataclass(slots=True)
class LiquidityModel:
    """Volume-based participation and fill behaviour.

    ``max_participation_pct`` is the share of a bar's volume a single order may
    consume. ``allow_partial_fills`` controls what happens beyond it: either
    the order is rejected, or it fills partially over subsequent bars.

    ``fill_probability`` is deliberately explicit. A limit order at a price the
    bar only touched for one minute of a 390-minute session should not fill at
    100%. Without a probabilistic fill, limit orders are systematically
    over-optimistic because they are only *placed* when they would have filled.
    """

    max_participation_pct: float = 0.05
    min_adv_shares: float = 0.0
    min_dollar_volume: float = 0.0
    allow_partial_fills: bool = True
    partial_fill_min_fraction: float = 0.10
    limit_fill_probability: float = 0.85
    #: Volume-weighted fill price offset, in fraction of the bar range, when a
    #: market order consumes a large share of the bar.
    impact_range_coefficient: float = 0.10
    min_bar_volume_for_fill: float = 0.0

    def max_fillable_quantity(
        self, bar_volume: Optional[float], requested: float,
    ) -> float:
        """Quantity that can actually be filled in this bar."""
        qty = abs(float(requested))
        if bar_volume is None or bar_volume <= 0:
            # No volume information: refuse to fill rather than assume infinite
            # liquidity, which is how backtests acquire unachievable returns.
            return 0.0
        if self.min_bar_volume_for_fill > 0 and bar_volume < self.min_bar_volume_for_fill:
            return 0.0
        cap = bar_volume * self.max_participation_pct
        return float(min(qty, cap))

    def is_tradable(
        self, bar_volume: Optional[float], dollar_volume: Optional[float] = None
    ) -> bool:
        """Whether the instrument meets minimum liquidity for this strategy."""
        if bar_volume is None or bar_volume <= 0:
            return False
        if self.min_adv_shares > 0 and bar_volume < self.min_adv_shares:
            return False
        if self.min_dollar_volume > 0 and (dollar_volume or 0.0) < self.min_dollar_volume:
            return False
        return True

    def decide_fill(
        self,
        requested: float,
        bar_volume: Optional[float],
        order_type: str = "market",
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Decide what happens to an order in this bar.

        Returns a decision dict; the caller applies it. Kept separate from
        execution so the policy can be inspected and tested on its own.
        """
        qty = abs(float(requested))
        if qty == 0:
            return {"filled": 0.0, "rejected": False, "reason": "zero quantity",
                    "partial": False}

        if not self.is_tradable(bar_volume):
            return {
                "filled": 0.0, "rejected": not self.allow_partial_fills,
                "reason": (
                    f"insufficient liquidity: bar volume "
                    f"{0.0 if bar_volume is None else bar_volume:,.0f} below the "
                    f"configured minimum"
                ),
                "partial": False,
            }

        max_qty = self.max_fillable_quantity(bar_volume, qty)
        if max_qty <= 0:
            return {"filled": 0.0, "rejected": True,
                    "reason": "max fillable quantity is zero", "partial": False}

        if order_type in ("limit", "stop_limit"):
            # A limit order's fill is uncertain even when volume suffices.
            rng = rng or np.random.default_rng()
            if rng.random() > self.limit_fill_probability:
                # Reduce the fill rather than rejecting outright: the order
                # stays working, it simply did not fill this bar.
                return {
                    "filled": 0.0, "rejected": False,
                    "reason": (
                        f"limit order did not fill (price not reached with "
                        f"{self.limit_fill_probability:.0%} fill probability)"
                    ),
                    "partial": False,
                }

        if max_qty >= qty:
            return {"filled": qty, "rejected": False,
                    "reason": "filled in full", "partial": False,
                    "participation": qty / bar_volume}

        if not self.allow_partial_fills:
            return {
                "filled": 0.0, "rejected": True,
                "reason": (
                    f"requested {qty:,.0f} exceeds the participation cap of "
                    f"{self.max_participation_pct:.1%} of {bar_volume:,.0f} shares "
                    f"(max {max_qty:,.0f}) and partial fills are disabled"
                ),
                "partial": False,
            }

        if max_qty < self.partial_fill_min_fraction * qty:
            return {
                "filled": 0.0, "rejected": False,
                "reason": (
                    f"fillable {max_qty:,.0f} is below the minimum partial fill of "
                    f"{self.partial_fill_min_fraction:.0%} of the order; leaving the "
                    "order working for a later bar"
                ),
                "partial": False,
            }

        return {
            "filled": float(max_qty),
            "rejected": False,
            "partial": True,
            "reason": (
                f"partial fill: {max_qty:,.0f} of {qty:,.0f} shares "
                f"({max_qty / bar_volume:.2%} of bar volume)"
            ),
            "participation": max_qty / bar_volume,
        }

    def market_impact_offset(
        self,
        quantity: float,
        bar_volume: Optional[float],
        bar_high: float,
        bar_low: float,
        side: str = "long",
    ) -> float:
        """Price concession from consuming a large share of the bar.

        Models impact as a fraction of the bar's own range, scaled by the
        square root of participation. Returns a *signed* offset that always
        moves against the trader.
        """
        if bar_volume is None or bar_volume <= 0:
            return 0.0
        rng = bar_high - bar_low
        if rng <= 0:
            return 0.0
        participation = min(1.0, abs(quantity) / bar_volume)
        if participation <= self.max_participation_pct:
            return 0.0
        excess = participation - self.max_participation_pct
        offset = self.impact_range_coefficient * np.sqrt(excess) * rng
        return float(offset if side == "long" else -offset)

    def as_dict(self) -> dict:
        return {
            "max_participation_pct": self.max_participation_pct,
            "min_adv_shares": self.min_adv_shares,
            "min_dollar_volume": self.min_dollar_volume,
            "allow_partial_fills": self.allow_partial_fills,
            "partial_fill_min_fraction": self.partial_fill_min_fraction,
            "limit_fill_probability": self.limit_fill_probability,
            "impact_range_coefficient": self.impact_range_coefficient,
            "note": (
                "A backtest that fills any order size at the bar close assumes "
                "infinite liquidity, which no real market provides. These caps "
                "make the assumption explicit and testable."
            ),
        }


# --------------------------------------------------------------------------
# Fill price simulation
# --------------------------------------------------------------------------


def simulate_fill_price(
    side: str,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    quantity: float,
    bar_volume: Optional[float],
    liquidity: Optional[LiquidityModel] = None,
    order_type: str = "market",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    signal_time: Optional[str] = None,
) -> dict:
    """Determine the fill price for an order within one bar.

    The model is intentionally conservative:

    * A **market** order submitted after the bar's close (or at any time other
      than the open) fills near the close, plus impact. Filling at the *open*
      when the signal was computed on the close would be look-ahead bias, so
      the open is only used when the signal was available at the open.
    * A **limit** order fills only if the bar's range crosses the limit, and
      then at the limit price (not at a better price the trader could not
      have relied on).
    * A **stop** order triggers only if the bar's range crosses the stop, and
      fills at the stop price plus gap slippage. A gap through the stop fills
      at the *open*, not the stop, which is the single most common way a
      stop-loss backtest overstates performance.
    """
    liquidity = liquidity or LiquidityModel()
    qty = abs(float(quantity))
    result = {
        "filled": False, "fill_price": None, "quantity_filled": 0.0,
        "reason": "", "impact": 0.0, "gap_adjustment": 0.0,
    }
    if qty == 0:
        result["reason"] = "zero quantity"
        return result

    rng = bar_high - bar_low
    ref = bar_close

    if order_type == "market":
        base = bar_open if signal_time == "bar_open" else bar_close
        impact = liquidity.market_impact_offset(
            qty, bar_volume, bar_high, bar_low, side
        )
        fill = base + impact
        result.update(filled=True, fill_price=float(fill),
                      quantity_filled=qty, impact=float(impact),
                      reason="market order filled at close plus impact"
                      if signal_time != "bar_open" else "market order filled at open")

    elif order_type == "limit":
        if limit_price is None:
            result["reason"] = "limit order without a limit price"
            return result
        touched = (bar_low <= limit_price <= bar_high) if side == "long" \
            else (bar_high >= limit_price >= bar_low)
        if not touched:
            result["reason"] = f"limit {limit_price:g} not reached in the bar range"
            return result
        # A marketable limit that the bar traded through could fill better, but
        # relying on that is optimistic: assume the limit price.
        fill = limit_price
        result.update(filled=True, fill_price=float(fill), quantity_filled=qty,
                      reason="limit price reached within the bar")

    elif order_type in ("stop", "stop_limit"):
        if stop_price is None:
            result["reason"] = "stop order without a stop price"
            return result
        triggered = (bar_high >= stop_price) if side == "short" \
            else (bar_low <= stop_price)
        if not triggered:
            result["reason"] = f"stop {stop_price:g} not triggered in the bar range"
            return result
        # Gap handling: if the bar opened beyond the stop, the fill is the open.
        gapped = (side == "long" and bar_open <= stop_price) or \
                 (side == "short" and bar_open >= stop_price)
        if gapped:
            fill = bar_open
            gap_adj = abs(bar_open - stop_price)
            reason = (
                f"stop gapped through: filled at the open {bar_open:g}, not the "
                f"stop {stop_price:g} (adverse gap of {gap_adj:g})"
            )
        else:
            fill = stop_price
            gap_adj = 0.0
            reason = f"stop triggered at {stop_price:g}"
        impact = liquidity.market_impact_offset(
            qty, bar_volume, bar_high, bar_low, side
        )
        result.update(filled=True, fill_price=float(fill + impact),
                      quantity_filled=qty, impact=float(impact),
                      gap_adjustment=float(gap_adj), reason=reason)

    else:
        result["reason"] = f"order type {order_type!r} not supported by this simulator"
        return result

    if not np.isfinite(result["fill_price"]) or result["fill_price"] <= 0:
        result.update(filled=False, fill_price=None, quantity_filled=0.0,
                      reason="computed fill price is not positive; treated as unfilled")
        result["impact"] = 0.0
    return result


def average_daily_volume(
    volume: pd.Series, window: int = 20, min_periods: Optional[int] = None
) -> pd.Series:
    """Trailing ADV. Uses only bars up to and including each row."""
    return volume.rolling(window, min_periods=min_periods or window).mean()


def max_order_quantity(
    desired: float,
    bar_volume: Optional[float],
    participation_pct: float,
) -> float:
    """Size an order to the participation cap. Returns 0 when unknowable."""
    if bar_volume is None or bar_volume <= 0 or participation_pct <= 0:
        return 0.0
    return float(min(abs(desired), bar_volume * participation_pct))
