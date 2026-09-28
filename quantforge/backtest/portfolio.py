"""Portfolio accounting for backtests and paper trading.

One class serves both modes deliberately. Blueprint Section 58 requires paper
trading to use the same portfolio engine as live trading, and the only way to
guarantee that is for there to be a single implementation. A separate paper
portfolio is a parallel code path that will diverge.

Accounting conventions, all of which change reported P&L if violated:

* **Realised vs unrealised.** Realised P&L is booked at fill. Unrealised is
  marked to market and is *not* realised.
* **Cash is explicit.** A position's notional is not equity. Cash is tracked
  separately, so leverage and idle cash are visible rather than assumed away.
* **Fees reduce cash**, never cost-basis. Deducting fees from the entry price
  instead makes average cost disagree with actual cash spent.
* **Shorts** are handled by signed quantity, with proceeds credited to cash on
  the opening fill and the symmetric cost booked on close.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from quantforge.core.enums import OrderSide
from quantforge.core.errors import RiskError


@dataclass(slots=True)
class Position:
    """A single instrument's holding.

    ``quantity`` is signed: positive is long, negative is short. Keeping one
    signed field rather than separate long/short books is what makes an
    accidental unhedged short visible in a single number.
    """

    instrument_id: str
    quantity: float = 0.0
    average_entry_price: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    strategy_id: Optional[str] = None
    currency: str = "USD"
    opened_at: Optional[dt.datetime] = None
    last_price: float = 0.0
    entry_bar: Optional[object] = None

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-12

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.average_entry_price

    @property
    def unrealized_pnl(self) -> float:
        if self.is_flat or self.last_price <= 0:
            return 0.0
        return (self.last_price - self.average_entry_price) * self.quantity

    @property
    def gross_exposure(self) -> float:
        """Absolute notional. The quantity at risk regardless of direction."""
        return abs(self.market_value)

    @property
    def net_exposure(self) -> float:
        return self.market_value

    def as_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
            "last_price": self.last_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_fees": self.total_fees,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "strategy_id": self.strategy_id,
            "currency": self.currency,
        }


@dataclass(slots=True)
class Trade:
    """A completed round trip."""

    instrument_id: str
    entry_time: object
    exit_time: object
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    fees: float
    return_pct: float
    bars_held: int
    strategy_id: Optional[str] = None
    exit_reason: str = ""
    mae: float = 0.0     # maximum adverse excursion, in currency
    mfe: float = 0.0     # maximum favourable excursion

    def as_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time),
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "fees": self.fees,
            "return_pct": self.return_pct,
            "bars_held": self.bars_held,
            "strategy_id": self.strategy_id,
            "exit_reason": self.exit_reason,
            "mae": self.mae,
            "mfe": self.mfe,
        }


class Portfolio:
    """Cash, positions, and their accounting."""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        base_currency: str = "USD",
        allow_shorting: bool = False,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.base_currency = base_currency
        self.allow_shorting = allow_shorting
        self.cash = float(initial_cash)
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[object, float]] = []
        self._open_entries: dict[str, list[tuple[object, float, float, int]]] = {}

    # -- valuation --------------------------------------------------------

    @property
    def market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def equity(self) -> float:
        """Cash plus marked position value."""
        return self.cash + self.market_value

    @property
    def gross_exposure(self) -> float:
        return sum(p.gross_exposure for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return sum(p.net_exposure for p in self.positions.values())

    @property
    def leverage(self) -> float:
        eq = self.equity
        return float("inf") if eq <= 0 else self.gross_exposure / eq

    @property
    def position_count(self) -> int:
        return sum(1 for p in self.positions.values() if not p.is_flat)

    def exposure_by(self, key_fn) -> dict[str, float]:
        """Aggregate gross exposure by an arbitrary attribute (sector, class)."""
        out: dict[str, float] = {}
        for p in self.positions.values():
            if p.is_flat:
                continue
            k = key_fn(p.instrument_id) or "UNKNOWN"
            out[k] = out.get(k, 0.0) + p.gross_exposure
        return out

    def position_weight(self, instrument_id: str) -> float:
        eq = self.equity
        p = self.positions.get(instrument_id)
        if p is None or eq <= 0:
            return 0.0
        return p.gross_exposure / eq

    # -- mutation ---------------------------------------------------------

    def get_or_create(self, instrument_id: str, **kwargs) -> Position:
        if instrument_id not in self.positions:
            self.positions[instrument_id] = Position(
                instrument_id=instrument_id, **kwargs
            )
        return self.positions[instrument_id]

    def apply_fill(
        self,
        instrument_id: str,
        side: OrderSide | str,
        quantity: float,
        price: float,
        fees: float = 0.0,
        when=None,
        strategy_id: Optional[str] = None,
        bar_index: int = 0,
        exit_reason: str = "",
    ) -> Optional[Trade]:
        """Book a fill. Returns a :class:`Trade` when the position closes.

        Handles three cases: opening, adding, reducing, and reversing. A
        reversal is split so the closed portion books a trade and the residual
        opens a new position at the fill price rather than blending the two,
        which would corrupt the average entry price.
        """
        side = side if isinstance(side, OrderSide) else OrderSide(side)
        qty = float(quantity)
        if qty <= 0:
            raise RiskError(f"fill quantity must be positive, got {qty}")
        if price <= 0:
            raise RiskError(f"fill price must be positive, got {price}")
        signed = qty if side == OrderSide.LONG else -qty

        pos = self.get_or_create(instrument_id, strategy_id=strategy_id)
        if strategy_id and not pos.strategy_id:
            pos.strategy_id = strategy_id

        self.total_fees += fees
        pos.total_fees += fees
        closed_trade: Optional[Trade] = None

        current = pos.quantity
        if abs(current) < 1e-12:
            # Opening a fresh position.
            pos.quantity = signed
            pos.average_entry_price = price
            pos.opened_at = when
            pos.entry_bar = bar_index
            pos.last_price = price
            self._open_entries[instrument_id] = [
                (when, price, qty, bar_index)
            ]
            self.cash -= signed * price
            self.cash -= fees
            return None

        same_direction = (current > 0) == (signed > 0)

        if same_direction:
            # Adding: weighted-average the entry price.
            total_cost = abs(current) * pos.average_entry_price + abs(signed) * price
            new_qty = current + signed
            pos.average_entry_price = total_cost / abs(new_qty) if new_qty else 0.0
            pos.quantity = new_qty
            self.cash -= signed * price
            self.cash -= fees
            entries = self._open_entries.setdefault(instrument_id, [])
            entries.append((when, price, qty, bar_index))
            return None

        # Opposite direction: reduce, possibly close, possibly reverse.
        reduce_qty = min(abs(current), qty)
        direction = 1.0 if current > 0 else -1.0
        pnl = (price - pos.average_entry_price) * reduce_qty * direction
        pos.realized_pnl += pnl
        self.realized_pnl += pnl
        self.cash += direction * reduce_qty * price
        self.cash -= fees

        entries = self._open_entries.get(instrument_id, [])
        entry = entries[0] if entries else (when, pos.average_entry_price,
                                           reduce_qty, bar_index)

        # A position is only closed when the *entire* holding was consumed.
        # Confusing "the order was fully applied" with "the position was
        # fully closed" books a spurious trade every time a position is
        # partially scaled out, and leaves the remaining size with a zero
        # average entry price.
        fully_closed = reduce_qty >= abs(current) - 1e-12
        remaining = qty - reduce_qty

        if not fully_closed:
            pos.quantity = current + signed
            if entries:
                entries[0] = (entry[0], entry[1], entry[2] - reduce_qty, entry[3])
            return None

        pos.quantity = 0.0
        realised_entry_price = pos.average_entry_price
        pos.average_entry_price = 0.0
        self._open_entries.pop(instrument_id, None)
        # The P&L above was computed against the weighted-average entry price,
        # so the trade record must use that same price. Recording the first
        # fill's price instead would make `return_pct` inconsistent with `pnl`
        # whenever a position was built from more than one fill.
        cost = realised_entry_price * reduce_qty
        closed_trade = Trade(
            instrument_id=instrument_id,
            entry_time=entry[0], exit_time=when,
            quantity=direction * reduce_qty,
            entry_price=realised_entry_price, exit_price=price,
            pnl=pnl, fees=fees,
            return_pct=(pnl / cost) if cost else float("nan"),
            bars_held=max(0, bar_index - entry[3]),
            strategy_id=pos.strategy_id, exit_reason=exit_reason,
        )
        self.trades.append(closed_trade)

        if remaining > 1e-12:
            # Reversal: the residual opens a fresh position in the other
            # direction at the fill price, rather than blending into the
            # average entry of the position that just closed.
            pos.quantity = signed
            pos.average_entry_price = price
            pos.opened_at = when
            pos.entry_bar = bar_index
            pos.last_price = price
            self._open_entries[instrument_id] = [(when, price, remaining, bar_index)]
            self.cash -= signed * price
        return closed_trade

    def mark_to_market(self, prices: dict[str, float], when=None) -> None:
        """Update mark prices. Instruments absent from ``prices`` keep their
        last mark rather than being valued at zero.

        Valuing an unpriced holding at zero would show a fabricated 100% loss
        and could trip a drawdown kill switch on a data gap rather than a
        market move.
        """
        for iid, px in prices.items():
            pos = self.positions.get(iid)
            if pos is not None and px and px > 0:
                pos.last_price = float(px)
        if when is not None:
            self.equity_curve.append((when, self.equity))

    def record_equity(self, when, equity: Optional[float] = None) -> None:
        self.equity_curve.append((when, self.equity if equity is None else equity))

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype="float64")
        s = pd.Series([v for _, v in self.equity_curve],
                      index=pd.Index([t for t, _ in self.equity_curve], name="timestamp"))
        return s[~s.index.duplicated(keep="last")].sort_index()

    def close_all(
        self, prices: dict[str, float], when=None,
        reason: str = "end_of_backtest",
    ) -> list[Trade]:
        """Flatten everything. Open P&L is realised at the given prices."""
        out: list[Trade] = []
        for iid, pos in list(self.positions.items()):
            if pos.is_flat:
                continue
            px = prices.get(iid)
            if not px or px <= 0:
                continue
            # Closing a long means selling it, and closing a short means buying
            # it back. Passing the *same* side as the holding would open more
            # of it instead of flattening the position.
            side = OrderSide.SHORT if pos.quantity > 0 else OrderSide.LONG
            trade = self.apply_fill(
                iid, side, abs(pos.quantity), float(px), fees=0.0,
                when=when, strategy_id=pos.strategy_id,
                bar_index=len(self.equity_curve), exit_reason=reason,
            )
            if trade is not None:
                out.append(trade)
        return out

    def returns(self) -> pd.Series:
        """Equity returns, aligned to the equity series.

        The first return is measured against ``initial_cash`` so the series
        accounts for the full life of the portfolio including its first bar.
        Computed positionally rather than via index arithmetic so a string,
        integer, or datetime index all work; subtracting a Timedelta from a
        label only makes sense for a time index.
        """
        eq = self.equity_series()
        if eq.empty:
            return pd.Series(dtype="float64", name="return")
        prev = eq.to_numpy(dtype="float64").copy()
        prev[1:] = eq.to_numpy(dtype="float64")[:-1]
        prev[0] = self.initial_cash
        denom = np.where(np.abs(prev) > 0, prev, np.nan)
        out = (eq.to_numpy(dtype="float64") - prev) / denom
        return pd.Series(out, index=eq.index, name="return")

    def trade_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.as_dict() for t in self.trades])

    def position_frame(self) -> pd.DataFrame:
        rows = [p.as_dict() for p in self.positions.values() if not p.is_flat]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def snapshot(self) -> dict:
        return {
            "timestamp": None,
            "cash": self.cash,
            "equity": self.equity,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_fees": self.total_fees,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "leverage": self.leverage,
            "position_count": self.position_count,
        }

    def __repr__(self) -> str:
        return (
            f"Portfolio(equity={self.equity:,.2f}, cash={self.cash:,.2f}, "
            f"positions={self.position_count}, realized={self.realized_pnl:,.2f})"
        )
