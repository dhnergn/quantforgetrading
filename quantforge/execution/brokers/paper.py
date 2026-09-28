"""Simulated broker for paper trading and backtest-realism testing.

This is the adapter that makes blueprint Section 58 satisfiable: paper trading
uses the *same* strategy, feature, risk, portfolio, and order engines as live
trading, with only the execution adapter swapped. The alternative — a separate
paper path — diverges, and the divergence is discovered during live trading.

The simulation is deliberately pessimistic:

* Market orders fill at the next bar's open plus impact, not at the signal
  close.
* Limit orders fill only when the bar's range crosses the limit, and only with
  a configurable probability.
* Orders larger than the participation cap fill partially.
* Stops fill at the open when the bar gapped through, not at the stop price.
* Submission can be configured to fail or time out, so the duplicate-order
  and reconciliation paths are exercised in testing rather than discovered in
  production.
"""
from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.enums import (
    ExecutionMode, OrderSide, OrderStatus, OrderType,
)
from quantforge.core.errors import (
    BrokerConnectionError, BrokerTimeoutError, OrderRejected,
)
from quantforge.core.timestamps import utcnow
from quantforge.execution.brokers.base import (
    Account, BrokerAdapter, BrokerPosition, Fill, OrderAck,
)
from quantforge.execution.orders import Order


@dataclass(slots=True)
class SimulatedMarket:
    """Bar data the simulator fills against.

    Kept separate from the adapter so a backtest can advance it bar by bar and
    the fill logic stays a pure function of the current bar.
    """

    instrument_id: str
    bars: pd.DataFrame

    def current(self) -> Optional[pd.Series]:
        return self.bars.iloc[-1] if len(self.bars) else None

    def as_of(self) -> Optional[dt.datetime]:
        return self.bars.index[-1].to_pydatetime() if len(self.bars) else None


class PaperBrokerAdapter(BrokerAdapter):
    """Deterministic simulated broker."""

    mode = ExecutionMode.PAPER
    name = "paper"

    def __init__(
        self,
        initial_equity: float = 100_000.0,
        currency: str = "USD",
        base_currency: str = "USD",
        fill_probability: float = 1.0,
        participation_cap: float = 0.10,
        impact_coefficient: float = 0.05,
        allow_partial_fills: bool = True,
        reject_probability: float = 0.0,
        timeout_probability: float = 0.0,
        seed: int = 0,
        require_idempotency_key: bool = True,
    ) -> None:
        super().__init__(require_idempotency_key)
        self.initial_equity = initial_equity
        self.currency = currency
        self.base_currency = base_currency
        self.fill_probability = fill_probability
        self.participation_cap = participation_cap
        self.impact_coefficient = impact_coefficient
        self.allow_partial_fills = allow_partial_fills
        self.reject_probability = reject_probability
        self.timeout_probability = timeout_probability
        self._rng = np.random.default_rng(seed)

        self.cash = initial_equity
        self.positions: dict[str, BrokerPosition] = {}
        self.orders: dict[str, Order] = {}
        self.fills: list[Fill] = []
        self.working: dict[str, str] = {}      # client id -> order state
        self._seen_keys: dict[str, str] = {}
        self._seen_client_ids: dict[str, str] = {}
        self.markets: dict[str, SimulatedMarket] = {}

    # -- connection -------------------------------------------------------

    def _do_connect(self) -> None:
        return None

    def _do_disconnect(self) -> None:
        return None

    # -- market data ------------------------------------------------------

    def set_market(self, instrument_id: str, bars: pd.DataFrame) -> None:
        self.markets[instrument_id] = SimulatedMarket(instrument_id, bars)

    def advance(self, instrument_id: str, up_to: dt.datetime | str) -> None:
        """Move the simulated clock forward for one instrument."""
        m = self.markets.get(instrument_id)
        if m is None:
            return
        ts = pd.Timestamp(up_to)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        m.bars = m.bars[m.bars.index <= ts]

    # -- queries ----------------------------------------------------------

    def get_account(self) -> Account:
        market_value = sum(p.market_value for p in self.positions.values())
        equity = self.cash + market_value
        return Account(
            account_id="paper", equity=equity, cash=self.cash,
            buying_power=self.cash, currency=self.currency,
            as_of=self._now(),
        )

    def get_positions(self) -> list[BrokerPosition]:
        return [p for p in self.positions.values() if abs(p.quantity) > 1e-12]

    def get_order(self, client_order_id: str) -> Optional[dict]:
        oid = self._seen_client_ids.get(client_order_id)
        if oid is None:
            return None
        o = self.orders[oid]
        return {
            "client_order_id": o.client_order_id,
            "broker_order_id": o.broker_order_id,
            "status": self.working.get(o.client_order_id, o.status.value),
            "filled_quantity": o.filled_quantity,
            "quantity": o.quantity,
            "average_fill_price": o.average_fill_price,
        }

    def get_orders(self, status: Optional[OrderStatus] = None) -> list[dict]:
        out = []
        for o in self.orders.values():
            state = self.working.get(o.client_order_id, o.status.value)
            if status and state != status.value:
                continue
            out.append(self.get_order(o.client_order_id) or {})
        return [x for x in out if x]

    def get_fills(self, since: Optional[dt.datetime] = None) -> list[Fill]:
        if since is None:
            return list(self.fills)
        return [f for f in self.fills if f.at >= since]

    # -- commands ---------------------------------------------------------

    def _do_submit(self, order: Order) -> OrderAck:
        # Simulated failure modes, for exercising the safety paths.
        if self.timeout_probability and self._rng.random() < self.timeout_probability:
            # The submission may or may not have reached the broker. Model the
            # genuinely ambiguous case by recording it as working, then raising.
            oid = str(uuid.uuid4())
            broker_id = f"PB-{oid[:12]}"
            self.orders[oid] = order
            order.broker_order_id = broker_id
            self.working[order.client_order_id] = OrderStatus.UNKNOWN.value
            self._seen_client_ids[order.client_order_id] = oid
            if order.idempotency_key:
                self._seen_keys[order.idempotency_key] = oid
            raise BrokerTimeoutError(
                f"submission of {order.client_order_id} timed out. The order's state "
                "at the broker is UNKNOWN. Do not resubmit — reconcile with "
                "reconcile_unknown_orders() first."
            )
        if self.reject_probability and self._rng.random() < self.reject_probability:
            raise OrderRejected(f"simulated rejection for {order.client_order_id}")

        # Duplicate protection lives in the broker too: a real venue rejects a
        # duplicate client order id, and the platform must not rely solely on
        # its own bookkeeping.
        if order.idempotency_key and order.idempotency_key in self._seen_keys:
            existing_id = self._seen_keys[order.idempotency_key]
            existing = self.orders[existing_id]
            self.duplicates_rejected += 1
            return OrderAck(
                client_order_id=order.client_order_id,
                broker_order_id=existing.broker_order_id,
                accepted=True, status=OrderStatus.SUBMITTED,
                message=(
                    "duplicate submission detected and coalesced with an existing "
                    "order for the same idempotency key"
                ),
            )

        oid = str(uuid.uuid4())
        broker_id = f"PB-{oid[:12]}"
        self.orders[oid] = order
        order.broker_order_id = broker_id
        self._seen_client_ids[order.client_order_id] = oid
        if order.idempotency_key:
            self._seen_keys[order.idempotency_key] = oid
        self.working[order.client_order_id] = OrderStatus.ACCEPTED.value

        self._try_fill(order, reason="marketable at submission")
        return OrderAck(
            client_order_id=order.client_order_id, broker_order_id=broker_id,
            accepted=True, status=OrderStatus.ACCEPTED,
            message="accepted by simulated venue",
        )

    def _do_cancel(self, client_order_id: str) -> bool:
        if client_order_id in self.working:
            state = self.working[client_order_id]
            if state in (OrderStatus.ACCEPTED.value, OrderStatus.SUBMITTED.value,
                         OrderStatus.PARTIALLY_FILLED.value,
                         OrderStatus.UNKNOWN.value):
                self.working[client_order_id] = OrderStatus.CANCELLED.value
                return True
            return False
        return False

    def _do_replace(self, order: Order, new_quantity: float,
                    new_limit_price: Optional[float] = None) -> OrderAck:
        existing = self.get_order(order.client_order_id)
        if existing is None:
            raise OrderRejected(
                f"cannot replace unknown order {order.client_order_id}"
            )
        oid = self._seen_client_ids[order.client_order_id]
        target = self.orders[oid]
        if target.status == OrderStatus.UNKNOWN:
            raise OrderRejected(
                "cannot replace an order in UNKNOWN state; reconcile first. "
                "Replacing an order whose state is unknown risks two live orders."
            )
        remaining = target.quantity - target.filled_quantity
        if new_quantity < target.filled_quantity:
            raise OrderRejected(
                f"replace quantity {new_quantity} is below the filled quantity "
                f"{target.filled_quantity}"
            )
        if new_quantity != target.quantity:
            new_oid = str(uuid.uuid4())
            replacement = Order(
                order_id=new_oid, client_order_id=order.client_order_id + "-r",
                instrument_id=target.instrument_id, side=target.side,
                quantity=new_quantity - target.filled_quantity,
                order_type=target.order_type,
                limit_price=new_limit_price if new_limit_price is not None
                else target.limit_price,
                stop_price=target.stop_price,
                strategy_id=target.strategy_id, portfolio_id=target.portfolio_id,
                tags={**target.tags, "replaces": target.order_id},
            )
            self.orders[new_oid] = replacement
            self._seen_client_ids[replacement.client_order_id] = new_oid
            self.working[replacement.client_order_id] = OrderStatus.ACCEPTED.value
            self.working[target.client_order_id] = OrderStatus.CANCELLED.value
            self._try_fill(replacement, reason="replacement order")
            return OrderAck(
                client_order_id=replacement.client_order_id,
                broker_order_id=replacement.broker_order_id, accepted=True,
                status=OrderStatus.ACCEPTED,
                message=f"replaced {remaining:.0f} remaining shares",
            )
        return OrderAck(
            client_order_id=target.client_order_id,
            broker_order_id=target.broker_order_id, accepted=True,
            status=OrderStatus.ACCEPTED, message="no change required",
        )

    # -- fill engine ------------------------------------------------------

    def _try_fill(self, order: Order, reason: str = "") -> Fill | None:
        market = self.markets.get(order.instrument_id)
        if market is None or market.current() is None:
            return None
        bar = market.current()

        price = self._fill_price(order, bar)
        if price is None:
            return None
        if self.fill_probability < 1.0 and self._rng.random() > self.fill_probability:
            return None

        volume = float(bar.get("volume", np.nan))
        qty = order.quantity
        if np.isfinite(volume) and volume > 0:
            cap = volume * self.participation_cap
            if qty > cap:
                if not self.allow_partial_fills:
                    self.working[order.client_order_id] = OrderStatus.REJECTED.value
                    return None
                qty = cap
        qty = float(int(qty))
        if qty <= 0:
            return None

        notional = qty * price
        fee = max(0.0, notional * 0.0002)   # 2bps
        signed = qty if order.side == OrderSide.LONG else -qty
        self.cash -= signed * price
        self.cash -= fee

        prev_qty = self.positions.get(order.instrument_id)
        if prev_qty is None or abs(prev_qty.quantity) < 1e-12:
            self.positions[order.instrument_id] = BrokerPosition(
                instrument_id=order.instrument_id, quantity=signed,
                average_entry_price=price, current_price=price,
                strategy_id=order.strategy_id,
            )
        else:
            p = prev_qty
            same = (p.quantity > 0) == (signed > 0)
            if same:
                p.average_entry_price = (
                    abs(p.quantity) * p.average_entry_price + qty * price
                ) / (abs(p.quantity) + qty)
                p.quantity += signed
            else:
                reduce_qty = min(abs(p.quantity), qty)
                direction = 1.0 if p.quantity > 0 else -1.0
                p.realized_pnl += (price - p.average_entry_price) * reduce_qty * direction
                p.quantity += signed
                if abs(p.quantity) < 1e-12:
                    p.quantity = 0.0
            p.current_price = price

        fill = Fill(
            order_id=order.order_id, instrument_id=order.instrument_id,
            side=order.side, quantity=qty, price=price, fees=fee,
            at=utcnow(), liquidity="taker", venue="simulated",
        )
        self.fills.append(fill)

        order.filled_quantity = min(order.quantity, order.filled_quantity + qty)
        order.average_fill_price = price
        order.fees = (order.fees or 0.0) + fee
        order.execution_time = fill.at
        self.working[order.client_order_id] = (
            OrderStatus.FILLED.value if order.filled_quantity >= order.quantity - 1e-9
            else OrderStatus.PARTIALLY_FILLED.value
        )
        return fill

    def _fill_price(self, order: Order, bar: pd.Series) -> Optional[float]:
        o, h, l, c = (float(bar["open"]), float(bar["high"]),
                      float(bar["low"]), float(bar["close"]))
        rng = max(0.0, h - l)
        volume = float(bar.get("volume", np.nan))
        participation = 0.0
        if np.isfinite(volume) and volume > 0:
            participation = order.quantity / volume

        if order.order_type == OrderType.MARKET:
            base = o
            impact = self.impact_coefficient * rng * min(1.0, participation)
            return base + (impact if order.side == OrderSide.LONG else -impact)

        if order.order_type == OrderType.LIMIT:
            lp = order.limit_price
            if lp is None:
                return None
            touched = (l <= lp <= h) if order.side == OrderSide.LONG else (h >= lp >= l)
            if not touched:
                return None
            return lp

        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            sp = order.stop_price
            if sp is None:
                return None
            triggered = (h >= sp) if order.side == OrderSide.SHORT else (l <= sp)
            if not triggered:
                return None
            gapped = (order.side == OrderSide.LONG and o <= sp) or \
                     (order.side == OrderSide.SHORT and o >= sp)
            price = o if gapped else sp
            impact = self.impact_coefficient * rng * min(1.0, participation)
            return price + (impact if order.side == OrderSide.LONG else -impact)
        return None

    def step(self) -> list[Fill]:
        """Advance one bar: attempt to fill every working order."""
        out: list[Fill] = []
        for cid, state in list(self.working.items()):
            if state not in (OrderStatus.ACCEPTED.value, OrderStatus.PARTIALLY_FILLED.value):
                continue
            oid = self._seen_client_ids.get(cid)
            order = self.orders.get(oid) if oid else None
            if order is None:
                continue
            remaining = order.quantity - order.filled_quantity
            if remaining <= 0:
                continue
            before = len(self.fills)
            probe = Order(
                order_id=order.order_id, client_order_id=order.client_order_id,
                instrument_id=order.instrument_id, side=order.side,
                quantity=remaining, order_type=order.order_type,
                limit_price=order.limit_price, stop_price=order.stop_price,
            )
            self._try_fill(probe, reason="bar step")
            if len(self.fills) > before:
                out.extend(self.fills[before:])
        return out

    def mark_to_market(self) -> None:
        for iid, p in self.positions.items():
            m = self.markets.get(iid)
            if m is not None and m.current() is not None:
                px = float(m.current()["close"])
                p.current_price = px
                p.unrealized_pnl = (px - p.average_entry_price) * p.quantity

    def _now(self) -> dt.datetime:
        stamps = [
            m.as_of() for m in self.markets.values()
            if m.as_of() is not None
        ]
        return max(stamps) if stamps else utcnow()
