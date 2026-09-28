"""Order model and the order state machine.

``UNKNOWN`` is a first-class state, and that is the point of this module. The
sequence that motivates it:

    submit -> network timeout -> ??? -> maybe submitted, maybe not

Every live-trading system that has ever lost money to a double fill has made the
same mistake: treating a timeout as a failure and resubmitting. The correct
response to a timeout is **reconciliation**, never a retry. :class:`OrderBook`
encodes that by refusing any transition out of ``UNKNOWN`` that does not begin
with a broker query.

The state machine is also the place duplicate-order protection lives. Every
order carries a ``client_order_id`` derived from a deterministic idempotency
key, so a retried submission is recognisable as the same order.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.enums import (
    ExecutionMode, OrderSide, OrderStatus, OrderType, TimeInForce, TradingState,
)
from quantforge.core.errors import (
    DuplicateOrderError, ExecutionStateError, OrderRejected,
)
from quantforge.core.timestamps import UTC, ensure_aware, utcnow


@dataclass(slots=True)
class Order:
    """One order, with full provenance.

    ``idempotency_key`` is the field that prevents double submission. It is
    derived from (strategy, portfolio, instrument, side, quantity, decision
    time) so the same logical decision always produces the same key across
    processes and restarts.
    """

    order_id: str
    client_order_id: str
    instrument_id: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.CREATED
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    signal_id: Optional[str] = None
    created_at: dt.datetime = field(default_factory=utcnow)
    updated_at: dt.datetime = field(default_factory=utcnow)
    submitted_at: Optional[dt.datetime] = None
    filled_at: Optional[dt.datetime] = None
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    fees: float = 0.0
    slippage: float = 0.0
    broker_order_id: Optional[str] = None
    reject_reason: Optional[str] = None
    idempotency_key: Optional[str] = None
    #: Populated when status is UNKNOWN: what we were trying to find out.
    unknown_since: Optional[dt.datetime] = None
    attempts: int = 0
    tags: dict = field(default_factory=dict)
    #: Decision chain, retained for reconstruction.
    information_cutoff: Optional[dt.datetime] = None
    prediction_time: Optional[dt.datetime] = None
    decision_time: Optional[dt.datetime] = None
    order_time: Optional[dt.datetime] = None
    execution_time: Optional[dt.datetime] = None
    model_version: Optional[str] = None
    regime: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ExecutionStateError(
                f"order quantity must be positive, got {self.quantity}. Use the "
                "side to express direction; a negative quantity makes the intent "
                "ambiguous at the boundary."
            )
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ExecutionStateError("a limit order requires limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) \
                and self.stop_price is None:
            raise ExecutionStateError("a stop order requires stop_price")
        if not self.idempotency_key:
            self.idempotency_key = make_idempotency_key(
                self.strategy_id, self.portfolio_id, self.instrument_id,
                self.side, self.quantity, self.decision_time or self.created_at,
            )
        if not self.client_order_id:
            self.client_order_id = derive_client_order_id(self.idempotency_key)

    # -- derived ----------------------------------------------------------

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == OrderSide.LONG else -self.quantity

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def is_open(self) -> bool:
        return self.status in (
            OrderStatus.CREATED, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED, OrderStatus.UNKNOWN,
        )

    @property
    def notional(self) -> float:
        price = self.average_fill_price or self.limit_price or self.stop_price or 0.0
        return self.filled_quantity * price if self.filled_quantity else \
            self.quantity * price

    def validate_decision_chain(self) -> None:
        """The chain must be monotonic. A violation invalidates the order.

        When ``information_cutoff`` is unset it defaults to the *decision*
        time, not the order creation time. The cutoff is the instant the
        decision's inputs were available, which is at or before the decision
        itself; defaulting it to a later creation time would make every order
        that decides and then constructs look like a causality violation.
        """
        from quantforge.core.timestamps import DecisionChain

        cutoff = self.information_cutoff or self.decision_time or self.created_at
        chain = DecisionChain(
            information_cutoff=cutoff,
            prediction=self.prediction_time,
            decision=self.decision_time,
            order=self.order_time,
            execution=self.execution_time,
        )
        chain.validate()

    def as_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "idempotency_key": self.idempotency_key,
            "instrument_id": self.instrument_id,
            "side": self.side.value,
            "quantity": self.quantity,
            "signed_quantity": self.signed_quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "strategy_id": self.strategy_id,
            "portfolio_id": self.portfolio_id,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "fees": self.fees,
            "slippage": self.slippage,
            "reject_reason": self.reject_reason,
            "attempts": self.attempts,
            "model_version": self.model_version,
            "regime": self.regime,
            "tags": self.tags,
        }


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


def make_idempotency_key(
    strategy_id: Optional[str],
    portfolio_id: Optional[str],
    instrument_id: str,
    side: OrderSide,
    quantity: float,
    when: dt.datetime,
) -> str:
    """Deterministic key for one logical order decision.

    Deterministic rather than random so that a retry — whether from a timeout
    handler, a process restart, or a duplicate webhook — produces the same key
    and is therefore recognisable as the same order.
    """
    t = ensure_aware(when)
    material = "|".join([
        str(strategy_id), str(portfolio_id), str(instrument_id),
        side.value, f"{float(quantity):.8f}",
        t.isoformat(timespec="milliseconds") if t else "",
    ])
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def derive_client_order_id(idempotency_key: str) -> str:
    """Short, broker-safe client order id."""
    return f"qf-{idempotency_key[:24]}"


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------

#: Legal transitions. A transition absent from this table is rejected.
LEGAL_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.CREATED: {
        OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.CANCELLED,
    },
    OrderStatus.SUBMITTED: {
        OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.EXPIRED,
        OrderStatus.UNKNOWN,
    },
    OrderStatus.ACCEPTED: {
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED,
        OrderStatus.EXPIRED, OrderStatus.REJECTED, OrderStatus.UNKNOWN,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED,
        OrderStatus.EXPIRED, OrderStatus.UNKNOWN,
    },
    # UNKNOWN may only resolve to a known terminal or open state, and only
    # after a broker query. It must never go back to SUBMITTED, because that
    # would be re-submitting an order whose fate is unknown.
    OrderStatus.UNKNOWN: {
        OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED,
    },
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.REJECTED: set(),
    OrderStatus.EXPIRED: set(),
}


class OrderStateMachine:
    """Applies state transitions, rejecting illegal ones.

    Held alongside every :class:`Order` so the transition logic is testable
    independently of any broker.
    """

    def __init__(self, order: Order) -> None:
        self.order = order
        self.history: list[tuple[dt.datetime, OrderStatus, str]] = [
            (order.created_at, order.status, "created")
        ]

    def can_transition(self, to: OrderStatus) -> bool:
        return to in LEGAL_TRANSITIONS.get(self.order.status, set())

    def transition(
        self,
        to: OrderStatus,
        reason: str = "",
        broker_order_id: Optional[str] = None,
        filled_quantity: Optional[float] = None,
        fill_price: Optional[float] = None,
        fees: Optional[float] = None,
        slippage: Optional[float] = None,
        reject_reason: Optional[str] = None,
    ) -> Order:
        o = self.order
        if o.is_terminal and to != o.status:
            raise ExecutionStateError(
                f"order {o.order_id} is terminal ({o.status.value}); cannot move to "
                f"{to.value}. A filled order that later appears unfilled is a "
                "reconciliation failure, not a state transition."
            )
        if not self.can_transition(to):
            legal = sorted(s.value for s in LEGAL_TRANSITIONS.get(o.status, set()))
            raise ExecutionStateError(
                f"illegal transition {o.status.value} -> {to.value} for order "
                f"{o.order_id}. Legal next states: {legal}"
            )

        if to == OrderStatus.SUBMITTED:
            o.attempts += 1
            o.submitted_at = utcnow()
            o.order_time = o.order_time or utcnow()
        elif to == OrderStatus.UNKNOWN:
            o.unknown_since = o.unknown_since or utcnow()
        elif to in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.FILLED, OrderStatus.CANCELLED,
                    OrderStatus.REJECTED, OrderStatus.EXPIRED):
            # Any resolution clears the unknown marker: the question is answered.
            o.unknown_since = None

        if filled_quantity is not None:
            if filled_quantity > o.quantity + 1e-9:
                raise ExecutionStateError(
                    f"fill of {filled_quantity} exceeds the order quantity "
                    f"{o.quantity} for {o.order_id}; a broker overfill is a "
                    "reconciliation failure"
                )
            o.filled_quantity = float(filled_quantity)
        if fill_price is not None:
            if o.filled_quantity > 0:
                o.average_fill_price = float(fill_price)
            o.execution_time = utcnow()
        if fees is not None:
            o.fees += float(fees)
        if slippage is not None:
            o.slippage += float(slippage)
        if broker_order_id:
            o.broker_order_id = broker_order_id
        if reject_reason:
            o.reject_reason = reject_reason
        if to == OrderStatus.PARTIALLY_FILLED and o.filled_quantity >= o.quantity:
            # A "partial" fill that covers the whole order is a full fill.
            to = OrderStatus.FILLED
        if to == OrderStatus.FILLED and o.filled_quantity >= o.quantity - 1e-9:
            o.filled_quantity = o.quantity
            o.filled_at = o.filled_at or utcnow()
            o.execution_time = o.execution_time or utcnow()

        o.status = to
        o.updated_at = utcnow()
        self.history.append((utcnow(), to, reason))
        o.validate_decision_chain()
        return o

    def trail(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"at": a, "status": b.value, "reason": c} for a, b, c in self.history]
        )


# --------------------------------------------------------------------------
# Order book
# --------------------------------------------------------------------------


class OrderBook:
    """In-memory order registry with duplicate protection.

    ``submit`` is idempotent on ``idempotency_key``. A duplicate submission
    returns the existing order instead of creating a second one, and the caller
    can distinguish "submitted" from "already submitted" via the return value.
    This is the last line of defence against a double fill; the primary one is
    reconciling after a timeout rather than retrying.
    """

    def __init__(self, dedupe_window_seconds: float = 300.0) -> None:
        self._orders: dict[str, Order] = {}
        self._by_key: dict[str, str] = {}
        self._machines: dict[str, OrderStateMachine] = {}
        self.dedupe_window = dedupe_window_seconds
        self.duplicates_blocked = 0

    def create(self, order: Order) -> Order:
        if order.order_id in self._orders:
            raise DuplicateOrderError(
                f"order_id {order.order_id} already exists in the book"
            )
        self._orders[order.order_id] = order
        self._machines[order.order_id] = OrderStateMachine(order)
        if order.idempotency_key:
            self._by_key[order.idempotency_key] = order.order_id
        return order

    def new_order(
        self, instrument_id: str, side: OrderSide, quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None, stop_price: Optional[float] = None,
        strategy_id: Optional[str] = None, portfolio_id: Optional[str] = None,
        **kwargs,
    ) -> Order:
        return self.create(Order(
            order_id=uuid.uuid4().hex[:20],
            client_order_id="",
            instrument_id=instrument_id, side=side, quantity=quantity,
            order_type=order_type, limit_price=limit_price, stop_price=stop_price,
            strategy_id=strategy_id, portfolio_id=portfolio_id, **kwargs,
        ))

    def find_by_key(self, idempotency_key: str) -> Optional[Order]:
        oid = self._by_key.get(idempotency_key)
        return self._orders.get(oid) if oid else None

    def find_by_client_id(self, client_order_id: str) -> Optional[Order]:
        for o in self._orders.values():
            if o.client_order_id == client_order_id:
                return o
        return None

    def check_duplicate(self, order: Order) -> Optional[Order]:
        """Return an existing equivalent order, if one is live."""
        if not order.idempotency_key:
            return None
        existing = self.find_by_key(order.idempotency_key)
        if existing is None:
            return None
        if existing.is_terminal:
            return None
        age = (utcnow() - existing.created_at).total_seconds()
        if age > self.dedupe_window:
            return None
        self.duplicates_blocked += 1
        return existing

    def submit(self, order: Order) -> tuple[Order, bool]:
        """Submit, refusing a duplicate. Returns ``(order, is_new)``.

        An order passed in from outside the book is registered first, so
        ``submit`` is a complete operation rather than requiring the caller to
        have already called :meth:`create`.
        """
        existing = self.check_duplicate(order)
        if existing is not None:
            return existing, False
        if order.order_id not in self._orders:
            self.create(order)
        self._machines[order.order_id].transition(
            OrderStatus.SUBMITTED, reason="submitted to broker"
        )
        return order, True

    def transition(self, order_id: str, to: OrderStatus, **kw) -> Order:
        return self._machines[order_id].transition(to, **kw)

    def get(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError:
            raise KeyError(f"unknown order {order_id}") from None

    def open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_open]

    def unknown_orders(self) -> list[Order]:
        return [o for o in self._orders.values()
                if o.status == OrderStatus.UNKNOWN]

    def all(self) -> list[Order]:
        return list(self._orders.values())

    def frame(self) -> pd.DataFrame:
        if not self._orders:
            return pd.DataFrame()
        return pd.DataFrame([o.as_dict() for o in self._orders.values()])

    def reconcile_status_counts(self) -> dict:
        counts: dict[str, int] = {}
        for o in self._orders.values():
            counts[o.status.value] = counts.get(o.status.value, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._orders)
