"""Broker adapter abstraction.

``BrokerAdapter`` is the only interface the execution engine knows. No broker
specific behaviour is permitted anywhere above this interface — that is the
requirement in blueprint Section 61, and the reason the whole execution layer
is written against an abstract base class rather than against a vendor SDK.

Adapters communicate in three ways, and the distinction matters for safety:

* **Queries** return data or raise.
* **Commands** (``submit``/``cancel``) return an acknowledgement, which is
  *not* a fill.
* **Events** arrive asynchronously and are reconciled into the order book.

A submission that times out raises :class:`BrokerTimeoutError`. It never
returns a synthetic rejection, because a rejection would invite a retry, and a
retry after an unknown outcome is how a position gets doubled.
"""
from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.enums import ExecutionMode, OrderSide, OrderStatus, OrderType
from quantforge.core.errors import (
    BrokerConnectionError, BrokerError, BrokerTimeoutError, OrderRejected,
)
from quantforge.core.timestamps import utcnow
from quantforge.execution.orders import Order


# ==========================================================================
# Value objects
# ==========================================================================


@dataclass(slots=True)
class Account:
    account_id: str
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"
    margin_used: float = 0.0
    day_trading_buying_power: Optional[float] = None
    pattern_day_trader: bool = False
    as_of: dt.datetime = field(default_factory=utcnow)

    def as_dict(self) -> dict:
        return {
            "account_id": self.account_id, "equity": self.equity, "cash": self.cash,
            "buying_power": self.buying_power, "currency": self.currency,
            "margin_used": self.margin_used,
            "day_trading_buying_power": self.day_trading_buying_power,
            "pattern_day_trader": self.pattern_day_trader,
            "as_of": self.as_of.isoformat(),
        }


@dataclass(slots=True)
class BrokerPosition:
    instrument_id: str
    quantity: float
    average_entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    currency: str = "USD"
    strategy_id: Optional[str] = None

    @property
    def market_value(self) -> float:
        px = self.current_price or self.average_entry_price
        return self.quantity * px

    def as_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id, "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl, "realized_pnl": self.realized_pnl,
            "currency": self.currency, "strategy_id": self.strategy_id,
        }


@dataclass(slots=True)
class Fill:
    order_id: str
    instrument_id: str
    side: OrderSide
    quantity: float
    price: float
    fees: float = 0.0
    at: dt.datetime = field(default_factory=utcnow)
    liquidity: str = "unknown"      # maker or taker
    venue: Optional[str] = None

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == OrderSide.LONG else -self.quantity

    def as_dict(self) -> dict:
        return {
            "order_id": self.order_id, "instrument_id": self.instrument_id,
            "side": self.side.value, "quantity": self.quantity, "price": self.price,
            "fees": self.fees, "at": self.at.isoformat(),
            "liquidity": self.liquidity, "venue": self.venue,
        }


@dataclass(slots=True)
class OrderAck:
    """Acknowledgement of a *submission*, not a fill.

    ``accepted`` is true when the broker took responsibility for the order. The
    order is then live: it may fill partially, fully, or be rejected later.
    Treating an ack as a fill is the second most common cause of a position
    mismatch after double submission.
    """

    client_order_id: str
    broker_order_id: Optional[str]
    accepted: bool
    status: OrderStatus = OrderStatus.SUBMITTED
    message: str = ""
    received_at: dt.datetime = field(default_factory=utcnow)
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id, "accepted": self.accepted,
            "status": self.status.value, "message": self.message,
            "received_at": self.received_at.isoformat(),
        }


# ==========================================================================
# Abstract adapter
# ==========================================================================


class BrokerAdapter(abc.ABC):
    """The only broker interface the platform knows.

    Adapters are responsible for translating the broker's vocabulary into these
    types and nothing else. The base class supplies the safety behaviour that
    every adapter must have: idempotent submission keys, no retry after a
    timeout, and a query for every order whose state is unknown.
    """

    #: Reported by the adapter; ``live`` requires the authorisation gate.
    mode: ExecutionMode = ExecutionMode.PAPER
    name: str = "abstract"

    def __init__(self, require_idempotency_key: bool = True) -> None:
        self.require_idempotency_key = require_idempotency_key
        self._connected = False
        self.submission_log: list[dict] = []
        self.duplicates_rejected = 0

    # -- connection -------------------------------------------------------

    @abc.abstractmethod
    def _do_connect(self) -> None:
        ...

    def connect(self) -> bool:
        self._do_connect()
        self._connected = True
        return True

    @abc.abstractmethod
    def _do_disconnect(self) -> None:
        ...

    def disconnect(self) -> None:
        self._do_disconnect()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def _require_connection(self) -> None:
        if not self._connected:
            raise BrokerConnectionError(
                f"{self.name} adapter is not connected. Open orders submitted while "
                "disconnected have unknown state and require reconciliation, so "
                "submission is refused rather than queued."
            )

    def health_check(self) -> dict:
        return {
            "adapter": self.name, "mode": self.mode.value,
            "connected": self._connected, "at": utcnow().isoformat(),
        }

    # -- queries ----------------------------------------------------------

    @abc.abstractmethod
    def get_account(self) -> Account:
        ...

    @abc.abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        ...

    @abc.abstractmethod
    def get_order(self, client_order_id: str) -> Optional[dict]:
        """Query one order's authoritative state. Used to resolve UNKNOWN."""

    @abc.abstractmethod
    def get_orders(self, status: Optional[OrderStatus] = None) -> list[dict]:
        ...

    @abc.abstractmethod
    def get_fills(self, since: Optional[dt.datetime] = None) -> list[Fill]:
        ...

    def get_market_data(self, symbols: Sequence[str]) -> dict:
        raise NotImplementedError(
            f"{self.name} does not provide market data through this adapter. "
            "Quote data should come from a market-data provider, not the broker."
        )

    # -- commands ---------------------------------------------------------

    @abc.abstractmethod
    def _do_submit(self, order: Order) -> OrderAck:
        ...

    def submit_order(self, order: Order) -> OrderAck:
        """Submit an order. Never retries, and never converts a timeout to a
        rejection."""
        self._require_connection()
        if self.require_idempotency_key and not order.idempotency_key:
            raise BrokerError(
                "order has no idempotency key. Without one, a retry after a "
                "timeout cannot be recognised as a duplicate, so submission is "
                "refused."
            )
        ack = self._do_submit(order)
        self.submission_log.append({
            "client_order_id": order.client_order_id,
            "idempotency_key": order.idempotency_key,
            "at": utcnow().isoformat(),
            "accepted": ack.accepted,
            "status": ack.status.value,
            "broker_order_id": ack.broker_order_id,
        })
        return ack

    @abc.abstractmethod
    def _do_cancel(self, client_order_id: str) -> bool:
        ...

    def cancel_order(self, client_order_id: str) -> bool:
        self._require_connection()
        return self._do_cancel(client_order_id)

    @abc.abstractmethod
    def _do_replace(self, order: Order, new_quantity: float,
                    new_limit_price: Optional[float] = None) -> OrderAck:
        ...

    def replace_order(self, order: Order, new_quantity: float,
                      new_limit_price: Optional[float] = None) -> OrderAck:
        self._require_connection()
        return self._do_replace(order, new_quantity, new_limit_price)

    def reconcile_unknown_orders(self, order_ids: Sequence[str]) -> dict:
        """Resolve orders whose state is unknown. The only correct response to
        a submission timeout.

        Returns a per-order resolution so a caller can distinguish "it was
        never sent" (safe to retry) from "it is live" (must cancel, not resubmit).
        """
        out: dict[str, Any] = {}
        for cid in order_ids:
            try:
                remote = self.get_order(cid)
            except BrokerTimeoutError:
                out[cid] = {"resolved": False, "action": "retry_query",
                            "detail": "query also timed out; the broker's state is "
                                      "still unknown. Do not resubmit."}
                continue
            except BrokerConnectionError as exc:
                out[cid] = {"resolved": False, "action": "halt",
                            "detail": str(exc)}
                continue
            if remote is None:
                out[cid] = {
                    "resolved": True, "state": "not_found",
                    "action": "safe_to_resubmit",
                    "detail": (
                        "the broker has no record of this client order id, so the "
                        "original submission did not reach the broker. Resubmission "
                        "is safe."
                    ),
                }
            else:
                state = str(remote.get("status", "")).lower()
                # A state the platform does not recognise is NOT a resolution.
                # Treating an unrecognised status as "accepted" would convert an
                # unknown outcome into a live order, which is the double-fill
                # failure this whole path exists to prevent.
                recognised = {"new", "accepted", "open", "pending", "working",
                              "partially_filled", "filled", "cancelled", "canceled",
                              "rejected", "expired", "done_for"}
                if state not in recognised:
                    out[cid] = {
                        "resolved": False, "state": state, "remote": remote,
                        "action": "escalate",
                        "detail": (
                            f"the broker reported status {state!r}, which this "
                            "platform does not map to a known state. It is not safe "
                            "to assume the order is live or that it is not. Escalate "
                            "to a human; do not resubmit."
                        ),
                    }
                    continue
                out[cid] = {
                    "resolved": True, "state": state,
                    "remote": remote,
                    "action": ("cancel" if state in ("new", "partially_filled",
                                                     "accepted", "pending",
                                                     "open", "working")
                               else "none"),
                    "detail": (
                        "the order is live at the broker. Do NOT resubmit: issue a "
                        "cancel if the position is not wanted."
                    ),
                }
        return out

    # -- diagnostics ------------------------------------------------------

    def submissions(self) -> pd.DataFrame:
        return pd.DataFrame(self.submission_log) if self.submission_log else pd.DataFrame()

    def __enter__(self) -> "BrokerAdapter":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
