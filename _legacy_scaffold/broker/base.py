"""Broker adapter interface — abstraction over broker-specific implementations.

Never hard-code broker-specific logic throughout the application.
All broker-specific code lives only in the adapter implementation.
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


class OrderSide(enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(enum.Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BrokerCapabilities:
    """What a broker supports."""
    supported_order_types: Set[OrderType]
    supports_bracket_orders: bool
    supports_stop_limit: bool
    supports_trailing_stops: bool
    market_data_available: bool
    sandbox_available: bool
    max_orders_per_second: int


@dataclass
class OrderRequest:
    """A request to submit an order."""
    instrument_id: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    client_order_id: Optional[str] = None
    idempotency_key: Optional[str] = None


@dataclass
class OrderResult:
    """Result of an order submission."""
    success: bool
    order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    commission: float = 0.0
    message: Optional[str] = None
    latency_ms: Optional[int] = None


@dataclass
class Position:
    """A broker position."""
    instrument_id: str
    quantity: float
    average_entry_price: float
    current_price: float
    unrealized_pnl: float
    currency: str = "USD"


@dataclass
class AccountState:
    """Broker account state."""
    account_id: str
    cash: float
    equity: float
    buying_power: float
    currency: str = "USD"


class BrokerAdapter(ABC):
    """Abstract broker adapter — the interface all brokers must implement.

    The rest of the system depends only on this interface.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    def get_account(self) -> AccountState:
        """Get current account state."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get recent orders."""
        pass

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit an order.

        Never assume order submitted = order filled.
        Unknown states must trigger reconciliation.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    def get_market_data(self, instrument_id: str) -> Optional[Dict[str, Any]]:
        """Get current market data for an instrument."""
        pass

    @abstractmethod
    def get_capabilities(self) -> BrokerCapabilities:
        """Get broker capabilities."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the broker."""
        pass
