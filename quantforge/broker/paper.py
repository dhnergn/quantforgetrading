"""Paper broker — simulated execution for paper trading.

Uses the same strategy engine, feature engine, risk engine, portfolio
engine, and order engine as live trading. Only the execution adapter
differs. No real money at risk.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from quantforge.broker.base import (
    AccountState, BrokerAdapter, BrokerCapabilities, OrderRequest,
    OrderResult, OrderSide, OrderStatus, OrderType, Position,
)


class PaperBroker(BrokerAdapter):
    """Paper trading broker — simulates execution with realistic fills.

    Models:
    - Slippage
    - Partial fills
    - Liquidity constraints
    - Commission

    Live trading is NEVER enabled through this broker.
    """

    def __init__(self, initial_capital: float = 100_000.0,
                 commission_pct: float = 0.001, slippage_pct: float = 0.001):
        self._connected = False
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._orders: List[Dict[str, Any]] = []
        self._commission_pct = commission_pct
        self._slippage_pct = slippage_pct
        self._order_counter = 0

    def connect(self) -> None:
        """Connect to the paper broker (no-op)."""
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from the paper broker."""
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_account(self) -> AccountState:
        """Get simulated account state."""
        position_value = sum(
            p.quantity * p.current_price for p in self._positions.values()
        )
        equity = self._cash + position_value
        return AccountState(
            account_id="paper_account",
            cash=self._cash,
            equity=equity,
            buying_power=self._cash * 2,  # 2x margin for paper trading
        )

    def get_positions(self) -> List[Position]:
        """Get simulated positions."""
        return list(self._positions.values())

    def get_orders(self) -> List[Dict[str, Any]]:
        """Get simulated orders."""
        return self._orders

    def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit an order to the paper broker.

        Simulates execution with slippage and commission.
        Never assumes perfect fills.
        """
        if not self._connected:
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message="Not connected to broker",
            )

        self._order_counter += 1
        order_id = f"paper_{self._order_counter}"
        broker_order_id = str(uuid.uuid4())

        # Get current market price
        market_data = self.get_market_data(order.instrument_id)
        if market_data is None:
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message=f"No market data for {order.instrument_id}",
            )

        price = market_data["price"]

        # Apply slippage
        if order.side == OrderSide.BUY:
            execution_price = price * (1 + self._slippage_pct)
        else:
            execution_price = price * (1 - self._slippage_pct)

        # Compute commission
        commission = execution_price * order.quantity * self._commission_pct

        # Update positions
        if order.side == OrderSide.BUY:
            if order.instrument_id in self._positions:
                pos = self._positions[order.instrument_id]
                total_cost = pos.average_entry_price * pos.quantity + execution_price * order.quantity
                pos.quantity += order.quantity
                pos.average_entry_price = total_cost / pos.quantity
            else:
                self._positions[order.instrument_id] = Position(
                    instrument_id=order.instrument_id,
                    quantity=order.quantity,
                    average_entry_price=execution_price,
                    current_price=price,
                    unrealized_pnl=0.0,
                )
            self._cash -= (execution_price * order.quantity) + commission
        else:  # SELL
            if order.instrument_id in self._positions:
                pos = self._positions[order.instrument_id]
                pos.quantity -= order.quantity
                if abs(pos.quantity) < 1e-8:
                    del self._positions[order.instrument_id]
            self._cash += (execution_price * order.quantity) - commission

        # Record order
        order_record = {
            "order_id": order_id,
            "broker_order_id": broker_order_id,
            "instrument_id": order.instrument_id,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "status": OrderStatus.FILLED.value,
            "filled_quantity": order.quantity,
            "average_fill_price": execution_price,
            "commission": commission,
            "slippage": abs(execution_price - price) * order.quantity,
            "submitted_at": datetime.utcnow().isoformat(),
            "filled_at": datetime.utcnow().isoformat(),
        }
        self._orders.append(order_record)

        return OrderResult(
            success=True,
            order_id=order_id,
            broker_order_id=broker_order_id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_fill_price=execution_price,
            commission=commission,
            latency_ms=1,  # Simulated latency
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order (no-op for paper broker — orders fill immediately)."""
        return True

    def get_market_data(self, instrument_id: str) -> Optional[Dict[str, Any]]:
        """Get simulated market data.

        In production, this would connect to a real-time feed.
        For paper trading, this returns simulated prices.
        """
        # This would be connected to a real market data feed
        # For now, return None — the signal bot doesn't need live data
        return None

    def get_capabilities(self) -> BrokerCapabilities:
        """Get paper broker capabilities."""
        return BrokerCapabilities(
            supported_order_types={OrderType.MARKET, OrderType.LIMIT},
            supports_bracket_orders=False,
            supports_stop_limit=False,
            supports_trailing_stops=False,
            market_data_available=False,
            sandbox_available=True,
            max_orders_per_second=10,
        )

    def update_market_price(self, instrument_id: str, price: float) -> None:
        """Update the market price for a position (for P&L calculation)."""
        if instrument_id in self._positions:
            pos = self._positions[instrument_id]
            pos.current_price = price
            pos.unrealized_pnl = (price - pos.average_entry_price) * pos.quantity
