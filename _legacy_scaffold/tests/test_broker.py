"""Unit tests for broker abstraction — paper trading and order management."""
import pytest

from quantforge.broker.base import OrderRequest, OrderSide, OrderType, OrderStatus
from quantforge.broker.paper import PaperBroker


class TestPaperBroker:
    def test_connect_disconnect(self):
        broker = PaperBroker()
        assert not broker.is_connected()
        broker.connect()
        assert broker.is_connected()
        broker.disconnect()
        assert not broker.is_connected()

    def test_submit_market_order(self):
        broker = PaperBroker(initial_capital=100_000)
        broker.connect()

        order = OrderRequest(
            instrument_id="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        result = broker.submit_order(order)
        assert result.success
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 100
        assert result.average_fill_price is not None

    def test_account_state(self):
        broker = PaperBroker(initial_capital=100_000)
        broker.connect()

        account = broker.get_account()
        assert account.cash == 100_000
        assert account.equity == 100_000

    def test_position_tracking(self):
        broker = PaperBroker(initial_capital=100_000)
        broker.connect()

        # Buy 100 shares
        order = OrderRequest(
            instrument_id="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        broker.submit_order(order)

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].instrument_id == "AAPL"
        assert positions[0].quantity == 100

    def test_order_rejection_when_not_connected(self):
        broker = PaperBroker()
        # Not connected

        order = OrderRequest(
            instrument_id="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        result = broker.submit_order(order)
        assert not result.success
        assert result.status == OrderStatus.REJECTED

    def test_capabilities(self):
        broker = PaperBroker()
        caps = broker.get_capabilities()

        assert OrderType.MARKET in caps.supported_order_types
        assert caps.sandbox_available
        assert not caps.market_data_available  # Paper broker doesn't provide live data
