"""Broker abstraction — never hard-code broker-specific logic.

Live trading is DISABLED by default. The paper broker is always used
unless live trading is explicitly authorized.
"""
from quantforge.broker.base import BrokerAdapter, BrokerCapabilities, OrderType, OrderSide, OrderStatus
from quantforge.broker.paper import PaperBroker

__all__ = [
    "BrokerAdapter", "BrokerCapabilities", "OrderType", "OrderSide", "OrderStatus",
    "PaperBroker",
]
