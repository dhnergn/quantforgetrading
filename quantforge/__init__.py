"""QuantForge AI - research-grade quantitative trading platform."""

__version__ = "0.1.0"

#: Live trading is disabled by default. Enabling it additionally requires a
#: signed authorisation record (see quantforge.execution.live_gate); this flag
#: alone is deliberately insufficient.
LIVE_TRADING_ENABLED = False

__all__ = ["__version__", "LIVE_TRADING_ENABLED"]
