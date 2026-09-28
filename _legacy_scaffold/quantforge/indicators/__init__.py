"""Versioned technical indicator library."""
from quantforge.indicators.moving_averages import SMA, EMA, WMA, AdaptiveMA
from quantforge.indicators.momentum import RSI, MACD, Stochastic, ROC
from quantforge.indicators.volatility import BollingerBands, ATR, KeltnerChannels, RollingVolatility
from quantforge.indicators.volume import OBV, VWAP, ChaikinMoneyFlow, RelativeVolume

__all__ = [
    "SMA", "EMA", "WMA", "AdaptiveMA",
    "RSI", "MACD", "Stochastic", "ROC",
    "BollingerBands", "ATR", "KeltnerChannels", "RollingVolatility",
    "OBV", "VWAP", "ChaikinMoneyFlow", "RelativeVolume",
]
