"""Momentum indicators — RSI, MACD, Stochastic, ROC."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class IndicatorMetadata:
    name: str
    version: str
    formula: str
    inputs: list
    lookback_period: int
    warmup_period: int
    missing_data_behavior: str = "propagate_nan"
    time_alignment: str = "right"
    nan_behavior: str = "propagate"
    precision: int = 8

class RSI:
    METADATA = IndicatorMetadata("RSI", "1.0.0", "RSI = 100 - 100/(1 + RS)", ["close"], 14, 15)
    def __init__(self, period: int = 14):
        self.period = period
        self.METADATA = IndicatorMetadata("RSI", "1.0.0", "RSI = 100 - 100/(1 + RS)", ["close"], period, period + 1)
    def calculate(self, data: pd.Series) -> pd.Series:
        delta = data.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi[avg_loss == 0] = 100.0
        return rsi

class MACD:
    METADATA = IndicatorMetadata("MACD", "1.0.0", "MACD = EMA(12) - EMA(26)", ["close"], 26, 35)
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast; self.slow = slow; self.signal = signal
    def calculate(self, data: pd.Series) -> pd.DataFrame:
        ema_fast = data.ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
        ema_slow = data.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False, min_periods=self.signal).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})

class Stochastic:
    METADATA = IndicatorMetadata("Stochastic", "1.0.0", "%K = 100 * (C - LL) / (HH - LL)", ["high", "low", "close"], 14, 16)
    def __init__(self, period: int = 14, smooth: int = 3):
        self.period = period; self.smooth = smooth
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        lowest_low = low.rolling(self.period, min_periods=self.period).min()
        highest_high = high.rolling(self.period, min_periods=self.period).max()
        denom = highest_high - lowest_low
        pct_k = 100 * (close - lowest_low) / denom.replace(0, np.nan)
        pct_d = pct_k.rolling(self.smooth, min_periods=self.smooth).mean()
        return pd.DataFrame({"pct_k": pct_k, "pct_d": pct_d})

class ROC:
    METADATA = IndicatorMetadata("ROC", "1.0.0", "ROC = (Close[i] / Close[i-n] - 1) * 100", ["close"], 10, 11)
    def __init__(self, period: int = 10):
        self.period = period
        self.METADATA = IndicatorMetadata("ROC", "1.0.0", "ROC = (Close[i] / Close[i-n] - 1) * 100", ["close"], period, period + 1)
    def calculate(self, data: pd.Series) -> pd.Series:
        return ((data / data.shift(self.period)) - 1) * 100
