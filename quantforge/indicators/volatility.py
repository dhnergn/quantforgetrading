"""Volatility indicators — Bollinger Bands, ATR, Keltner Channels, Rolling Volatility."""
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

class BollingerBands:
    METADATA = IndicatorMetadata("BollingerBands", "1.0.0", "Mid = SMA(n), Upper = Mid + k*σ", ["close"], 20, 20)
    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period; self.num_std = num_std
    def calculate(self, data: pd.Series) -> pd.DataFrame:
        mid = data.rolling(self.period, min_periods=self.period).mean()
        std = data.rolling(self.period, min_periods=self.period).std(ddof=1)
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std
        bandwidth = (upper - lower) / mid.replace(0, np.nan)
        return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "bandwidth": bandwidth})

class ATR:
    METADATA = IndicatorMetadata("ATR", "1.0.0", "TR = max(H-L, |H-C_prev|, |L-C_prev|)", ["high", "low", "close"], 14, 15)
    def __init__(self, period: int = 14):
        self.period = period
        self.METADATA = IndicatorMetadata("ATR", "1.0.0", "TR = max(H-L, |H-C_prev|, |L-C_prev|)", ["high", "low", "close"], period, period + 1)
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        TR = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return TR.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

class KeltnerChannels:
    METADATA = IndicatorMetadata("KeltnerChannels", "1.0.0", "Mid = EMA(n), Upper = EMA + k*ATR", ["high", "low", "close"], 20, 34)
    def __init__(self, period: int = 20, atr_period: int = 14, multiplier: float = 2.0):
        self.period = period; self.atr_period = atr_period; self.multiplier = multiplier
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        mid = close.ewm(span=self.period, adjust=False, min_periods=self.period).mean()
        prev_close = close.shift(1)
        TR = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = TR.ewm(alpha=1 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()
        upper = mid + self.multiplier * atr
        lower = mid - self.multiplier * atr
        return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})

class RollingVolatility:
    METADATA = IndicatorMetadata("RollingVolatility", "1.0.0", "σ = std(returns, ddof=1)", ["close"], 20, 21)
    def __init__(self, period: int = 20, annualize: bool = False, periods_per_year: int = 252):
        self.period = period; self.annualize = annualize; self.periods_per_year = periods_per_year
    def calculate(self, data: pd.Series) -> pd.Series:
        returns = data.pct_change()
        vol = returns.rolling(self.period, min_periods=self.period).std(ddof=1)
        if self.annualize:
            vol *= np.sqrt(self.periods_per_year)
        return vol
