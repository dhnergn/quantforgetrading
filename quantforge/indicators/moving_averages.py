"""Moving average indicators."""
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

class SMA:
    METADATA = IndicatorMetadata("SMA", "1.0.0", "SMA(n) = (1/n) * Σ Close[i]", ["close"], 20, 20)
    def __init__(self, period: int = 20):
        self.period = period
        self.METADATA = IndicatorMetadata("SMA", "1.0.0", "SMA(n) = (1/n) * Σ Close[i]", ["close"], period, period)
    def calculate(self, data: pd.Series) -> pd.Series:
        return data.rolling(window=self.period, min_periods=self.period).mean()

class EMA:
    METADATA = IndicatorMetadata("EMA", "1.0.0", "EMA[i] = α * Close[i] + (1-α) * EMA[i-1]", ["close"], 20, 20)
    def __init__(self, period: int = 20):
        self.period = period
        self.METADATA = IndicatorMetadata("EMA", "1.0.0", "EMA[i] = α * Close[i] + (1-α) * EMA[i-1]", ["close"], period, period)
    def calculate(self, data: pd.Series) -> pd.Series:
        return data.ewm(span=self.period, adjust=False, min_periods=self.period).mean()

class WMA:
    METADATA = IndicatorMetadata("WMA", "1.0.0", "WMA(n) = (Σ w[i] * Close[i]) / Σ w[i]", ["close"], 20, 20)
    def __init__(self, period: int = 20):
        self.period = period
        self.METADATA = IndicatorMetadata("WMA", "1.0.0", "WMA(n) = (Σ w[i] * Close[i]) / Σ w[i]", ["close"], period, period)
    def calculate(self, data: pd.Series) -> pd.Series:
        weights = np.arange(1, self.period + 1, dtype=float)
        weights /= weights.sum()
        def _wma(x):
            if len(x) < self.period: return np.nan
            return float(np.dot(x, weights[-len(x):]))
        return data.rolling(window=self.period).apply(_wma, raw=True)

class AdaptiveMA:
    METADATA = IndicatorMetadata("AdaptiveMA", "1.0.0", "AMA[i] = AMA[i-1] + SC * (Close[i] - AMA[i-1])", ["close"], 20, 30)
    def __init__(self, period: int = 20, er_period: int = 10, fast_sc: float = 2.0, slow_sc: float = 30.0):
        self.period = period; self.er_period = er_period; self.fast_sc = fast_sc; self.slow_sc = slow_sc
    def calculate(self, data: pd.Series) -> pd.Series:
        n = self.period; er_n = self.er_period
        fastest = 2.0 / (self.fast_sc + 1.0); slowest = 2.0 / (self.slow_sc + 1.0)
        direction = data.diff().abs(); volatility = data.diff().abs().rolling(er_n).sum()
        er = (direction.rolling(er_n).sum() / volatility.replace(0, np.nan)).fillna(0)
        sc = (er * (fastest - slowest) + slowest) ** 2
        ama = pd.Series(np.nan, index=data.index, dtype=float)
        if len(data) < n: return ama
        ama.iloc[n-1] = data.iloc[:n].mean()
        values = data.values; sc_vals = sc.values
        for i in range(n, len(values)):
            ama.iloc[i] = ama.iloc[i-1] + sc_vals[i] * (values[i] - ama.iloc[i-1])
        return ama
