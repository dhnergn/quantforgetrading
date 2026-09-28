"""Volume indicators — OBV, VWAP, Chaikin Money Flow, Relative Volume."""
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

class OBV:
    METADATA = IndicatorMetadata("OBV", "1.0.0", "OBV[i] = OBV[i-1] + sign(ΔClose) * Volume[i]", ["close", "volume"], 1, 1)
    def calculate(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

class VWAP:
    METADATA = IndicatorMetadata("VWAP", "1.0.0", "VWAP = Σ(Price * Volume) / Σ Volume", ["high", "low", "close", "volume"], 1, 1)
    def __init__(self, period: int = 20):
        self.period = period
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical_price = (high + low + close) / 3
        pv = typical_price * volume
        rolling_pv = pv.rolling(self.period, min_periods=1).sum()
        rolling_vol = volume.rolling(self.period, min_periods=1).sum()
        return rolling_pv / rolling_vol.replace(0, np.nan)

class ChaikinMoneyFlow:
    METADATA = IndicatorMetadata("ChaikinMoneyFlow", "1.0.0", "CMF = Σ(MFV * V) / Σ V", ["high", "low", "close", "volume"], 20, 20)
    def __init__(self, period: int = 20):
        self.period = period
        self.METADATA = IndicatorMetadata("ChaikinMoneyFlow", "1.0.0", "CMF = Σ(MFV * V) / Σ V", ["high", "low", "close", "volume"], period, period)
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        denom = high - low
        mfv = ((close - low) - (high - close)) / denom.replace(0, np.nan)
        mfv = mfv.fillna(0)
        rolling_mfv_vol = (mfv * volume).rolling(self.period, min_periods=self.period).sum()
        rolling_vol = volume.rolling(self.period, min_periods=self.period).sum()
        return rolling_mfv_vol / rolling_vol.replace(0, np.nan)

class RelativeVolume:
    METADATA = IndicatorMetadata("RelativeVolume", "1.0.0", "RV = Volume[i] / SMA(Volume, n)[i-1]", ["volume"], 20, 21)
    def __init__(self, period: int = 20):
        self.period = period
        self.METADATA = IndicatorMetadata("RelativeVolume", "1.0.0", "RV = Volume[i] / SMA(Volume, n)[i-1]", ["volume"], period, period + 1)
    def calculate(self, volume: pd.Series) -> pd.Series:
        avg_vol = volume.shift(1).rolling(self.period, min_periods=self.period).mean()
        return volume / avg_vol.replace(0, np.nan)
