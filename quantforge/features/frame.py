"""Build the feature frame a strategy needs, from an OHLCV frame."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quantforge.quant import indicators as ind


def build_feature_frame(
    data: pd.DataFrame,
    sma_periods: tuple = (10, 20, 50, 200),
    ema_periods: tuple = (12, 26),
    extra: bool = True,
) -> pd.DataFrame:
    """Compute the indicator set the preset strategies reference.

    Output columns are all causal: every value at row ``i`` uses only rows
    ``<= i``. Warmup periods are NaN and stay NaN — shortening a window to
    produce a value on an early bar changes the statistic being measured.
    """
    if not {"open", "high", "low", "close", "volume"} <= set(data.columns):
        raise KeyError("data must contain open/high/low/close/volume")
    out = pd.DataFrame(index=data.index)
    out["open"] = data["open"]
    out["close"] = data["close"]
    out["high"] = data["high"]
    out["low"] = data["low"]
    out["volume"] = data["volume"]

    for p in sma_periods:
        out[f"sma_{p}"] = ind.SMA(p).calculate(data)["sma"]
    for p in ema_periods:
        out[f"ema_{p}"] = ind.EMA(p).calculate(data)["ema"]
    out["wma_20"] = ind.WMA(20).calculate(data)["wma"]

    rsi = ind.RSI(14).calculate(data)
    out["rsi_14"] = rsi["rsi"]
    out["atr_14"] = ind.ATR(14).calculate(data)["atr"]
    out["roc_10"] = ind.ROC(10).calculate(data)["roc"]
    out["adx_14"] = ind.ADX(14).calculate(data)["adx"]

    macd = ind.MACD().calculate(data)
    out["macd"] = macd["macd"]
    out["macd_signal"] = macd["macd_signal"]
    out["macd_hist"] = macd["macd_hist"]

    bb = ind.BollingerBands(20).calculate(data)
    out["bb_upper"] = bb["bb_upper"]
    out["bb_lower"] = bb["bb_lower"]
    out["bb_mid"] = bb["bb_mid"]
    out["bb_width"] = bb["bb_width"]
    out["bb_pct_b"] = bb["bb_pct_b"]

    st = ind.Stochastic().calculate(data)
    out["stoch_k"] = st["stoch_k"]
    out["stoch_d"] = st["stoch_d"]

    tr = ind.TrueRange().calculate(data)["tr"]
    out["true_range"] = tr

    if extra:
        out["rel_volume_20"] = ind.RelativeVolume(20).calculate(data)["rel_volume"]
        out["volume_sma_20"] = ind.RelativeVolume(20).calculate(data)["volume_ma"]
        out["cmf_20"] = ind.CMF(20).calculate(data)["cmf"]
        out["obv"] = ind.OBV().calculate(data)["obv"]
        out["mfi_14"] = ind.MFI(14).calculate(data)["mfi"]
        vwap = ind.VWAP(20).calculate(data)["vwap"]
        out["vwap_20"] = vwap
        out["vwap_distance"] = data["close"] / vwap.replace(0, np.nan) - 1.0
        out["realized_vol_20"] = ind.RealizedVolatility(20).calculate(data)["realized_vol"]
        out["realized_vol_60"] = ind.RealizedVolatility(60).calculate(data)["realized_vol"]
        out["hl_range"] = data["high"] - data["low"]
        out["gap"] = data["open"] - data["close"].shift(1)
        out["simple_return"] = data["close"].pct_change(fill_method=None)
        out["log_return"] = np.log(data["close"] / data["close"].shift(1))
        out["return_lag_1"] = out["simple_return"].shift(1)
        out["return_lag_5"] = out["simple_return"].shift(5)
        out["cumulative_return_20"] = data["close"].pct_change(20, fill_method=None)
        out["cumulative_return_60"] = data["close"].pct_change(60, fill_method=None)
        out["cumulative_return_252"] = data["close"].pct_change(252, fill_method=None)
    return out
