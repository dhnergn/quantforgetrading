"""Feature engineering with explicit availability semantics.

Each builder returns a frame of raw values plus, critically, the
``available_at`` timestamp for every column. A builder that returns values
without availability information cannot be used in a point-in-time backtest,
so the type makes that visible at the call site.

The lag convention is uniform: a feature computed from bars up to and
including bar ``T`` is available at ``T + publication_delay``, never at ``T``.
For daily bars on a US exchange that means the close-based features are
available at 21:00 UTC (close 20:00 + processing), and a daily-bar strategy
deciding at 20:00 is already too early.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.errors import FeatureError
from quantforge.features.store import FeatureSet, FeatureSpec, PointInTimeFeatureStore
from quantforge.quant import indicators as ind
from quantforge.quant.returns import log_returns, simple_returns

#: Bars are assumed to close at this hour UTC when the caller does not say
#: otherwise. Overridable per exchange via :func:`session_close_utc`.
DEFAULT_DAILY_CLOSE_UTC_HOUR = 21


def session_close_utc(
    index: pd.DatetimeIndex,
    close_hour_utc: int = DEFAULT_DAILY_CLOSE_UTC_HOUR,
) -> pd.DatetimeIndex:
    """Availability boundary for close-derived features on a daily bar.

    The feature derived from a close is usable from the close *plus* a small
    processing allowance. Using the bar timestamp itself would let a backtest
    trade on a close it could not have observed, which is look-ahead bias in
    its purest form.
    """
    stamps = index
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    return pd.DatetimeIndex(
        [ts.replace(hour=close_hour_utc, minute=0, second=0, microsecond=0)
         for ts in stamps]
    )


def build_price_features(
    data: pd.DataFrame,
    feature_set: FeatureSet,
    instrument_id: str,
    close_hour_utc: int = DEFAULT_DAILY_CLOSE_UTC_HOUR,
    include_indicators: bool = True,
    store: Optional[PointInTimeFeatureStore] = None,
) -> PointInTimeFeatureStore:
    """Build and register the MVP feature set for one instrument.

    ``data`` must carry open/high/low/close/volume on a tz-aware UTC index.
    Everything derived here uses only bars at or before each row, so the frame
    itself is causal; the availability boundary is what makes it *usable*
    causally.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise FeatureError("price data must be indexed by DatetimeIndex")
    if data.index.tz is None:
        raise FeatureError(
            "price index must be timezone-aware (UTC). Naive bar timestamps are "
            "the most common source of session misalignment."
        )
    for col in ("open", "high", "low", "close", "volume"):
        if col not in data.columns:
            raise FeatureError(f"missing column {col!r}")

    # The store must carry the *same* FeatureSet instance being populated.
    # Constructing a fresh one here leaves the specs unregistered and every
    # subsequent write fails.
    if store is None:
        store = PointInTimeFeatureStore(feature_set)
    elif store.feature_set is not feature_set:
        store.feature_set = feature_set
    avail = session_close_utc(data.index, close_hour_utc)
    fs = feature_set

    def reg(name, category, method, inputs=(), window=None, units="", lag=dt.timedelta(0),
            notes="") -> None:
        fs.register(FeatureSpec(
            name=name, category=category, source="price_data",
            calculation_method=method, inputs=inputs, window=window,
            publication_time="session_close", availability_lag=lag,
            units=units, notes=notes,
        ))

    # -- price-derived ----------------------------------------------------
    c = data["close"].astype(float)
    h = data["high"].astype(float)
    low = data["low"].astype(float)
    o = data["open"].astype(float)
    v = data["volume"].astype(float)

    reg("close", "price", "session close", units="currency")
    reg("adj_close", "price", "split/dividend-adjusted close", units="currency",
        notes="adjusted basis; see AdjustmentResult.describe()")
    reg("open", "price", "session open", units="currency")
    reg("high", "price", "session high", units="currency")
    reg("low", "price", "session low", units="currency")
    reg("volume", "price", "reported session volume", units="shares")
    reg("hl_range", "price", "high - low", inputs=("high", "low"), units="currency")
    reg("true_range", "price", "max(H-L, |H-Cp|, |L-Cp|)",
        inputs=("high", "low", "close"), window=1, units="currency")
    reg("gap", "price", "open - prior close", inputs=("open", "close"), window=1,
        units="currency",
        notes="overnight move; positive gaps frequently mean information arrived "
              "while the market was closed")

    # -- returns ----------------------------------------------------------
    reg("simple_return", "return", "close/close[-1] - 1", inputs=("close",), window=1)
    reg("log_return", "return", "ln(close/close[-1])", inputs=("close",), window=1)
    reg("return_lag_1", "return", "simple_return shifted by 1 bar", window=1,
        notes="deliberately lagged: a model using this is not using today's move")
    reg("return_lag_2", "return", "simple_return shifted by 2 bars", window=2)
    reg("return_lag_5", "return", "simple_return shifted by 5 bars", window=5)
    reg("cumulative_return_20", "return", "close/close[-20] - 1", window=20)
    reg("cumulative_return_60", "return", "close/close[-60] - 1", window=60)
    reg("cumulative_return_252", "return", "close/close[-252] - 1", window=252,
        notes="12-month momentum; one of the most widely replicated anomalies, "
              "and correspondingly one of the most arbitraged away")

    lr = log_returns(c)
    r1 = simple_returns(c)

    frames: dict[str, pd.Series] = {
        "close": c,
        "open": o,
        "high": h,
        "low": low,
        "volume": v,
        "hl_range": h - low,
        "true_range": ind.TrueRange().calculate(data)["tr"],
        "gap": o - c.shift(1),
        "simple_return": r1,
        "log_return": lr,
        "return_lag_1": r1.shift(1),
        "return_lag_2": r1.shift(2),
        "return_lag_5": r1.shift(5),
        "cumulative_return_20": c.pct_change(20, fill_method=None),
        "cumulative_return_60": c.pct_change(60, fill_method=None),
        "cumulative_return_252": c.pct_change(252, fill_method=None),
    }

    if include_indicators:
        # -- moving averages ---------------------------------------------
        for period in (10, 20, 50, 200):
            reg(f"sma_{period}", "technical", f"SMA({period}) of close",
                inputs=("close",), window=period, units="currency",
                notes="a moving average describes a smoothed past price; it is not "
                      "a predictor and provides no guarantee of continuation")
            frames[f"sma_{period}"] = ind.SMA(period).calculate(data)["sma"]
        for period in (12, 26):
            reg(f"ema_{period}", "technical", f"EMA({period}) of close",
                inputs=("close",), window=period, units="currency")
            frames[f"ema_{period}"] = ind.EMA(period).calculate(data)["ema"]
        reg("wma_20", "technical", "WMA(20) of close", inputs=("close",), window=20,
            units="currency")
        frames["wma_20"] = ind.WMA(20).calculate(data)["wma"]

        # -- momentum -----------------------------------------------------
        reg("rsi_14", "technical", "RSI(14), Wilder smoothing", inputs=("close",),
            window=14, units="percent",
            notes="bounded oscillator measuring recent directional strength; "
                  "overbought does not mean reversal is due")
        frames["rsi_14"] = ind.RSI(14).calculate(data)["rsi"]
        reg("roc_10", "technical", "ROC(10)", inputs=("close",), window=10, units="percent")
        frames["roc_10"] = ind.ROC(10).calculate(data)["roc"]
        reg("adx_14", "technical", "ADX(14)", inputs=("high", "low", "close"),
            window=14, units="percent",
            notes="trend strength without direction; a high ADX describes realised "
                  "movement, it does not forecast it")
        frames["adx_14"] = ind.ADX(14).calculate(data)["adx"]
        macd = ind.MACD().calculate(data)
        reg("macd", "technical", "EMA(12)-EMA(26)", inputs=("close",), window=26)
        reg("macd_signal", "technical", "EMA(9) of MACD", inputs=("close",), window=9)
        reg("macd_hist", "technical", "MACD - signal", inputs=("close",), window=9)
        frames["macd"] = macd["macd"]
        frames["macd_signal"] = macd["macd_signal"]
        frames["macd_hist"] = macd["macd_hist"]

        # -- volatility ---------------------------------------------------
        reg("atr_14", "technical", "ATR(14)", inputs=("high", "low", "close"),
            window=14, units="currency",
            notes="average true range; used for stop placement and volatility "
                  "sizing, and is not a directional signal")
        atr = ind.ATR(14).calculate(data)
        frames["atr_14"] = atr["atr"]
        reg("atr_pct", "technical", "ATR(14)/close", inputs=("close",), window=14,
            units="percent", notes="scale-free volatility measure")
        frames["atr_pct"] = atr["atr"] / c.replace(0, np.nan)
        bb = ind.BollingerBands(20).calculate(data)
        reg("bb_upper", "technical", "SMA(20)+2*sd", inputs=("close",), window=20,
            units="currency")
        reg("bb_lower", "technical", "SMA(20)-2*sd", inputs=("close",), window=20,
            units="currency")
        reg("bb_width", "technical", "2*k*sd", inputs=("close",), window=20,
            units="currency",
            notes="band width measures dispersion; a squeeze is not a directional "
                  "prediction")
        reg("bb_pct_b", "technical", "(close-lower)/(upper-lower)", inputs=("close",),
            window=20, units="ratio")
        frames["bb_upper"] = bb["bb_upper"]
        frames["bb_lower"] = bb["bb_lower"]
        frames["bb_width"] = bb["bb_width"]
        frames["bb_pct_b"] = bb["bb_pct_b"]
        reg("realized_vol_20", "technical", "sqrt(sum(logret^2)) over 20 bars, annualised",
            inputs=("close",), window=20, units="annualised_fraction")
        frames["realized_vol_20"] = ind.RealizedVolatility(20).calculate(data)["realized_vol"]
        reg("realized_vol_60", "technical", "sqrt(sum(logret^2)) over 60 bars, annualised",
            inputs=("close",), window=60, units="annualised_fraction")
        frames["realized_vol_60"] = ind.RealizedVolatility(60).calculate(data)["realized_vol"]

        # -- volume -------------------------------------------------------
        reg("rel_volume_20", "volume", "volume / SMA(20, volume)", inputs=("volume",),
            window=20, units="ratio",
            notes="participation relative to own history; says nothing about direction")
        frames["rel_volume_20"] = ind.RelativeVolume(20).calculate(data)["rel_volume"]
        reg("volume_sma_20", "volume", "SMA(20) of volume", inputs=("volume",), window=20)
        frames["volume_sma_20"] = ind.RelativeVolume(20).calculate(data)["volume_ma"]
        reg("obv", "volume", "cumulative signed volume", inputs=("close", "volume"),
            window=None,
            notes="cumulative, therefore not comparable across instruments without "
                  "normalisation")
        frames["obv"] = ind.OBV().calculate(data)["obv"]
        reg("cmf_20", "volume", "Chaikin Money Flow over 20 bars",
            inputs=("high", "low", "close", "volume"), window=20, units="ratio")
        frames["cmf_20"] = ind.CMF(20).calculate(data)["cmf"]
        reg("vwap_20", "volume", "volume-weighted average price over 20 bars",
            inputs=("high", "low", "close", "volume"), window=20, units="currency",
            notes="used as an execution benchmark; a price relative to VWAP is a "
                  "cost observation, not a signal")
        frames["vwap_20"] = ind.VWAP(20).calculate(data)["vwap"]
        reg("vwap_distance", "volume", "close/vwap_20 - 1", inputs=("close",), window=20,
            units="fraction")
        frames["vwap_distance"] = c / frames["vwap_20"].replace(0, np.nan) - 1.0

    # -- write, with a uniform availability boundary ---------------------
    av_idx = pd.DatetimeIndex(avail)
    for name, series in frames.items():
        if name not in fs:
            continue
        s = series.reindex(data.index)
        store.write(instrument_id, name, data.index, s.to_numpy(), available_at=av_idx)
    return store


# --------------------------------------------------------------------------
# Feature set definitions
# --------------------------------------------------------------------------


def mvp_feature_set(version: str = "mvp-1.0.0") -> FeatureSet:
    """The blueprint's Section 101 MVP feature set."""
    return FeatureSet(
        version=version,
        description=(
            "MVP technical feature set: OHLCV, returns with explicit lags, "
            "SMA/EMA/WMA, RSI/MACD, Bollinger Bands, ATR, volume features. "
            "All values are close-derived and become available at the session "
            "close plus a processing allowance."
        ),
    )


def build_cross_sectional_features(
    panel: dict[str, pd.DataFrame],
    feature_set: FeatureSet,
    close_hour_utc: int = DEFAULT_DAILY_CLOSE_UTC_HOUR,
) -> PointInTimeFeatureStore:
    """Build features for many instruments into one store."""
    store = PointInTimeFeatureStore(feature_set)
    for iid, df in panel.items():
        build_price_features(df, feature_set, iid, close_hour_utc, store=store)
    return store


def build_target(
    prices: pd.Series,
    horizon: int = 1,
    kind: str = "simple",
    label_thresholds: Optional[tuple[float, float]] = None,
    total_return: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Construct a forward-looking target with an explicit horizon.

    The target is deliberately *not* a feature. A common and catastrophic
    mistake is to place the forward return in the feature matrix, which is
    target leakage that produces a near-perfect in-sample model and a useless
    out-of-sample one. Keeping target construction in a separate function with
    an explicit horizon makes the separation visible in the code.

    With ``label_thresholds=(down, up)`` the continuous return is mapped to
    ``up``/``down``/``flat``. The thresholds are an input, fixed before
    evaluation; tuning them on a test set is a form of overfitting that is easy
    to do accidentally and hard to detect afterwards.
    """
    if horizon < 1:
        raise FeatureError(f"horizon must be >= 1, got {horizon}")
    series = total_return if total_return is not None else prices
    if kind == "simple":
        fwd = series.shift(-horizon) / series - 1.0
    elif kind == "log":
        fwd = np.log(series.shift(-horizon) / series)
    else:
        raise FeatureError(f"kind must be 'simple' or 'log', got {kind!r}")

    out = pd.DataFrame({"forward_return": fwd}, index=prices.index)
    out.attrs["horizon"] = horizon
    out.attrs["kind"] = kind
    if label_thresholds is not None:
        down, up = label_thresholds
        if down >= up:
            raise FeatureError(
                f"thresholds must satisfy down < up, got down={down}, up={up}"
            )
        labels = pd.Series("flat", index=prices.index, dtype="object")
        labels[fwd > up] = "up"
        labels[fwd < down] = "down"
        out["label"] = labels
        out.attrs["thresholds"] = (down, up)
        out.attrs["label_rule"] = (
            f"up if forward_return > {up}, down if < {down}, else flat. "
            "Thresholds are fixed before evaluation; optimising them on the "
            "test set invalidates the test."
        )
    # The final `horizon` rows have no future observation, so their target is
    # NaN. Recording them as a training target would teach a model to predict
    # from a target that does not exist.
    out = out.iloc[:-horizon] if horizon > 0 else out
    return out


def training_frame(
    store: PointInTimeFeatureStore,
    instrument_id: str,
    feature_names: Sequence[str],
    prices: pd.Series,
    horizon: int = 1,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Assemble a leakage-safe training frame.

    Every feature is fetched at its own availability time, and the label is
    built from a *future* return. The join is on bar timestamp with the
    feature's own availability, so a row can only contain values that existed
    at that bar.
    """
    from quantforge.core.timestamps import UTC

    pieces = {}
    for name in feature_names:
        s = store.get_history(instrument_id, name)
        if s.empty:
            continue
        # The label for bar T is decided at the close of bar T+horizon, so a
        # model predicting it must be trained on features available at T.
        pieces[name] = s
    if not pieces:
        return pd.DataFrame()
    X = pd.DataFrame(pieces)
    y = build_target(prices, horizon=horizon)
    frame = X.join(y[["forward_return"] + ([ "label"] if "label" in y else [])],
                   how="inner")
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start).tz_localize(UTC)]
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end).tz_localize(UTC)]
    if drop_missing:
        n0 = len(frame)
        frame = frame.dropna()
        if n0 and len(frame) < n0 * 0.5:
            raise FeatureError(
                f"{instrument_id}: dropping missing values removed "
                f"{100 * (1 - len(frame) / n0):.0f}% of rows "
                f"({n0} -> {len(frame)}). Choose imputation explicitly rather than "
                "letting an incomplete feature set silently shrink the sample."
            )
    return frame
