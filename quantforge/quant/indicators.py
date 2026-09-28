"""Versioned technical indicator library.

Every indicator is a class with declared metadata rather than a bare function,
so that a feature store can record *how* a value was produced and can refuse
values computed by a different version. Metadata is the contract:

``name``/``version``
    Identity. A value tagged ``EMA@2.0.0`` is not interchangeable with
    ``EMA@1.0.0`` if the seeding convention changed.

``lookback``/``warmup``
    The number of observations required before the first defined value. A
    window of length ``n`` first produces a value at index ``n-1``, so the
    number of leading NaNs is ``warmup - 1`` and the last value is at
    ``warmup - 1``. Warmup periods emit NaN rather than a shorter-window
    fallback, because silently shortening the window is how spurious alpha
    gets manufactured.

``nan_behavior``
    Indicators propagate NaN by default. A missing bar inside a window yields
    NaN for that window rather than an average over the surviving bars, which
    would be a different statistic than the one documented.

No indicator in this module is a predictor. Each is a deterministic transform
of past observations whose value is an empirical question, not a guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

LIBRARY_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class IndicatorMetadata:
    name: str
    version: str
    formula: str
    inputs: tuple[str, ...]
    lookback: int
    warmup: int
    missing_data_behavior: str
    time_alignment: str
    nan_behavior: str
    precision: int = 8
    category: str = "generic"
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "formula": self.formula,
            "inputs": list(self.inputs),
            "lookback": self.lookback,
            "warmup": self.warmup,
            "missing_data_behavior": self.missing_data_behavior,
            "time_alignment": self.time_alignment,
            "nan_behavior": self.nan_behavior,
            "precision": self.precision,
            "category": self.category,
        }


class Indicator:
    """Base class. Subclasses implement :meth:`_compute`."""

    METADATA: IndicatorMetadata

    def __init__(self, **params):
        self.params = dict(params)
        meta = self.METADATA
        overrides = {k: v for k, v in params.items() if k in ("lookback", "warmup")}
        self.METADATA = replace(meta, **overrides) if overrides else meta

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the indicator. ``data`` must contain the declared inputs."""
        if isinstance(data, pd.Series):
            data = data.to_frame(self.METADATA.inputs[0])
        missing = [c for c in self.METADATA.inputs if c not in data.columns]
        if missing:
            raise KeyError(
                f"{self.METADATA.name} requires columns {self.METADATA.inputs}; "
                f"missing {missing}"
            )
        out = self._compute(data)
        out.attrs["indicator"] = self.METADATA.as_dict()
        return out

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.calculate(data)

    @property
    def warmup(self) -> int:
        return self.METADATA.warmup

    def is_warm(self, n_observations: int) -> bool:
        return n_observations >= self.warmup


def _require(data: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")


# ==========================================================================
# Moving averages
# ==========================================================================


class SMA(Indicator):
    """Simple moving average.

    Formula: ``SMA_n(t) = (1/n) * sum_{i=0}^{n-1} C(t-i)``
    Warmup: ``n`` observations. A window containing NaN yields NaN.
    """

    METADATA = IndicatorMetadata(
        name="SMA", version="1.0.0",
        formula="SMA_n(t) = (1/n) * sum_{i=0}^{n-1} Close(t-i)",
        inputs=("close",), lookback=20, warmup=20,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="moving_average",
        notes="Equal-weighted mean. Slowest to react of the moving averages.",
    )

    def __init__(self, period: int = 20):
        super().__init__(period=period)
        self.period = int(period)
        if self.period < 1:
            raise ValueError("period must be >= 1")
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"sma": data["close"].rolling(self.period, min_periods=self.period).mean()},
            index=data.index,
        )


class EMA(Indicator):
    """Exponential moving average, seeded with an SMA.

    Formula: ``alpha = 2/(n+1); EMA(t) = alpha*C(t) + (1-alpha)*EMA(t-1)``

    Seeding matters and is versioned: seeding with the first observation rather
    than the n-period SMA produces visibly different values for the first
    ~3n periods, which changes backtest results. This version uses the SMA seed.
    """

    METADATA = IndicatorMetadata(
        name="EMA", version="1.0.0",
        formula="alpha = 2/(n+1); EMA(t) = alpha*Close(t) + (1-alpha)*EMA(t-1)",
        inputs=("close",), lookback=20, warmup=20,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="moving_average",
        notes="SMA-seeded (pandas adjust=False). Recency-weighted.",
    )

    def __init__(self, period: int = 20, seed: str = "sma"):
        super().__init__(period=period, seed=seed)
        self.period = int(period)
        self.seed = seed
        if self.period < 1:
            raise ValueError("period must be >= 1")
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"ema": data["close"].ewm(span=self.period, adjust=False,
                                       min_periods=self.period).mean()},
            index=data.index,
        )


class WMA(Indicator):
    """Linearly weighted moving average.

    Formula: ``WMA_n(t) = sum_{i=0}^{n-1} w(i)*C(t-i) / sum w(i)`` with ``w(i)=i+1``
    Warmup: ``n`` observations.
    """

    METADATA = IndicatorMetadata(
        name="WMA", version="1.0.0",
        formula="WMA_n(t) = sum_{i=0}^{n-1} (i+1)*Close(t-i) / (n(n+1)/2)",
        inputs=("close",), lookback=20, warmup=20,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="moving_average",
    )

    def __init__(self, period: int = 20):
        super().__init__(period=period)
        self.period = int(period)
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        w = np.arange(1, self.period + 1, dtype=float)
        w /= w.sum()
        return pd.DataFrame(
            {"wma": data["close"].rolling(self.period).apply(
                lambda x: float(np.dot(x, w)), raw=True)},
            index=data.index,
        )


class VWMA(Indicator):
    """Volume-weighted moving average.

    Formula: ``VWMA_n(t) = sum(P*V) / sum(V)`` over the window.
    """

    METADATA = IndicatorMetadata(
        name="VWMA", version="1.0.0",
        formula="VWMA_n(t) = sum(Price*Volume) / sum(Volume)",
        inputs=("close", "volume"), lookback=20, warmup=20,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="moving_average",
    )

    def __init__(self, period: int = 20):
        super().__init__(period=period)
        self.period = int(period)
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        pv = (data["close"] * data["volume"]).rolling(self.period, min_periods=self.period).sum()
        v = data["volume"].rolling(self.period, min_periods=self.period).sum()
        return pd.DataFrame({"vwma": pv / v.replace(0, np.nan)}, index=data.index)


class KAMA(Indicator):
    """Kaufman Adaptive Moving Average.

    Formula: efficiency ratio ``ER = |C(t) - C(t-n)| / sum_{j} |C(j) - C(j-1)|``,
    smoothing constant ``SC = (ER*(fast-slow) + slow)^2``,
    ``KAMA(t) = KAMA(t-1) + SC * (C(t) - KAMA(t-1))``.

    The numerator of ER is the *net* move over n periods while the denominator
    sums the *path length*. Confusing the two makes ER identically 1 and turns
    the adaptive weight into a constant, i.e. a plain EMA. The two are kept
    separate here.
    """

    METADATA = IndicatorMetadata(
        name="KAMA", version="1.0.0",
        formula="ER = |C(t)-C(t-n)| / sum|C(j)-C(j-1)|; SC = (ER*(f-s)+s)^2; KAMA += SC*(C(t)-KAMA)",
        inputs=("close",), lookback=20, warmup=30,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="moving_average",
        notes="Efficiency ratio uses net displacement over path length; a flat "
              "series yields ER=0 by construction.",
    )

    def __init__(self, period: int = 10, er_period: int = 10,
                 fast_sc: float = 2.0, slow_sc: float = 30.0):
        super().__init__(period=period, er_period=er_period)
        self.period = int(period)
        self.er_period = int(er_period)
        self.fast = 2.0 / (fast_sc + 1.0)
        self.slow = 2.0 / (slow_sc + 1.0)
        # The efficiency ratio needs `er_period` bars of path length, so the
        # first defined KAMA lands at index er_period (1-based count
        # er_period + 1); the `period`-bar displacement in the numerator lags
        # the same amount. Leading NaNs therefore number er_period.
        self.METADATA = replace(
            self.METADATA, lookback=er_period + 1, warmup=er_period + 1
        )

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"].astype(float)
        n, er_n = self.period, self.er_period
        if len(close) < n + er_n:
            return pd.DataFrame({"kama": np.nan}, index=data.index)

        # Net displacement over n bars (numerator).
        displacement = (close - close.shift(n)).abs()
        # Total path length over er_n bars (denominator).
        path = close.diff().abs().rolling(er_n, min_periods=er_n).sum()
        er = (displacement / path.replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.0)
        sc = (er * (self.fast - self.slow) + self.slow) ** 2

        out = np.full(len(close), np.nan)
        # Seed at the first index where a full ER window exists.
        start = er_n
        if start >= len(close):
            return pd.DataFrame({"kama": out}, index=data.index)
        out[start] = close.iloc[: start + 1].mean()
        sc_v = sc.to_numpy()
        c_v = close.to_numpy()
        for i in range(start + 1, len(close)):
            prev = out[i - 1]
            s = sc_v[i]
            if not np.isfinite(prev) or not np.isfinite(s):
                out[i] = np.nan
            else:
                out[i] = prev + s * (c_v[i] - prev)
        return pd.DataFrame({"kama": out}, index=data.index)


# ==========================================================================
# Momentum
# ==========================================================================


class RSI(Indicator):
    """Relative Strength Index (Wilder smoothing).

    Formula: ``RS = avg_gain/avg_loss; RSI = 100 - 100/(1+RS)``

    Uses Wilder's original smoothing (``alpha = 1/period``), not a simple
    moving average of gains/losses. Constant price series produce ``RSI = NaN``
    rather than 50, because the ratio is 0/0; see ``constant_rsi_value``.
    """

    METADATA = IndicatorMetadata(
        name="RSI", version="1.0.0",
        formula="RS = avg_gain/avg_loss (Wilder); RSI = 100 - 100/(1+RS)",
        inputs=("close",), lookback=14, warmup=15,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="momentum",
        notes="Wilder smoothing. A 70/30 threshold is a convention, not a "
              "signal of reversal; RSI can stay overbought for extended periods.",
    )

    def __init__(self, period: int = 14, constant_value: Optional[float] = None):
        super().__init__(period=period)
        self.period = int(period)
        self.constant_value = constant_value
        self.METADATA = replace(self.METADATA, lookback=self.period,
                                warmup=self.period + 1)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / self.period, adjust=False,
                            min_periods=self.period).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.period, adjust=False,
                            min_periods=self.period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # avg_gain == 0 and avg_loss == 0  =>  flat series, undefined ratio.
        flat = (avg_gain == 0.0) & (avg_loss == 0.0)
        rsi = rsi.mask(flat, self.constant_value)
        # avg_loss == 0 with gains present => RSI saturates at 100.
        rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
        return pd.DataFrame({"rsi": rsi}, index=data.index)


class MACD(Indicator):
    """Moving Average Convergence Divergence.

    Formula: ``MACD = EMA(fast) - EMA(slow)``;
    ``signal = EMA(MACD, signal_period)``; ``hist = MACD - signal``.

    The MACD line is only defined once the slow EMA is seeded, and the signal
    line only once the MACD series has a full signal window; both are NaN
    before that rather than back-filled.
    """

    METADATA = IndicatorMetadata(
        name="MACD", version="1.0.0",
        formula="MACD = EMA(f)-EMA(s); signal = EMA9(MACD); hist = MACD - signal",
        inputs=("close",), lookback=26, warmup=34,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="momentum",
    )

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(fast=fast, slow=slow, signal=signal)
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast, self.slow, self.signal = int(fast), int(slow), int(signal)
        # The slow EMA seeds at index slow-1, so MACD is defined there. The
        # signal EMA then needs `signal` further observations, giving a first
        # defined signal at index slow+signal-2 and leading NaNs of
        # slow+signal-2. MACD itself leads NaNs of slow-1.
        self.METADATA = replace(self.METADATA, lookback=self.slow,
                                warmup=self.slow + self.signal - 1)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        c = data["close"].astype(float)
        ef = c.ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
        es = c.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
        macd = ef - es
        sig = macd.ewm(span=self.signal, adjust=False, min_periods=self.signal).mean()
        return pd.DataFrame(
            {"macd": macd, "macd_signal": sig, "macd_hist": macd - sig},
            index=data.index,
        )


class Stochastic(Indicator):
    """Stochastic oscillator.

    Formula: ``%K = 100*(C - LL_n)/(HH_n - LL_n)``;
    ``%D = SMA3(%K)``; ``%J = 3*%K - 2*%D``.

    ``HH_n == LL_n`` (flat range) yields NaN, not a divide-by-zero or an
    arbitrary midpoint.
    """

    METADATA = IndicatorMetadata(
        name="Stochastic", version="1.0.0",
        formula="%K = 100*(C-LL_n)/(HH_n-LL_n); %D = SMA3(%K); %J = 3%K-2%D",
        inputs=("high", "low", "close"), lookback=14, warmup=16,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="momentum",
        notes="Flat-range periods yield NaN by design.",
    )

    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 1):
        super().__init__(k_period=k_period, d_period=d_period)
        self.k_period = int(k_period)
        self.d_period = int(d_period)
        self.smooth_k = int(smooth_k)
        self.METADATA = replace(self.METADATA, lookback=self.k_period,
                                warmup=self.k_period + self.d_period - 1)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        ll = data["low"].rolling(self.k_period, min_periods=self.k_period).min()
        hh = data["high"].rolling(self.k_period, min_periods=self.k_period).max()
        rng = (hh - ll).replace(0.0, np.nan)
        k = 100.0 * (data["close"] - ll) / rng
        if self.smooth_k > 1:
            k = k.rolling(self.smooth_k, min_periods=self.smooth_k).mean()
        d = k.rolling(self.d_period, min_periods=self.d_period).mean()
        return pd.DataFrame(
            {"stoch_k": k, "stoch_d": d, "stoch_j": 3.0 * k - 2.0 * d},
            index=data.index,
        )


class ROC(Indicator):
    """Rate of change.

    Formula: ``ROC_n(t) = 100 * (C(t) - C(t-n)) / C(t-n)``
    """

    METADATA = IndicatorMetadata(
        name="ROC", version="1.0.0",
        formula="ROC_n(t) = 100*(C(t)-C(t-n))/C(t-n)",
        inputs=("close",), lookback=10, warmup=10,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="momentum",
    )

    def __init__(self, period: int = 10):
        super().__init__(period=period)
        self.period = int(period)
        # shift(period) is undefined for the first `period` rows, and the
        # first value therefore lands at index `period` -> warmup = period + 1.
        self.METADATA = replace(self.METADATA, lookback=self.period,
                                warmup=self.period + 1)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        c = data["close"].astype(float)
        prev = c.shift(self.period)
        return pd.DataFrame({"roc": 100.0 * (c - prev) / prev.replace(0.0, np.nan)},
                            index=data.index)


class CCI(Indicator):
    """Commodity Channel Index.

    Formula: ``CCI = (TP - SMA(TP)) / (0.015 * mean_abs_deviation(TP))``
    """

    METADATA = IndicatorMetadata(
        name="CCI", version="1.0.0",
        formula="CCI = (TP - SMA_n(TP)) / (0.015 * MAD_n(TP)); TP = (H+L+C)/3",
        inputs=("high", "low", "close"), lookback=20, warmup=20,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="momentum",
    )

    def __init__(self, period: int = 20):
        super().__init__(period=period)
        self.period = int(period)
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        tp = (data["high"] + data["low"] + data["close"]) / 3.0
        sma = tp.rolling(self.period, min_periods=self.period).mean()
        mad = tp.rolling(self.period, min_periods=self.period).apply(
            lambda x: float(np.abs(x - x.mean()).mean()), raw=True)
        denom = (0.015 * mad).replace(0.0, np.nan)
        return pd.DataFrame({"cci": (tp - sma) / denom}, index=data.index)


# ==========================================================================
# Volatility
# ==========================================================================


class TrueRange(Indicator):
    """True range.

    Formula: ``TR = max(H-L, |H-C_prev|, |L-C_prev|)``
    """

    METADATA = IndicatorMetadata(
        name="TrueRange", version="1.0.0",
        formula="TR(t) = max(H(t)-L(t), |H(t)-C(t-1)|, |L(t)-C(t-1)|)",
        inputs=("high", "low", "close"), lookback=1, warmup=1,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="volatility",
    )

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        pc = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - pc).abs(),
                (data["low"] - pc).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return pd.DataFrame({"tr": tr}, index=data.index)


class ATR(Indicator):
    """Average True Range (Wilder smoothing).

    Formula: ``ATR = WilderEWM(mean(TR), alpha=1/n)``

    ATR is a measure of price *range*, not directional volatility, and it is
    used here for stop placement and volatility sizing. It does not indicate
    direction and is not a risk forecast.
    """

    METADATA = IndicatorMetadata(
        name="ATR", version="1.0.0",
        formula="TR = max(H-L,|H-Cp|,|L-Cp|); ATR = WilderEWM(TR, alpha=1/n)",
        inputs=("high", "low", "close"), lookback=14, warmup=14,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="volatility",
    )

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = int(period)
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        tr = TrueRange()._compute(data)["tr"]
        atr = tr.ewm(alpha=1.0 / self.period, adjust=False, min_periods=self.period).mean()
        return pd.DataFrame({"atr": atr, "tr": tr}, index=data.index)


class BollingerBands(Indicator):
    """Bollinger Bands.

    Formula: ``mid = SMA_n(C); upper = mid + k*sd; lower = mid - k*sd``

    ``sd`` is the *population* standard deviation (ddof=0), matching the
    convention in John Bollinger's original definition. Sample (ddof=1) shifts
    the bands; the choice is versioned.
    """

    METADATA = IndicatorMetadata(
        name="BollingerBands", version="1.0.0",
        formula="mid = SMA_n(C); upper = mid + k*sd_pop; lower = mid - k*sd_pop",
        inputs=("close",), lookback=20, warmup=20,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="volatility",
        notes="sd uses ddof=0. A band touch is a statistical event, not "
              "evidence of mean reversion.",
    )

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(period=period, num_std=num_std)
        self.period = int(period)
        self.num_std = float(num_std)
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        c = data["close"].astype(float)
        mid = c.rolling(self.period, min_periods=self.period).mean()
        sd = c.rolling(self.period, min_periods=self.period).std(ddof=0)
        return pd.DataFrame(
            {
                "bb_mid": mid,
                "bb_upper": mid + self.num_std * sd,
                "bb_lower": mid - self.num_std * sd,
                "bb_width": (2.0 * self.num_std * sd).abs(),
                "bb_pct_b": (c - (mid - self.num_std * sd))
                / (2.0 * self.num_std * sd).replace(0.0, np.nan),
            },
            index=data.index,
        )


class KeltnerChannels(Indicator):
    """Keltner Channels.

    Formula: ``mid = EMA_n(C); upper = mid + m*ATR_k; lower = mid - m*ATR_k``

    Defaults differ between sources; ``atr_period``/``multiplier`` are explicit
    parameters rather than hidden constants.
    """

    METADATA = IndicatorMetadata(
        name="KeltnerChannels", version="1.0.0",
        formula="mid = EMA_n(C); upper/lower = mid +/- m*ATR_k",
        inputs=("high", "low", "close"), lookback=20, warmup=20,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="volatility",
    )

    def __init__(self, period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        super().__init__(period=period, atr_period=atr_period, multiplier=multiplier)
        self.period, self.atr_period = int(period), int(atr_period)
        self.multiplier = float(multiplier)
        self.METADATA = replace(self.METADATA, lookback=max(self.period, self.atr_period),
                                warmup=max(self.period, self.atr_period))

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        c = data["close"].astype(float)
        mid = c.ewm(span=self.period, adjust=False, min_periods=self.period).mean()
        atr = ATR(self.atr_period)._compute(data)["atr"]
        return pd.DataFrame(
            {
                "kc_mid": mid,
                "kc_upper": mid + self.multiplier * atr,
                "kc_lower": mid - self.multiplier * atr,
            },
            index=data.index,
        )


class RealizedVolatility(Indicator):
    """Close-to-close realised volatility, annualised.

    Formula: ``RV_n = sqrt(mean_i(logret_i^2)) * sqrt(bars_per_year)``

    The division by ``n`` inside the square root is what makes this a
    *per-period* volatility. Omitting it yields ``sqrt(sum)``, which is the
    realised volatility of the n-period return and is therefore larger by a
    factor of ``sqrt(n)`` — for a 60-day window that is a 7.7x overstatement of
    annualised risk, which then propagates into every volatility-scaled
    position size and risk limit derived from it.

    Log returns are used so multi-period aggregation is consistent: the sum of
    squared log returns telescopes correctly with the price ratio.
    """

    METADATA = IndicatorMetadata(
        name="RealizedVolatility", version="1.1.0",
        formula="RV_n = sqrt( mean(logret^2) ) * sqrt(bars_per_year)",
        inputs=("close",), lookback=20, warmup=21,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="volatility",
        notes="Annualised by sqrt(bars_per_year) under square-root-of-time "
              "scaling, an assumption that fails for autocorrelated returns. "
              "Version 1.1.0 divides by the window length inside the square "
              "root; 1.0.0 overstated annualised volatility by sqrt(window).",
    )

    def __init__(self, period: int = 20, bars_per_year: int = 252,
                 annualize: bool = True):
        super().__init__(period=period, bars_per_year=bars_per_year)
        self.period = int(period)
        self.bars_per_year = int(bars_per_year)
        self.annualize = bool(annualize)
        # The first log return needs C(t-1), so it appears at index 1; the
        # `period`-bar rolling window then ends at index `period` -> warmup
        # = period + 1.
        self.METADATA = replace(self.METADATA, lookback=self.period,
                                warmup=self.period + 1)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        c = data["close"].astype(float)
        lr = np.log(c / c.shift(1))
        # mean, not sum: see the class docstring. A sum of squares cannot be
        # negative, so clip guards float round-off before the square root.
        sq = (lr.pow(2).rolling(self.period, min_periods=self.period)
              .mean().clip(lower=0.0))
        rv = np.sqrt(sq)
        if self.annualize:
            rv = rv * np.sqrt(self.bars_per_year)
        return pd.DataFrame({"realized_vol": rv}, index=data.index)


class ParkinsonVolatility(Indicator):
    """Parkinson high-low range volatility.

    Formula: ``sigma^2 = (1/(4 ln2 * n)) * sum (ln(H/L))^2``

    Uses only the high and low, so it is less contaminated by closing-price
    microstructure noise. Requires high > low > 0.
    """

    METADATA = IndicatorMetadata(
        name="ParkinsonVolatility", version="1.0.0",
        formula="sigma = sqrt( sum(ln(H/L)^2) / (4*ln(2)*n) ) * sqrt(bars_per_year)",
        inputs=("high", "low"), lookback=20, warmup=20,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="volatility",
    )

    def __init__(self, period: int = 20, bars_per_year: int = 252,
                 annualize: bool = True):
        super().__init__(period=period)
        self.period = int(period)
        self.bars_per_year = int(bars_per_year)
        self.annualize = bool(annualize)
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        hl = np.log(data["high"] / data["low"])
        var = (hl.pow(2).rolling(self.period, min_periods=self.period)
               .sum() / (4.0 * np.log(2.0) * self.period))
        vol = np.sqrt(var)
        if self.annualize:
            vol = vol * np.sqrt(self.bars_per_year)
        return pd.DataFrame({"parkinson_vol": vol}, index=data.index)


class ADX(Indicator):
    """Average Directional Index (Wilder).

    Formula: ``+DI = 100*smoothed(+DM)/ATR; -DI = 100*smoothed(-DM)/ATR;
    DX = 100*|+DI - -DI|/(+DI + -DI); ADX = WilderEWM(DX)``

    The regime engine uses ADX as a *trend-strength* descriptor. A high ADX
    describes realised trend, and says nothing about future direction.
    """

    METADATA = IndicatorMetadata(
        name="ADX", version="1.0.0",
        formula="ADX = WilderEWM(100*|+DI-(-DI)|/(+DI+(-DI)))",
        inputs=("high", "low", "close"), lookback=14, warmup=27,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="trend",
    )

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = int(period)
        # DI is defined once the ATR seed completes (index period-1); the ADX
        # smoothing window then needs `period` further observations, so the
        # first defined value lands at 2*period-1 and leading NaNs number
        # 2*period-1... i.e. warmup = 2*period-1.
        self.METADATA = replace(
            self.METADATA, lookback=2 * self.period - 1, warmup=2 * self.period - 1
        )

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        h, l, c = data["high"], data["low"], data["close"]
        up = h.diff()
        dn = -l.diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)

        atr = ATR(self.period)._compute(data)["atr"]
        a = 1.0 / self.period
        pdi = 100.0 * plus_dm.ewm(alpha=a, adjust=False, min_periods=self.period).mean() / atr.replace(0.0, np.nan)
        mdi = 100.0 * minus_dm.ewm(alpha=a, adjust=False, min_periods=self.period).mean() / atr.replace(0.0, np.nan)
        dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
        adx = dx.ewm(alpha=a, adjust=False, min_periods=self.period).mean()
        return pd.DataFrame({"adx": adx, "plus_di": pdi, "minus_di": mdi},
                            index=data.index)


# ==========================================================================
# Volume
# ==========================================================================


class OBV(Indicator):
    """On-Balance Volume.

    Formula: ``OBV(t) = OBV(t-1) +/- Volume(t)`` by close direction.

    OBV is cumulative and therefore path-dependent: it is not comparable
    across instruments without normalising, and a level shift from a share
    split corrupts it unless the series is split-adjusted.
    """

    METADATA = IndicatorMetadata(
        name="OBV", version="1.0.0",
        formula="OBV(t) = OBV(t-1) + sign(C(t)-C(t-1))*Volume(t)",
        inputs=("close", "volume"), lookback=1, warmup=1,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="volume",
        notes="Cumulative and not level-comparable across instruments.",
    )

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        c, v = data["close"].astype(float), data["volume"].astype(float)
        sign = np.sign(c.diff()).fillna(0.0)
        return pd.DataFrame({"obv": (sign * v).cumsum()}, index=data.index)


class CMF(Indicator):
    """Chaikin Money Flow.

    Formula: ``MFV = ((C-L)-(H-C))/(H-L) * V; CMF = SMA_n(MFV)/SMA_n(V)``

    ``H == L`` (no range) yields 0 contribution, documented rather than NaN,
    because the volume-weighted money flow is genuinely undefined-but-cancelling
    on a zero-range bar; the flag is surfaced via ``cmf_flat_range``.
    """

    METADATA = IndicatorMetadata(
        name="CMF", version="1.0.0",
        formula="CMF = sum_n(MFV)/sum_n(V); MFV = ((C-L)-(H-C))/(H-L)*V",
        inputs=("high", "low", "close", "volume"), lookback=20, warmup=20,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="zero_range_contributes_zero", category="volume",
    )

    def __init__(self, period: int = 20):
        super().__init__(period=period)
        self.period = int(period)
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        h, l, c, v = data["high"], data["low"], data["close"], data["volume"]
        rng = (h - l).replace(0.0, np.nan)
        mfm = ((c - l) - (h - c)) / rng
        mfv = (mfm * v).fillna(0.0)
        num = mfv.rolling(self.period, min_periods=self.period).sum()
        den = v.rolling(self.period, min_periods=self.period).sum()
        return pd.DataFrame(
            {"cmf": num / den.replace(0.0, np.nan),
             "cmf_flat_range": (rng.isna() & (h == l))},
            index=data.index,
        )


class VWAP(Indicator):
    """Volume-Weighted Average Price.

    Formula: ``VWAP_n = sum(TP*V)/sum(V)`` where ``TP = (H+L+C)/3``.

    Within a bar this is the volume-weighted mean traded price. Across a
    multi-bar window it is the mean cost basis of that window's volume, which
    is why it is commonly used as an execution benchmark rather than a signal.
    """

    METADATA = IndicatorMetadata(
        name="VWAP", version="1.0.0",
        formula="VWAP = sum(TP*V)/sum(V); TP = (H+L+C)/3",
        inputs=("high", "low", "close", "volume"), lookback=1, warmup=1,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="volume",
        notes="Used as an execution benchmark, not as a directional predictor.",
    )

    def __init__(self, period: Optional[int] = None):
        super().__init__()
        self.period = int(period) if period else None
        if self.period:
            self.METADATA = replace(self.METADATA, lookback=self.period,
                                    warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        tp = (data["high"] + data["low"] + data["close"]) / 3.0
        pv = tp * data["volume"]
        if self.period is None:
            num, den = pv.cumsum(), data["volume"].cumsum()
        else:
            num = pv.rolling(self.period, min_periods=self.period).sum()
            den = data["volume"].rolling(self.period, min_periods=self.period).sum()
        return pd.DataFrame({"vwap": num / den.replace(0.0, np.nan)}, index=data.index)


class RelativeVolume(Indicator):
    """Volume relative to its own recent average.

    Formula: ``RelVol = V(t) / SMA_n(V)``

    Values above 1 indicate above-average participation. This is a liquidity
    observation; it does not indicate direction.
    """

    METADATA = IndicatorMetadata(
        name="RelativeVolume", version="1.0.0",
        formula="RelVol = V(t)/SMA_n(V)",
        inputs=("volume",), lookback=20, warmup=20,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="volume",
    )

    def __init__(self, period: int = 20):
        super().__init__(period=period)
        self.period = int(period)
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        v = data["volume"].astype(float)
        ma = v.rolling(self.period, min_periods=self.period).mean()
        return pd.DataFrame({"rel_volume": v / ma.replace(0.0, np.nan),
                             "volume_ma": ma}, index=data.index)


class MFI(Indicator):
    """Money Flow Index (RSI applied to money flow).

    Formula: ``MFI = 100 - 100/(1 + posMF/negMF)`` over ``n`` periods.
    """

    METADATA = IndicatorMetadata(
        name="MFI", version="1.0.0",
        formula="MFI = 100 - 100/(1 + sum(posMF_n)/sum(negMF_n)); TP=(H+L+C)/3",
        inputs=("high", "low", "close", "volume"), lookback=14, warmup=15,
        missing_data_behavior="propagate_nan", time_alignment="right",
        nan_behavior="propagate", category="volume",
    )

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = int(period)
        self.METADATA = replace(self.METADATA, lookback=self.period, warmup=self.period)

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        tp = (data["high"] + data["low"] + data["close"]) / 3.0
        raw = tp * data["volume"]
        d = tp.diff()
        pos = raw.where(d > 0, 0.0).rolling(self.period, min_periods=self.period).sum()
        neg = raw.where(d < 0, 0.0).rolling(self.period, min_periods=self.period).sum()
        ratio = pos / neg.replace(0.0, np.nan)
        mfi = 100.0 - 100.0 / (1.0 + ratio)
        mfi = mfi.mask((neg == 0.0) & (pos > 0.0), 100.0)
        return pd.DataFrame({"mfi": mfi}, index=data.index)


# ==========================================================================
# Registry
# ==========================================================================

INDICATORS: dict[str, type[Indicator]] = {
    "SMA": SMA, "EMA": EMA, "WMA": WMA, "VWMA": VWMA, "KAMA": KAMA,
    "RSI": RSI, "MACD": MACD, "Stochastic": Stochastic, "ROC": ROC, "CCI": CCI,
    "TrueRange": TrueRange, "ATR": ATR, "BollingerBands": BollingerBands,
    "KeltnerChannels": KeltnerChannels, "RealizedVolatility": RealizedVolatility,
    "ParkinsonVolatility": ParkinsonVolatility, "ADX": ADX,
    "OBV": OBV, "CMF": CMF, "VWAP": VWAP, "RelativeVolume": RelativeVolume, "MFI": MFI,
}


def get_indicator(name: str, **params) -> Indicator:
    try:
        cls = INDICATORS[name]
    except KeyError:
        raise KeyError(
            f"unknown indicator {name!r}; available: {sorted(INDICATORS)}"
        ) from None
    return cls(**params)


def catalogue() -> list[dict]:
    """All indicators with default metadata. Feeds the Indicator Laboratory."""
    out = []
    for name, cls in sorted(INDICATORS.items()):
        inst = cls()
        d = inst.METADATA.as_dict()
        d["library_version"] = LIBRARY_VERSION
        out.append(d)
    return out


# ==========================================================================
# Self-verification
# ==========================================================================


def _probe_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """A non-degenerate OHLCV frame used to measure warmup empirically.

    Random-walk closes with a strictly positive range, so indicators whose
    denominators involve ``high - low`` do not degenerate to NaN everywhere.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    high = close + np.abs(rng.normal(0.0, 0.5, n)) + 0.01
    low = close - np.abs(rng.normal(0.0, 0.5, n)) - 0.01
    return pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )


def measure_warmup(indicator: Indicator, probe_n: int = 400,
                   column: Optional[str] = None) -> int:
    """Empirically count leading NaNs: the observations required.

    Returned value equals ``indicator.warmup`` when metadata is correct. This
    exists because composite indicators (ADX, MACD, stochastic) have warmups
    that are awkward to derive by hand and easy to declare wrongly, and a
    wrong warmup silently changes a backtest's first tradable bar.

    For a multi-column indicator the *maximum* across value columns is
    returned, because ``warmup`` describes when the indicator is fully formed.
    ``MACD`` is the motivating case: its ``macd`` line is available well
    before ``macd_signal`` and ``macd_hist`` exist.
    """
    frame = _probe_frame(probe_n)
    out = indicator.calculate(frame)
    if column:
        columns = [column]
    else:
        columns = [c for c in out.columns if not c.endswith("flat_range")]
    worst = 0
    for col in columns:
        arr = out[col].to_numpy(dtype="float64")
        if np.all(np.isnan(arr)):
            continue
        first_valid = int(np.argmax(~np.isnan(arr)))
        worst = max(worst, first_valid + 1)
    return worst


def validate_metadata() -> list[str]:
    """Check every indicator's declared warmup against observed behaviour.

    Returns a list of human-readable discrepancies. Empty means the library's
    declared metadata is self-consistent. Called by the test suite so a
    metadata regression cannot reach a backtest unnoticed.
    """
    problems: list[str] = []
    frame = _probe_frame()
    for name, cls in sorted(INDICATORS.items()):
        inst = cls()
        try:
            out = inst.calculate(frame)
        except Exception as exc:  # pragma: no cover
            problems.append(f"{name}: raised {type(exc).__name__}: {exc}")
            continue
        for col in out.columns:
            if col.endswith("flat_range"):
                continue  # boolean diagnostic column, not a value series
            arr = out[col].to_numpy(dtype="float64")
            finite = arr[~np.isnan(arr)]
            if finite.size == 0:
                problems.append(f"{name}.{col}: entirely NaN on a valid probe")
                continue
            if not np.isfinite(finite).all():
                problems.append(f"{name}.{col}: contains non-finite values")
    # Warmup cross-check on the primary output column.
    for name, cls in sorted(INDICATORS.items()):
        inst = cls()
        observed = measure_warmup(inst)
        if observed != inst.warmup:
            problems.append(
                f"{name}: declared warmup={inst.warmup} but first value appears "
                f"at {observed}"
            )
    return problems
