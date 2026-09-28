"""Market regime detection.

A *regime* is a label describing the market's character at a point in time. Two
things make this harder than it looks, and both are temporal.

**A label must be knowable when it is used.** A regime computed from data up
to bar ``t`` can be acted on at ``t``, but the label that eventually gets
attached to bar ``t`` in a research dataset may have been computed with a
fuller history. :class:`RegimeLabel` carries the timestamp it was determined
at and the features it used, and :meth:`RegimeDetector.detect_series` lags the
labels so a strategy consuming them at bar ``t`` is not reading a regime
described using bar ``t+1``.

**A regime is not a forecast.** ``HIGH_VOLATILITY`` says the market has been
unusual, not that it will continue to be. Regime labels are used to *decompose*
a backtest - does this rule work in every regime, or only in the one that
happened to dominate the sample? - and a strategy that only ever worked in one
regime is a regime bet. That is a legitimate choice, but it has to be made
deliberately with knowledge of that regime's historical frequency, not
discovered afterwards.

The rule-based detector is deterministic and has no fitted parameters, so it
cannot be overfitted. The ML detector is a classifier over the same features and
is subject to everything in
[docs/MACHINE_LEARNING.md](docs/MACHINE_LEARNING.md), including the
requirement that its own version travel with every label.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.timestamps import UTC, utcnow


# ==========================================================================
# Labels
# ==========================================================================


class Regime:
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

    ALL = (BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY, LOW_VOLATILITY)


@dataclass(frozen=True, slots=True)
class RegimeLabel:
    """One regime observation, with the provenance needed to trust it."""

    regime: str
    method: str                       # "rule_based" | "ml_based"
    timestamp: pd.Timestamp           # the bar this label describes
    #: When the label became knowable. Equal to ``timestamp`` shifted by the
    #: labelling lag; a strategy may act on it from this instant onward.
    available_at: Optional[pd.Timestamp] = None
    parameters: dict = field(default_factory=dict)
    model_version: Optional[str] = None
    confidence: Optional[float] = None
    features: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.regime not in Regime.ALL:
            raise ValueError(
                f"unknown regime {self.regime!r}; expected one of {Regime.ALL}"
            )
        if self.method not in ("rule_based", "ml_based"):
            raise ValueError(
                f"unknown method {self.method!r}; expected 'rule_based' or 'ml_based'"
            )
        if self.method == "ml_based" and not self.model_version:
            # Without a version, a label cannot be tied to the model that
            # produced it, and a model swap silently changes every label
            # attributed to the same "version" of the strategy.
            raise ValueError(
                "an ml_based regime label must carry the model_version that "
                "produced it"
            )
        if self.available_at is None:
            object.__setattr__(self, "available_at", self.timestamp)
        elif self.available_at < self.timestamp:
            raise ValueError(
                f"available_at {self.available_at} precedes the label timestamp "
                f"{self.timestamp}: a label cannot be knowable before the bar it "
                "describes"
            )

    def as_dict(self) -> dict:
        return {
            "regime": self.regime,
            "method": self.method,
            "timestamp": str(self.timestamp),
            "available_at": str(self.available_at),
            "parameters": dict(self.parameters),
            "model_version": self.model_version,
            "confidence": self.confidence,
            "features": dict(self.features),
        }


# ==========================================================================
# Features
# ==========================================================================


def regime_features(prices: pd.Series, volume: Optional[pd.Series] = None) -> pd.DataFrame:
    """Causal features describing trend, volatility, and momentum.

    Every column uses only data at or before its own timestamp. Warmup periods
    are NaN and stay NaN: shortening a window to produce a value on an early
    bar measures a different thing.

    The parameters are the conventional ones, chosen because they are
    interpretable rather than optimal. They are exposed in
    ``RegimeDetector`` so a change is visible in the label provenance.
    """
    close = pd.Series(prices, dtype="float64")
    out = pd.DataFrame(index=close.index)
    log_ret = np.log(close / close.shift(1))

    for n in (20, 50, 200):
        out[f"sma_{n}"] = close.rolling(n, min_periods=n).mean()
    out["price_over_sma_200"] = close / out["sma_200"] - 1.0

    # Slope of the 50-day average, expressed per day so it is comparable
    # across price levels and instruments.
    out["sma_50_slope"] = out["sma_50"].diff() / out["sma_50"].shift(1)
    out["trend_strength"] = (close - out["sma_50"]) / out["sma_50"]

    out["realized_vol_20"] = log_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    out["realized_vol_60"] = log_ret.rolling(60, min_periods=60).std() * np.sqrt(252)
    # Volatility ratio: current against its own trailing level. A regime is a
    # statement about unusualness, which needs a reference.
    out["vol_ratio"] = out["realized_vol_20"] / out["realized_vol_60"]

    out["return_20"] = close.pct_change(20, fill_method=None)
    out["return_60"] = close.pct_change(60, fill_method=None)
    out["drawdown_from_high"] = close / close.rolling(252, min_periods=252).max() - 1.0

    if volume is not None:
        vol = pd.Series(volume, dtype="float64")
        out["volume_change_20"] = vol.pct_change(20, fill_method=None)
    return out


# ==========================================================================
# Rule-based detector
# ==========================================================================


@dataclass(slots=True)
class RegimeThresholds:
    """Thresholds for the rule-based detector.

    Volatility cutoffs are **expanding percentiles of the input series** rather
    than fixed numbers. A fixed "20% annualised volatility" is high for one
    instrument and unremarkable for another; the 75th percentile of an
    instrument's own history is a statement about *its* unusualness, which is
    what a regime is.
    """

    trend_window: int = 200
    fast_window: int = 50
    high_vol_percentile: float = 75.0
    low_vol_percentile: float = 25.0
    min_bars: int = 60
    #: Volatility is checked first, so a violent bear market is labelled by its
    #: defining characteristic rather than by its direction.
    volatility_takes_precedence: bool = True

    def as_dict(self) -> dict:
        return {
            "trend_window": self.trend_window,
            "fast_window": self.fast_window,
            "high_vol_percentile": self.high_vol_percentile,
            "low_vol_percentile": self.low_vol_percentile,
            "min_bars": self.min_bars,
            "volatility_takes_precedence": self.volatility_takes_precedence,
        }


class RuleBasedRegimeDetector:
    """Deterministic regime labels with no fitted parameters.

    Because nothing is fitted, nothing can be overfitted - which is the reason
    to prefer this over the ML detector when the two disagree, until there is
    out-of-sample evidence that the ML version is actually better.
    """

    method = "rule_based"

    def __init__(self, thresholds: Optional[RegimeThresholds] = None) -> None:
        self.t = thresholds or RegimeThresholds()

    def detect_series(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        labelling_lag: int = 1,
    ) -> pd.DataFrame:
        """Label every bar.

        ``labelling_lag`` shifts the ``available_at`` column forward. A label
        computed from bar ``t``'s close cannot be acted on before bar
        ``t + labelling_lag`` opens, and the default of one bar reflects that.
        Setting it to zero makes the labels look actionable in the same bar they
        were computed from, which is look-ahead bias with extra steps.
        """
        f = regime_features(prices, volume)
        if len(f) < self.t.min_bars:
            raise ValueError(
                f"{len(f)} bars is too few to classify a regime; "
                f"{self.t.min_bars} is the floor. Volatility percentiles need a "
                "distribution to compare against."
            )

        vol = f["realized_vol_20"]
        # Expanding percentiles: bar t is compared against the distribution of
        # everything up to t, not the whole sample. Using the full-sample
        # distribution would let a later crash change how an earlier calm period
        # was labelled.
        hi = vol.expanding(min_periods=self.t.min_bars).quantile(
            self.t.high_vol_percentile / 100.0
        )
        lo = vol.expanding(min_periods=self.t.min_bars).quantile(
            self.t.low_vol_percentile / 100.0
        )
        slow = f[f"sma_{self.t.trend_window}"]
        fast = f[f"sma_{self.t.fast_window}"]
        close = pd.Series(prices, dtype="float64").reindex(f.index)

        bull = (close > slow) & (fast > slow)
        bear = (close < slow) & (fast < slow)

        regime = pd.Series(Regime.SIDEWAYS, index=f.index, dtype="object")
        regime = regime.mask(bull, Regime.BULL).mask(bear, Regime.BEAR)
        high_vol = vol > hi
        low_vol = vol < lo
        if self.t.volatility_takes_precedence:
            regime = regime.mask(high_vol, Regime.HIGH_VOLATILITY)
            regime = regime.mask(low_vol, Regime.LOW_VOLATILITY)
        else:
            regime = regime.mask(high_vol, Regime.HIGH_VOLATILITY)
            regime = regime.mask(low_vol, Regime.LOW_VOLATILITY)

        # Before the volatility reference exists, direction is the only thing
        # that can be said. Leaving those bars unlabelled would drop the warmup
        # from every regime-conditional statistic without saying so.
        warm = vol.isna() | hi.isna() | lo.isna()
        regime = regime.mask(warm, np.nan)

        ts = pd.Series(f.index, index=f.index)
        return pd.DataFrame({
            "regime": regime,
            "available_at": ts + pd.Timedelta(days=labelling_lag),
            "realized_vol_20": vol,
            "vol_high_threshold": hi,
            "vol_low_threshold": lo,
            "trend_strength": f["trend_strength"],
        })

    def detect(
        self, market_state: pd.Series, timestamp: Optional[pd.Timestamp] = None
    ) -> RegimeLabel:
        """Label a single bar from a state carrying its own history.

        ``market_state`` is a series of closes ending at the bar of interest,
        so the thresholds are derived from the history actually available.
        """
        s = pd.Series(market_state, dtype="float64").dropna()
        if len(s) < self.t.min_bars:
            raise ValueError(
                f"{len(s)} observations supplied; {self.t.min_bars} are needed to "
                "form a volatility reference"
            )
        frame = self.detect_series(s)
        ts = timestamp if timestamp is not None else s.index[-1]
        row = frame.iloc[-1]
        if pd.isna(row["regime"]):
            raise ValueError(
                f"no regime could be assigned at {ts}: the volatility reference "
                "has not accumulated enough history"
            )
        return RegimeLabel(
            regime=str(row["regime"]),
            method=self.method,
            timestamp=pd.Timestamp(ts),
            parameters=self.t.as_dict(),
            features={
                "realized_vol_20": _f(row["realized_vol_20"]),
                "vol_high_threshold": _f(row["vol_high_threshold"]),
                "vol_low_threshold": _f(row["vol_low_threshold"]),
                "trend_strength": _f(row["trend_strength"]),
            },
        )


def _f(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ==========================================================================
# Regime-conditioned analysis
# ==========================================================================


def regime_breakdown_report(
    returns: pd.Series,
    regimes: pd.Series,
    periods_per_year: int = 252,
    trade_pnls: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Performance within each regime, with an explicit dominance warning.

    A strategy that works in one regime and not the others has a regime bet.
    That may be deliberate, but the historical frequency of the favourable
    regime is then part of the thesis and belongs in front of the reader, not
    buried in an appendix.
    """
    from quantforge.quant import risk_metrics as rm

    r = pd.Series(returns).astype("float64")
    labels = pd.Series(regimes).reindex(r.index)
    rows = []
    for regime in sorted(set(labels.dropna().unique())):
        sub = r[labels == regime].dropna()
        if sub.empty:
            rows.append({"regime": regime, "n_periods": 0,
                         "share_of_history": 0.0})
            continue
        equity = (1.0 + sub).cumprod()
        dd = rm.max_drawdown(equity)
        row = {
            "regime": regime,
            "n_periods": len(sub),
            "share_of_history": len(sub) / len(r),
            "total_return": float(equity.iloc[-1] - 1.0),
            "mean_period": float(sub.mean()),
            "volatility": (float(sub.std(ddof=1) * np.sqrt(periods_per_year))
                           if len(sub) > 1 else float("nan")),
            "sharpe": rm.sharpe_ratio(sub, periods_per_year=periods_per_year).value,
            "max_drawdown": dd.max_drawdown,
            "win_rate": float((sub > 0).mean()),
        }
        if trade_pnls is not None:
            tp = pd.Series(trade_pnls).reindex(sub.index).dropna()
            if len(tp):
                row["n_trades"] = len(tp)
                row["profit_factor"] = rm.profit_factor(tp).value
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(
        "n_periods", ascending=False
    ).reset_index(drop=True)

    # Unlabelled periods - the volatility warmup - get their own row rather
    # than being dropped or absorbed into a regime. Absorbing them would make
    # the shares sum to 1 while quietly mislabelling warmup as, say,
    # SIDEWAYS; dropping them would make the shares sum to less than 1 with no
    # statement of where the remainder went.
    labelled = int(labels.notna().sum())
    if labelled < len(r):
        frame = pd.concat([frame, pd.DataFrame([{
            "regime": "UNLABELLED",
            "n_periods": int(len(r) - labelled),
            "share_of_history": (len(r) - labelled) / len(r),
            "total_return": float("nan"),
            "note": ("warmup: the volatility percentile has no reference yet, "
                     "so no regime can be assigned"),
        }])], ignore_index=True)
    return frame


def regime_dominance(returns: pd.Series, regimes: pd.Series) -> dict:
    """How much of the result depends on one regime.

    Reports the best regime's share of the total and of the absolute P&L. A
    strategy whose entire outperformance comes from a regime that occupied 8%
    of the history is a regime bet, and this is the number that says so.
    """
    from quantforge.quant import risk_metrics as rm

    r = pd.Series(returns).astype("float64")
    labels = pd.Series(regimes).reindex(r.index)
    per = {}
    for regime in sorted(set(labels.dropna().unique())):
        sub = r[labels == regime].dropna()
        if sub.empty:
            continue
        per[regime] = {
            "n_periods": len(sub),
            "share_of_periods": len(sub) / len(r),
            "total_return": float((1.0 + sub).prod() - 1.0),
            "sharpe": rm.sharpe_ratio(sub).value,
        }
    if not per:
        return {"per_regime": {}, "dominant": None, "note": "no labelled periods"}

    overall = float((1.0 + r.dropna()).prod() - 1.0)
    ranked = sorted(per.items(), key=lambda kv: kv[1]["total_return"], reverse=True)
    best, best_stats = ranked[0]
    return {
        "per_regime": per,
        "overall_return": overall,
        "dominant_regime": best,
        "dominant_regime_share_of_periods": best_stats["share_of_periods"],
        "dominant_regime_return": best_stats["total_return"],
        "note": (
            "If the best regime occupied a small share of the history, the "
            "strategy is a bet on that regime recurring at that frequency. That "
            "frequency is not a forecast: regimes are not equally likely in the "
            "future, and a regime absent from the sample cannot appear in the "
            "breakdown at all."
        ),
    }
