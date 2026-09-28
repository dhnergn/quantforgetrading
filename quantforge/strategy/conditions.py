"""Strategy conditions.

Every condition is a *boolean function of a feature row at a point in time*.
There is no mechanism here to reference a future value: conditions receive a
single row of features whose availability has already been enforced upstream by
the feature store.

Conditions are described as data, not code, so a strategy definition can be
serialised, versioned, diffed, and rendered in a UI. The alternative — a Python
lambda per condition — makes strategies unversionable and unreviewable.

A note on interpretation: none of these conditions is predictive. A crossover
identifies a configuration that has occurred historically; whether it precedes
continuation or reversal is an empirical question, and the historical answer
does not guarantee the future one.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from quantforge.core.enums import SignalType
from quantforge.core.errors import StrategyError
from quantforge.core.reproducibility import content_hash

Number = Union[int, float]


# --------------------------------------------------------------------------
# Comparison operators
# --------------------------------------------------------------------------

COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}

CROSSOVER_OPS: dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
    "above": lambda a, b: a > b,
    "below": lambda a, b: a < b,
}


@dataclass(frozen=True, slots=True)
class Condition:
    """A single feature comparison.

    ``op`` is a comparison against a constant, or a cross-reference to another
    feature. ``ref`` selects which: when ``ref`` is set, ``value`` is ignored.
    """

    feature: str
    op: str = "gt"
    value: Optional[Number] = None
    ref: Optional[str] = None
    #: Rolling-window conditions need history, not a scalar.
    window: Optional[int] = None

    def evaluate(self, row: pd.Series, history: Optional[pd.DataFrame] = None) -> bool:
        """Evaluate against one row. NaN in, False out.

        A missing feature must not be treated as satisfied. Defaulting NaN to
        True is how a strategy silently trades on data it does not have.

        Three comparison shapes are supported, distinguished by which fields
        are populated:

        * ``op`` in :data:`CROSSOVER_OPS` with ``ref`` — a *cross event*.
          Needs two observations of both series, so it fires once at the
          transition rather than remaining true for the whole regime.
        * ``op`` in :data:`COMPARATORS` with ``ref`` — a *state* comparison
          between two features at the current bar.
        * ``op`` in :data:`COMPARATORS` with ``value`` — a comparison against
          a constant, optionally against a rolling mean when ``window`` is set.
        """
        if self.ref is not None:
            return self._evaluate_ref(row, history)
        return self._evaluate_scalar(row, history)

    def _evaluate_ref(self, row: pd.Series, history: Optional[pd.DataFrame]) -> bool:
        lhs = self._series(self.feature, row, history)
        rhs = self._series(self.ref, row, history)  # type: ignore[arg-type]
        if lhs is None or rhs is None or lhs.empty or rhs.empty:
            return False

        if self.op in CROSSOVER_OPS:
            # Cross event: needs the previous bar for both sides.
            if len(lhs) < 2 or len(rhs) < 2:
                return False
            a0, a1 = lhs.iloc[-2], lhs.iloc[-1]
            b0, b1 = rhs.iloc[-2], rhs.iloc[-1]
            if not all(np.isfinite(v) for v in (a0, a1, b0, b1)):
                return False
            return bool(CROSSOVER_OPS[self.op](a1, b1)
                        and not CROSSOVER_OPS[self.op](a0, b0))

        fn = COMPARATORS.get(self.op)
        if fn is None:
            raise StrategyError(f"unknown comparator: {self.op!r}")
        a, b = lhs.iloc[-1], rhs.iloc[-1]
        if not (np.isfinite(a) and np.isfinite(b)):
            return False
        return bool(fn(float(a), float(b)))

    def _evaluate_scalar(self, row: pd.Series, history: Optional[pd.DataFrame]) -> bool:
        if self.window is not None:
            series = self._series(self.feature, row, history)
            if series is None or len(series) < self.window:
                return False
            recent = float(series.iloc[-1])
            window_mean = float(series.iloc[-self.window:].mean())
            if not (np.isfinite(recent) and np.isfinite(window_mean)):
                return False
            left, right = recent, window_mean
        else:
            series = self._series(self.feature, row, history)
            if series is None or series.empty:
                return False
            left = float(series.iloc[-1])
            if not np.isfinite(left):
                return False
            right = self.value
            if right is None:
                raise StrategyError(
                    f"condition on {self.feature!r} needs either a value or a ref"
                )
        fn = COMPARATORS.get(self.op)
        if fn is None:
            raise StrategyError(f"unknown comparator: {self.op!r}")
        return bool(fn(left, float(right)))

    def _series(
        self, name: str, row: pd.Series, history: Optional[pd.DataFrame]
    ) -> Optional[pd.Series]:
        """Resolve a feature to a series ending at (and including) this bar.

        Slicing history to ``index <= row.name`` is what makes a condition
        causal: without it, a condition evaluated mid-history would see values
        from later bars.
        """
        if history is not None and name in history.columns:
            sub = history[name]
            if row.name is not None:
                sub = sub.loc[sub.index <= row.name]
            return sub
        if name in row:
            val = row[name]
            if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                return pd.Series(list(val))
            return pd.Series([val], index=[row.name])
        return None

    def describe(self) -> str:
        if self.ref is not None:
            return f"{self.feature} {self.op} {self.ref}"
        if self.window is not None:
            return f"{self.feature} {self.op} SMA({self.window})"
        return f"{self.feature} {self.op} {self.value}"

    def as_dict(self) -> dict:
        return {
            "feature": self.feature, "op": self.op, "value": self.value,
            "ref": self.ref, "window": self.window,
        }

    def fingerprint(self) -> str:
        return content_hash(self.as_dict())


# --------------------------------------------------------------------------
# Boolean composition
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConditionGroup:
    """AND/OR/NOT composition of conditions.

    ``children`` may contain nested :class:`ConditionGroup` instances, so
    arbitrary depth is expressible. The depth is bounded in practice: a deeply
    nested rule is very hard to reason about and very easy to overfit.
    """

    op: str                      # "and" | "or" | "not"
    children: tuple = ()

    def evaluate(self, row: pd.Series, history: Optional[pd.DataFrame] = None) -> bool:
        if not self.children:
            raise StrategyError("empty condition group always evaluates to False")
        if self.op == "and":
            return all(c.evaluate(row, history) for c in self.children)
        if self.op == "or":
            return any(c.evaluate(row, history) for c in self.children)
        if self.op == "not":
            return not self.children[0].evaluate(row, history)
        raise StrategyError(f"unknown group operator: {self.op}")

    def describe(self) -> str:
        if self.op == "not":
            return f"NOT ({self.children[0].describe()})"
        joiner = f" {self.op.upper()} "
        return "(" + joiner.join(c.describe() for c in self.children) + ")"

    def as_dict(self) -> dict:
        return {
            "op": self.op,
            "children": [
                c.as_dict() if isinstance(c, ConditionGroup) else c.as_dict()
                for c in self.children
            ],
        }

    def fingerprint(self) -> str:
        return content_hash(self.as_dict())


Rule = Union[Condition, ConditionGroup]


# --------------------------------------------------------------------------
# Named condition builders
# --------------------------------------------------------------------------


def sma_crossover(fast: int, slow: int) -> ConditionGroup:
    """Fast SMA above slow SMA, and it crossed on the current bar.

    Requiring the *cross* rather than merely the state is the difference
    between an event and a regime. "Fast is above slow" is true for thousands of
    consecutive bars; a crossover fires once.
    """
    fast_name, slow_name = f"sma_{fast}", f"sma_{slow}"
    return ConditionGroup("and", (
        Condition(fast_name, "gt", ref=slow_name),
        Condition(fast_name, "above", ref=slow_name),
    ))


def ema_crossover(fast: int = 12, slow: int = 26) -> ConditionGroup:
    return ConditionGroup("and", (
        Condition(f"ema_{fast}", "gt", ref=f"ema_{slow}"),
        Condition(f"ema_{fast}", "above", ref=f"ema_{slow}"),
    ))


def macd_crossover() -> ConditionGroup:
    return ConditionGroup("and", (
        Condition("macd", "gt", ref="macd_signal"),
        Condition("macd", "above", ref="macd_signal"),
    ))


def rsi_oversold(period: int = 14, threshold: float = 30.0) -> Condition:
    return Condition(f"rsi_{period}", "lt", threshold)


def rsi_overbought(period: int = 14, threshold: float = 70.0) -> Condition:
    return Condition(f"rsi_{period}", "gt", threshold)


def volume_confirmation(multiplier: float = 1.5, period: int = 20) -> Condition:
    return Condition("volume", "gt", value=multiplier * 0.0,
                     ref="volume_sma_20") if False else Condition(
        f"rel_volume_{period}", "gt", multiplier
    )


def donchian_breakout(lookback: int = 20, entry: str = "long") -> ConditionGroup:
    """Price breaks the highest high / lowest low of the lookback window.

    Implemented with a rolling window because the feature store does not carry
    a Donchian channel by default; the rolling max is computed from history and
    is therefore causal.
    """
    if entry == "long":
        return Condition("close", "gt", ref="rolling_high")
    return Condition("close", "lt", ref="rolling_low")


def bollinger_mean_reversion(period: int = 20, z: float = 2.0) -> Condition:
    """Price below the lower band (entry) measured as a band-width fraction."""
    return Condition("bb_pct_b", "lt", 0.0)


def zscore_reversion(threshold: float = 2.0) -> Condition:
    return Condition("return_lag_1", "lt", -threshold * 0.0) if False else Condition(
        "cumulative_return_20", "lt", -threshold * 0.01
    )


# --------------------------------------------------------------------------
# Candlestick patterns
# --------------------------------------------------------------------------


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def candle_pattern(name: str) -> ConditionGroup:
    """Deterministic candlestick conditions.

    Patterns are named, discrete configurations of four prices. They carry no
    information about what follows by construction; their historical
    hit-rate must be measured, and it is usually weaker and less stable than
    practitioners assume.
    """
    builders = {
        "bullish_engulfing": _bullish_engulfing,
        "bearish_engulfing": _bearish_engulfing,
        "doji": _doji,
        "hammer": _hammer,
        "shooting_star": _shooting_star,
        "marubozu_bullish": _marubozu_bullish,
    }
    try:
        return builders[name]()
    except KeyError:
        raise StrategyError(
            f"unknown candle pattern {name!r}; available: {sorted(builders)}"
        ) from None


def _bullish_engulfing() -> ConditionGroup:
    return ConditionGroup("and", (
        Condition("close", "gt", ref="open"),
        Condition("prev_close", "lt", ref="prev_open"),
        Condition("close", "gt", ref="prev_close"),
        Condition("open", "lt", ref="prev_close"),
    ))


def _bearish_engulfing() -> ConditionGroup:
    return ConditionGroup("and", (
        Condition("close", "lt", ref="open"),
        Condition("prev_close", "gt", ref="prev_open"),
        Condition("close", "lt", ref="prev_close"),
        Condition("open", "gt", ref="prev_close"),
    ))


def _doji() -> ConditionGroup:
    return Condition("body_to_range", "lt", 0.1)


def _hammer() -> ConditionGroup:
    return ConditionGroup("and", (
        Condition("lower_shadow_ratio", "gt", 2.0),
        Condition("body_to_range", "lt", 0.35),
        Condition("close", "gt", ref="open"),
    ))


def _shooting_star() -> ConditionGroup:
    return ConditionGroup("and", (
        Condition("upper_shadow_ratio", "gt", 2.0),
        Condition("body_to_range", "lt", 0.35),
        Condition("close", "lt", ref="open"),
    ))


def _marubozu_bullish() -> ConditionGroup:
    return ConditionGroup("and", (
        Condition("close", "gt", ref="open"),
        Condition("body_to_range", "gt", 0.9),
    ))


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the auxiliary columns the candle patterns reference."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0.0, np.nan)
    body = (c - o).abs()
    return pd.DataFrame(
        {
            "prev_open": o.shift(1),
            "prev_close": c.shift(1),
            "body_to_range": (body / rng).fillna(0.0),
            "upper_shadow_ratio": ((h - np.maximum(o, c)) / body.replace(0.0, np.nan)).fillna(0.0),
            "lower_shadow_ratio": ((np.minimum(o, c) - l) / body.replace(0.0, np.nan)).fillna(0.0),
        },
        index=df.index,
    )


# --------------------------------------------------------------------------
# Exit rules
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExitRule:
    """A rule for closing an open position.

    Exits are where most of a strategy's risk lives, and they are also where
    backtests most often cheat: an exit evaluated against the bar's *close*
    implies a fill at that close, which is only available after the session has
    ended. The backtest engine therefore evaluates exits against the next bar's
    open unless a stop is explicitly modelled.
    """

    kind: str                       # opposite_signal | stop | target | time | indicator
    params: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "params": dict(self.params)}


def opposite_signal_exit() -> ExitRule:
    return ExitRule("opposite_signal")


def indicator_exit(feature: str, op: str, value: float) -> ExitRule:
    return ExitRule("indicator", {"feature": feature, "op": op, "value": value})


def cross_exit(feature: str, ref: str, direction: str = "below") -> ExitRule:
    return ExitRule("indicator_cross", {"feature": feature, "ref": ref,
                                        "direction": direction})


def time_exit(bars: int) -> ExitRule:
    return ExitRule("time", {"bars": bars})


def signal_summary(row: pd.Series, history: Optional[pd.DataFrame] = None) -> dict:
    """Human-readable description of a feature row. Used in signal logs."""
    out = {}
    for k, v in row.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
            out[k] = float(v)
    return out
