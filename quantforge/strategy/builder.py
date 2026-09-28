"""Versioned, serialisable strategy definitions.

A strategy is data: a versioned name, an entry rule, exit rules, a sizing
configuration, and the stop policy. It can be hashed, stored, diffed between
versions, and rendered in a UI without executing anything.

Keeping strategies as data rather than Python is what makes the
"strategy version locked" requirement in the live-trading gate meaningful: the
exact rule that was validated in paper trading is byte-identical to the one
promoted to live.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.enums import SignalType, SizingMethod, StopType, StrategyKind
from quantforge.core.errors import StrategyError
from quantforge.core.reproducibility import content_hash
from quantforge.core.timestamps import utcnow
from quantforge.strategy.conditions import (
    Condition, ConditionGroup, ExitRule, Rule, signal_summary,
)


@dataclass(slots=True)
class StopConfig:
    """Stop policy.

    ``atr_multiple`` is the default because a percentage stop ignores the
    instrument's own volatility: a 5% stop is a very different proposition on
    a 1%-daily-volatility mega-cap and on a 6%-volatility small cap.
    """

    stop_type: StopType = StopType.ATR
    atr_multiple: float = 2.0
    percentage: float = 0.05
    trailing_atr_multiple: Optional[float] = None
    trailing_percentage: Optional[float] = None
    time_stop_bars: Optional[int] = None
    use_exit_stop: bool = False

    def as_dict(self) -> dict:
        return {
            "stop_type": self.stop_type.value,
            "atr_multiple": self.atr_multiple,
            "percentage": self.percentage,
            "trailing_atr_multiple": self.trailing_atr_multiple,
            "trailing_percentage": self.trailing_percentage,
            "time_stop_bars": self.time_stop_bars,
            "use_exit_stop": self.use_exit_stop,
        }


@dataclass(slots=True)
class Strategy:
    """A complete, versioned trading rule."""

    strategy_id: str
    name: str
    version: str
    entry_rule: Rule
    long_exit_rules: tuple[ExitRule, ...] = ()
    short_exit_rules: tuple[ExitRule, ...] = ()
    kind: StrategyKind = StrategyKind.CROSSOVER
    stop: StopConfig = field(default_factory=StopConfig)
    sizing: dict = field(default_factory=dict)
    description: str = ""
    created_at: dt.datetime = field(default_factory=utcnow)
    #: Set by the registry once a version is locked for live trading.
    locked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.entry_rule, (Condition, ConditionGroup)):
            raise StrategyError(
                "entry_rule must be a Condition or ConditionGroup, not "
                f"{type(self.entry_rule).__name__}"
            )
        self.long_exit_rules = tuple(self.long_exit_rules)
        self.short_exit_rules = tuple(self.short_exit_rules)

    # -- versioning -------------------------------------------------------

    def fingerprint(self) -> str:
        """Content hash of the rule, excluding the timestamp.

        Two strategies with the same fingerprint are behaviourally identical and
        can share cached results.
        """
        return content_hash({
            "strategy_id": self.strategy_id,
            "version": self.version,
            "entry_rule": self.entry_rule.as_dict(),
            "long_exits": [r.as_dict() for r in self.long_exit_rules],
            "short_exits": [r.as_dict() for r in self.short_exit_rules],
            "stop": self.stop.as_dict(),
            "sizing": self.sizing,
            "kind": self.kind.value,
        })

    def as_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "version": self.version,
            "kind": self.kind.value,
            "entry_rule": self.entry_rule.as_dict(),
            "entry_description": self.entry_rule.describe(),
            "long_exit_rules": [r.as_dict() for r in self.long_exit_rules],
            "short_exit_rules": [r.as_dict() for r in self.short_exit_rules],
            "stop": self.stop.as_dict(),
            "sizing": self.sizing,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "fingerprint": self.fingerprint(),
            "locked": self.locked,
        }

    def lock(self) -> "Strategy":
        """Mark this exact version immutable for live execution."""
        self.locked = True
        return self

    # -- evaluation -------------------------------------------------------

    def evaluate_entry(
        self, row: pd.Series, history: Optional[pd.DataFrame] = None
    ) -> tuple[SignalType, dict]:
        """Evaluate the entry rule for one bar.

        Returns the signal and the feature values, which are logged together so
        a trade can be reconstructed after the fact.
        """
        try:
            fired = bool(self.entry_rule.evaluate(row, history))
        except StrategyError:
            raise
        except Exception as exc:  # a malformed row should not silently pass
            raise StrategyError(
                f"{self.strategy_id}@{self.version}: entry evaluation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        signal = SignalType.LONG if fired else SignalType.FLAT
        return signal, signal_summary(row, history)

    def should_exit(
        self,
        position_side: str,
        row: pd.Series,
        bars_held: int,
        history: Optional[pd.DataFrame] = None,
    ) -> tuple[bool, str]:
        """Evaluate exit rules for an open position.

        Exits are OR-combined: any one rule firing closes the position. Exit
        logic is intentionally simpler than entry logic; complexity there is
        usually a sign of curve-fitting the exit to the entry sample.
        """
        rules = self.long_exit_rules if position_side == "long" else self.short_exit_rules
        if not rules:
            return False, ""
        for rule in rules:
            if rule.kind == "time":
                bars = int(rule.params.get("bars", 0))
                if bars and bars_held >= bars:
                    return True, f"time stop after {bars_held} bars (limit {bars})"
            elif rule.kind == "indicator":
                cond = Condition(rule.params["feature"], rule.params["op"],
                                 rule.params.get("value"))
                if cond.evaluate(row, history):
                    return True, f"indicator exit: {cond.describe()}"
            elif rule.kind == "indicator_cross":
                cond = Condition(rule.params["feature"], rule.params["direction"],
                                 ref=rule.params["ref"])
                if cond.evaluate(row, history):
                    return True, f"cross exit: {cond.describe()}"
        return False, ""

    def stop_price(
        self, entry_price: float, side: str, atr: Optional[float],
        peak_price: Optional[float] = None,
    ) -> Optional[float]:
        """Compute the stop level for a position.

        Returns ``None`` when no stop is configured. The stop is a *trigger*,
        not a fill price: the engine models the gap between the trigger and the
        achievable fill separately, because a stop that gaps through fills
        worse than its level.
        """
        st = self.stop
        if st.stop_type == StopType.NONE:
            return None
        if st.stop_type == StopType.ATR:
            if atr is None or not np.isfinite(atr) or atr <= 0:
                return None
            dist = st.atr_multiple * atr
            return entry_price - dist if side == "long" else entry_price + dist
        if st.stop_type == StopType.PERCENTAGE:
            dist = entry_price * st.percentage
            return entry_price - dist if side == "long" else entry_price + dist
        if st.stop_type == StopType.TRAILING_ATR:
            if atr is None or not np.isfinite(atr) or atr <= 0:
                return None
            ref = peak_price if peak_price is not None else entry_price
            dist = (st.trailing_atr_multiple or st.atr_multiple) * atr
            return ref - dist if side == "long" else ref + dist
        if st.stop_type == StopType.TRAILING_PERCENT:
            ref = peak_price if peak_price is not None else entry_price
            dist = ref * (st.trailing_percentage or st.percentage)
            return ref - dist if side == "long" else ref + dist
        return None

    def describe(self) -> str:
        lines = [
            f"{self.name} ({self.strategy_id}@{self.version})",
            f"  kind: {self.kind.value}",
            f"  entry: {self.entry_rule.describe()}",
        ]
        if self.long_exit_rules:
            lines.append("  long exits: " + ", ".join(
                f"{r.kind}{r.params if r.params else ''}" for r in self.long_exit_rules
            ))
        if self.short_exit_rules:
            lines.append("  short exits: " + ", ".join(
                f"{r.kind}{r.params if r.params else ''}" for r in self.short_exit_rules
            ))
        lines.append(f"  stop: {self.stop.stop_type.value}")
        lines.append(f"  fingerprint: {self.fingerprint()}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Strategy({self.strategy_id}@{self.version}, {self.kind.value})"


# --------------------------------------------------------------------------
# Strategy registry
# --------------------------------------------------------------------------


class StrategyRegistry:
    """Versioned store. A locked version cannot be modified in place."""

    def __init__(self) -> None:
        self._versions: dict[str, list[Strategy]] = {}

    def register(self, strategy: Strategy) -> Strategy:
        bucket = self._versions.setdefault(strategy.strategy_id, [])
        for existing in bucket:
            if existing.version == strategy.version:
                if existing.fingerprint() != strategy.fingerprint():
                    raise StrategyError(
                        f"{strategy.strategy_id} version {strategy.version} is "
                        "already registered with a different definition. Create a "
                        "new version; silently replacing a version destroys the "
                        "reproducibility of prior results."
                    )
                return existing
        bucket.append(strategy)
        bucket.sort(key=lambda s: s.version)
        return strategy

    def get(self, strategy_id: str, version: Optional[str] = None) -> Strategy:
        bucket = self._versions.get(strategy_id)
        if not bucket:
            raise StrategyError(f"unknown strategy: {strategy_id}")
        if version is None:
            return bucket[-1]
        for s in bucket:
            if s.version == version:
                return s
        raise StrategyError(
            f"unknown version {version!r} for {strategy_id}; have "
            f"{[s.version for s in bucket]}"
        )

    def latest(self, strategy_id: str) -> Strategy:
        return self.get(strategy_id)

    def all(self) -> list[Strategy]:
        return [s for b in self._versions.values() for s in b]

    def versions(self, strategy_id: str) -> list[str]:
        return [s.version for s in self._versions.get(strategy_id, [])]

    def lock(self, strategy_id: str, version: Optional[str] = None) -> Strategy:
        s = self.get(strategy_id, version)
        return s.lock()

    def __len__(self) -> int:
        return sum(len(b) for b in self._versions.values())


# --------------------------------------------------------------------------
# Preset builders
# --------------------------------------------------------------------------


def sma_crossover_strategy(
    fast: int = 20, slow: int = 50,
    strategy_id: str = "sma_cross", version: str = "1.0.0",
) -> Strategy:
    from quantforge.strategy.conditions import Condition, ConditionGroup

    entry = ConditionGroup("and", (
        Condition(f"sma_{fast}", "gt", ref=f"sma_{slow}"),
        Condition(f"sma_{fast}", "above", ref=f"sma_{slow}"),
    ))
    return Strategy(
        strategy_id=strategy_id, name=f"SMA {fast}/{slow} Crossover",
        version=version, entry_rule=entry, kind=StrategyKind.CROSSOVER,
        long_exit_rules=(Condition(f"sma_{fast}", "below", ref=f"sma_{slow}")
                         if False else ExitRule("indicator_cross", {
                             "feature": f"sma_{fast}", "ref": f"sma_{slow}",
                             "direction": "below"}),),
        short_exit_rules=(ExitRule("indicator_cross", {
            "feature": f"sma_{fast}", "ref": f"sma_{slow}", "direction": "below"}),),
        stop=StopConfig(stop_type=StopType.ATR, atr_multiple=2.5),
        sizing={"method": SizingMethod.VOLATILITY_ATR.value, "risk_fraction": 0.01},
        description=(
            f"Long when the {fast}-bar SMA is above the {slow}-bar SMA and has just "
            f"crossed above. Exit on a cross back below or a {2.5:g}x ATR stop. "
            "The rule describes a configuration that has occurred historically; it "
            "does not predict that continuation will follow."
        ),
    )


def rsi_mean_reversion_strategy(
    period: int = 14, oversold: float = 30.0, overbought: float = 70.0,
    strategy_id: str = "rsi_reversion", version: str = "1.0.0",
) -> Strategy:
    from quantforge.strategy.conditions import Condition, ConditionGroup

    entry = ConditionGroup("and", (
        Condition(f"rsi_{period}", "lt", oversold),
        Condition("close", "gt", ref="open"),
    ))
    return Strategy(
        strategy_id=strategy_id, name=f"RSI({period}) Mean Reversion",
        version=version, entry_rule=entry, kind=StrategyKind.MEAN_REVERSION,
        long_exit_rules=(ExitRule("indicator", {
            "feature": f"rsi_{period}", "op": "gte", "value": overbought}),),
        stop=StopConfig(stop_type=StopType.ATR, atr_multiple=2.0),
        sizing={"method": SizingMethod.FIXED_FRACTIONAL.value, "fraction": 0.10},
        description=(
            f"Buy when RSI({period}) is below {oversold:g} and the bar closed up. "
            f"Exit at RSI {overbought:g} or a 2x ATR stop. An RSI extreme is a "
            "statement about recent price movement, not about what happens next; "
            "an oversold instrument can stay oversold for months."
        ),
    )


def bollinger_mean_reversion_strategy(
    period: int = 20, k: float = 2.0, strategy_id: str = "bb_reversion",
    version: str = "1.0.0",
) -> Strategy:
    from quantforge.strategy.conditions import Condition, ConditionGroup

    entry = ConditionGroup("and", (
        Condition("bb_pct_b", "lt", 0.0),
        Condition("bb_pct_b", "gte", 0.0 - 0.02),
    ))
    return Strategy(
        strategy_id=strategy_id, name=f"Bollinger({period}, {k:g}) Reversion",
        version=version, entry_rule=entry, kind=StrategyKind.MEAN_REVERSION,
        long_exit_rules=(ExitRule("indicator", {
            "feature": "bb_pct_b", "op": "gte", "value": 0.5}),),
        stop=StopConfig(stop_type=StopType.PERCENTAGE, percentage=0.08),
        sizing={"method": SizingMethod.FIXED_FRACTIONAL.value, "fraction": 0.10},
        description=(
            "Buy when the close is at or below the lower Bollinger band. Exit at "
            "the band midpoint. A band touch is a statistical event; whether it "
            "precedes reversion must be measured out of sample, not assumed."
        ),
    )
