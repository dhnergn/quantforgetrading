# Archived prior scaffold

These files predate the current architecture and reference a module layout
(`quantforge.signals`, `quantforge.broker`, `quantforge.risk.portfolio`, etc.)
that no longer exists. They are kept for reference only and are
not importable. The current implementation lives in `quantforge/`.

## Do not use any of this

The current platform reimplements all of this with the safety properties the
original lacked: point-in-time availability, leakage-guarded preprocessing, a
documented order state machine, and enforced risk limits. Where the two
disagree, the current implementation is correct. See `../docs/`.

## Completeness: this archive is PARTIAL

`quantforge/indicators/` and `quantforge/strategies/` were recovered intact and
their internal references resolve. The other package subtrees the tests import
were lost before this archive was made and are **not** here:

| Module referenced by the archived tests | Status |
|---|---|
| `quantforge.indicators` | present |
| `quantforge.strategies.{base,crossover,mean_reversion,breakout}` | present |
| `quantforge.backtest.engine` | **missing** |
| `quantforge.broker.{base,paper}` | **missing** |
| `quantforge.data.{ingestion,validation}` | **missing** |
| `quantforge.features.engine` | **missing** |
| `quantforge.risk.{portfolio,position_sizing,stops}` | **missing** |
| `quantforge.signals.generator` | **missing** |

`tests/` therefore will not run. They are kept as a record of what the original
scaffold intended to verify, not as an executable suite. To revive it you would
need to reconstruct the missing modules from `QUANTFORGE_COMPLETE.md` and the
tests' own expectations.

## Why `quantforge.indicators.RSI` must not be used

The original `calculate()` returned the **entire input frame** with the
open/high/low/close columns overwritten by the indicator value, instead of a
result frame containing just the indicator's own column:

```python
stray.RSI().calculate(df).columns
# ['open', 'high', 'low', 'close', 'volume']   <- wrong: the input, clobbered
tested.RSI(14).calculate(df).columns
# ['rsi']                                       <- correct
```

So an import from this path silently returns malformed data with the right
values in the wrong shape. This is why the archive must not be importable
alongside the real package.

## History: this directory was recreated, then removed again

`quantforge/indicators/` was moved here on request, then reappeared in the live
package over several minutes — one file at a time, with mtimes advancing while
being observed. A concurrent process was writing it.

The recreated copy restored four of the five files byte-identically but wrote
`volatility.py` with a **regression**: `ATR.calculate` referenced the
module-level True Range class `TR` instead of the local `tr` series.

```python
# live copy:    NameError: name 'TR' is not defined
# this archive: 1.25
```

That is strictly worse than either previous state: the package imported
successfully and then failed at runtime. The archived `volatility.py` is the
correct `tr.ewm(...)` version, so removing the live copy lost nothing.

**If `quantforge/indicators/` or `quantforge/strategies/` reappear in the live
package, something else is writing to this tree.** Check for another agent or
session working here before assuming the move did not take.

