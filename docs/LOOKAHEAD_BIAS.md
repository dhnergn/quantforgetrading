# Look-Ahead Bias

The most expensive class of bug in quantitative research. A backtest with
look-ahead bias produces numbers that look entirely reasonable and describe a
system that could not have traded.

## The five sources

| Source | Example | Platform defence |
|---|---|---|
| **Bar close as a decision input** | Using bar `T`'s close to trade at bar `T`'s close | Signals fill on `T+1` at the open |
| **Restated fundamentals** | Using a Q3 figure revised in Q4 during Q3 | `publication_timestamp` filter, revision tracking |
| **Revised macro** | Using today's revised GDP for a 2019 decision | Vintage-aware `available_at` |
| **Survivorship** | Backtesting today's index members in 2005 | Point-in-time universe membership |
| **Preprocessing** | Fitting a scaler on the full dataset before splitting | `assert_no_future_fit` |

---

## Mechanism 1: the availability boundary

A feature computed from bar `T` is available at `T` **plus the session close
and a processing allowance** — never at `T`.

```python
def session_close_utc(index, close_hour_utc=21):
    """Availability boundary for close-derived daily features."""
    return index.map(lambda ts: ts.replace(hour=close_hour_utc))
```

At 20:00 UTC on a US session day, the day's close is not yet observable. A
query returns the *previous* bar's value:

```python
bar = ohlcv.index[100]
store.get_point_in_time("EQ001", bar, ["close"])                  # previous bar
store.get_point_in_time("EQ001", bar + pd.Timedelta(hours=21), ["close"])  # this bar
```

The `available_at` column is what makes this work, and it is stamped at write
time rather than derived at read time so it cannot be forgotten.

---

## Mechanism 2: the fill gap

The engine separates a signal bar from a fill bar.

```
bar T:  features available at T+21:00 → signal → risk check → order queued
bar T+1: order fills at the open, plus impact, spread, and commission
```

A stop that triggers mid-bar fills at the stop only if the bar traded through
it, and at the **open** if the bar gapped past it:

```python
# A bar gapping through a buy stop at 95, opening at 88:
res = simulate_fill_price("long", 88.0, 89.0, 87.0, 88.5, 100, 1e6,
                          LiquidityModel(), order_type="stop", stop_price=95.0)
assert res["fill_price"] < 95.0     # filling at the stop would be the
assert res["gap_adjustment"] > 0    # classic stop-loss backtest error
```

---

## Mechanism 3: the machine check

The definitive test is that **a truncated run reproduces the full run's
prefix exactly**. If any future bar influenced a past decision, they diverge.

```python
full = engine.run(ohlcv, features)
full_equity = full.portfolio.equity_series()

for k in (300, 450, 550):
    part = engine.run(ohlcv.iloc[:k], features.iloc[:k])
    common = part.portfolio.equity_series().index.intersection(full_equity.index)
    assert (part.portfolio.equity_series().reindex(common)
            - full_equity.reindex(common)).abs().max() == 0.0
```

This is a stronger property than checking that no future *value* was read. It
catches subtle influences: a warmup computed over the whole series, a
normalisation constant fitted globally, a universe filtered by future
survival.

---

## Mechanism 4: corporate actions respect the announcement date

A split is known at its **announcement**, not its ex-date. `known_at()` returns
only actions public by the query instant:

```python
register.known_at("X", datetime(2024, 5, 1))    # [] - not announced yet
register.known_at("X", datetime(2024, 5, 20))   # [split] - announced 5/15
```

Using an action before its announcement would let a backtest restate history
with knowledge the market did not have.

---

## Mechanism 5: fail loudly rather than trade nothing

A strategy referencing a feature the frame lacks would evaluate every
condition to `False` and produce a flat, plausible-looking backtest. The engine
refuses:

```python
eng = BacktestEngine(sma_crossover_strategy(5, 10), config)  # needs sma_5
eng.run(ohlcv, features_without_sma_5)
# BacktestError: strategy references features absent from the feature frame:
#   ['sma_5']. Without them every condition evaluates False and the backtest
#   would report a flat result rather than failing.
```

A zero-signal backtest is the most dangerous failure mode, because it looks
like a valid result.

---

## Tests

`tests/test_leakage.py` — 40 tests covering:

- `TemporalEnvelope` rejects impossible orderings
- `DecisionChain` rejects non-monotonic chains
- a close feature is unavailable at its own bar
- an injected future value is never returned
- `NaN` is never stored as an available value
- truncated runs match the full-run prefix
- fills never occur on the signal bar
- stop gaps fill at the open
- missing features abort the run
- split factors apply to prior bars, not forward
- dividends scale correctly across a split

---

## Common ways to reintroduce it

1. **Fitting a transform on the full dataset.** Guarded by `assert_no_future_fit`.
2. **`pct_change()` with the default forward fill.** Manufactures a return
   across a price gap. Use `fill_method=None`.
3. **Using a vendor's "adjusted" series without checking the basis.** Ask
   whether the adjustment used the ex-date or the announcement, and whether
   dividends are included.
4. **Building a universe from current constituents.** Use point-in-time
   membership with delisted names included.
5. **Survivorship in cross-sectional studies.** A symbol absent from a
   historical panel because it delisted later is a silent bias.
6. **Reshuffling time series with `sample(frac=1)`** for a "shuffled
   benchmark" destroys serial dependence and produces an unrealistically
   favourable result.
7. **Reading a full-history file into a rolling feature function.** The rolling
   window must operate on the slice available at each point.
