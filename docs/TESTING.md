# Testing

```bash
.venv/bin/python -m pytest tests/ -q                 # 228 tests
.venv/bin/python -m pytest tests/ -m leakage -q      # causality only
.venv/bin/python -m pytest tests/test_leakage.py -q  # 40 tests
```

Nothing in the suite contacts a network, a broker, or a data vendor, and no
test can place a real order. All data is synthetic.

## Structure

| File | Tests | Covers |
|---|---|---|
| `test_indicators.py` | 42 | Mathematical reference values, edge cases, metadata self-consistency |
| `test_leakage.py` | 40 | Temporal semantics, point-in-time retrieval, preprocessing guards, corporate-action causality |
| `test_risk.py` | 62 | Portfolio accounting, sizing, risk engine, kill switches, VaR/ES, costs, liquidity |
| `test_execution_and_validation.py` | 84 | Order state machine, idempotency, reconciliation, live gate, security, statistics, validation |

---

## Indicator tests

Reference values are **hand-computed**, not taken from another library, so a
library change would be caught as well as a bug in this code.

```python
def test_sma_matches_hand_computed(self):
    s = ind.SMA(3).calculate(frame(range(1, 11)))["sma"]
    assert s.iloc[2] == pytest.approx(2.0)      # (1+2+3)/3
```

Every indicator's declared warmup is verified against observed behaviour:

```python
def test_all_declared_warmups_are_correct(self):
    assert ind.validate_metadata() == []
```

`validate_metadata()` recomputes the first defined index for every indicator
and reports discrepancies. This caught four off-by-one warmups (ADX, KAMA, MACD
via the multi-column rule, ROC) and one real bug
(`RealizedVolatility` overstated annualised volatility by √window).

### Edge cases asserted

- RSI on a constant series is `NaN`, not 50 (0/0 is undefined)
- CCI on a constant series is `NaN`, not 0
- Stochastic on a zero-range bar is `NaN`
- Bollinger `%B` on zero volatility is `NaN`
- OBV signs, CMF sign, VWAP within range, MFI bounded
- A `NaN` inside a window does **not** shorten it

---

## Causality tests

The most important file in the suite.

### The machine check

```python
def test_truncated_run_matches_the_full_run_prefix(...):
    for k in (300, 450, 550):
        part = self._run(ohlcv.iloc[:k], features.iloc[:k], ...)
        common = ...intersection...
        assert (part_equity - full_equity).abs().max() == 0.0
```

A truncated run must reproduce the full run's prefix **exactly**. This is
stronger than checking that no future value was read: it catches warmups
computed over the whole series, global normalisation constants, and
future-filtered universes.

### The fill gap

```python
assert o.filled_bar > o.signal_bar      # never fills on the signal bar
```

plus a distance comparison, because a tolerance-based "not equal" check is
unreliable when overnight gaps are small.

### Stop gaps

```python
res = simulate_fill_price("long", 88.0, 89.0, 87.0, 88.5, 100, 1e6,
                          liquidity, order_type="stop", stop_price=95.0)
assert res["fill_price"] < 95.0        # not the stop
assert res["gap_adjustment"] > 0
```

### Feature availability

```python
at_bar = store.get_point_in_time("EQ001", bar, ["close"])["close"]
assert at_bar != ohlcv.loc[bar, "close"]          # previous bar
after = store.get_point_in_time("EQ001", bar + 21h, ["close"])["close"]
assert after == ohlcv.loc[bar, "close"]           # this bar
```

### Corporate-action causality

- announcement date bounds availability
- an action announced after its ex-date is rejected
- `known_at` excludes unannounced actions
- raw prices are never modified
- the split factor applies to **prior** bars
- split-adjusted returns contain no jump
- dividends scale across a split in both directions

---

## Risk tests

- realised and unrealised P&L are kept separate
- average entry is a weighted average across scale-ins
- a partial scale-out does not book a trade
- a reversal leaves `order - closed`, not the full order
- unpriced holdings are not valued at zero
- returns reconcile with equity differences
- sizing never exceeds a position limit
- Kelly is capped, fractional, and takes no position on a negative edge
- limits are evaluated against the **post-fill** state
- a permission check that raises is denied
- the kill switch fails closed with no context
- repeated triggers do not flap
- ES ≥ VaR; VaR rises with confidence; Student-t > normal on fat tails
- undefined ratios are `NaN` with an explanation
- a zero-cost model is loudly flagged
- an unknown volume refuses to fill

---

## Execution safety tests

- terminal states are final
- `UNKNOWN` cannot return to `SUBMITTED`
- an overfill raises
- duplicate submissions coalesce
- dedupe survives a restart via the database
- a submission timeout leaves state unknown and does not resolve
- an unmapped broker state escalates rather than resolving to accepted
- a disconnected broker refuses submission
- a missing idempotency key refuses submission
- position mismatches block trading; corrections are explicit
- every live-gate refusal path refuses

---

## Statistical tests

- correlation of a constant series is `NaN`
- variance of a constant series is `0.0`
- too-few-observations is `NaN`
- the two downside-deviation conventions differ materially
- beta against a constant benchmark is `NaN`
- shrinkage fixes a rank-deficient covariance and refuses a hopeless one
- **the variance-ratio test is calibrated**: ≤4/20 false positives on iid data

That last one caught a badly miscalibrated normal-approximation p-value (37%
false positives) and an inverted sign. The fix was an iid bootstrap under the
null.

---

## Validation tests

- temporal windows never overlap
- a test window is mandatory
- an embargo creates a measurable gap
- walk-forward produces windows and an aggregate
- Monte Carlo on zero drift is symmetric; a positive drift compounds
- **a bootstrap cannot invent a worse single period** (documented limitation)
- overfitting flags fire on a suspicious result
- the effective parameter count counts *configurations*, not varied numbers
- `E[max]` is monotonic in trials and scales with dispersion
- grid search filters low-trade-count optima
- sensitivity distinguishes a plateau from a narrow peak
- the report never declares robustness

---

## Bugs the suite caught

Each is a regression test, and each was a real defect:

| Bug | Impact |
|---|---|
| `RealizedVolatility` omitted `÷ n` inside the sqrt | 4.4x overstatement of annualised vol at n=20; corrupts every vol-scaled position size |
| `split_factor_series` applied the factor forward | a fictitious −75% move at every split |
| Portfolio reversal opened the full order size | short 300 instead of short 200 |
| `clamp_to_limits` left notional unchanged | reported a cap that was not applied |
| Sharpe/Sortino zero-volatility guards used exact zero | float residue produced Sharpe ≈ 1e16 |
| Historical VaR took the wrong tail | reported a loss as a gain (negative VaR) |
| Student-t VaR used `σ/√ν` instead of `σ√((ν−2)/ν)` | fat-tailed fit looked *less* conservative than normal |
| Student-t and MC ES used the wrong side of the distribution | ES < VaR, violating a mathematical identity |
| MC ES threshold used the confidence quantile | averaged over 95% of paths |
| `monte_carlo_returns` appended the observed path unreshaped | crashed whenever `horizon != len(returns)` |
| GARCH simulator omitted `omega` | variance decayed to zero; prices froze |
| A falsy-but-valid `FeatureSet` was replaced | every store write failed |
| `SandboxViolation` defined twice | one `except` clause could miss violations |
| Paper broker: marketable limit did not fill | non-marketable behaviour on a marketable order |
| Variance-ratio test used Bartlett weights | 37% false positives; sign inverted |
| Coordinate descent double-counted the current coefficient | iteration oscillated; lasso never converged to OLS |
| `_expected_max_sharpe` used within-series variance | deflated Sharpe of −inf to +41 |

---

## Coverage philosophy

Tests assert **properties**, not just outputs:

- `ES >= VaR` for every confidence
- `adjusted returns contain no jump` across corporate actions
- `truncated run == full run prefix`
- `ES >= VaR` and `VaR increasing in confidence`
- `E[max] monotonic in trials`
- `chain valid` and `chain detects tampering`

A test that only checks a number was computed will pass while the number is
wrong in a way nobody noticed. The property tests are the ones that catch
mistakes.
