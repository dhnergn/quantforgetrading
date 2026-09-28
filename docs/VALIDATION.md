# Validation

Backtesting tells you what happened under assumptions you chose. Validation is
the machinery for finding out whether those assumptions survive contact with
data you did not select. This platform is deliberately weighted toward
falsification.

---

## Temporal splits

```python
split = temporal_split(index, train_frac=0.60, validation_frac=0.20,
                       embargo_bars=5)
```

| Window | Range | Bars |
|---|---|---|
| train | 2018-01-01 → 2021-06-04 | 895 |
| validation | 2021-06-21 → 2022-07-29 | 290 |
| test | 2022-08-15 → 2023-09-29 | 295 |

The **embargo** matters more than it appears. A label with a 5-day horizon at
the end of training uses prices up to 5 days into validation, so the windows
share observations and validation performance is partly in-sample. The embargo
must be at least the label horizon.

A test window is mandatory:

```python
temporal_split(index, 0.6, 0.4)
# ValidationError: train_frac + validation_frac = 1.0 leaves no test period;
#   an out-of-sample test is not optional
```

`seal_test(split)` marks the window consumed. It is procedural discipline made
explicit: any subsequent change to features, parameters, or model choice
invalidates the result and requires a new split.

---

## Walk-forward analysis

```
Training window ─► validation (parameter selection) ─► out-of-sample test
        advance
Training window ─► validation ─► out-of-sample test
        ...
```

```python
wf = walk_forward(index, fit_predict, param_grid=grids,
                  train_bars=500, validation_bars=150, test_bars=150,
                  step=150, objective="sharpe")
```

`fit_predict(train_idx, eval_idx, params)` must fit on `train_idx` and score on
`eval_idx`. Parameter selection uses **only** the validation score. The
out-of-sample segment is then produced once with the selected parameters, and
concatenating those segments gives a return series that informed no decision.

Passing a raw return Series works directly:

```python
def fit_predict(tr, ev, params):
    r = engine.run(data.loc[tr.union(ev)], features.loc[tr.union(ev)])
    return r.portfolio.returns().reindex(ev).dropna()
```

`tr.union(ev)` rather than joining the frames: the two index ranges do not
overlap, so a join yields an all-NaN frame.

### Output

```
windows: 5   failed: 0

AGGREGATE OUT-OF-SAMPLE
  total_return                -0.0285
  sharpe                      -0.3144
  max_drawdown                 0.0515
  worst_window_score          -1.6105
  best_window_score            1.2936
  positive_window_fraction     0.25

PARAMETER STABILITY
  fast_modal_share   0.4
  slow_modal_share   0.6
```

**Parameter stability is itself a result.** A modal share near 1.0 indicates a
stable optimum; near `1/len(windows)` indicates the search is arbitrary. A
strategy whose parameters are re-selected every window is fitting noise.

Window-level dispersion matters as much as the aggregate: a good mean with
large dispersion was driven by a few windows and is not a dependable process.

---

## Monte Carlo

Four methods, with their limits stated rather than glossed:

| Method | Property | Limitation |
|---|---|---|
| `bootstrap` | Preserves the empirical distribution including fat tails | **Cannot invent a worse period** |
| `block_bootstrap` | Preserves short-range dependence | Fewer effective samples |
| `shuffle` | Exact multiset, order destroyed | No new information at all |
| `parametric` | Arbitrarily extreme values possible | Only as far as the assumption permits |

```python
mc = monte_carlo_returns(returns, n_simulations=5000, method="bootstrap")
mc.statistics     # mean, median, p05, p95, worst, best, drawdowns
mc.percentiles    # 1st..99th for final return and max drawdown
mc.probability_of_loss
mc.warnings
```

The bootstrap limitation is asserted in the test suite so it cannot be
forgotten:

```python
# horizon=1: no resampled period can be worse than the worst on record
assert mc.statistics["worst_final_return"] >= float(returns.min()) - 1e-12
```

Resampling a historical sample reproduces its distribution; it does not reveal
outcomes the sample never contained. A fat-tailed history resampled will not
produce a crisis it never saw.

Insufficient data refuses to simulate rather than producing a thin
distribution:

```python
monte_carlo_returns(pd.Series([0.01, -0.01] * 4))
# n_simulations: 0
# "only 8 return observations; 20 is the minimum ..."
```

---

## Grid search

```python
best, table = grid_search(evaluate, grids, objective="sharpe", min_trades=30)
```

`min_trades` filters **before** the optimum is chosen, not reports afterwards. A
configuration with two trades and a Sharpe of 4 is not a finding; it is a
division of two observations.

Selecting the top of this table is the first step of a selection process, not
the end. The chosen point is still in-sample.

Count the configurations honestly:

```python
parameter_count_effective({"fast": [5,10,20,40,80],
                           "slow": [20,30,50,100,200]})   # 25
```

Twenty-five configurations, not two degrees of freedom. This is `n_trials` for
the deflated Sharpe.

---

## Sensitivity analysis

One parameter varied at a time, distinguishing a **plateau** from a **narrow
peak**:

```python
table, stability = sensitivity_analysis(evaluate, base, axes, "sharpe")
stability["fast"]["verdict"]
# "stable plateau: performance persists across the neighbourhood"
# "NARROW PEAK: performance depends on precise parameter values"
```

A strategy that works only at `fast == 10` exactly was fitted to a point in
parameter space, not to a phenomenon.

---

## Robustness

```python
robustness_tests(run_variant, base_config)
# base, costs_2x, costs_5x, slippage_2x, delay_1bar, delay_2bar,
# no_costs, half_capital
```

Each variant perturbs one assumption. A strategy that survives higher costs,
delayed execution, and missing bars is more likely to be measuring something
real than one that works only at the chosen settings.

---

## Regime breakdown

```python
regime_breakdown(returns, regime_labels)
# regime              n    sharpe   max_dd
# BULL              208     1.211     0.148
# SIDEWAYS          194    -0.712     0.201
# BEAR              177     1.752     0.130
# HIGH_VOLATILITY   177     1.847     0.088
```

A strategy that works only in one regime is a regime bet. That may be a
legitimate choice, but it has to be made deliberately with knowledge of that
regime's historical frequency — not discovered after seeing that one slice
worked.

---

## Benchmarking

```python
compare_to_benchmark(returns, benchmark_returns, risk_free_annual=0.03)
# excess_total_return, excess_cagr, beta, correlation,
# tracking_error, information_ratio, relative_drawdown,
# up_capture, down_capture
```

The join is on index, and observations missing from either side are dropped.
Comparing over different date ranges produces meaningless level differences
even when the strategies are identical.

---

## Recommended sequence

1. **Data quality report** — do not proceed on corrupt data
2. **In-sample backtest** with costs, to confirm the code runs and the
   assumptions are stated
3. **Parameter sensitivity** — plateau or narrow peak?
4. **IS/OOS split** with an embargo at least as long as the label horizon
5. **Walk-forward** — aggregate plus window dispersion plus parameter stability
6. **Monte Carlo** — the distribution of outcomes, and its limits
7. **Robustness** — costs, slippage, execution delay
8. **Overfitting diagnostics** — deflated Sharpe, flag review
9. **Regime breakdown** — where does it work, and how often is that?
10. **Out-of-sample test** — once, at the very end

The final out-of-sample period is touched exactly once. A researcher who peeks
and then adjusts has converted it into a validation set, and the reported
performance is no longer out-of-sample regardless of what the report says.
