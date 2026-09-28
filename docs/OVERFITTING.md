# Overfitting

The most common way a quantitative result is wrong is that it was found by
looking. Every configuration examined, every threshold tried, and every feature
considered is a degree of freedom, and the best of many noisy estimates is
biased upward even when every strategy has zero true edge.

These are heuristics that produce **flags**, not a verdict. Nothing here can
authorise capital.

---

## Deflated Sharpe ratio

```python
d = deflated_sharpe_ratio(returns, n_trials=100)
d.observed_sharpe
d.expected_max_sharpe_under_null
d.deflated_sharpe              # observed - expected max
d.p_value
d.significant_at_95
```

`n_trials` must be the **total number of configurations evaluated**, not the
number that survived screening. Undercounting produces a falsely impressive
statistic.

### Worked behaviour

Pure noise, 1000 observations:

| `n_trials` | observed | E[max under null] | deflated | significant |
|---|---|---|---|---|
| 1 | −0.78 | 0.00 | −0.78 | no |
| 10 | −0.78 | +1.58 | −2.36 | no |
| 100 | −0.78 | +2.53 | −3.31 | no |
| 1000 | −0.78 | +3.26 | −4.04 | no |

A real edge (t = 0.8), same data:

| `n_trials` | observed | deflated | significant |
|---|---|---|---|
| 1 | +1.12 | +1.12 | yes |
| 100 | +1.12 | −1.41 | no |
| 10000 | +1.12 | −2.74 | no |

A genuine edge stops looking significant once you account for how many things
you tried. **That is the point.**

### The expected maximum

```
E[max] = σ_trials · [(1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e))]
```

with γ the Euler–Mascheroni constant. `σ_trials` is the standard deviation of
the Sharpe ratio **across trials**.

This is an easy quantity to get wrong. The *within-series* variance of the
Sharpe estimator is roughly `1/observations`; using it here gives an expected
maximum in the tens for a few hundred observations. That mistake was made
during development and is why `_expected_max_sharpe` is a separate, documented
function with a `trials_stdev` parameter rather than an inline expression.

`trials_stdev` defaults to `1.0` (annualised Sharpe units), which is
deliberately conservative: it assumes the trials differ from each other by a
full Sharpe point, so the correction is at least as large as it would be for a
tighter trial set.

For `n_trials == 1` there is no selection to adjust for and the expected
maximum is exactly zero. `Z⁻¹(0)` is `−inf`, so this case must be special-cased
before the formula is applied.

---

## White reality check

Complements the deflated Sharpe. That answers *how much of the Sharpe is
selection bias*; this answers *how surprising is this Sharpe*.

```python
white_reality_check(strategy_sharpe=2.5, trials=100, observations=1000)
# p_value: 0.988
# "A p-value at or above 0.05 means the observed Sharpe is within what
#  selecting the best of 100 configurations produces by chance alone."
```

The p-value is floored at `1/trials`, because a result that survives only
because the best of N was selected cannot claim more confidence than that.

---

## Diagnostic flags

```python
report = analyse(n_parameters=25, n_trades=18,
                 in_sample={"sharpe": 4.2},
                 out_of_sample={"sharpe": -0.3},
                 parameter_stability={"fast": {"fraction_near_optimal": 0.1}},
                 robustness={"costs_5x": {"degradation": 0.8},
                             "positive_window_fraction": 0.35})
```

| Flag | Severity | Fires when |
|---|---|---|
| `few_trades` | critical | fewer than ~30 trades |
| `few_observations` | warning | fewer than ~100 observations |
| `parameter_complexity` | critical | more than 0.5 parameters per trade |
| `extreme_in_sample_sharpe` | critical | in-sample Sharpe above 3.0 |
| `oos_degradation` | critical | OOS Sharpe more than 50% below IS |
| `negative_oos_sharpe` | critical | the strategy lost money out of sample |
| `suspicious_oos_improvement` | warning | OOS better than IS |
| `unstable_<param>` | warning | a narrow slice of parameter space performs |
| `cost_sensitivity` | warning | 5x costs degrade performance badly |
| `slippage_sensitivity` | warning | 2x slippage degrades performance badly |
| `inconsistent_windows` | warning | fewer than 60% of windows profitable |

Every flag records the observed value next to the threshold, so a reader can
judge the margin rather than accept a binary.

Thresholds live in `OverfittingThresholds` and are changeable. They are
heuristics: a low-frequency strategy will trip `few_trades` legitimately, and
a high-frequency one with 10 parameters and 2,000 trades is not overfitted
merely because it has 10 parameters.

---

## Reading the flags

The summary always ends with:

> These are heuristics, not a verdict. A strategy that trips nothing here can
> still be overfitted in ways these diagnostics cannot detect, and a strategy
> that trips several can still be sound if the thresholds are misapplied to
> its type. No automated score should be used to authorise capital deployment.

Two specific notes:

**`extreme_in_sample_sharpe` usually means a bug.** An in-sample Sharpe above
3 is far above what a real strategy achieves on unseen data. In order of
likelihood: look-ahead bias, a bug in the return calculation, or fitting the
sample. Check the first two before considering the third.

**`parameter_complexity` is a count of configurations, not parameters.** A
strategy with `fast` in {5,10,20,40,80} searched against `slow` in
{20,30,50,100,200} is 25 configurations, and selecting the best of 25 is a
25-way selection even though only two numbers vary.

---

## Two detection failures worth knowing

**In-sample metrics cannot detect leakage.** A PCA fitted on the full dataset
and one fitted on training data produce *identical* in-sample fit quality. The
leak only appears out of sample. This is why leakage prevention is structural
([DATA_LEAKAGE.md](DATA_LEAKAGE.md)) rather than a matter of checking metrics.

**A bootstrap cannot reveal what the sample never contained.** Resampling a
five-year history produces no crisis if the history contains none. Tail risk
estimated this way is understated by construction, which is why the limitation
is asserted in the test suite.

---

## What none of this can do

These diagnostics measure whether a result is *consistent with noise*. They
cannot tell you:

- whether a real edge, if present, will persist,
- whether the costs are modelled correctly,
- whether the data has vendor errors,
- whether the strategy will be crowded once discovered,
- whether a strategy's good period is a coincidence of regime.

A clean overfitting report means no listed indicator tripped. It does not mean
the strategy works.
