# Statistics

The discipline this module enforces is narrow and unglamorous: **a statistic on
a degenerate input is `NaN` with a reason, never a confident-looking number.**

Every function here has a documented behaviour for zero variance, constant
series, too-few observations, and rank-deficient covariance. Those are not
edge cases to be handled quietly — they are the inputs that produce a table
full of zeros which reads as "ordinary results" while meaning something entirely
different.

---

## The governing rule

| Situation | Naive result | This module |
|---|---|---|
| Correlation of a constant series | `0.0` (or a crash) | `NaN` — undefined, no variation to correlate |
| Variance of a constant series | `0.0` | `0.0` — genuinely correct |
| Variance of 1 observation | `NaN` or crash | `NaN` — degrees of freedom is zero |
| Beta against a constant benchmark | `inf` or crash | `NaN` — the regression is unidentified |
| Downside deviation, two conventions | silently one of them | both available, materially different |
| Covariance with more features than observations | singular matrix | Ledoit–Wolf shrinkage, with the condition number reported |
| Outlier removal | rows deleted | masks returned; **nothing is ever deleted** |

The asymmetry in row two is deliberate. A constant series genuinely has zero
variance, and reporting that is correct. The distinction is between "this
statistic is zero" and "this statistic does not exist".

---

## Dispersion

```python
mean, median, variance, std, sem, iqr, mad
```

`variance` and `std` require at least two observations and return `NaN`
otherwise. `mad` (median absolute deviation) is the robust alternative to the
standard deviation and is the preferred scale estimator for financial data,
because it does not break down on the very outliers it is looking for.

```python
outlier_mask(x)        # Tukey fences
mad_outlier_mask(x)    # robust to heavy tails
iqr_bounds(x)          # bounds for inspection, not filtering
```

These return **masks**. An anomaly may be a feed error or the single most
informative observation in the sample, and only a human with source access can
tell them apart. See [DATA_QUALITY.md](DATA_QUALITY.md).

---

## Association

```python
correlation(a, b)      # NaN when either side is constant
covariance(a, b)
beta(returns, benchmark, risk_free=None)
tracking_error(returns, benchmark, periods_per_year)
information_ratio(returns, benchmark, periods_per_year)
```

Beta against a constant benchmark is `NaN`, not infinity: the regression
coefficient is unidentified when the regressor has no variation.

Returns are aligned on index before any pairwise statistic. Comparing over
mismatched date ranges produces meaningless level differences even when the
series are identical.

---

## Shape

```python
skewness, kurtosis, tail_ratio
```

Excess kurtosis — normal is 0. A positive value means fat tails, so Gaussian
risk models understate the probability of extreme outcomes. Every risk metric in
this platform carries the shape of the distribution it was computed on.

`tail_ratio` compares the right and left tails. Large asymmetry means a
position's downside differs from its upside in a way a symmetric model misses.

---

## Downside deviation: two conventions

```python
downside_deviation(returns, mar=0.0, kind="sqrt")
downside_deviation(returns, mar=0.0, kind="simple")
```

| Convention | Definition |
|---|---|
| `"sqrt"` | `sqrt(mean(min(r - mar, 0)^2))` — the standard Sortino basis |
| `"simple"` | `mean(max(mar - r, 0))` — the original Sortino definition |

They differ **materially**, not marginally, and practitioners use the same word
for both. Choosing one silently and reporting it as *the* downside deviation
makes two results incomparable.

The default is `"sqrt"`; the choice is always reported.

---

## Conditioning

```python
condition_number(cov)
shrinkage_covariance(returns, method="ledoit_wolf")
```

A sample covariance matrix in high dimensions is near-singular, and its inverse
— the thing a minimum-variance optimiser needs — is wildly unstable. Worse, it
becomes unstable *in a crisis*, exactly when portfolio risk matters most.

`shrinkage_covariance` applies Ledoit–Wolf shrinkage toward a scaled identity
matrix, which is well defined in every dimension.

It **refuses hopeless input** rather than producing a plausible matrix:

```python
shrinkage_covariance(X)   # 30 observations, 100 features
# ValueError: the sample covariance of 30 observations across 100 features is
# rank-deficient. Shrinkage cannot recover structure that is not present; add
# observations or reduce dimensionality first.
```

Shrinkage regularises a poorly-conditioned matrix. It cannot invent information
that was never in the data, and a function that quietly returned a shrunk matrix
for a hopeless input would be reporting a result with no meaning.

The condition number is reported alongside any covariance so a reader can see
how much to trust it.

---

## Outlier and serial-dependence diagnostics

```python
variance_ratio_test(returns, q=5, n_bootstrap=1000, seed=0)
```

Tests whether a return series is consistent with a random walk, by comparing
`q`-period realised variance to `q` times one-period variance. `VR > 1` indicates
positive autocorrelation, which invalidates the square-root-of-time scaling
used everywhere else in the platform.

**The p-value is calibrated by an iid bootstrap under the null, not by a normal
approximation.** This is not a stylistic choice:

```python
# 20 independent iid series, alpha = 0.05
rejects = sum(variance_ratio_test(rng.normal(0, 0.01, 1000), 5,
                                  n_bootstrap=1000)["p_value"] < 0.05
              for _ in range(20))
assert rejects <= 4      # <= 4/20
```

A normal-approximation p-value on this test produced 37% false positives — more
than seven times the nominal rate — and had the sign inverted, so it rejected
the *absence* of autocorrelation. A statistical test that misreports its own
error rate is worse than no test, because it is trusted.

The bootstrap version is verified calibrated in the test suite, and detected
dependence correctly on a generated AR(1) process.

---

## Numerical guards

The recurring bug class in this codebase, recorded here so it is not
rediscovered:

**Never test a dispersion for exact equality with zero.** Repeated summation of
a constant leaves residue around `1e-18`. A zero check misses it, the Sharpe
division proceeds, and the result is `1e16` rather than `NaN`. Every guard uses
a scale-relative epsilon:

```python
if sd <= 1e-9 * max(1.0, abs(mu)):
    return NaNResult("standard deviation is zero to numerical precision, ...")
```

Other guards of the same kind:

- VaR and ES quantile selection must take the *correct* tail. Taking the wrong
  side reports a loss as a gain.
- Expected shortfall must average the observations **beyond** the VaR threshold.
  Averaging the wrong side gives ES < VaR, violating a mathematical identity.
- A Monte-Carlo ES threshold must use the confidence quantile, not a fixed
  cutoff, or it averages over 95% of paths.
- Student-t degrees of freedom affect the scale as `σ·sqrt((ν−2)/ν)`, not
  `σ/sqrt(ν)`. The wrong form makes a fat-tailed fit look *less* conservative
  than a normal one, inverting the entire point of using it.
- A perfect-fit F-statistic needs a relative SSE floor, since an exact fit gives
  `SSE = 0` and an infinite t-statistic.
- Coordinate descent must add back `beta[j] * col[j]` before correlating with
  the partial residual, or the iteration oscillates and lasso never converges
  to the OLS solution.

---

## What this module does not do

- It does not correct data. It reports.
- It does not test stationarity, structural breaks, or cointegration. The
  regime engine and walk-forward dispersion cover that ground.
- It does not model serial correlation. The variance-ratio test *detects* it;
  making risk estimates robust to it is a modelling decision the caller takes.
- It does not make small samples large. Every function states the sample size
  it was computed from, and the metrics layer warns below 30 observations.
