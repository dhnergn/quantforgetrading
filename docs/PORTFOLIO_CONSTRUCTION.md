# Portfolio Construction

How much of each thing. Several individually reasonable positions can be
correlated and behave like one concentrated bet, and that is where most of the
real risk in a multi-asset book lives.

```python
from quantforge.portfolio.construction import (
    PortfolioConstructor, CovarianceEstimator, component_risk_contribution,
)

target = PortfolioConstructor("risk_parity").construct(returns)
print(target.summary())
```

---

## The three methods

| Method | Needs | Assumes | Use when |
|---|---|---|---|
| `equal_weight` | nothing | nothing | first look; often permanently |
| `inverse_volatility` | marginal vol | assets are not correlated | you want stability over optimality |
| `risk_parity` | covariance | correlation is estimable | you need risk genuinely spread |

**Equal weight is the default for a reason.** It has nothing to estimate, so
there is no input that can be overfitted. With a short sample that is worth
more than optimality.

**Inverse volatility** weights by each asset's own volatility, so holdings
contribute comparable risk. It uses no correlation estimate, so it stays stable
with few observations. Its limitation is structural: two highly correlated
assets with the same volatility get the same weight, and the portfolio
concentrates anyway.

**Risk parity** equalises each position's *contribution to total portfolio
volatility*. It requires a covariance matrix, and that is where the danger is.

---

## The correlation estimate is the weak link

Not the optimiser. Risk parity iterates on the inverse covariance, and:

- sample covariance in high dimensions is near-singular;
- its inverse is wildly unstable, and **least stable in a crisis** — exactly
  when the balance matters most;
- a portfolio that looks optimally balanced on an ill-conditioned covariance is
  a portfolio whose balance is an artefact of noise.

So shrinkage is applied by default and the conditioning is reported:

```python
cov, diag = CovarianceEstimator().estimate(returns)
diag["condition_number"]    # 7.075
diag["ill_conditioned"]     # False, against a 1e8 threshold
diag["estimator"]           # "shrunk:constant_correlation"
```

When the covariance is **not positive definite**, risk parity raises. There is
no risk-parity solution for such a matrix, and substituting the equal-weight
answer would hand back a portfolio nobody evaluated:

```python
# PortfolioConstructionError: covariance is not positive definite
# (min eigenvalue -1.000e+00). Shrinkage did not resolve it; there is no
# risk-parity portfolio for this input.
```

---

## Risk contribution, and a bug worth knowing about

```python
rc = component_risk_contribution(weights, cov)
rc.sum()      # exactly 1.0
```

The divisor is the **portfolio variance** (`sigma^2`), not the portfolio
volatility (`sigma`). Dividing by `sigma` yields a set that sums to `sigma`:

```python
# a 15%-volatility portfolio, contributions summing to 0.154 instead of 1.0
rc.sum() == 0.154
```

That is a set that looks like contributions without being them, and it is the
kind of defect that survives review because nothing visibly breaks. The test
`test_divisor_is_variance_not_volatility` pins it.

A 30% weight carrying 60% of the risk is the case worth catching, and that
comparison is only meaningful if the contributions sum to 1.

---

## Convergence is judged on the objective

```python
w, diag = risk_parity(returns)
diag["risk_contribution_dispersion"]   # 1.1e-08
diag["converged"]                      # True
diag["target_contributions"]           # what it was aiming for
```

`risk_parity` stops when the **dispersion of the contributions** is small, not
when the weight step is small. An earlier version stopped on the step, and
returned a point flagged `converged` whose contributions spanned 0.03 to 0.91
— a risk-parity result that was not risk parity, and whose weights looked
entirely plausible. Only checking the contributions reveals that.

The step is damped (`damping=0.5`) to bound the step size. Damping is a
robustness measure, not a fix for an observed failure: the undamped update also
converges on these inputs, in about three iterations. An earlier version of this
document claimed the opposite; that claim came from a flawed experiment and was
wrong.

---

## Does it actually help?

Verified on a four-asset panel with a shared factor:

| Construction | max risk share | contribution dispersion |
|---|---|---|
| equal weight | 0.302 | 8.3e-02 |
| inverse volatility | 0.299 | 5.7e-02 |
| **risk parity** | **0.250** | **1.1e-08** |

Risk parity reaches 0.25, and `1/n` is the floor for four assets — so this is
the best any construction can do here. If your risk-parity result does *not*
beat equal weight on max risk share, the iteration has not converged; check
`risk_contribution_dispersion` first.

---

## Constraints

Applied **after** construction:

```python
PortfolioConstructor("risk_parity", constraints={"max_weight": 0.30}).construct(R)
```

A constrained risk-parity problem needs a different solver, so this caps and
renormalises rather than pretending to solve the constrained problem. An
infeasible cap raises:

```python
# every instrument is at the weight cap, so the weights cannot be renormalised
# to sum to one; a cap below 1/n is infeasible
```

Post-construction capping also redistributes to the *unconstrained* names only,
which is a defensible choice and not the same as a true constrained optimum. It
is reported rather than presented as one.

---

## What this does not do

- **It does not forecast returns.** No method here has an expected-return input.
  These allocate *risk*, and a risk-parity portfolio is not thereby expected to
  earn anything.
- **It does not short.** Negative weights raise. Shorting is a separate decision
  with its own risk limits.
- **It does not respect the risk engine's sector or asset-class limits.**
  Combining a target with `RiskEngine.check_order` is the caller's job, and
  skipping that step is how a diversified-looking book breaches a concentration
  limit.
- **It does not model transaction costs.** Rebalancing a risk-parity book is
  frequent, and the turnover can consume the risk premium it allocates.
- **It does not shrink toward the equal-weight portfolio.** That is a
  reasonable robustness measure and it is not implemented.

---

## What is verified vs. asserted

| Claim | How |
|---|---|
| Contributions sum to 1 | `test_contributions_sum_to_one` over three weight vectors |
| Divisor is variance, not volatility | `test_divisor_is_variance_not_volatility` |
| Risk parity actually equalises | `test_contributions_actually_equalise` |
| It beats equal weight | `test_beats_equal_weight_on_risk_concentration` |
| Convergence is on the objective | reported in `diagnostics`, asserted in tests |
| Singular covariance survives shrinkage | `test_singular_input_is_survivable` |
| Non positive-definite blocks risk parity | `test_non_positive_definite_blocks_risk_parity` |
| Weight cap applied and renormalised | `test_cap_is_applied_and_renormalised` |
| Infeasible cap raises | `test_infeasible_cap_raises` |
