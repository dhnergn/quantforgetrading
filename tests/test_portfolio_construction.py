"""Portfolio construction and risk attribution.

The invariants here are the point. A construction method that returns plausible
weights is not the same as one that achieves what it claims, and three of the
defects these tests cover produced entirely plausible-looking output:

* component risk contributions dividing by sigma instead of sigma^2, so they
  summed to 0.15 rather than 1.0 - a set that looks like contributions without
  being them;
* a risk-parity iteration judged converged on the size of the weight step
  rather than on the objective, so a point that was not balanced could be
  reported as balanced;
* a covariance routine that accepted a non positive-definite matrix and
  returned a balanced-looking portfolio that had no risk-parity solution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.core.errors import PortfolioConstructionError
from quantforge.portfolio.construction import (
    ConstructionMethod, CovarianceEstimator, PortfolioConstructor, PortfolioTarget,
    component_risk_contribution, effective_betas, equal_weight,
    inverse_volatility, marginal_risk_contribution, risk_parity,
)


@pytest.fixture(scope="module")
def panel():
    """Four assets with a shared factor, so correlation is genuinely present."""
    rng = np.random.default_rng(3)
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    f = rng.normal(0, 0.008, n)
    return pd.DataFrame({
        "A": 0.5 * f + rng.normal(0, 0.006, n),
        "B": 1.2 * f + rng.normal(0, 0.006, n),
        "C": 0.3 * f + rng.normal(0, 0.015, n),
        "D": 0.8 * f + rng.normal(0, 0.010, n),
    }, index=idx)


class TestEqualWeight:
    def test_sums_to_one(self, panel):
        w = equal_weight(list(panel.columns))
        assert float(w.sum()) == pytest.approx(1.0)
        assert (w > 0).all()

    def test_needs_no_history(self):
        """Equal weighting has nothing to estimate, so nothing to overfit."""
        w = equal_weight(["A", "B", "C", "D"])
        assert len(w) == 4

    def test_empty_raises(self):
        with pytest.raises(PortfolioConstructionError):
            equal_weight([])


class TestInverseVolatility:
    def test_lower_vol_gets_more(self, panel):
        w = inverse_volatility(panel)
        vol = panel.std()
        corr = w.corr(vol)
        assert corr < 0, "a higher-volatility asset received a larger weight"

    def test_sums_to_one(self, panel):
        assert float(inverse_volatility(panel).sum()) == pytest.approx(1.0)

    def test_zero_volatility_gets_no_weight(self):
        """A constant price carries no information; it must not be allocated to."""
        flat = pd.DataFrame({"A": [0.01, -0.01, 0.02] * 10,
                             "FLAT": [0.0] * 30})
        w = inverse_volatility(flat)
        assert w.get("FLAT", 0.0) == 0.0
        assert float(w.sum()) == pytest.approx(1.0)

    def test_all_zero_volatility_raises(self):
        with pytest.raises(PortfolioConstructionError, match="zero volatility"):
            inverse_volatility(pd.DataFrame({"A": [0.0] * 30, "B": [0.0] * 30}))


class TestRiskParity:
    def test_contributions_actually_equalise(self, panel):
        """The defining property. Weights alone would not reveal a failure."""
        w, diag = risk_parity(panel)
        cov, _ = CovarianceEstimator().estimate(panel)
        rc = component_risk_contribution(w, cov)
        assert float(rc.std()) < 1e-6, (
            f"risk parity did not equalise contributions: {dict(rc)}"
        )
        assert diag["converged"] is True

    def test_beats_equal_weight_on_risk_concentration(self, panel):
        """It must actually improve on the naive construction, or be pointless."""
        w, _ = risk_parity(panel)
        cov, _ = CovarianceEstimator().estimate(panel)
        rp_max = component_risk_contribution(w, cov).max()
        ew_max = component_risk_contribution(
            equal_weight(list(panel.columns)), cov).max()
        assert rp_max < ew_max
        # 1/N is the floor: with n assets, perfect parity is 1/n.
        assert rp_max == pytest.approx(1.0 / len(panel.columns), abs=1e-4)

    def test_damping_bounds_the_step_size(self, panel):
        """What damping actually does.

        An earlier version of this test asserted the undamped iteration
        diverges. It does not: on this panel and on a near-collinear one it
        converges in about three iterations. That assertion came from a
        throwaway experiment that clamped a negative contribution and kept
        iterating, and it was wrong. Damping is a step-size bound, not a fix for
        a demonstrated failure, and this test pins the property that is true.
        """
        import math

        cov, _ = CovarianceEstimator().estimate(panel)
        sigma = cov.to_numpy()
        k = sigma.shape[0]
        target = np.full(k, 1.0 / k)
        w = np.full(k, 1.0 / k)
        mrc = sigma @ w
        rc = w * mrc / float(w @ mrc)
        step = np.log(target / rc)
        undamped = float(np.max(np.abs(np.exp(step))))
        damped = float(np.max(np.abs(np.exp(0.5 * step))))
        assert damped <= undamped
        # A damping of 0.5 must at least halve the exponent, so the worst-case
        # multiplicative change is the square root of the undamped one.
        assert math.isclose(damped, undamped ** 0.5, rel_tol=1e-9)

    @pytest.mark.parametrize("damping", [0.2, 0.5, 0.9])
    def test_equalises_regardless_of_damping(self, panel, damping):
        """Convergence is judged on the objective, so any damping must work."""
        w, diag = risk_parity(panel, damping=damping)
        cov, _ = CovarianceEstimator().estimate(panel)
        rc = component_risk_contribution(w, cov)
        assert float(rc.std()) < 1e-6, f"damping {damping}: {dict(rc)}"
        assert diag["converged"] is True

    def test_reports_dispersion_alongside_weights(self, panel):
        _, diag = risk_parity(panel)
        assert "risk_contribution_dispersion" in diag
        assert "target_contributions" in diag


class TestRiskContribution:
    def test_contributions_sum_to_one(self, panel):
        """The invariant the sigma-vs-sigma-squared bug violated."""
        cov, _ = CovarianceEstimator().estimate(panel)
        for w in (equal_weight(list(panel.columns)),
                  inverse_volatility(panel),
                  pd.Series({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05})):
            rc = component_risk_contribution(w, cov)
            assert float(rc.sum()) == pytest.approx(1.0, abs=1e-12)
            assert (rc > 0).all()

    def test_divisor_is_variance_not_volatility(self, panel):
        """A 15%-vol portfolio must still yield contributions summing to 1."""
        cov, _ = CovarianceEstimator().estimate(panel)
        w = equal_weight(list(panel.columns))
        sigma_p = float(np.sqrt(w @ marginal_risk_contribution(w, cov)))
        assert sigma_p < 1.0, "test premise: portfolio vol must be below 1 for this to bite"
        rc = component_risk_contribution(w, cov)
        # Dividing by sigma instead of sigma^2 would give a sum of sigma_p.
        assert float(rc.sum()) == pytest.approx(1.0, abs=1e-12)
        assert float(rc.sum()) != pytest.approx(sigma_p, abs=1e-6)

    def test_marginal_and_component_are_consistent(self, panel):
        cov, _ = CovarianceEstimator().estimate(panel)
        w = inverse_volatility(panel)
        mrc = marginal_risk_contribution(w, cov)
        rc = component_risk_contribution(w, cov)
        assert np.allclose(rc.values, (w * mrc / float(w @ mrc)).values)

    def test_unknown_instrument_raises(self, panel):
        cov, _ = CovarianceEstimator().estimate(panel)
        with pytest.raises(PortfolioConstructionError, match="no covariance"):
            component_risk_contribution(pd.Series({"ZZZ": 1.0}), cov)


class TestCovariance:
    def test_reports_conditioning(self, panel):
        _, diag = CovarianceEstimator().estimate(panel)
        assert diag["condition_number"] > 0
        assert "ill_conditioned" in diag
        assert "min_eigenvalue" in diag

    def test_shrinkage_is_applied_by_default(self, panel):
        _, diag = CovarianceEstimator().estimate(panel)
        assert diag["estimator"].startswith("shrunk:")

    def test_singular_input_is_survivable(self):
        """Perfectly collinear columns have a singular sample covariance."""
        rng = np.random.default_rng(1)
        base = rng.normal(0, 0.01, 200)
        X = pd.DataFrame({"A": base, "B": base, "C": base, "D": base})
        cov, diag = CovarianceEstimator().estimate(X)
        assert np.all(np.isfinite(cov.to_numpy()))
        assert diag["ill_conditioned"] is True

    def test_non_positive_definite_blocks_risk_parity(self):
        """There is no risk-parity solution; say so rather than substitute one."""
        from quantforge.portfolio import construction as C

        class BadEstimator:
            def estimate(self, returns):
                idx = pd.Index(["A", "B"])
                # Indefinite: one negative eigenvalue.
                return pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=idx,
                                    columns=idx), {"estimator": "bad"}

        with pytest.raises(PortfolioConstructionError, match="positive definite"):
            C.risk_parity(pd.DataFrame({"A": [0.1, -0.1], "B": [0.2, -0.2]}),
                          estimator=BadEstimator())

    def test_single_observation_raises(self, panel):
        with pytest.raises(PortfolioConstructionError, match="cannot produce"):
            CovarianceEstimator().estimate(panel.head(1))

    def test_empty_frame_raises(self):
        with pytest.raises(PortfolioConstructionError):
            CovarianceEstimator().estimate(pd.DataFrame())


class TestPortfolioTarget:
    def test_rejects_nan_weight(self):
        with pytest.raises(PortfolioConstructionError, match="NaN"):
            PortfolioTarget(pd.Series({"A": np.nan, "B": 1.0}), "t")

    def test_rejects_negative_weight(self):
        with pytest.raises(PortfolioConstructionError, match="negative"):
            PortfolioTarget(pd.Series({"A": -0.5, "B": 1.5}), "t")

    def test_rejects_zero_sum(self):
        with pytest.raises(PortfolioConstructionError, match="sum to"):
            PortfolioTarget(pd.Series({"A": 0.0, "B": 0.0}), "t")

    def test_effective_n(self, panel):
        """1 for one holding, N for equal weight."""
        assert PortfolioTarget(pd.Series({"A": 1.0}), "t").effective_n == \
            pytest.approx(1.0)
        eq = PortfolioTarget(equal_weight(list(panel.columns)), "t")
        assert eq.effective_n == pytest.approx(4.0)
        assert eq.concentration == pytest.approx(0.25)

    def test_concentration_flags_a_lopsided_book(self, panel):
        eq = PortfolioTarget(equal_weight(list(panel.columns)), "t")
        lop = PortfolioTarget(pd.Series({"A": 0.8, "B": 0.07, "C": 0.07, "D": 0.06}), "t")
        assert lop.concentration > eq.concentration

    def test_summary_carries_the_caveat(self, panel):
        s = PortfolioTarget(risk_parity(panel)[0], "risk_parity").summary()
        assert "allocate risk, not expected return" in s
        assert "does not" in s


class TestConstraints:
    def test_cap_is_applied_and_renormalised(self, panel):
        c = PortfolioConstructor("equal_weight",
                                 constraints={"max_weight": 0.30}).construct(panel)
        assert c.weights.max() <= 0.30 + 1e-9
        assert float(c.weights.sum()) == pytest.approx(1.0)

    def test_infeasible_cap_raises(self, panel):
        c = PortfolioConstructor("equal_weight",
                                 constraints={"max_weight": 0.10})
        with pytest.raises(PortfolioConstructionError, match="infeasible"):
            c.construct(panel)

    def test_invalid_cap_raises(self, panel):
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(PortfolioConstructionError, match="max_weight"):
                PortfolioConstructor("equal_weight",
                                     constraints={"max_weight": bad}).construct(panel)


class TestConstructor:
    def test_unknown_method_raises(self, panel):
        with pytest.raises(PortfolioConstructionError, match="unknown method"):
            PortfolioConstructor("variance_minimisation").construct(panel)

    def test_equal_weight_requires_a_frame(self):
        with pytest.raises(PortfolioConstructionError, match="instrument list"):
            PortfolioConstructor("equal_weight").construct()

    def test_all_three_methods_run(self, panel):
        for m in (ConstructionMethod.EQUAL_WEIGHT,
                  ConstructionMethod.INVERSE_VOLATILITY,
                  ConstructionMethod.RISK_PARITY):
            t = PortfolioConstructor(m).construct(panel)
            assert float(t.weights.sum()) == pytest.approx(1.0)
            assert (t.weights > 0).all()
            assert t.method == m

    def test_as_dict_is_serialisable(self, panel):
        import json

        t = PortfolioConstructor("risk_parity").construct(panel)
        payload = json.dumps(t.as_dict())
        assert "weights" in payload and "diagnostics" in payload
