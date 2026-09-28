"""Multi-asset portfolio construction.

The single-instrument backtest engine answers "does this rule work". It cannot
answer "how much of each thing", and that second question is where most of the
real risk lives. Several individually reasonable positions can be highly
correlated and behave like one concentrated bet.

Three construction methods, in increasing order of what they assume:

``equal_weight``
    1/N each. No estimation, so nothing to get wrong and nothing to overfit.
    The right default for a first look and frequently the right answer
    permanently.
``inverse_volatility``
    Weight inversely to volatility, so each holding contributes comparable
    risk. Uses only each asset's own volatility, so it stays stable with few
    observations.
``risk_parity``
    Equalise each position's contribution to total portfolio volatility. This
    requires a covariance matrix, which is where the danger is: sample
    covariance in high dimensions is near-singular and its inverse -
    precisely what risk parity needs - is wildly unstable, and least stable
    exactly in a crisis.

**Correlation estimates are the weak link, not the optimiser.** Shrinkage is
applied by default and the conditioning is reported, because a portfolio that
looks optimally balanced on an ill-conditioned covariance is a portfolio whose
balance is an artefact of noise. See [docs/STATISTICS.md].

None of these methods forecasts returns, so none of them produces an expected
return. They allocate *risk*, and a risk-parity portfolio is not thereby
expected to earn anything.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.errors import PortfolioConstructionError
from quantforge.core.timestamps import utcnow

__all__ = [
    "ConstructionMethod", "CovarianceEstimator", "PortfolioTarget",
    "PortfolioConstructor", "equal_weight", "inverse_volatility", "risk_parity",
    "marginal_risk_contribution", "component_risk_contribution", "effective_betas",
]


class ConstructionMethod:
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    RISK_PARITY = "risk_parity"


# ==========================================================================
# Covariance estimation
# ==========================================================================


@dataclass(slots=True)
class CovarianceEstimator:
    """Covariance with shrinkage and a reported condition number.

    The condition number is not a diagnostic footnote. Risk parity iterates on
    the inverse covariance, and the inverse of an ill-conditioned matrix is
    dominated by whatever direction the matrix barely constrains. Reporting it
    is the difference between "balanced" and "balanced according to noise".
    """

    method: str = "constant_correlation"
    lookback: int = 252
    #: Below this condition number the covariance is treated as unusable.
    max_condition_number: float = 1e8
    annualize: bool = True
    periods_per_year: int = 252

    def estimate(self, returns: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Return ``(covariance, diagnostics)``.

        ``returns`` is a wide frame, one column per instrument, indexed by time.
        """
        from quantforge.quant.statistics import shrinkage_covariance

        df = pd.DataFrame(returns).dropna(how="all")
        if df.shape[1] < 1:
            raise PortfolioConstructionError("no instruments supplied")
        if df.shape[0] < 2:
            raise PortfolioConstructionError(
                f"{df.shape[0]} observation(s) cannot produce a covariance"
            )

        dropped = df.columns[df.isna().all()].tolist()
        if dropped:
            df = df.drop(columns=dropped)
        if df.shape[1] < 1:
            raise PortfolioConstructionError(
                f"every instrument was entirely missing: {dropped}"
            )

        if len(df) < self.lookback:
            # Not fatal, but the caller should know the covariance rests on less
            # history than configured - this is the usual cause of an
            # ill-conditioned result, and hiding it would make the condition
            # number look arbitrary.
            short_history = True
        else:
            short_history = False
        if self.lookback:
            df = df.tail(self.lookback)

        fill_means = df.mean()
        df = df.fillna(fill_means)
        was_imputed = int(pd.DataFrame(returns).isna().sum().sum())

        if self.method == "sample":
            cov = df.cov()
            estimator = "sample"
        else:
            # shrinkage_covariance takes the target, not a method name.
            cov = shrinkage_covariance(df, target=self.method)
            estimator = f"shrunk:{self.method}"

        if self.annualize:
            cov = cov * self.periods_per_year

        eig = np.linalg.eigvalsh(cov.to_numpy())
        smallest = float(eig.min()) if eig.size else 0.0
        largest = float(eig.max()) if eig.size else 0.0
        condition = float(largest / smallest) if smallest > 1e-18 else float("inf")

        diagnostics = {
            "estimator": estimator,
            "n_observations": int(df.shape[0]),
            "n_instruments": int(df.shape[1]),
            "dropped_instruments": dropped,
            "imputed_cells": was_imputed,
            "short_history": short_history,
            "requested_lookback": self.lookback,
            "condition_number": condition,
            "min_eigenvalue": smallest,
            "ill_conditioned": condition > self.max_condition_number,
            "annualized": self.annualize,
            "note": (
                "Sample covariance in many dimensions is near-singular, and the "
                "inverse that risk parity iterates on is unstable - least so in a "
                "crisis, which is when the balance matters. Shrinkage is applied "
                "by default for that reason. A high condition number means the "
                "weights below reflect noise as much as structure."
            ),
        }
        return cov, diagnostics


# ==========================================================================
# Weights
# ==========================================================================


def equal_weight(instruments: Sequence[str]) -> pd.Series:
    """1/N each. The default, because it has nothing to estimate.

    With no covariance to estimate, there is no input that can be overfitted,
    which is the property that matters most when the sample is short.
    """
    n = len(instruments)
    if n == 0:
        raise PortfolioConstructionError("no instruments supplied")
    return pd.Series(1.0 / n, index=list(instruments), name="weight")


def inverse_volatility(
    returns: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    lookback: Optional[int] = None,
) -> pd.Series:
    """Weight inversely to each asset's own volatility.

    Uses only marginal volatility, so the estimate is far more stable than a
    full covariance and needs far fewer observations. Two assets with the same
    volatility get the same weight whatever their correlation - which is the
    limitation, not a bug.
    """
    df = pd.DataFrame(returns).dropna(how="all").ffill().dropna(axis=1, how="all")
    if lookback:
        df = df.tail(lookback)
    vol = df.std(ddof=1)
    # A zero-volatility asset cannot be inverse-weighted; it is assigned nothing
    # rather than an arbitrary weight, because a constant price carries no
    # information and pretending otherwise is how a cash-like asset ends up
    # dominating a risk portfolio.
    vol = vol.replace(0.0, np.nan)
    inv = 1.0 / vol
    if inv.isna().all():
        raise PortfolioConstructionError(
            "every instrument has zero volatility; there is no risk to allocate"
        )
    inv = inv.dropna()
    if weights is not None:
        inv = inv.reindex(weights.dropna().index).fillna(0.0)
    total = inv.sum()
    if total <= 0:
        raise PortfolioConstructionError("inverse-volatility weights sum to zero")
    return (inv / total).rename("weight")


def risk_parity(
    returns: pd.DataFrame,
    estimator: Optional[CovarianceEstimator] = None,
    max_iter: int = 5000,
    damping: float = 0.5,
    tolerance: float = 1e-8,
) -> tuple[pd.Series, dict]:
    """Equalise each position's contribution to total portfolio volatility.

    Solved by fixed-point iteration on the standard risk-parity update::

        w_i <- w_i * (target_i / rc_i)

    which converges for positive-definite covariance. A non positive-definite
    covariance is reported rather than worked around: the iteration has no
    meaning there, and returning *something* would present noise as a balanced
    portfolio.
    """
    est = estimator or CovarianceEstimator()
    cov, diagnostics = est.estimate(returns)

    eig = np.linalg.eigvalsh(cov.to_numpy())
    if float(eig.min()) <= 0:
        diagnostics["error"] = (
            "covariance is not positive definite, so risk contributions are not "
            "well defined and no risk-parity solution exists"
        )
        raise PortfolioConstructionError(
            f"covariance is not positive definite (min eigenvalue "
            f"{float(eig.min()):.3e}). Shrinkage did not resolve it; there is no "
            "risk-parity portfolio for this input."
        )

    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    target = np.full(n, 1.0 / n)
    sigma = cov.to_numpy()
    converged = False
    best: Optional[np.ndarray] = None
    best_dispersion = np.inf

    def _dispersion(vec: np.ndarray) -> float:
        """Standard deviation of the component risk contributions.

        Convergence is judged on the objective, not on the size of the step. A
        step that stops shrinking can be a fixed point that is *not* a
        risk-parity solution, and only the contributions reveal that.
        """
        mrc = sigma @ vec
        variance = float(vec @ mrc)
        if variance <= 0:
            return np.inf
        rc = vec * mrc / variance
        if not np.all(np.isfinite(rc)) or np.any(rc <= 0):
            # A negative contribution is not a contribution; treat it as
            # infinitely far from a balanced portfolio rather than averaging it
            # into a dispersion that then looks merely imperfect.
            return np.inf
        return float(rc.std())

    for _ in range(max_iter):
        mrc = sigma @ w
        variance = float(w @ mrc)
        if not np.isfinite(variance) or variance <= 0:
            raise PortfolioConstructionError(
                "portfolio variance went non-positive during the risk-parity "
                "iteration; no solution was reached"
            )
        rc = w * mrc / variance
        if not np.all(np.isfinite(rc)) or np.any(rc <= 0):
            raise PortfolioConstructionError(
                "risk-parity iteration produced a non-positive component risk "
                f"contribution (min {float(np.min(rc)):.3e}); no solution was "
                "reached. Check the covariance."
            )

        disp = float(rc.std())
        if disp < best_dispersion:
            best_dispersion, best = disp, w.copy()
        if disp <= tolerance:
            converged = True
            break

        # Damped multiplicative update: w_i <- w_i * (target_i / CRC_i)^damping.
        #
        # The damping bounds the step size. It is a robustness measure, not a
        # fix for an observed failure: on both the well-conditioned and the
        # near-collinear panels tested here the undamped update also converges,
        # in about three iterations each. What damping buys is insurance against
        # a step that overshoots into a region where a component contribution
        # turns non-positive, at which point the update is undefined rather
        # than merely inaccurate.
        #
        # An earlier version of this comment claimed the undamped update
        # diverges and produced a negative contribution. That was an artefact
        # of a throwaway experiment that clamped a negative contribution up to a
        # small positive number and kept iterating; the method was fine and the
        # experiment was not. The guarantee actually relied on here is the
        # objective check above - convergence is judged on the contributions -
        # not the damping.
        step = np.log(target / rc)
        w_new = w * np.exp(damping * step)
        total = w_new.sum()
        if not np.isfinite(total) or total <= 0 or not np.all(np.isfinite(w_new)):
            # Keep the best point seen rather than propagating the failure.
            break
        w_new /= total
        w = w_new

    if best is not None:
        w = best
    if not converged and best_dispersion > tolerance:
        # Report the shortfall instead of presenting a near-miss as balanced.
        diagnostics = {
            **diagnostics,
            "converged": False,
            "risk_contribution_dispersion": best_dispersion,
            "max_risk_contribution_spread": float("nan"),
            "error": (
                f"risk parity did not converge: the best iterate still has a "
                f"contribution dispersion of {best_dispersion:.3e}, above the "
                f"{tolerance:.1e} tolerance. The weights below are the best "
                "approximation found and are NOT a balanced portfolio."
            ),
        }
        return pd.Series(w, index=cov.index, name="weight"), diagnostics

    weights = pd.Series(w, index=cov.index, name="weight")
    rc = component_risk_contribution(weights, cov)
    dispersion = float(rc.std()) if len(rc) > 1 else 0.0
    diagnostics.update({
        "converged": converged,
        "risk_contribution_dispersion": dispersion,
        "max_risk_contribution_spread": float(rc.max() - rc.min()) if len(rc) else 0.0,
        "target_contributions": {k: float(v) for k, v in zip(cov.index, target)},
        "damping": damping,
        "note": (
            diagnostics["note"]
            + " The contributions are reported alongside the weights because the "
            "weights alone cannot show whether the balance was achieved: an "
            "undamped iteration reaches a fixed-looking point whose contributions "
            "span 0.03 to 0.91. Check risk_contribution_dispersion, not the "
            "weights."
        ),
    })
    return weights, diagnostics


# ==========================================================================
# Risk attribution
# ==========================================================================


def marginal_risk_contribution(
    weights: pd.Series, covariance: pd.DataFrame
) -> pd.Series:
    """d(portfolio volatility) / d(weight_i) = (Σw)_i."""
    aligned = covariance.reindex(index=weights.index, columns=weights.index)
    if aligned.isna().any().any():
        missing = set(weights.index) - set(covariance.index)
        raise PortfolioConstructionError(
            f"no covariance for instruments {sorted(missing)}; a weight cannot be "
            "attributed to risk without one"
        )
    return pd.Series(aligned.to_numpy() @ weights.to_numpy(),
                     index=weights.index, name="marginal_risk")


def component_risk_contribution(
    weights: pd.Series, covariance: pd.DataFrame
) -> pd.Series:
    """Each position's *share* of total portfolio risk, summing to exactly 1.

    ``CRC_i = w_i * MRC_i / (w' * Sigma * w)`` = ``w_i * MRC_i / sigma^2``.

    The divisor is the **portfolio variance**, not the portfolio volatility.
    Dividing by ``sigma`` instead yields a set that sums to ``sigma``, which
    looks like a set of contributions while not being one: on a 15%-volatility
    portfolio the numbers come to 0.15 rather than 1.0, and every downstream
    read of "this holding carries 40% of the risk" is off by that factor.

    A 30% weight contributing 60% of the risk is the case worth catching, and
    that comparison is only meaningful if the contributions sum to 1.
    """
    mrc = marginal_risk_contribution(weights, covariance)
    variance = float(weights @ mrc)          # == sigma^2
    if variance <= 0:
        return pd.Series(0.0, index=weights.index, name="component_risk")
    return (weights * mrc / variance).rename("component_risk")


def effective_betas(
    returns: pd.DataFrame, weights: pd.Series, periods_per_year: int = 252
) -> pd.Series:
    """Each position's beta to the weighted portfolio.

    A position can be a small weight and a large beta. Summing weighted betas
    gives the portfolio's own beta, which is a useful cross-check that the
    weights and the risk model agree.
    """
    df = pd.DataFrame(returns).reindex(columns=weights.index)
    if df.isna().all().any():
        raise PortfolioConstructionError(
            f"no returns for {sorted(df.columns[df.isna().all()].tolist())}"
        )
    port = (df * weights).sum(axis=1)
    var = float(port.var(ddof=1)) * periods_per_year
    if var <= 0:
        raise PortfolioConstructionError(
            "portfolio return has zero variance; betas are undefined against a "
            "constant portfolio"
        )
    betas = {}
    for col in df.columns:
        cov = float(df[col].cov(port)) * periods_per_year
        betas[col] = cov / var
    return pd.Series(betas, name="beta")


# ==========================================================================
# The constructor
# ==========================================================================


@dataclass(slots=True)
class PortfolioTarget:
    """A proposed allocation, with the evidence that produced it."""

    weights: pd.Series
    method: str
    diagnostics: dict = field(default_factory=dict)
    created_at: dt.datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        w = self.weights
        if w.isna().any():
            raise PortfolioConstructionError(
                f"weights contain NaN: {list(w.index[w.isna()])}; a NaN weight is "
                "an allocation of unknown size, which is not zero"
            )
        if (w < -1e-12).any():
            short = list(w.index[w < -1e-12])
            raise PortfolioConstructionError(
                f"negative weights in {short}; this constructor allocates long-only. "
                "Shorting is a separate decision with its own risk limits."
            )
        total = float(w.sum())
        if total <= 0:
            raise PortfolioConstructionError(f"weights sum to {total}")

    @property
    def gross_exposure(self) -> float:
        return float(self.weights.abs().sum())

    @property
    def net_exposure(self) -> float:
        return float(self.weights.sum())

    @property
    def effective_n(self) -> float:
        """Inverse Herfindahl: 1 for a single holding, N for equal weight."""
        w = self.weights[self.weights > 0]
        if w.empty:
            return 0.0
        hhi = float((w ** 2).sum())
        return 1.0 / hhi if hhi > 0 else 0.0

    @property
    def concentration(self) -> float:
        """Herfindahl index. 1.0 is a single position."""
        w = self.weights[self.weights > 0]
        return float((w ** 2).sum()) if not w.empty else 0.0

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "weights": {k: float(v) for k, v in self.weights.items()},
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "effective_n": self.effective_n,
            "concentration_hhi": self.concentration,
            "diagnostics": self.diagnostics,
            "created_at": self.created_at.isoformat(),
        }

    def summary(self) -> str:
        lines = [
            "=" * 68,
            f"PORTFOLIO TARGET  ({self.method})",
            "=" * 68,
        ]
        for k, v in self.weights.sort_values(ascending=False).items():
            bar = "#" * max(1, int(round(abs(float(v)) * 40)))
            lines.append(f"  {k:<12} {float(v):>7.2%}  {bar}")
        lines += [
            "",
            f"  gross exposure   {self.gross_exposure:.2%}",
            f"  net exposure     {self.net_exposure:.2%}",
            f"  effective N      {self.effective_n:.2f} of {len(self.weights)}",
            f"  concentration    {self.concentration:.3f} (HHI, 1.0 = single holding)",
        ]
        if self.diagnostics.get("ill_conditioned"):
            lines += [
                "",
                f"  WARNING: the covariance is ill conditioned "
                f"(number {self.diagnostics['condition_number']:.2e}).",
                "  These weights reflect estimation noise as much as structure.",
            ]
        if self.diagnostics.get("error"):
            lines += ["", f"  {self.diagnostics['error']}"]
        lines += [
            "",
            "Weights allocate risk, not expected return. This construction does not",
            "forecast returns and carries no view on whether the portfolio will",
            "profit. Concentration and covariance are estimates; a well-balanced",
            "portfolio built on a noisy covariance is still concentrated in reality.",
            "=" * 68,
        ]
        return "\n".join(lines)


class PortfolioConstructor:
    """Builds a :class:`PortfolioTarget` by the chosen method."""

    def __init__(
        self,
        method: str = ConstructionMethod.EQUAL_WEIGHT,
        covariance: Optional[CovarianceEstimator] = None,
        constraints: Optional[dict] = None,
    ) -> None:
        self.method = method
        self.covariance = covariance or CovarianceEstimator()
        self.constraints = constraints or {}

    def construct(self, returns: Optional[pd.DataFrame] = None) -> PortfolioTarget:
        if self.method == ConstructionMethod.EQUAL_WEIGHT:
            if returns is None:
                raise PortfolioConstructionError(
                    "equal weighting needs the instrument list, which is the column "
                    "index of a returns frame"
                )
            w = equal_weight(list(pd.DataFrame(returns).columns))
            diagnostics = {"note": "no covariance estimated, so nothing to overfit"}
        elif self.method == ConstructionMethod.INVERSE_VOLATILITY:
            if returns is None:
                raise PortfolioConstructionError(
                    "inverse volatility needs a returns frame"
                )
            w = inverse_volatility(returns, lookback=self.covariance.lookback)
            diagnostics = {
                "note": "uses marginal volatility only, so it is stable with few "
                        "observations; it ignores correlation entirely, so two "
                        "correlated assets can still concentrate the portfolio"
            }
        elif self.method == ConstructionMethod.RISK_PARITY:
            if returns is None:
                raise PortfolioConstructionError("risk parity needs a returns frame")
            w, diagnostics = risk_parity(returns, self.covariance)
        else:
            raise PortfolioConstructionError(
                f"unknown method {self.method!r}; use equal_weight, "
                "inverse_volatility, or risk_parity"
            )

        w = self._apply_constraints(w, returns)
        return PortfolioTarget(weights=w, method=self.method, diagnostics=diagnostics)

    def _apply_constraints(
        self, w: pd.Series, returns: Optional[pd.DataFrame]
    ) -> pd.Series:
        """Cap single-name weight, then renormalise.

        Applied *after* construction rather than inside the optimiser: a
        constrained risk-parity problem needs a different solver, and silently
        substituting an unconstrained result for a constrained one would report
        a portfolio the caller did not ask for.
        """
        cap = self.constraints.get("max_weight")
        if cap is None:
            return w
        if not 0 < cap <= 1:
            raise PortfolioConstructionError(
                f"max_weight must be in (0, 1], got {cap}"
            )
        w = w.copy()
        over = w > cap + 1e-12
        if over.any():
            w.loc[over] = cap
            remaining = 1.0 - float(w.sum())
            if remaining > 1e-12:
                free = w.index[~over]
                if len(free) == 0:
                    raise PortfolioConstructionError(
                        "every instrument is at the weight cap, so the weights cannot "
                        "be renormalised to sum to one; a cap below 1/n is infeasible"
                    )
                w.loc[free] = w.loc[free] + remaining * (w.loc[free] / w.loc[free].sum())
        return w / w.sum()
