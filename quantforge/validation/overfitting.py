"""Overfitting diagnostics.

Blueprint Section 87 is a list of things to *flag*, not a pass/fail gate. This
module produces warnings with severity and never declares a strategy robust —
that judgement belongs to a person who understands what the strategy is trying
to do, and a mechanical score cannot supply it.

The diagnostics fall into three families:

* **Sample adequacy** — too few trades or observations to distinguish an edge
  from noise.
* **Degradation** — a large gap between in-sample and out-of-sample
  performance is the classic signature of overfitting.
* **Fragility** — performance that depends on precise parameters or vanishes
  under realistic costs.

Every threshold is a default the caller can change, and every result states the
observed value next to the threshold so a reader can judge the margin.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.errors import InsufficientDataError
from quantforge.core.timestamps import utcnow


@dataclass(slots=True)
class Flag:
    """One diagnostic finding."""

    name: str
    severity: str            # info | warning | critical
    observed: Any
    threshold: Any
    message: str

    def as_dict(self) -> dict:
        return {"name": self.name, "severity": self.severity,
                "observed": self.observed, "threshold": self.threshold,
                "message": self.message}


@dataclass(slots=True)
class OverfittingReport:
    flags: list[Flag] = field(default_factory=list)
    n_parameters: int = 0
    n_trades: int = 0
    n_observations: int = 0
    in_sample: dict = field(default_factory=dict)
    out_of_sample: dict = field(default_factory=dict)
    parameter_stability: dict = field(default_factory=dict)
    robustness: dict = field(default_factory=dict)

    @property
    def critical(self) -> list[Flag]:
        return [f for f in self.flags if f.severity == "critical"]

    @property
    def warnings(self) -> list[Flag]:
        return [f for f in self.flags if f.severity == "warning"]

    def add(self, name: str, severity: str, observed: Any, threshold: Any,
            message: str) -> None:
        self.flags.append(Flag(name, severity, observed, threshold, message))

    def as_dict(self) -> dict:
        return {
            "n_parameters": self.n_parameters,
            "n_trades": self.n_trades,
            "n_observations": self.n_observations,
            "in_sample": self.in_sample,
            "out_of_sample": self.out_of_sample,
            "parameter_stability": self.parameter_stability,
            "robustness": self.robustness,
            "n_flags": len(self.flags),
            "n_critical": len(self.critical),
            "n_warnings": len(self.warnings),
            "flags": [f.as_dict() for f in self.flags],
        }

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "OVERFITTING DIAGNOSTICS",
            "=" * 72,
            f"parameters: {self.n_parameters}   trades: {self.n_trades}   "
            f"observations: {self.n_observations}",
            f"flags: {len(self.critical)} critical, {len(self.warnings)} warning",
            "",
        ]
        if self.flags:
            for f in self.flags:
                lines.append(f"  [{f.severity.upper():8s}] {f.name}: {f.message}")
        else:
            lines.append("  No flags raised. This means no listed indicator tripped,")
            lines.append("  not that the strategy is robust.")
        lines.append("")
        lines.append(
            "These are heuristics, not a verdict. A strategy that trips nothing here "
            "can still be overfitted in ways these diagnostics cannot detect, and a "
            "strategy that trips several can still be sound if the thresholds are "
            "misapplied to its type. No automated score should be used to authorise "
            "capital deployment."
        )
        lines.append("=" * 72)
        return "\n".join(lines)


# ==========================================================================
# Thresholds
# ==========================================================================


@dataclass(slots=True)
class OverfittingThresholds:
    min_trades: int = 30
    min_observations: int = 100
    max_parameters_per_trade: float = 0.5
    max_in_sample_sharpe: float = 3.0
    max_oos_sharpe_degradation: float = 0.5
    max_sensitivity_cost_multiplier: float = 0.5
    max_sensitivity_slippage_multiplier: float = 0.5
    min_stable_parameter_share: float = 0.4
    min_walk_forward_positive_fraction: float = 0.6

    def as_dict(self) -> dict:
        return {
            "min_trades": self.min_trades,
            "min_observations": self.min_observations,
            "max_parameters_per_trade": self.max_parameters_per_trade,
            "max_in_sample_sharpe": self.max_in_sample_sharpe,
            "max_oos_sharpe_degradation": self.max_oos_sharpe_degradation,
            "min_stable_parameter_share": self.min_stable_parameter_share,
        }


# ==========================================================================
# The checks
# ==========================================================================


def analyse(
    n_parameters: int,
    n_trades: int,
    in_sample: Optional[dict] = None,
    out_of_sample: Optional[dict] = None,
    parameter_stability: Optional[dict] = None,
    robustness: Optional[dict] = None,
    n_observations: int = 0,
    thresholds: Optional[OverfittingThresholds] = None,
) -> OverfittingReport:
    """Run every diagnostic and collect the flags."""
    t = thresholds or OverfittingThresholds()
    r = OverfittingReport(
        n_parameters=n_parameters, n_trades=n_trades, n_observations=n_observations,
        in_sample=in_sample or {}, out_of_sample=out_of_sample or {},
        parameter_stability=parameter_stability or {}, robustness=robustness or {},
    )

    # 1. Sample adequacy ---------------------------------------------------
    if n_trades < t.min_trades:
        r.add("few_trades", "critical", n_trades, t.min_trades,
              f"only {n_trades} trades. Below roughly {t.min_trades} the sample "
              "cannot distinguish a persistent edge from luck, and performance "
              "statistics have confidence intervals too wide to act on.")
    if n_observations and n_observations < t.min_observations:
        r.add("few_observations", "warning", n_observations, t.min_observations,
              f"{n_observations} observations is a small sample for estimating "
              "even a simple parameter.")

    # 2. Parameter-to-trade ratio -----------------------------------------
    if n_trades > 0:
        ratio = n_parameters / n_trades
        if ratio > t.max_parameters_per_trade:
            r.add("parameter_complexity", "critical", round(ratio, 3),
                  t.max_parameters_per_trade,
                  f"{n_parameters} parameters for {n_trades} trades "
                  f"({ratio:.2f} parameters per trade). A strategy with more "
                  "degrees of freedom than independent opportunities will fit "
                  "noise by construction.")

    # 3. Implausibly good in-sample performance ----------------------------
    is_sharpe = _get(in_sample, "sharpe")
    if is_sharpe is not None and np.isfinite(is_sharpe):
        if is_sharpe > t.max_in_sample_sharpe:
            r.add("extreme_in_sample_sharpe", "critical", is_sharpe,
                  t.max_in_sample_sharpe,
                  f"an in-sample Sharpe of {is_sharpe:.2f} is far above what a "
                  "real strategy achieves on unseen data. The most common causes "
                  "are look-ahead bias, a bug in the return calculation, or "
                  "fitting the sample rather than a process.")

    # 4. In-sample / out-of-sample degradation ------------------------------
    oos_sharpe = _get(out_of_sample, "sharpe")
    if (is_sharpe is not None and oos_sharpe is not None
            and np.isfinite(is_sharpe) and np.isfinite(oos_sharpe)):
        if is_sharpe > 0:
            degradation = (is_sharpe - oos_sharpe) / is_sharpe
            if degradation > t.max_oos_sharpe_degradation:
                r.add("oos_degradation", "critical", round(degradation, 3),
                      t.max_oos_sharpe_degradation,
                      f"out-of-sample Sharpe is {degradation:.0%} below "
                      f"in-sample ({is_sharpe:.2f} -> {oos_sharpe:.2f}). A large "
                      "gap is the characteristic signature of a strategy fitted "
                      "to noise.")
            elif oos_sharpe < 0:
                r.add("negative_oos_sharpe", "critical", oos_sharpe, 0.0,
                      f"out-of-sample Sharpe is {oos_sharpe:.2f}: the strategy lost "
                      "money on data it never saw.")
        if oos_sharpe > 0 and is_sharpe > 0 and oos_sharpe > is_sharpe:
            r.add("suspicious_oos_improvement", "warning", oos_sharpe, is_sharpe,
                  "out-of-sample performance exceeds in-sample. Occasional in "
                  "small samples, but if large it usually indicates the "
                  "out-of-sample window is not actually independent.")

    # 5. Parameter stability ----------------------------------------------
    for key, value in (parameter_stability or {}).items():
        if not isinstance(value, dict):
            continue
        share = value.get("fraction_near_optimal")
        if share is not None and np.isfinite(share) and share < t.min_stable_parameter_share:
            r.add(f"unstable_{key}", "warning", round(share, 3),
                  t.min_stable_parameter_share,
                  f"only {share:.0%} of tested {key} values performed near the "
                  "optimum. Performance is concentrated on a narrow slice of "
                  "parameter space, which is what fitting noise looks like.")

    # 6. Cost and slippage sensitivity -------------------------------------
    for name, key, thr in (
        ("cost_sensitivity", "costs_5x", t.max_sensitivity_cost_multiplier),
        ("slippage_sensitivity", "slippage_2x", t.max_sensitivity_slippage_multiplier),
    ):
        entry = (robustness or {}).get(key)
        if not isinstance(entry, dict):
            continue
        degradation = entry.get("degradation")
        if degradation is None or not np.isfinite(degradation):
            continue
        if degradation > thr:
            r.add(name, "warning", round(degradation, 3), thr,
                  f"performance degrades {degradation:.0%} under the '{key}' "
                  "assumption. If the edge does not survive a plausible cost "
                  "assumption, it is not an edge.")

    # 7. Walk-forward consistency ------------------------------------------
    pos = (robustness or {}).get("positive_window_fraction")
    if pos is not None and np.isfinite(pos) and pos < t.min_walk_forward_positive_fraction:
        r.add("inconsistent_windows", "warning", round(pos, 3),
              t.min_walk_forward_positive_fraction,
              f"only {pos:.0%} of walk-forward windows were profitable. A strategy "
              "relying on a few good periods is not a dependable process.")
    return r


def _get(d: dict, key: str) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return None
    try:
        f = float(v if not hasattr(v, "value") else v.value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ==========================================================================
# Statistical tests
# ==========================================================================


#: Euler-Mascheroni constant, used in the expected-maximum formula.
EULER_GAMMA = 0.5772156649015329


def _expected_max_sharpe(n_trials: int, trials_stdev: float) -> float:
    """Expected maximum of ``n_trials`` standard-normal trials.

    ``E[max] = sigma * [(1-gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - 1/(N*e))]``

    ``trials_stdev`` is the standard deviation of the Sharpe ratio **across
    trials**, not within a return series. Confusing the two is the error this
    helper exists to prevent: the within-series variance is roughly
    1/observations, and using it here produces an expected maximum in the tens
    for a few hundred observations, which is meaningless.
    """
    from scipy import stats as _st

    if n_trials <= 1:
        # With a single trial there is no selection to adjust for: the observed
        # value *is* the maximum. Z^-1(0) is -inf, so this case must be handled
        # before the formula is applied.
        return 0.0
    n = float(n_trials)
    term_a = _st.norm.ppf(1.0 - 1.0 / n)
    term_b = _st.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(trials_stdev * ((1.0 - EULER_GAMMA) * term_a
                                 + EULER_GAMMA * term_b))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    periods_per_year: int = 252,
    trials_stdev: Optional[float] = None,
) -> dict:
    """Sharpe adjusted for the number of strategy variants tried.

    Selecting the best of ``n_trials`` backtests inflates the reported Sharpe by
    construction: the maximum of many noisy estimates is biased upward even when
    every strategy has zero true edge. This reports the expected maximum under
    the null and subtracts it.

    ``n_trials`` should be the **total number of configurations evaluated**,
    not the number that survived screening. Understating it produces a falsely
    impressive statistic.

    ``trials_stdev`` is the standard deviation of Sharpe across those trials.
    When it is not supplied a default of 1.0 (annualised Sharpe units) is used,
    which is deliberately conservative: it assumes the trials differ from each
    other by a full Sharpe point, so the correction is at least as large as it
    would be for a tighter trial set. Pass the measured value when available.
    """
    from scipy import stats as _st

    r = pd.Series(returns).astype("float64").dropna()
    n = r.size
    if n < 20:
        raise InsufficientDataError(
            f"{n} observations is too few to compute a deflated Sharpe ratio; 20 "
            "is the floor"
        )
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return {"deflated_sharpe": float("nan"),
                "note": "zero volatility makes the ratio undefined"}
    sharpe = float(r.mean()) / sd * np.sqrt(periods_per_year)

    sigma = float(trials_stdev) if trials_stdev else 1.0
    expected_max = _expected_max_sharpe(n_trials, sigma)
    # Single-trial probability of exceeding the observed Sharpe, using the
    # distribution of the estimator itself (variance 1/(n-1) in Sharpe units).
    estimator_sd = 1.0 / np.sqrt(max(1, n - 1))
    p_value = float(1.0 - _st.norm.cdf((sharpe - expected_max) / estimator_sd))
    p_value = float(min(1.0, max(0.0, p_value)))
    return {
        "observed_sharpe": sharpe,
        "expected_max_sharpe_under_null": expected_max,
        "deflated_sharpe": float(sharpe - expected_max),
        "p_value": p_value,
        "n_trials": int(n_trials),
        "n_observations": int(n),
        "trials_stdev_used": sigma,
        "skew": float(_st.skew(r)),
        "excess_kurtosis": float(_st.kurtosis(r, fisher=True)),
        "significant_at_95": bool(p_value < 0.05),
        "note": (
            f"A raw Sharpe of {sharpe:.2f} across {n_trials} configurations is not "
            f"unusual by chance alone: the best of {n_trials} independent trials "
            f"reaches roughly {expected_max:.2f} under the null of no edge. A deflated "
            f"figure at or below zero means the observed result is consistent with "
            "having found noise rather than a process."
        ),
    }


def white_reality_check(
    strategy_sharpe: float,
    trials: int,
    observations: int,
    trials_stdev: float = 1.0,
) -> dict:
    """How surprising is this Sharpe, given how many configurations were tried?

    Complements :func:`deflated_sharpe_ratio`: that answers "how much of the
    Sharpe is selection bias", this answers "what is the probability that a
    single configuration reaches this Sharpe under the null".

    The p-value is floored at ``1/trials`` because a result that survives only
    because the best of ``trials`` was selected cannot claim more confidence
    than that.
    """
    from scipy import stats as _st

    if trials < 1 or observations < 20:
        raise ValueError("trials must be >= 1 and observations >= 20")
    expected_max = _expected_max_sharpe(trials, float(trials_stdev))
    estimator_sd = 1.0 / np.sqrt(max(1, observations - 1))
    p_single = float(1.0 - _st.norm.cdf(
        (strategy_sharpe - expected_max) / estimator_sd
    ))
    return {
        "strategy_sharpe": strategy_sharpe,
        "expected_max_under_null": expected_max,
        "p_value": float(min(1.0, max(1.0 / trials, p_single))),
        "n_trials": int(trials),
        "n_observations": int(observations),
        "note": (
            f"A p-value at or above 0.05 means the observed Sharpe is within what "
            f"selecting the best of {trials} configurations produces by chance alone."
        ),
    }


def parameter_count_effective(
    parameters: dict[str, Any],
) -> int:
    """Count the degrees of freedom actually used by a configuration.

    Counts grid *points*, not parameters: a strategy with fast in
    {5,10,20,40,80} searched against slow in {20,30,50,100,200} is 25
    configurations, and selecting the best of 25 is a 25-way selection even
    though only two numbers are being varied.
    """
    count = 1
    for v in parameters.values():
        if isinstance(v, (list, tuple, set, frozenset)):
            count *= max(1, len(v))
        elif isinstance(v, dict):
            count *= max(1, len(v))
        else:
            count *= 1
    return count


def minimum_detectable_sharpe(
    returns: pd.Series,
    n_periods: Optional[int] = None,
    periods_per_year: int = 252,
) -> float:
    """Smallest annualised Sharpe this sample could have detected.

    Reported alongside an insignificant result so that "no edge found" is
    distinguished from "an edge this size could not have been found".
    """
    from scipy import stats as _st

    r = pd.Series(returns).astype("float64").dropna()
    n = n_periods or r.size
    if n < 20:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return float("nan")
    # Two-sided 5% test, 80% power.
    z = _st.norm.ppf(0.975) + _st.norm.ppf(0.80)
    daily_mde = z * sd / np.sqrt(n)
    return float(daily_mde * np.sqrt(periods_per_year))
