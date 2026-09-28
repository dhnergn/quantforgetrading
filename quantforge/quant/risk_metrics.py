"""Risk metrics: VaR, CVaR/Expected Shortfall, drawdown, risk ratios.

Every function here reports the four things blueprint Section 43 requires to be
identified: **confidence, horizon, method, and dataset**. A VaR number without
those is not a risk measurement, it is a number, and the four change it by
orders of magnitude.

Two conventions are enforced throughout:

1. **Losses are positive.** ``VaR`` is reported as a positive loss magnitude.
   Mixing a positive-loss VaR with a negative-return P&L series is the most
   common source of a sign-flipped risk report.

2. **Undefined is not zero.** Zero downside deviation gives an infinite
   Sortino; a constant series gives a zero-volatility VaR that means "no
   variation observed", not "no risk". Both return NaN and say why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from quantforge.core.enums import VaRMethod

# Below this many observations, distributional tail estimates are not
# credible. The threshold is exposed so a report can state why a number is
# flagged rather than silently trusting a 12-point tail.
MIN_TAIL_OBSERVATIONS = 30
MIN_ES_OBSERVATIONS = 20


# --------------------------------------------------------------------------
# Value at Risk
# --------------------------------------------------------------------------


@dataclass(slots=True)
class VaRResult:
    """A VaR measurement with the metadata needed to interpret it."""

    value: float                 # positive loss magnitude, in return units
    confidence: float
    horizon_periods: int
    method: VaRMethod
    dataset: str
    n_observations: int
    period_length: str = "period"
    caveat: str = ""

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "horizon_periods": self.horizon_periods,
            "method": self.method.value,
            "dataset": self.dataset,
            "n_observations": self.n_observations,
            "caveat": self.caveat,
        }

    def __float__(self) -> float:
        return self.value


def _losses(returns, periods: int = 1) -> np.ndarray:
    """Convert a return series to a positive loss vector."""
    r = pd.Series(returns).astype("float64").dropna()
    return (-r).to_numpy()


def historical_var(
    returns,
    confidence: float = 0.95,
    horizon_periods: int = 1,
    dataset: str = "unspecified",
    min_observations: int = MIN_TAIL_OBSERVATIONS,
    scale_by_horizon: bool = True,
) -> VaRResult:
    """Empirical percentile of historical losses. No distributional assumption.

    With ``horizon_periods > 1`` the default scales the single-period VaR by
    ``sqrt(horizon)``. That is an approximation resting on independent,
    identically-distributed, finite-variance returns. Real return series have
    fat tails and volatility clustering, both of which make sqrt-of-time
    understate multi-day risk. Set ``scale_by_horizon=False`` and supply
    already-aggregated multi-period returns to avoid the assumption.
    """
    _check_confidence(confidence)
    losses = _losses(returns)
    if losses.size < min_observations:
        return VaRResult(
            value=float("nan"), confidence=confidence, horizon_periods=horizon_periods,
            method=VaRMethod.HISTORICAL, dataset=dataset, n_observations=int(losses.size),
            caveat=f"only {losses.size} observations; {min_observations} required for a "
                   "credible tail estimate. Reported as undefined rather than estimated.",
        )
    q = 1.0 - confidence
    # The upper tail of the *loss* distribution is the q-quantile of losses,
    # which is the (1-q) quantile of returns. Taking the q-quantile of the loss
    # vector instead picks the wrong tail and yields a negative number that
    # silently reads as "VaR is a gain".
    base = float(np.quantile(losses, 1.0 - q))
    value = base * np.sqrt(horizon_periods) if scale_by_horizon else base
    caveat = (
        f"{int(round((1 - q) * losses.size))} of {losses.size} observations fall "
        "beyond this loss. The estimate is only as good as the sample's tail, "
        "which is by definition the part with fewest observations."
    )
    if scale_by_horizon and horizon_periods > 1:
        caveat += (
            " Multi-period value uses sqrt-of-time scaling, which assumes iid "
            "finite-variance returns and understates risk for fat-tailed, "
            "volatility-clustering series."
        )
    return VaRResult(value, confidence, horizon_periods, VaRMethod.HISTORICAL,
                     dataset, int(losses.size), caveat=caveat)


def parametric_var(
    returns,
    confidence: float = 0.95,
    horizon_periods: int = 1,
    dataset: str = "unspecified",
    distribution: Literal["normal", "student_t"] = "normal",
    min_observations: int = MIN_TAIL_OBSERVATIONS,
) -> VaRResult:
    """Variance-covariance VaR under a fitted normal or Student-t.

    The normal assumption is the weak point: real returns have excess kurtosis
    (a typical equity index shows 5-15 versus 0 for a normal), so a normal VaR
    at 99% is materially optimistic. ``student_t`` fits degrees of freedom to
    the data and is materially more conservative, and the fitted parameters are
    returned so the assumption is auditable.
    """
    _check_confidence(confidence)
    r = pd.Series(returns).astype("float64").dropna()
    n = r.size
    if n < min_observations:
        return VaRResult(
            value=float("nan"), confidence=confidence, horizon_periods=horizon_periods,
            method=VaRMethod.PARAMETRIC, dataset=dataset, n_observations=int(n),
            caveat=f"only {n} observations; {min_observations} required. Undefined, not estimated.",
        )
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma <= 0.0:
        return VaRResult(
            value=float("nan"), confidence=confidence, horizon_periods=horizon_periods,
            method=VaRMethod.PARAMETRIC, dataset=dataset, n_observations=int(n),
            caveat="constant return series: zero volatility, so parametric VaR is "
                   "undefined. Zero measured variance is not zero risk.",
        )
    tail = 1.0 - confidence
    scale = np.sqrt(horizon_periods)
    if distribution == "normal":
        # Loss = -(mu - sigma*z) = sigma*z - mu, scaled by sqrt(horizon).
        loss = (sigma * _scipy_stats.norm.ppf(confidence) - mu) * scale
        caveat = (f"Assumes normal returns with sigma={sigma:.6g}. Excess kurtosis "
                  f"{_excess_kurtosis(r):.2f} indicates whether that assumption holds; "
                  "a normal assumption understates the 99% tail for fat-tailed series.")
    elif distribution == "student_t":
        df, t_scale = _student_t_fit(r, sigma)
        # Var(nu_t * s^2) = nu/(nu-2) * s^2, so the scale that reproduces the
        # sample sigma is sigma*sqrt((nu-2)/nu). Using sigma/sqrt(nu) instead
        # gives a distribution with roughly half the intended variance, which
        # shrinks the quantiles and makes the fat-tailed fit appear *less*
        # conservative than a normal. That is backwards and can hide real risk.
        loss = (-(mu + t_scale * _scipy_stats.t.ppf(tail, df))) * scale
        caveat = (f"Student-t with fitted df={df:.1f} (scale {t_scale:.3g} chosen so the "
                  "fitted variance matches the sample). Fitted from the same sample used "
                  "for the estimate, so the fit is optimistic. With finite df the t has "
                  "heavier tails than a normal, so this should exceed the normal VaR at "
                  "high confidence; if it does not, the fit failed.")
    else:
        raise ValueError(f"distribution must be 'normal' or 'student_t', got {distribution!r}")
    return VaRResult(float(loss), confidence, horizon_periods,
                     VaRMethod.PARAMETRIC, dataset, int(n), caveat=caveat)


def monte_carlo_var(
    returns,
    confidence: float = 0.95,
    horizon_periods: int = 1,
    dataset: str = "unspecified",
    n_simulations: int = 20_000,
    seed: int = 0,
    distribution: Literal["normal", "student_t", "empirical"] = "normal",
) -> VaRResult:
    """VaR from simulated return paths.

    ``empirical`` resamples historical returns with replacement, which
    inherits real fat tails but no new extremes — it cannot produce a loss
    worse than the worst observed day. ``normal`` and ``student_t`` produce
    arbitrarily extreme values but only as far as the parametric assumption
    permits. Neither is a good model of a genuine crisis, which is the
    practical limitation of every parametric tail method.
    """
    _check_confidence(confidence)
    r = pd.Series(returns).astype("float64").dropna()
    n = r.size
    if n < 10:
        return VaRResult(
            value=float("nan"), confidence=confidence, horizon_periods=horizon_periods,
            method=VaRMethod.MONTE_CARLO, dataset=dataset, n_observations=int(n),
            caveat="fewer than 10 observations; undefined.",
        )
    rng = np.random.default_rng(seed)
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    if distribution == "empirical":
        paths = rng.choice(r.to_numpy(), size=(n_simulations, horizon_periods),
                           replace=True).sum(axis=1)
    elif distribution == "student_t":
        df = _fit_student_t_df(r)
        paths = (mu + sigma / np.sqrt(df)
                 * rng.standard_t(df, size=(n_simulations, horizon_periods))).sum(axis=1)
    else:
        if sigma <= 0:
            return VaRResult(
                value=float("nan"), confidence=confidence, horizon_periods=horizon_periods,
                method=VaRMethod.MONTE_CARLO, dataset=dataset, n_observations=int(n),
                caveat="constant series: zero volatility, undefined.",
            )
        paths = (rng.normal(mu, sigma, size=(n_simulations, horizon_periods))).sum(axis=1)
    # Simulated paths are returns, so the loss tail is the *lower* tail. The
    # confidence-level quantile must be taken directly (already correct for a
    # return distribution); negating it, or taking the wrong tail, yields a
    # negative "loss" that reads as a gain.
    value = float(-np.quantile(paths, 1.0 - confidence))
    return VaRResult(
        value, confidence, horizon_periods, VaRMethod.MONTE_CARLO, dataset, int(n),
        caveat=f"{n_simulations} simulated paths of {horizon_periods} period(s) under "
               f"{distribution}. Simulation reproduces the generating assumption and "
               "adds no information about tail behaviour beyond it.",
    )


def var(
    returns,
    confidence: float = 0.95,
    horizon_periods: int = 1,
    method: VaRMethod | str = VaRMethod.HISTORICAL,
    dataset: str = "unspecified",
    **kwargs,
) -> VaRResult:
    """Dispatch across VaR methods."""
    m = VaRMethod(method) if not isinstance(method, VaRMethod) else method
    if m == VaRMethod.HISTORICAL:
        return historical_var(returns, confidence, horizon_periods, dataset, **kwargs)
    if m == VaRMethod.PARAMETRIC:
        return parametric_var(returns, confidence, horizon_periods, dataset, **kwargs)
    if m == VaRMethod.MONTE_CARLO:
        return monte_carlo_var(returns, confidence, horizon_periods, dataset, **kwargs)
    raise ValueError(f"unknown VaR method: {m}")


# --------------------------------------------------------------------------
# Expected Shortfall / CVaR
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CVaRResult:
    """Expected shortfall, with explicit metadata and sample adequacy."""

    value: float                 # positive loss magnitude
    confidence: float
    horizon_periods: int
    method: str
    dataset: str
    n_observations: int
    n_tail_observations: int
    caveat: str = ""

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "horizon_periods": self.horizon_periods,
            "method": self.method,
            "dataset": self.dataset,
            "n_observations": self.n_observations,
            "n_tail_observations": self.n_tail_observations,
            "caveat": self.caveat,
        }

    def __float__(self) -> float:
        return self.value


def historical_expected_shortfall(
    returns,
    confidence: float = 0.95,
    horizon_periods: int = 1,
    dataset: str = "unspecified",
    min_observations: int = MIN_ES_OBSERVATIONS,
    scale_by_horizon: bool = True,
) -> CVaRResult:
    """Mean loss conditional on exceeding the historical VaR threshold.

    Uses the standard Rockafellar-Uryasev formulation, optimised at the VaR
    quantile, which is a lower bound that coincides with the tail mean exactly
    when the quantile is one of the observations (which it is, for the
    empirical method).

    Note the failure mode: if the tail holds a single observation, this returns
    that one value, which is not an estimate of anything. ``n_tail_observations``
    is reported so a reader can see the fragility.
    """
    _check_confidence(confidence)
    losses = _losses(returns)
    n = losses.size
    if n < min_observations:
        return CVaRResult(
            float("nan"), confidence, horizon_periods, "historical", dataset, int(n), 0,
            caveat=f"only {n} observations; {min_observations} required. Tail mean is "
                   "undefined on a sample this small, not approximately zero.",
        )
    q = 1.0 - confidence
    threshold = float(np.quantile(losses, 1.0 - q))
    tail = losses[losses >= threshold]
    if tail.size == 0:
        return CVaRResult(
            float("nan"), confidence, horizon_periods, "historical", dataset, int(n), 0,
            caveat="no observations beyond the threshold; undefined.",
        )
    base = float(tail.mean())
    value = base * np.sqrt(horizon_periods) if scale_by_horizon else base
    caveat = (
        f"Averaged over only {tail.size} of {n} observations. Expected shortfall is "
        "always >= VaR by construction; if the two are close, the tail is thin and "
        "the estimate rests on very few points."
    )
    if tail.size < 5:
        caveat += " Fewer than 5 tail points: treat as indicative only."
    return CVaRResult(value, confidence, horizon_periods, "historical", dataset,
                      int(n), int(tail.size), caveat)


def parametric_expected_shortfall(
    returns,
    confidence: float = 0.95,
    horizon_periods: int = 1,
    dataset: str = "unspecified",
    distribution: Literal["normal", "student_t"] = "normal",
) -> CVaRResult:
    """Expected shortfall under a fitted normal or Student-t.

    For a normal, ``ES = -(mu + sigma*phi(z_a)/alpha)`` where ``phi`` is the
    standard normal density. Unlike VaR this does grow as the tail thins, which
    is the behaviour that makes it preferable for risk limits.
    """
    _check_confidence(confidence)
    r = pd.Series(returns).astype("float64").dropna()
    n = r.size
    alpha = 1.0 - confidence
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    if n < MIN_ES_OBSERVATIONS or sigma <= 0.0:
        return CVaRResult(
            float("nan"), confidence, horizon_periods, f"parametric_{distribution}",
            dataset, int(n), 0,
            caveat="insufficient observations or zero volatility; undefined.",
        )
    scale = np.sqrt(horizon_periods)
    if distribution == "normal":
        z = _scipy_stats.norm.ppf(confidence)
        es = (sigma * _scipy_stats.norm.pdf(z) / alpha - mu) * scale
        caveat = ("Normal assumption. Excess kurtosis of "
                  f"{_excess_kurtosis(r):.2f} suggests the normal tail is too thin.")
    else:
        df, t_scale = _student_t_fit(r, sigma)
        z = _scipy_stats.t.ppf(confidence, df)
        pdf = _scipy_stats.t.pdf(z, df)
        # (df + z^2)/(df - 1) * pdf(z;df) / alpha is the t expected-shortfall
        # loading. Using t_scale (the variance-matching scale) rather than
        # sigma/sqrt(df) keeps ES consistent with the t VaR above.
        loading = (df + z**2) / (df - 1) * pdf / alpha
        es = (t_scale * loading - mu) * scale
        caveat = f"Student-t with fitted df={df:.1f}; parameters fitted on the same sample."
    return CVaRResult(float(es), confidence, horizon_periods,
                      f"parametric_{distribution}", dataset, int(n), int(max(1, alpha * n)),
                      caveat)


def monte_carlo_expected_shortfall(
    returns,
    confidence: float = 0.95,
    horizon_periods: int = 1,
    dataset: str = "unspecified",
    n_simulations: int = 20_000,
    seed: int = 0,
    distribution: Literal["normal", "student_t", "empirical"] = "normal",
) -> CVaRResult:
    """Mean simulated loss beyond the simulated VaR."""
    _check_confidence(confidence)
    r = pd.Series(returns).astype("float64").dropna()
    n = r.size
    if n < 10:
        return CVaRResult(float("nan"), confidence, horizon_periods, "monte_carlo",
                          dataset, int(n), 0, caveat="fewer than 10 observations; undefined.")
    rng = np.random.default_rng(seed)
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    if distribution == "empirical":
        paths = rng.choice(r.to_numpy(), size=(n_simulations, horizon_periods),
                           replace=True).sum(axis=1)
    elif distribution == "student_t":
        df, t_scale = _student_t_fit(r, sigma)
        paths = (mu + t_scale
                 * rng.standard_t(df, size=(n_simulations, horizon_periods))).sum(axis=1)
    else:
        if sigma <= 0:
            return CVaRResult(float("nan"), confidence, horizon_periods, "monte_carlo",
                              dataset, int(n), 0, caveat="zero volatility; undefined.")
        paths = rng.normal(mu, sigma, size=(n_simulations, horizon_periods)).sum(axis=1)
    # The loss tail is the *lower* tail of returns, so its threshold is the
    # (1 - confidence) quantile. Using the confidence quantile selects the
    # upper tail instead, so "expected shortfall" averages over nearly the
    # whole sample and collapses toward the mean return.
    threshold = np.quantile(paths, 1.0 - confidence)
    tail = paths[paths <= threshold]
    if tail.size == 0:
        return CVaRResult(float("nan"), confidence, horizon_periods, "monte_carlo",
                          dataset, int(n), 0, caveat="no simulated path beyond threshold.")
    return CVaRResult(
        float(-tail.mean()), confidence, horizon_periods, "monte_carlo", dataset,
        int(n), int(tail.size),
        caveat=f"{n_simulations} simulated paths under {distribution}; "
               f"{tail.size} fell beyond the simulated VaR.",
    )


# --------------------------------------------------------------------------
# Drawdown
# --------------------------------------------------------------------------


@dataclass(slots=True)
class DrawdownResult:
    max_drawdown: float          # positive fraction
    peak_index: object
    trough_index: object
    recovery_index: Optional[object]
    duration_periods: Optional[int]
    time_underwater: Optional[int]
    current_drawdown: float

    def as_dict(self) -> dict:
        return {
            "max_drawdown": self.max_drawdown,
            "peak_index": str(self.peak_index),
            "trough_index": str(self.trough_index),
            "recovery_index": str(self.recovery_index) if self.recovery_index is not None else None,
            "duration_periods": self.duration_periods,
            "time_underwater": self.time_underwater,
            "current_drawdown": self.current_drawdown,
        }


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak. Always <= 0."""
    e = equity.astype("float64")
    peak = e.cummax()
    return (e - peak) / peak.replace(0.0, np.nan)


def max_drawdown(equity: pd.Series) -> DrawdownResult:
    """Maximum drawdown with peak, trough, and recovery timing.

    Duration is measured peak-to-recovery. When the series ends before recovery
    (the common case for a live or recent backtest), the drawdown is still
    reported but recovery is ``None`` and the duration is marked as unrecovered
    rather than being quietly truncated to the sample end.
    """
    e = equity.astype("float64").dropna()
    if e.empty:
        return DrawdownResult(float("nan"), None, None, None, None, None, float("nan"))
    dd = drawdown_series(e)
    trough_i = dd.idxmin()
    mdd = float(dd.min())
    if np.isnan(mdd):
        return DrawdownResult(float("nan"), None, None, None, None, None, float("nan"))
    peak_i = e.loc[:trough_i].idxmax()
    recovery_i: Optional[object] = None
    after = e.loc[trough_i:]
    recovered = after[after >= e.loc[peak_i]]
    if not recovered.empty:
        recovery_i = recovered.index[0]
    pos = e.index.get_loc
    duration = int(pos(recovery_i) - pos(peak_i)) if recovery_i is not None else None
    underwater = int(pos(e.index[-1]) - pos(peak_i))
    return DrawdownResult(abs(mdd), peak_i, trough_i, recovery_i, duration,
                          underwater, abs(float(dd.iloc[-1])))


# --------------------------------------------------------------------------
# Risk-adjusted performance ratios
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RatioResult:
    value: float
    definition: str
    caveat: str = ""

    def __float__(self) -> float:
        return self.value


def sharpe_ratio(
    returns,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    downside: bool = False,
) -> RatioResult:
    """Annualised Sharpe ratio.

    Returns NaN when the denominator is zero or the sample is too small, with
    the reason attached. A ratio of 0.0 would be a claim of "no risk-adjusted
    return" when the truth is "not computable".
    """
    r = pd.Series(returns).astype("float64").dropna()
    n = r.size
    rf_p = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0 if risk_free else 0.0
    excess = r - rf_p
    if n < 2:
        return RatioResult(float("nan"), "sharpe", f"only {n} observations.")
    denom = (
        float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
        if downside else float(excess.std(ddof=1))
    )
    # A constant series does not give an exactly-zero standard deviation in
    # floating point: repeated summation of the same value leaves residue at
    # the 1e-18 level, and the resulting ratio overflows to ~1e16. Comparing
    # against zero exactly therefore misses the case entirely. The threshold
    # is relative to the series' own scale, so it triggers for genuinely
    # constant input without flagging legitimately low-volatility input.
    scale = max(float(np.abs(excess).mean()), abs(rf_p), 1e-12)
    if not np.isfinite(denom) or denom <= 1e-9 * scale:
        kind = "downside deviation" if downside else "standard deviation"
        return RatioResult(
            float("nan"), f"sharpe_{'downside' if downside else 'standard'}",
            f"{kind} is zero to numerical precision, so the ratio is undefined. "
            "Zero dispersion is not evidence of a riskless strategy.",
        )
    per = float(excess.mean()) / denom
    return RatioResult(per * np.sqrt(periods_per_year), "sharpe",
                       f"annualised from {n} periods at {periods_per_year}/year")


def sortino_ratio(
    returns,
    mar: float = 0.0,
    periods_per_year: int = 252,
    kind: Literal["sqrt", "simple"] = "sqrt",
) -> RatioResult:
    """Annualised Sortino ratio, penalising only downside deviation."""
    r = pd.Series(returns).astype("float64").dropna()
    n = r.size
    if n < 2:
        return RatioResult(float("nan"), "sortino", f"only {n} observations.")
    short = np.minimum(r - mar, 0.0)
    denom = (float(np.sqrt(np.mean(short**2))) if kind == "sqrt"
             else float(-np.mean(short)))
    scale = max(float(np.abs(r).mean()), abs(mar), 1e-12)
    if not np.isfinite(denom) or denom <= 1e-9 * scale:
        return RatioResult(
            float("nan"), "sortino",
            "no observation fell below the minimum acceptable return, so downside "
            "deviation is zero and the ratio is undefined. This is not a statement "
            "that the strategy is riskless; it is a statement that MAR was never "
            "breached in this sample.",
        )
    per = float(r.mean() - mar) / denom
    return RatioResult(per * np.sqrt(periods_per_year), "sortino",
                       f"downside convention: {kind}")


def calmar_ratio(equity_or_returns, periods_per_year: int = 252,
                 is_equity: bool = True, max_years: Optional[float] = None) -> RatioResult:
    """Annualised return divided by maximum drawdown."""
    if is_equity:
        equity = pd.Series(equity_or_returns).astype("float64").dropna()
        dd = max_drawdown(equity)
        mdd = dd.max_drawdown
        if mdd is None or np.isnan(mdd) or mdd <= 0.0:
            return RatioResult(float("nan"), "calmar",
                               "zero or undefined maximum drawdown; ratio undefined.")
        years = max_years or (len(equity) / periods_per_year)
        if years <= 0:
            return RatioResult(float("nan"), "calmar", "non-positive elapsed period.")
        cagr = (float(equity.iloc[-1]) / float(equity.iloc[0])) ** (1.0 / years) - 1.0
        return RatioResult(cagr / mdd, "calmar",
                           f"over {years:.2f} years, max drawdown {mdd:.2%}")
    r = pd.Series(equity_or_returns).astype("float64").dropna()
    wealth = (1.0 + r).cumprod()
    dd = max_drawdown(wealth)
    mdd = dd.max_drawdown
    if not mdd or np.isnan(mdd) or mdd <= 0.0:
        return RatioResult(float("nan"), "calmar", "zero drawdown; undefined.")
    years = max_years or (len(r) / periods_per_year)
    cagr = float(wealth.iloc[-1]) ** (1.0 / years) - 1.0
    return RatioResult(cagr / mdd, "calmar", f"max drawdown {mdd:.2%}")


def profit_factor(trade_pnls) -> RatioResult:
    """Gross profit divided by gross loss, from per-trade P&L."""
    pnl = pd.Series(trade_pnls).astype("float64").dropna()
    if pnl.empty:
        return RatioResult(float("nan"), "profit_factor", "no trades.")
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    scale = max(float(pnl.abs().mean()), 1e-12)
    if losses <= 1e-9 * scale:
        return RatioResult(
            float("nan"), "profit_factor",
            f"no losing trades in {len(pnl)} samples. The ratio has no finite value; "
            "this is a small-sample artefact, not evidence of a perfect strategy.",
        )
    return RatioResult(gains / losses, "profit_factor", f"over {len(pnl)} trades")


def win_rate(trade_pnls) -> RatioResult:
    """Fraction of trades with positive P&L, with the counts attached."""
    pnl = pd.Series(trade_pnls).astype("float64").dropna()
    if pnl.empty:
        return RatioResult(float("nan"), "win_rate", "no trades.")
    wins = int((pnl > 0).sum())
    return RatioResult(wins / len(pnl), "win_rate",
                       f"{wins} winning of {len(pnl)} trades; breakeven excluded from wins")


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def _excess_kurtosis(r: pd.Series) -> float:
    v = r.to_numpy()
    if v.size < 4 or np.all(v == v[0]):
        return float("nan")
    return float(_scipy_stats.kurtosis(v, fisher=True))


def _fit_student_t_df(r: pd.Series) -> float:
    """Method-of-moments degrees-of-freedom estimate for a Student-t.

    ``kurtosis_t = 6/(df-4) + 3`` for df > 4. Solved for df. If the sample has
    excess kurtosis <= 0 the series is lighter-tailed than normal and the fit
    is clamped to a large finite df rather than returning infinity, which would
    poison downstream arithmetic.
    """
    exc = _excess_kurtosis(r)
    if not np.isfinite(exc) or exc <= 1e-6:
        return 500.0
    df = 6.0 / exc + 4.0
    return float(np.clip(df, 4.5, 500.0))


def _student_t_fit(r: pd.Series, sigma: float) -> tuple[float, float]:
    """Return ``(df, scale)`` for a Student-t matching the sample's sigma.

    A standard t with ``df`` degrees of freedom has variance ``df/(df-2)``, so
    the scale factor that makes the fitted distribution reproduce the sample
    standard deviation is ``sigma * sqrt((df-2)/df)``.

    This second return value is the part that is easy to get wrong. Using
    ``sigma/sqrt(df)`` is a common shortcut, but it defines a distribution with
    variance ``sigma^2 * (df-2)/df^2`` — for df=5.6 that is about 28% of the
    intended variance, and the resulting quantiles are small enough that the
    "fat-tailed" fit reports a *smaller* 99% VaR than the normal one.
    """
    df = _fit_student_t_df(r)
    t_scale = sigma * float(np.sqrt((df - 2.0) / df))
    return df, t_scale
