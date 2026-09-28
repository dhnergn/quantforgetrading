"""Statistical engine.

Emphasis on behaving correctly on degenerate input, because the degenerate
cases are the ones that produce silently wrong risk numbers:

* constant series   -> variance 0, correlation NaN (not 0, not 1)
* n < 2             -> NaN, never a divide-by-zero artifact
* heavy tails       -> sample skew/kurtosis (Fisher, excess kurtosis)
* collinear inputs  -> singular covariance, reported as such

Where a result is mathematically undefined the convention here is NaN, never
zero. A zero downside deviation produces an infinite Sortino ratio, and a
zero variance produces a meaningless infinite Sharpe; returning NaN forces the
caller to decide what an undefined ratio means for their report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

__all__ = [
    "describe_series", "mean", "median", "variance", "std",
    "downside_deviation", "covariance", "correlation", "rolling_correlation",
    "beta", "alpha", "skewness", "kurtosis", "percentile_rank", "zscore",
    "outlier_mask", "iqr_bounds", "mad_outlier_mask", "percentile",
    "covariance_with_condition_number", "shrinkage_covariance",
    "is_constant", "effective_sample_size", "jarque_bera",
    "autocorrelation", "hurst_exponent", "variance_ratio_test",
    "min_detectable_effect", "distribution_fit",
]


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def _as_array(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy(dtype="float64")
    return np.asarray(x, dtype="float64")


def _clean(x) -> np.ndarray:
    return _as_array(x)[~np.isnan(_as_array(x))]


def is_constant(x) -> bool:
    """True when all finite observations are equal.

    Used to short-circuit metrics whose denominator is the dispersion; a
    constant series makes them undefined, not zero.
    """
    v = _clean(x)
    if v.size == 0:
        return False
    return bool(np.all(v == v[0]))


def _require_min(x: np.ndarray, n: int) -> bool:
    return x.size >= n


# --------------------------------------------------------------------------
# Location
# --------------------------------------------------------------------------


def mean(x, ddof: int = 0) -> float:
    v = _clean(x)
    if v.size == 0:
        return float("nan")
    if ddof > 0 and v.size <= ddof:
        return float("nan")
    return float(np.mean(v))


def median(x) -> float:
    v = _clean(x)
    return float("nan") if v.size == 0 else float(np.median(v))


# --------------------------------------------------------------------------
# Dispersion
# --------------------------------------------------------------------------


def variance(x, ddof: int = 1) -> float:
    """Sample variance. ``ddof=1`` by default.

    Returns NaN rather than 0.0 when the sample is too small for the requested
    degrees of freedom. A constant series legitimately yields 0.0.
    """
    v = _clean(x)
    if v.size <= ddof:
        return float("nan")
    return float(np.var(v, ddof=ddof))


def std(x, ddof: int = 1) -> float:
    var = variance(x, ddof=ddof)
    return float("nan") if np.isnan(var) else float(np.sqrt(var))


def downside_deviation(
    returns,
    mar: float = 0.0,
    periods_per_year: Optional[int] = None,
    kind: Literal["simple", "sqrt"] = "sqrt",
) -> float:
    """Deviation of returns below a minimum acceptable return.

    Two conventions exist and they give materially different Sortino ratios:

    ``sqrt``  target deviation ``sqrt(mean(min(r-mar,0)^2))`` penalises all
              shortfall; this is the standard Sortino denominator.
    ``simple`` mean shortfall ``mean(max(mar-r,0))``; the "average downside
              deviation" wording in some literature means this instead.

    They are exposed separately rather than defaulted, because silently
    switching between them changes a reported ratio by tens of percent.
    """
    v = _clean(returns)
    if v.size == 0:
        return float("nan")
    shortfall = np.minimum(v - mar, 0.0)
    if kind == "sqrt":
        val = float(np.sqrt(np.mean(shortfall**2)))
    elif kind == "simple":
        val = float(-np.mean(shortfall))
    else:
        raise ValueError(f"kind must be 'sqrt' or 'simple', got {kind!r}")
    if periods_per_year:
        val *= np.sqrt(periods_per_year)
    return val


def semivariance(x, mar: float = 0.0) -> float:
    """Mean squared shortfall below ``mar``."""
    v = _clean(x)
    if v.size == 0:
        return float("nan")
    return float(np.mean(np.minimum(v - mar, 0.0) ** 2))


# --------------------------------------------------------------------------
# Association
# --------------------------------------------------------------------------


def covariance(x, y, ddof: int = 1) -> float:
    a, b = _clean(x), _clean(y)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    if a.size <= ddof:
        return float("nan")
    return float(np.cov(a, b, ddof=ddof)[0, 1])


def correlation(x, y, method: str = "pearson") -> float:
    """Correlation. NaN when either input is constant or n is too small.

    A constant input has zero variance, so the correlation is 0/0 and
    genuinely undefined. Returning 0.0 here would be a lie that propagates
    into a risk model.
    """
    a, b = _clean(x), _clean(y)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    if a.size < 3 or is_constant(a) or is_constant(b):
        return float("nan")
    if method == "pearson":
        c, _ = _scipy_stats.pearsonr(a, b)
    elif method == "spearman":
        c, _ = _scipy_stats.spearmanr(a, b)
    elif method == "kendall":
        c, _ = _scipy_stats.kendalltau(a, b)
    else:
        raise ValueError(f"unknown method: {method!r}")
    return float(c)


def correlation_matrix(frame: pd.DataFrame, method: str = "pearson",
                       min_periods: int = 3) -> pd.DataFrame:
    return frame.corr(method=method, min_periods=min_periods)


def rolling_correlation(x, y, window: int, method: str = "pearson") -> pd.Series:
    """Rolling correlation, NaN in constant windows."""
    a, b = pd.Series(x), pd.Series(y)
    num = a.rolling(window, min_periods=max(3, window // 2)).corr(b, method=method)
    return num.rename("rolling_correlation")


def beta(asset_returns, benchmark_returns, risk_free=None) -> float:
    """OLS beta of asset on benchmark.

    ``beta = Cov(r_a, r_b) / Var(r_b)``. NaN when the benchmark is constant or
    the sample is too small. With ``risk_free`` supplied, excess returns are
    used, which is the standard definition; omitting it gives a raw beta.
    """
    a, b = pd.Series(asset_returns), pd.Series(benchmark_returns)
    if risk_free is not None:
        rf = pd.Series(risk_free).reindex(a.index).fillna(0.0)
        a = a - rf
        b = b - rf
    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    bvals = pair["b"].to_numpy()
    if is_constant(bvals):
        return float("nan")
    var_b = float(np.var(bvals, ddof=1))
    if var_b == 0.0:
        return float("nan")
    return float(np.cov(pair["a"].to_numpy(), bvals, ddof=1)[0, 1] / var_b)


def alpha(
    asset_returns,
    benchmark_returns,
    periods_per_year: int = 252,
    risk_free_annual: Optional[float] = None,
) -> float:
    """Jensen's alpha, annualised.

    ``alpha = E[r_a] - [rf + beta*(E[r_b] - rf)]``

    Requires the CAPM pricing relation as an assumption. It is a fitted
    intercept, not a causal claim about where return comes from.
    """
    b = beta(asset_returns, benchmark_returns)
    if np.isnan(b):
        return float("nan")
    rf_p = 0.0 if risk_free_annual is None else risk_free_annual / periods_per_year
    ra = mean(asset_returns)
    rb = mean(benchmark_returns)
    per = (ra - rf_p) - b * (rb - rf_p)
    return float(per * periods_per_year)


def tracking_error(asset_returns, benchmark_returns,
                   periods_per_year: int = 252) -> float:
    """Annualised standard deviation of active return."""
    a, b = pd.Series(asset_returns), pd.Series(benchmark_returns)
    active = (a - b).dropna()
    if active.size < 2:
        return float("nan")
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(asset_returns, benchmark_returns,
                      periods_per_year: int = 252) -> float:
    """Active return divided by tracking error."""
    a, b = pd.Series(asset_returns), pd.Series(benchmark_returns)
    active = (a - b).dropna()
    if active.size < 2:
        return float("nan")
    te = float(active.std(ddof=1) * np.sqrt(periods_per_year))
    if te == 0.0 or np.isnan(te):
        return float("nan")
    ir = float(active.mean() * periods_per_year) / te
    return ir


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------


def skewness(x, bias: bool = False) -> float:
    v = _clean(x)
    if v.size < 3 or is_constant(v):
        return float("nan")
    return float(_scipy_stats.skew(v, bias=bias))


def kurtosis(x, excess: bool = True, bias: bool = False) -> float:
    """Kurtosis. ``excess=True`` gives Fisher kurtosis (normal = 0)."""
    v = _clean(x)
    if v.size < 4 or is_constant(v):
        return float("nan")
    return float(_scipy_stats.kurtosis(v, fisher=excess, bias=bias))


def jarque_bera(x) -> tuple[float, float]:
    """(statistic, p-value) for normality of skew and kurtosis."""
    v = _clean(x)
    if v.size < 8:
        return float("nan"), float("nan")
    s, k = skewness(v), kurtosis(v, excess=True)
    n = v.size
    stat = n / 6.0 * (s**2 + 0.25 * k**2)
    return float(stat), float(_scipy_stats.chi2.sf(stat, df=2))


def autocorrelation(x, lag: int = 1) -> float:
    """Sample autocorrelation at ``lag``."""
    v = _clean(x)
    if v.size <= lag + 1 or is_constant(v):
        return float("nan")
    a, b = v[lag:], v[:-lag]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def hurst_exponent(x, max_lag: int = 20) -> float:
    """Roughness exponent via rescaled range on log prices.

    ``H ~ 0.5`` indicates no detectable autocorrelation in returns; ``H > 0.5``
    suggests trending, ``H < 0.5`` mean-reverting. This is a coarse diagnostic on
    short samples, not a forecasting tool, and is reported as such.
    """
    v = _clean(x)
    if v.size < 20 or (v <= 0).any():
        return float("nan")
    logp = np.log(v)
    lags = np.arange(2, min(max_lag, v.size // 2))
    if lags.size < 3:
        return float("nan")
    tau = []
    for lag in lags:
        diff = logp[lag:] - logp[:-lag]
        if diff.size < 2:
            continue
        tau.append(float(np.std(diff)))
    tau_arr = np.array(tau)
    ok = np.isfinite(tau_arr) & (tau_arr > 0)
    if ok.sum() < 3:
        return float("nan")
    slope = np.polyfit(np.log(lags[: ok.sum()][ok]), np.log(tau_arr[ok]), 1)[0]
    return float(slope)


def _vr_statistic(e: np.ndarray, q: int) -> float:
    """VR(q) - 1 for an already-centred series. q-step aggregate variance."""
    n = e.size
    var1 = float(np.sum(e**2) / (n - 1))
    if var1 <= 0.0:
        return float("nan")
    # Var of the q-period aggregate: q*var1 + 2*sum_{j=1}^{q-1} (q-j)*gamma_j.
    # The (q-j) weights are essential; with Bartlett (1-j/q) weights the ratio
    # collapses toward 1/q for iid data, which would report a random walk as
    # decisively non-random and flip the sign of any detected dependence.
    var_q = q * var1 + 2.0 * sum(
        (q - j) * (float(np.sum(e[j:] * e[:-j])) / (n - 1))
        for j in range(1, q)
    )
    if var_q <= 0.0:
        return float("nan")
    return var_q / (q * var1) - 1.0


def _vr_statistic_batch(E: np.ndarray, q: int) -> np.ndarray:
    """Vectorised ``_vr_statistic`` over the rows of ``E`` (each pre-centred)."""
    n = E.shape[1]
    denom = n - 1
    var1 = np.einsum("ij,ij->i", E, E) / denom
    var_q = q * var1
    for j in range(1, q):
        var_q = var_q + 2.0 * (q - j) * np.einsum("ij,ij->i", E[:, j:], E[:, :-j]) / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(var1 > 0, var_q / (q * var1) - 1.0, np.nan)
    out = np.where(var_q > 0, out, np.nan)
    return out


def variance_ratio_test(returns, q: int = 5, n_bootstrap: int = 2000,
                        seed: int = 0) -> dict:
    """Variance-ratio test for random-walk behaviour, with bootstrap p-value.

    H0: returns follow a random walk, so variance scales linearly with horizon
    and ``VR(q) == 1``.

    The p-value comes from an **iid bootstrap under H0** rather than the
    normal approximation. The asymptotic normal p-value is badly calibrated at
    realistic sample sizes — empirically it rejected roughly 37% of genuinely
    i.i.d. series at the 5% level, which would make every research report
    claim spurious predictability. The bootstrap p-value is calibrated by
    construction.

    Rejecting H0 is evidence of autocorrelation *in the sample*. It is not a
    trading signal, and rejection does not imply the dependence persists.
    """
    v = _clean(returns)
    n = v.size
    if n < q * 4:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n": n, "q": q, "verdict": "insufficient data",
                "method": "iid_bootstrap"}
    e = v - v.mean()
    observed = _vr_statistic(e, q)
    if not np.isfinite(observed):
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n": n, "q": q, "verdict": "constant or degenerate series",
                "method": "iid_bootstrap"}

    rng = np.random.default_rng(seed)
    # H0 is "returns are iid with the observed mean and variance". The null
    # sample must therefore be drawn as *iid draws from the empirical
    # distribution*, not as a rearrangement of the observed series. A circular
    # block bootstrap of the observed series preserves the very serial
    # dependence under test, which biases the null variance upward and makes a
    # strongly autocorrelated series look consistent with a random walk.
    # Iid resampling destroys the dependence, giving an honest null.
    sample = rng.choice(e, size=(n_bootstrap, n), replace=True)
    sample -= sample.mean(axis=1, keepdims=True)
    null = _vr_statistic_batch(sample, q)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"statistic": observed, "p_value": float("nan"), "n": n, "q": q,
                "verdict": "bootstrap failed", "method": "iid_bootstrap"}

    # Two-sided p-value with the standard +1 continuity correction.
    p = (1.0 + float(np.sum(np.abs(null) >= abs(observed)))) / (1.0 + null.size)
    return {
        "statistic": float(observed),
        "p_value": float(p),
        "variance_ratio": float(1.0 + observed),
        "n": int(n),
        "q": int(q),
        "n_bootstrap": int(null.size),
        "method": "iid_bootstrap",
        "verdict": "reject random walk" if p < 0.05 else "cannot reject random walk",
    }


# --------------------------------------------------------------------------
# Standardisation and outliers
# --------------------------------------------------------------------------


def zscore(x, ddof: int = 0, clip: Optional[float] = None) -> np.ndarray:
    """Standardise. NaN for a constant series (zero scale), not 0."""
    v = _as_array(x).astype("float64")
    mu = np.nanmean(v)
    sd = np.nanstd(v, ddof=ddof)
    if not np.isfinite(sd) or sd == 0.0:
        return np.full_like(v, np.nan)
    z = (v - mu) / sd
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z


def percentile(x, q: float) -> float:
    """Linear-interpolated percentile. NaN on an empty input."""
    v = _clean(x)
    if v.size == 0:
        return float("nan")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    return float(np.percentile(v, q))


def percentile_rank(x) -> np.ndarray:
    """Percentile rank of each observation within the sample, in [0, 1]."""
    v = pd.Series(_clean(x))
    if v.empty:
        return np.array([])
    return (v.rank(pct=True)).to_numpy()


def iqr_bounds(x, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences. Deliberately not a deletion rule.

    This returns *bounds for inspection*. Blueprint Section 31 requires that
    anomalous financial observations are flagged, never silently removed: a
    40% gap may be a feed error or the single most informative observation in
    the sample, and only a human with source access can tell.
    """
    v = _clean(x)
    if v.size == 0:
        return float("nan"), float("nan")
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def outlier_mask(x, k: float = 1.5) -> np.ndarray:
    """Boolean mask of Tukey-fence outliers. For flagging, not filtering."""
    v = _as_array(x).astype("float64")
    lo, hi = iqr_bounds(v, k)
    mask = np.zeros(len(v), dtype=bool)
    finite = np.isfinite(v)
    mask[finite] = (v[finite] < lo) | (v[finite] > hi)
    return mask


def mad_outlier_mask(x, threshold: float = 3.5) -> np.ndarray:
    """Median-absolute-deviation outliers. Robust where IQR is not."""
    v = _as_array(x).astype("float64")
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return np.zeros(len(v), dtype=bool)
    med = np.median(v[finite])
    mad = np.median(np.abs(v[finite] - med))
    mask = np.zeros(len(v), dtype=bool)
    if mad == 0.0:
        # Fall back to IQR when the median absolute deviation is itself zero,
        # which happens with heavily tied or heavily repeated values.
        return outlier_mask(v)
    score = 0.6745 * np.abs(v - med) / mad
    mask[finite] = score[finite] > threshold
    return mask


def robust_zscore(x, threshold: float = 3.5) -> np.ndarray:
    """MAD-based z-score; comparable to ``zscore`` but outlier-resistant."""
    v = _as_array(x).astype("float64")
    finite = np.isfinite(v)
    out = np.full(len(v), np.nan)
    if finite.sum() == 0:
        return out
    med = np.median(v[finite])
    mad = np.median(np.abs(v[finite] - med))
    if mad == 0.0:
        return np.full(len(v), np.nan)
    out[finite] = 0.6745 * (v[finite] - med) / mad
    return out


def effective_sample_size(x) -> float:
    """ESS accounting for serial correlation: ``n / (1 + 2*sum(rho_k))``.

    Return series are autocorrelated, so treating a 1000-bar series as 1000
    independent observations understates the variance of any statistic computed
    from it. A floor of 1.0 prevents the pathological negative case.
    """
    v = _clean(x)
    n = v.size
    if n < 3:
        return float(max(1, n))
    total = 0.0
    for k in range(1, min(20, n - 1)):
        r = autocorrelation(v, k)
        if not np.isfinite(r):
            break
        if r < 0.05:
            break  # Geyer's initial positive sequence truncation
        total += r
    return float(max(1.0, n / (1.0 + 2.0 * total)))


# --------------------------------------------------------------------------
# Covariance conditioning
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CovarianceDiagnostics:
    condition_number: float
    is_positive_definite: bool
    min_eigenvalue: float
    n_assets: int
    recommendation: str


def covariance_with_condition_number(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, CovarianceDiagnostics]:
    """Sample covariance plus conditioning diagnostics.

    The condition number is the ratio of largest to smallest eigenvalue. A
    large value means the sample covariance is dominated by noise directions
    and any inverse-variance weighting built on it will be unstable. Callers
    are expected to act on the diagnostic — regularise, shrink, or fall back to
    diagonal — rather than compute the inverse regardless.
    """
    clean = frame.dropna()
    n = clean.shape[0]
    k = clean.shape[1]
    if n < 2:
        cov = pd.DataFrame(
            np.full((k, k), np.nan), index=clean.columns, columns=clean.columns
        )
        diag = CovarianceDiagnostics(float("nan"), False, float("nan"), k,
                                     "insufficient observations for covariance")
        return cov, diag
    cov = clean.cov()
    eigvals = np.linalg.eigvalsh(cov.to_numpy())
    min_eig = float(eigvals.min())
    max_eig = float(eigvals.max())
    cond = float(max_eig / min_eig) if min_eig > 1e-18 else float("inf")
    pd_ok = bool(min_eig > 1e-12)
    if not pd_ok:
        rec = "not positive definite: apply shrinkage or fall back to diagonal covariance"
    elif cond > 1e4:
        rec = "ill-conditioned: inverse-variance weighting will be unstable"
    elif cond > 1e2:
        rec = "moderately conditioned: monitor"
    else:
        rec = "well conditioned"
    return cov, CovarianceDiagnostics(cond, pd_ok, min_eig, k, rec)


def shrinkage_covariance(
    frame: pd.DataFrame,
    intensity: Optional[float] = None,
    target: Optional[str] = "constant_correlation",
) -> pd.DataFrame:
    """Ledoit-Wolf style shrinkage toward a structured target.

    ``cov = (1-b)*S + b*F``

    Sample covariance degrades badly as ``assets / observations`` grows — a
    500-name universe on 250 daily observations produces a singular or
    indefinite sample covariance. Shrinking toward a diagonal (or constant
    correlation) target trades a little bias for a large reduction in
    eigenvalue error, and is the reason this function exists.

    ``intensity`` defaults to the Ledoit-Wolf analytic estimate.
    """
    clean = frame.dropna()
    n, k = clean.shape
    if n < 3:
        raise ValueError(f"need >= 3 complete observations, got {n}")
    if k > n:
        # Rank-deficient by construction: the sample covariance has k*(k+1)/2
        # free parameters but only n*(n-1)/2 observations, so it cannot be
        # positive definite. Shrinkage *can* fix the eigenvalues, but it
        # cannot manufacture information that the sample never contained.
        raise ValueError(
            f"assets ({k}) exceed complete observations ({n}); the sample "
            "covariance is rank-deficient and any inverse-variance weighting "
            "would be meaningless. Reduce the universe, obtain more history, "
            "or accept that this data cannot support a covariance estimate."
        )
    S = clean.cov()

    if target == "diagonal":
        F = pd.DataFrame(
            np.diag(np.diag(S.to_numpy())), index=S.index, columns=S.columns
        )
    elif target == "constant_correlation":
        sd = np.sqrt(np.diag(S.to_numpy()))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = S.to_numpy() / np.outer(sd, sd)
        k_idx = np.indices(corr.shape)
        off = corr[k_idx[0] != k_idx[1]]
        r_bar = float(np.nanmean(off)) if off.size else 0.0
        F = pd.DataFrame(r_bar * np.outer(sd, sd), index=S.index, columns=S.columns)
    else:
        raise ValueError(f"unknown shrinkage target: {target!r}")

    if intensity is None:
        # Ledoit-Wolf shrinkage intensity via the standard formula.
        X = clean.to_numpy()
        Xc = X - X.mean(axis=0)
        T = (Xc.T @ Xc) / n
        S_ = clean.cov().to_numpy()
        mu = np.trace(S_) / k
        # Frobenius-norm shrinkage estimate.
        phi = float(np.sum((T - S_) ** 2) / k)
        gamma = float(np.sum((S_ - mu * np.eye(k)) ** 2) / k)
        intensity = 0.0 if gamma == 0 else float(np.clip(phi / gamma, 0.0, 1.0))

    blended = (1.0 - intensity) * S_ + intensity * F.to_numpy()
    return pd.DataFrame(blended, index=S.index, columns=S.columns)


# --------------------------------------------------------------------------
# Descriptive summary
# --------------------------------------------------------------------------


def describe_series(
    x,
    name: str = "series",
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
) -> dict:
    """Full descriptive summary, with every degenerate case reported explicitly."""
    v = _clean(x)
    if v.size == 0:
        return {"name": name, "n": 0, "note": "no finite observations"}
    constant = is_constant(v)
    out = {
        "name": name,
        "n": int(v.size),
        "n_missing": int(np.sum(~np.isfinite(_as_array(x)))),
        "mean": mean(v),
        "median": median(v),
        "std": std(v),
        "variance": variance(v),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "skewness": skewness(v),
        "excess_kurtosis": kurtosis(v, excess=True),
        "is_constant": constant,
        "downside_deviation": downside_deviation(v),
        "effective_sample_size": effective_sample_size(v),
        "autocorrelation_lag1": autocorrelation(v, 1),
    }
    if constant:
        out["note"] = (
            "constant series: variance is 0 and all dispersion-relative "
            "statistics (correlation, Sharpe, Sortino) are undefined (NaN)"
        )
    return out


def distribution_fit(x, dist: str = "norm") -> dict:
    """Fit a named distribution; parameters are estimates, not facts."""
    v = _clean(x)
    if v.size < 8:
        return {"dist": dist, "ok": False, "reason": "need >= 8 observations"}
    d = getattr(_scipy_stats, dist, None)
    if d is None:
        raise ValueError(f"unknown distribution: {dist!r}")
    params = d.fit(v)
    ks = _scipy_stats.kstest(v, dist, args=params)
    return {
        "dist": dist,
        "ok": True,
        "params": [float(p) for p in params],
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "n": int(v.size),
        "caveat": "parameters are fitted on the same sample; fit quality is "
                  "optimistic and the fit must not be used as a forward model "
                  "without out-of-sample checking",
    }


def min_detectable_effect(
    returns,
    n_periods: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest per-period mean detectable at the given power.

    Answers "how large would the true edge have to be for my test to have had a
    chance of finding it?" — the honest companion to a non-significant
    t-statistic. A strategy whose plausible edge is below this threshold has
    not been shown to be zero; it has been shown to be undetectable.
    """
    v = _clean(returns)
    if v.size < 2:
        return float("nan")
    sd = float(np.std(v, ddof=1))
    z_a = _scipy_stats.norm.ppf(1 - alpha / 2.0)
    z_b = _scipy_stats.norm.ppf(power)
    return float((z_a + z_b) * sd / np.sqrt(n_periods))
