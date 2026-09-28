"""Regression engine.

A recurring theme: statistical association is not causation. A significant
coefficient on a factor that happens to be popular in the period does not
establish that trading the factor produced the return. This module therefore
reports coefficients without causal language and attaches a machine-readable
``interpretation`` field to every result stating what the number is and is not.

The other recurring concern is numerical conditioning. Normal equations invert
``X'X``, which squares the condition number of ``X``. When features are on
wildly different scales (a 200-day price and a 0.02 yield spread) the direct
solve silently loses precision. QR via Householder reflections is used
throughout so the solve reflects the conditioning of ``X`` rather than ``X'X``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

__all__ = [
    "OLSResult", "fit_ols", "fit_ols_rolling", "fit_ridge", "fit_lasso",
    "fit_elastic_net", "factor_regression", "cross_sectional_regression",
    "linear_regression", "InterpretabilityNote",
]

#: The caveat attached to every regression result. Not optional.
NO_CAUSAL_CLAIM = (
    "These coefficients measure statistical association within the estimation "
    "sample. They do not establish a causal relationship, they are not "
    "guaranteed to persist out of sample, and a significant p-value is not "
    "evidence of a tradable edge once costs and slippage are applied."
)


@dataclass(frozen=True, slots=True)
class InterpretabilityNote:
    """Machine-readable statement of what a coefficient does and does not mean."""

    what_it_is: str
    what_it_is_not: str

    def as_dict(self) -> dict:
        return {"what_it_is": self.what_it_is, "what_it_is_not": self.what_it_is_not}


@dataclass(slots=True)
class OLSResult:
    params: pd.Series
    std_errors: pd.Series
    t_values: pd.Series
    p_values: pd.Series
    conf_int: pd.DataFrame
    r_squared: float
    adj_r_squared: float
    residuals: pd.Series
    fitted: pd.Series
    n_observations: int
    n_parameters: int
    condition_number: float
    residual_std_error: float
    f_statistic: Optional[float] = None
    f_p_value: Optional[float] = None
    interpretation: InterpretabilityNote = field(
        default_factory=lambda: InterpretabilityNote(
            "association conditional on the other included regressors",
            "causation or a guaranteed future relationship",
        )
    )
    warnings: list[str] = field(default_factory=list)

    def summary(self, confidence: float = 0.95) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.std_errors,
                "t": self.t_values,
                f"p({confidence:.2f})": self.p_values,
                f"ci_low_{confidence:.2f}": self.conf_int["low"],
                f"ci_high_{confidence:.2f}": self.conf_int["high"],
            }
        )

    def as_dict(self) -> dict:
        return {
            "params": {k: float(v) for k, v in self.params.items()},
            "std_errors": {k: float(v) for k, v in self.std_errors.items()},
            "p_values": {k: float(v) for k, v in self.p_values.items()},
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "n_observations": self.n_observations,
            "n_parameters": self.n_parameters,
            "condition_number": self.condition_number,
            "f_statistic": self.f_statistic,
            "f_p_value": self.f_p_value,
            "interpretation": self.interpretation.as_dict(),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# Ordinary least squares
# --------------------------------------------------------------------------


def _design(y: pd.Series | np.ndarray, X: Optional[pd.DataFrame | np.ndarray],
            add_intercept: bool = True):
    if isinstance(y, pd.Series):
        y_arr = y.astype("float64").to_numpy()
        index = y.index
    else:
        y_arr = np.asarray(y, dtype="float64")
        index = pd.RangeIndex(len(y_arr))
    finite = np.isfinite(y_arr)

    if X is None:
        X_df = pd.DataFrame({"const": np.ones(len(y_arr))}, index=index)
        names = ["const"]
    elif isinstance(X, pd.DataFrame):
        X_df = X.astype("float64").copy()
        names = list(X_df.columns)
    else:
        arr = np.asarray(X, dtype="float64")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        X_df = pd.DataFrame(arr, index=index,
                            columns=[f"x{i}" for i in range(arr.shape[1])])
        names = list(X_df.columns)

    finite &= np.isfinite(X_df.to_numpy()).all(axis=1)
    y_arr = y_arr[finite]
    X_df = X_df.loc[finite]
    index = index[finite]
    if add_intercept:
        X_df.insert(0, "const", 1.0)
        names = ["const"] + names
    return y_arr, X_df, names, index


def fit_ols(
    y,
    X,
    add_intercept: bool = True,
    confidence: float = 0.95,
    robust: bool = False,
) -> OLSResult:
    """OLS via Householder QR.

    ``robust=True`` uses heteroskedasticity-consistent (HC1) standard errors,
    which is the right default for financial returns: their variance is
    plainly not constant, and homoskedastic errors understate uncertainty in
    exactly the high-volatility periods that matter most.
    """
    y_arr, X_df, names, index = _design(y, X, add_intercept)
    n, k = X_df.shape
    if n <= k:
        raise ValueError(
            f"{n} observations for {k} parameters: the system is under-determined "
            "and the fit is unidentified. Add data or remove regressors."
        )
    if n - k == 0:
        raise ValueError("zero residual degrees of freedom")

    # Householder QR.
    A = X_df.to_numpy(dtype="float64")
    Q, R = np.linalg.qr(A)
    cond = float(np.linalg.cond(R)) if R.size else float("nan")

    warnings: list[str] = []
    if not np.all(np.isfinite(R)) or np.abs(np.diag(R)).min() < 1e-12:
        warnings.append(
            "design matrix is rank-deficient or near-singular; coefficients are "
            "not identified and the values shown are an artefact of the solver"
        )

    try:
        beta = np.linalg.solve(R, Q.T @ y_arr)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, y_arr, rcond=None)[0]
        warnings.append("QR solve failed; used least-norm solution")

    fitted = A @ beta
    resid = y_arr - fitted
    dof = n - k
    sse = float(resid @ resid)
    sst = float(((y_arr - y_arr.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / dof if np.isfinite(r2) and dof > 0 else float("nan")

    if robust:
        meat = (A * (resid**2)[:, None]).T @ A
        V = np.linalg.pinv(R) @ np.linalg.pinv(R).T
        cov = V @ meat @ V * (n / dof)
        warnings.append("heteroskedasticity-consistent (HC1) standard errors")
    else:
        sigma2 = sse / dof
        cov = sigma2 * (np.linalg.pinv(R) @ np.linalg.pinv(R).T)
        if k > 0 and A.shape[1] > 1:
            warnings.append("homoskedastic standard errors; returns are usually "
                            "heteroskedastic, so consider robust=True")

    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = np.where(se > 0, beta / se, np.nan)
    p_vals = 2.0 * (1.0 - _scipy_stats.t.cdf(np.abs(t_vals), df=dof))

    tcrit = _scipy_stats.t.ppf(0.5 + confidence / 2.0, df=dof)
    lo = beta - tcrit * se
    hi = beta + tcrit * se

    f_stat = f_p = None
    # A numerically perfect fit leaves an SSE of order 1e-27 (not exactly 0),
    # so the F statistic computed from it overflows to ~1e31. Compare the SSE
    # against a *relative* floor on the SST before dividing.
    sse_is_degenerate = sst > 0 and sse <= sst * 1e-15
    if k > 1 and sst > 0 and sse > 0 and not sse_is_degenerate:
        f_stat = float(((sst - sse) / (k - 1)) / (sse / dof))
        if np.isfinite(f_stat) and f_stat >= 0:
            f_p = float(1.0 - _scipy_stats.f.cdf(f_stat, k - 1, dof))
    elif k > 1 and sse_is_degenerate:
        warnings.append(
            f"residual sum of squares is {sse:.3g} against a total sum of squares "
            f"of {sst:.3g}: the fit is numerically perfect, so no F statistic is "
            "defined. This indicates overfitting, duplicated regressors, or a "
            "data-generation artefact rather than a good model."
        )
    if cond > 1e10:
        warnings.append(
            f"condition number {cond:.3g} exceeds 1e10: the regression is "
            "numerically unstable, likely due to unscaled or collinear regressors"
        )

    idx = pd.Index(names, name="variable")
    return OLSResult(
        params=pd.Series(beta, index=idx),
        std_errors=pd.Series(se, index=idx),
        t_values=pd.Series(t_vals, index=idx),
        p_values=pd.Series(p_vals, index=idx),
        conf_int=pd.DataFrame({"low": lo, "high": hi}, index=idx),
        r_squared=float(r2),
        adj_r_squared=float(adj),
        residuals=pd.Series(resid, index=index, name="residual"),
        fitted=pd.Series(fitted, index=index, name="fitted"),
        n_observations=n,
        n_parameters=k,
        condition_number=cond,
        residual_std_error=float(np.sqrt(sse / dof)),
        f_statistic=f_stat,
        f_p_value=f_p,
        warnings=warnings,
    )


def linear_regression(y, X, add_intercept: bool = True) -> OLSResult:
    """Alias for :func:`fit_ols` with homoskedastic errors."""
    return fit_ols(y, X, add_intercept=add_intercept, robust=False)


def fit_ols_rolling(
    y,
    X,
    window: int,
    min_periods: Optional[int] = None,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """Re-fit OLS over a rolling window, tracking coefficient stability.

    A coefficient that swings wildly across windows is a sign the relationship
    is not stable, which is a warning rather than a trading opportunity. The
    returned frame is intended to be inspected for that instability.
    """
    if isinstance(y, pd.Series):
        y_series = y
    else:
        y_series = pd.Series(np.asarray(y, dtype="float64"))
    mp = min_periods or max(10, window // 2)
    rows = []
    names: Optional[list[str]] = None
    for end in range(window, len(y_series) + 1):
        sl = slice(end - window, end)
        yw = y_series.iloc[sl]
        Xw = X.iloc[sl] if isinstance(X, pd.DataFrame) else (
            pd.DataFrame(np.asarray(X)[sl[0]:end]) if X is not None else None
        )
        if yw.isna().sum() > (1 - mp / window) * window:
            continue
        try:
            res = fit_ols(yw, Xw, add_intercept=add_intercept)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if names is None:
            names = list(res.params.index)
        rows.append({**res.params.to_dict(),
                     f"r2": res.r_squared,
                     "cond": res.condition_number})
    if not rows:
        return pd.DataFrame(columns=(names or []) + ["r2", "cond"])
    return pd.DataFrame(rows, index=y_series.index[window - 1: window - 1 + len(rows)])


# --------------------------------------------------------------------------
# Regularised regression
# --------------------------------------------------------------------------


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd, mu, sd


def fit_ridge(y, X, alpha: float = 1.0, add_intercept: bool = True) -> dict:
    """Ridge (L2) regression.

    The intercept is not penalised; penalising it would shrink the fit toward
    a zero mean when the target genuinely has a large mean, which is a
    modelling error rather than regularisation.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    y_arr, X_df, names, index = _design(y, X, add_intercept)
    A = X_df.to_numpy(dtype="float64")
    n, k = A.shape
    Xs, mu, sd = _standardise(A)
    # Unpenalise the intercept column.
    Xs[:, 0] = 1.0
    penalty = np.eye(k) * alpha
    penalty[0, 0] = 0.0
    beta_s = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ y_arr)
    beta = beta_s.copy()
    beta[1:] = beta_s[1:] / sd[1:]
    beta[0] = beta_s[0] - float((beta_s[1:] * mu[1:] / sd[1:]).sum())
    fitted = A @ beta
    resid = y_arr - fitted
    sst = float(((y_arr - y_arr.mean()) ** 2).sum())
    sse = float(resid @ resid)
    return {
        "params": pd.Series(beta, index=pd.Index(names, name="variable")),
        "r_squared": 1.0 - sse / sst if sst > 0 else float("nan"),
        "alpha": alpha,
        "n_observations": n,
        "interpretation": InterpretabilityNote(
            "penalised association; coefficients are shrunk toward zero and are "
            "not unbiased, so magnitudes are systematically understated",
            "an unbiased effect estimate or a significance test",
        ),
        "note": "ridge is useful for prediction under collinearity; for "
                "inference, penalisation invalidates the usual standard errors",
    }


def _cd_penalized(
    Xc: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Cyclic coordinate descent for elastic net on a centred design.

    Minimises ``(1/2n)||y - Xb||^2 + alpha*(l1_ratio*|b|_1 + (1-l1_ratio)/2*|b|^2)``.

    One implementation serves both lasso (``l1_ratio=1``) and elastic net,
    which keeps the two entry points from drifting apart in their scaling
    conventions — a class of bug that is easy to introduce and hard to spot
    when the coefficients merely look slightly different.
    """
    n, k = Xc.shape
    beta = np.zeros(k)
    # `r` is the working residual y - X@beta, kept incrementally for speed.
    r = y - Xc @ beta
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)
    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(k):
            col = Xc[:, j]
            denom = float(col @ col) / n + l2
            if denom < 1e-12:
                continue
            # The objective is (1/2n)||y - Xb||^2 + alpha*(l1_ratio*|b|_1 +
            # (1-l1_ratio)/2*|b|^2), so rho must be the correlation with the
            # residual *excluding this coordinate's own contribution*.
            # `r` still contains beta[j]*col, so it has to be added back;
            # correlating with `r` directly double-counts the current
            # coefficient and the iteration oscillates instead of converging.
            rho = float(col @ (r + beta[j] * col)) / n
            new = np.sign(rho) * max(abs(rho) - l1, 0.0) / denom
            delta = new - beta[j]
            if delta != 0.0:
                r -= col * delta
                beta[j] = new
                max_delta = max(max_delta, abs(delta))
        if max_delta < tol:
            break
    return beta


def _penalized_result(
    y, X, alpha: float, l1_ratio: float, label: str, note: str,
) -> dict:
    y_arr, X_df, names, index = _design(y, X, add_intercept=True)
    A = X_df.to_numpy(dtype="float64")
    n, k = A.shape
    if n <= k:
        raise ValueError(f"{n} observations for {k} parameters; under-determined")
    Xs, mu, sd = _standardise(A)
    Xc = Xs[:, 1:] - Xs[:, 1:].mean(axis=0, keepdims=True)
    y_c = y_arr - y_arr.mean()
    beta = np.zeros(k)
    beta[1:] = _cd_penalized(Xc, y_c, alpha, l1_ratio)
    # Undo standardisation; the intercept absorbs the centring.
    beta_full = np.zeros(k)
    beta_full[1:] = beta[1:] / sd[1:]
    beta_full[0] = y_arr.mean() - float((beta[1:] * mu[1:] / sd[1:]).sum())
    resid = y_arr - A @ beta_full
    sst = float(((y_arr - y_arr.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else float("nan")
    out = {
        "params": pd.Series(beta_full, index=pd.Index(names, name="variable")),
        "r_squared": r2,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "n_observations": n,
        "interpretation": InterpretabilityNote(
            "penalised predictive fit; coefficients are shrunk toward zero and "
            "the intercept is unpenalised",
            "an unbiased effect estimate, a significance test, or a causal claim",
        ),
        "note": note,
    }
    if l1_ratio > 0:
        selected = [names[j] for j in range(1, k) if abs(beta[j]) > 1e-10]
        out["selected_features"] = selected
        out["n_selected"] = len(selected)
    return out


def fit_lasso(y, X, alpha: float = 0.01, max_iter: int = 1000) -> dict:
    """Lasso (L1) regression via coordinate descent.

    L1 performs implicit variable selection, which is convenient but
    unstable: with two correlated predictors the penalty splits the
    coefficient arbitrarily between them, and that split changes from sample
    to sample. A stable-looking selection set is worth verifying before being
    relied upon.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    return _penalized_result(
        y, X, alpha, l1_ratio=1.0, label="lasso",
        note="L1 selection is unstable under correlated predictors: duplicated "
             "features will split the coefficient arbitrarily and the split will "
             "change between samples",
    )


def fit_elastic_net(y, X, alpha: float = 0.01, l1_ratio: float = 0.5) -> dict:
    """Elastic net: convex blend of L1 and L2 penalties.

    ``l1_ratio=1`` is lasso; ``l1_ratio=0`` is ridge. The L2 component keeps
    groups of correlated predictors together instead of splitting them
    arbitrarily, which is the main practical reason to prefer it over pure
    lasso on financial features, where collinearity is the norm.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
    return _penalized_result(
        y, X, alpha, l1_ratio=l1_ratio, label="elastic_net",
        note="the L2 component stabilises selection among correlated predictors",
    )


# --------------------------------------------------------------------------
# Factor and cross-sectional
# --------------------------------------------------------------------------


def factor_regression(
    returns: pd.Series | pd.DataFrame,
    factors: pd.DataFrame,
    add_intercept: bool = True,
    robust: bool = True,
) -> dict[str, OLSResult]:
    """Regress each column of ``returns`` on a set of factors.

    The classic asset-pricing specification. Note that R-squared here measures
    how much of the *historical* return variance the factors explain, which is
    not a claim that they will continue to, and that a high R-squared is
    compatible with a negative net alpha after costs.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame("asset")
    out: dict[str, OLSResult] = {}
    for col in returns.columns:
        pair_align = pd.concat(
            [returns[col].rename("y"), factors], axis=1
        ).dropna()
        if len(pair_align) < factors.shape[1] + 3:
            out[col] = OLSResult(
                params=pd.Series(dtype=float), std_errors=pd.Series(dtype=float),
                t_values=pd.Series(dtype=float), p_values=pd.Series(dtype=float),
                conf_int=pd.DataFrame(), r_squared=float("nan"),
                adj_r_squared=float("nan"),
                residuals=pd.Series(dtype=float), fitted=pd.Series(dtype=float),
                n_observations=int(len(pair_align)),
                n_parameters=factors.shape[1] + 1,
                condition_number=float("nan"), residual_std_error=float("nan"),
                warnings=["insufficient overlapping observations to fit the model"],
            )
            continue
        res = fit_ols(pair_align["y"], pair_align[factors.columns],
                     add_intercept=add_intercept, robust=robust)
        res.interpretation = InterpretabilityNote(
            f"exposure to {list(factors.columns)} and the intercept (alpha proxy) "
            "within the estimation sample",
            "a risk-adjusted expected return that can be relied on, or causation",
        )
        out[col] = res
    return out


def cross_sectional_regression(
    y: pd.Series,
    X: pd.DataFrame,
    add_intercept: bool = True,
    robust: bool = True,
) -> OLSResult:
    """Regress one target on cross-sectional features at a single point in time.

    ``X`` holds one row per asset (e.g. P/E, momentum, size) and ``y`` the
    forward return. This is a different animal from time-series regression:
    the error term is cross-sectional and typically far more heteroskedastic
    than a time series, so robust standard errors are the default.
    """
    return fit_ols(y, X, add_intercept=add_intercept, robust=robust)
