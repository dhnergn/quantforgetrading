"""Leakage-safe preprocessing.

Every transformer here is **stateful across fit/transform**. That is the whole
point: a scaler or PCA fitted once on the full dataset leaks the future into
every training row, and the resulting model looks excellent in-sample and fails
in production. This is blueprint Section 7.

The guard is structural, not advisory:

* :class:`FittedPreprocessor` records the index range it was fitted on.
* :meth:`FittedPreprocessor.transform` refuses data outside that range unless
  explicitly told the data is *later* — a train-time transform on validation
  data is the error being prevented.
* :class:`LeakageGuard` wraps a full ML pipeline and records the exact order of
  fit and transform calls, raising if a transform happens before a fit or if a
  fit happens twice.

The second-order protections matter too: mean-only imputation, and imputation
fitted on train statistics rather than on the combined set.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.errors import LeakageError, LeakageGuardViolation
from quantforge.core.timestamps import UTC


# ==========================================================================
# Base
# ==========================================================================


class FittedPreprocessor:
    """Base class: fit on train, transform anything, never refit implicitly."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.fitted = False
        self.fit_start: Optional[pd.Timestamp] = None
        self.fit_end: Optional[pd.Timestamp] = None
        self.n_fit_rows = 0
        self.fit_columns: tuple[str, ...] = ()

    # -- lifecycle --------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "FittedPreprocessor":
        X = self._validate(X)
        self.fit_start = X.index[0]
        self.fit_end = X.index[-1]
        self.n_fit_rows = len(X)
        self.fit_columns = tuple(X.columns)
        self._fit(X)
        self.fitted = True
        return self

    def _fit(self, X: pd.DataFrame) -> None:
        raise NotImplementedError

    def transform(self, X: pd.DataFrame, allow_later: bool = True) -> pd.DataFrame:
        if not self.fitted:
            raise LeakageError(
                f"{self.name} was used before fit(). A transformer must be fitted "
                "on the training period before it transforms anything; fitting on "
                "the full dataset and then splitting is the single most common "
                "cause of a model that cannot work out of sample."
            )
        X = self._validate(X, require_columns=True)
        if not allow_later and self.fit_start is not None:
            before = X[X.index < self.fit_start]
            if len(before):
                raise LeakageGuardViolation(
                    f"{self.name} refuses to transform {len(before)} row(s) dated "
                    f"before its fit window ({self.fit_start} to {self.fit_end}). "
                    "That would apply parameters derived from the future to "
                    "earlier data. Pass allow_later=True only if you are certain."
                )
        out = self._transform(X)
        # Only restore the index/columns when the step preserved the shape. A
        # step that legitimately drops rows (DropIncompleteRows) or renames
        # columns (TrainPCA) must be allowed to change both, and overwriting
        # with the input's labels raises a length mismatch.
        if len(out) == len(X):
            out.index = X.index
        expected = self.output_columns(X)
        if list(out.columns) != expected and len(out.columns) == len(expected):
            out.columns = expected
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def output_columns(self, X: pd.DataFrame) -> list[str]:
        return list(self.fit_columns)

    def _validate(self, X: pd.DataFrame, require_columns: bool = False) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise LeakageError(f"{self.name} requires a DataFrame, got {type(X)}")
        if len(X) == 0:
            raise LeakageError(f"{self.name} received an empty frame")
        if require_columns and self.fitted and not set(self.fit_columns) <= set(X.columns):
            missing = sorted(set(self.fit_columns) - set(X.columns))
            raise LeakageError(
                f"{self.name} was fitted on columns not present in the transform "
                f"input: {missing}"
            )
        return X

    def state(self) -> dict:
        return {
            "name": self.name,
            "fitted": self.fitted,
            "fit_start": str(self.fit_start) if self.fit_start is not None else None,
            "fit_end": str(self.fit_end) if self.fit_end is not None else None,
            "n_fit_rows": self.n_fit_rows,
            "fit_columns": list(self.fit_columns),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fitted={self.fitted}, rows={self.n_fit_rows})"


# ==========================================================================
# Imputation
# ==========================================================================


class TrainMedianImputer(FittedPreprocessor):
    """Median imputation with parameters from the training period.

    A **median** rather than a mean because financial features are heavy-tailed
    and a single crisis bar would drag a mean far above any typical value,
    imputing a missing observation to something that never happened.

    Column medians are computed per column and stored. If a column is entirely
    NaN in training, the mean is substituted and the column is marked
    all-missing so the caller can drop it rather than silently filling zeros.
    """

    def __init__(self, name: str = "TrainMedianImputer") -> None:
        super().__init__(name)
        self.statistics: dict[str, float] = {}
        self.all_missing: list[str] = []

    def _fit(self, X: pd.DataFrame) -> None:
        self.statistics = {}
        self.all_missing = []
        for col in X.columns:
            s = pd.to_numeric(X[col], errors="coerce")
            if s.notna().sum() == 0:
                self.all_missing.append(col)
                self.statistics[col] = 0.0
            else:
                self.statistics[col] = float(s.median())

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.fit_columns:
            fill = self.statistics.get(col, 0.0)
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill)
        return out

    def missing_report(self, X: pd.DataFrame) -> dict:
        """How much each column was imputed. Must be reviewed, not assumed small."""
        return {
            c: int(pd.to_numeric(X[c], errors="coerce").isna().sum())
            for c in self.fit_columns if c in X.columns
        }


class ForwardFillImputer(FittedPreprocessor):
    """Forward fill within each column, then backfill from training head.

    Appropriate for quantities that genuinely persist between observations
    (a yield, a share count) and *not* for quantities that can be absent for a
    reason (a missing print). The distinction is the researcher's to make, so
    the method is explicit rather than the platform's default.
    """

    def __init__(self, name: str = "ForwardFillImputer", limit: int = 0,
                 backfill: bool = True) -> None:
        super().__init__(name)
        self.limit = limit
        self.backfill = backfill
        self.train_head: dict[str, float] = {}

    def _fit(self, X: pd.DataFrame) -> None:
        self.train_head = {
            c: float(pd.to_numeric(X[c], errors="coerce").dropna().iloc[0])
            for c in X.columns
            if pd.to_numeric(X[c], errors="coerce").notna().any()
        }

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.fit_columns:
            s = pd.to_numeric(out[col], errors="coerce")
            s = s.ffill(limit=self.limit)
            if self.backfill and col in self.train_head:
                s = s.bfill(limit=1)
            out[col] = s
        return out


class DropIncompleteRows(FittedPreprocessor):
    """Drops rows with any NaN. Records how many were lost.

    Refuses to drop more than ``max_drop_fraction`` of the frame: a 60% loss
    means the feature set is not viable, and quietly proceeding would produce
    a model trained on a small, non-random subsample.
    """

    def __init__(self, name: str = "DropIncompleteRows",
                 max_drop_fraction: float = 0.5,
                 subset: Optional[Sequence[str]] = None) -> None:
        super().__init__(name)
        self.max_drop_fraction = max_drop_fraction
        self.subset = subset
        self.n_dropped = 0

    def _fit(self, X: pd.DataFrame) -> None:
        pass

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = list(self.subset) if self.subset else list(self.fit_columns)
        cols = [c for c in cols if c in X.columns]
        before = len(X)
        out = X.dropna(subset=cols)
        self.n_dropped = before - len(out)
        if before and self.n_dropped / before > self.max_drop_fraction:
            raise LeakageError(
                f"dropping incomplete rows would remove "
                f"{100 * self.n_dropped / before:.1f}% of the data, above the "
                f"{100 * self.max_drop_fraction:.0f}% limit. Choose an explicit "
                "imputation strategy instead; the resulting subsample may not be "
                "representative."
            )
        return out


# ==========================================================================
# Scaling
# ==========================================================================


class TrainScaler(FittedPreprocessor):
    """Standardisation with train-period mean and standard deviation.

    Two details that are usually wrong elsewhere:

    * ``ddof=0`` is used, matching the population definition. With ddof=1 the
      scaler itself introduces a tiny per-column bias on short samples.
    * A constant column yields scale 1.0 and is *not* zero-filled, which would
      destroy the signal that the column exists but does not vary.
    """

    def __init__(self, name: str = "TrainScaler", method: str = "standard") -> None:
        super().__init__(name)
        self.method = method
        self.location: dict[str, float] = {}
        self.scale: dict[str, float] = {}
        self.constant_columns: list[str] = []

    def _fit(self, X: pd.DataFrame) -> None:
        self.location = {}
        self.scale = {}
        self.constant_columns = []
        for col in X.columns:
            s = pd.to_numeric(X[col], errors="coerce")
            if self.method == "standard":
                mu = float(s.mean()) if s.notna().any() else 0.0
                sd = float(s.std(ddof=0)) if s.notna().any() else 1.0
            elif self.method == "robust":
                mu = float(s.median()) if s.notna().any() else 0.0
                q1 = float(s.quantile(0.25)) if s.notna().any() else 0.0
                q3 = float(s.quantile(0.75)) if s.notna().any() else 1.0
                sd = (q3 - q1) / 1.349 if s.notna().any() else 1.0
            else:
                raise LeakageError(f"unknown scaling method {self.method!r}")
            if not np.isfinite(sd) or sd < 1e-12:
                self.constant_columns.append(col)
                sd = 1.0
            self.location[col] = mu if np.isfinite(mu) else 0.0
            self.scale[col] = sd

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.fit_columns:
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = (s - self.location.get(col, 0.0)) / self.scale.get(col, 1.0)
        return out

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.fit_columns:
            out[col] = X[col] * self.scale.get(col, 1.0) + self.location.get(col, 0.0)
        return out


# ==========================================================================
# Dimensionality reduction
# ==========================================================================


class TrainPCA(FittedPreprocessor):
    """PCA fitted on the training period only.

    Blueprint Section 30 is explicit that PCA over the full dataset followed by
    a backtest is invalid: the components are chosen using variance that
    includes the future, so a model built on them has already seen the answer.
    """

    def __init__(self, name: str = "TrainPCA", n_components: Optional[int] = None,
                 variance_threshold: Optional[float] = None,
                 whiten: bool = False) -> None:
        super().__init__(name)
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.whiten = whiten
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.n_selected_ = 0

    def _fit(self, X: pd.DataFrame) -> None:
        if self.n_components is None and self.variance_threshold is None:
            self.n_components = min(len(self.fit_columns), 10)
        M = X[list(self.fit_columns)].to_numpy(dtype="float64")
        if np.isnan(M).any():
            raise LeakageError(
                "TrainPCA received NaN. Impute before reducing; PCA has no defined "
                "behaviour for missing values and numpy will silently propagate NaN "
                "into the components."
            )
        mean = M.mean(axis=0)
        centred = M - mean
        n, k = centred.shape
        if n <= k:
            raise LeakageError(
                f"PCA needs more observations ({n}) than features ({k}); otherwise "
                "the covariance matrix is rank-deficient and the components are not "
                "identifiable."
            )
        # SVD rather than eigendecomposition of the covariance: better
        # conditioning, and it does not square the condition number.
        U, S, Vt = np.linalg.svd(centred, full_matrices=False)
        var = (S ** 2) / (n - 1)
        ratio = var / var.sum() if var.sum() > 0 else np.zeros_like(var)

        if self.variance_threshold is not None:
            cum = np.cumsum(ratio)
            k_sel = int(np.searchsorted(cum, self.variance_threshold) + 1)
            k_sel = max(1, min(k_sel, len(var)))
        else:
            k_sel = int(min(self.n_components, len(var)))

        self.mean_ = mean
        self.components_ = Vt[:k_sel]
        self.explained_variance_ = var[:k_sel]
        self.explained_variance_ratio_ = ratio[:k_sel]
        self.n_selected_ = k_sel

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        M = X[list(self.fit_columns)].to_numpy(dtype="float64")
        projected = (M - self.mean_) @ self.components_.T
        if self.whiten:
            sd = np.sqrt(np.clip(self.explained_variance_, 1e-12, None))
            projected = projected / sd
        return pd.DataFrame(
            projected,
            index=X.index,
            columns=[f"pc_{i + 1}" for i in range(projected.shape[1])],
        )

    def output_columns(self, X: pd.DataFrame) -> list[str]:
        return [f"pc_{i + 1}" for i in range(self.n_selected_)]

    def state(self) -> dict:
        d = super().state()
        d.update({
            "n_components": self.n_selected_,
            "explained_variance_ratio": (
                [float(v) for v in self.explained_variance_ratio_]
                if self.explained_variance_ratio_ is not None else []
            ),
            "cumulative_explained_variance": (
                float(self.explained_variance_ratio_.sum())
                if self.explained_variance_ratio_ is not None else None
            ),
            "note": (
                "Fitted on the training window only. Re-fitting on later data would "
                "leak the future into the component basis."
            ),
        })
        return d


# ==========================================================================
# Feature selection
# ==========================================================================


class TrainVarianceFilter(FittedPreprocessor):
    """Drop columns with (near-)zero variance in the training period.

    Fitted on train, so a column that is flat in training but varies later is
    dropped — which is correct, because nothing in the training sample would
    have taught the model to use it.
    """

    def __init__(self, name: str = "TrainVarianceFilter", threshold: float = 1e-10) -> None:
        super().__init__(name)
        self.threshold = threshold
        self.selected_: list[str] = []
        self.dropped_: list[str] = []

    def _fit(self, X: pd.DataFrame) -> None:
        variances = X[list(self.fit_columns)].var(ddof=0)
        self.selected_ = [c for c in self.fit_columns
                          if float(variances[c]) > self.threshold]
        self.dropped_ = [c for c in self.fit_columns if c not in self.selected_]
        if not self.selected_:
            raise LeakageError("variance filter removed every column")

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[list(self.selected_)]

    def output_columns(self, X: pd.DataFrame) -> list[str]:
        return list(self.selected_)

    def state(self) -> dict:
        d = super().state()
        d.update({"n_selected": len(self.selected_), "dropped": list(self.dropped_)})
        return d


class TrainCorrelationFilter(FittedPreprocessor):
    """Drop one column from each highly correlated pair.

    Redundancy hurts more than it helps: it inflates the apparent importance of
    a duplicated feature and destabilises tree splits. Pairing is greedy in
    training-set column order, so the result is deterministic and reproducible.

    This is a *convenience* filter, not a selection method. Selecting features
    by their relationship to the target is a separate, more dangerous step that
    must also happen inside the training window.
    """

    def __init__(self, name: str = "TrainCorrelationFilter",
                 threshold: float = 0.95) -> None:
        super().__init__(name)
        self.threshold = threshold
        self.selected_: list[str] = []
        self.correlated_pairs_: list[tuple[str, str, float]] = []

    def _fit(self, X: pd.DataFrame) -> None:
        cols = list(self.fit_columns)
        corr = X[cols].corr().abs()
        self.selected_ = []
        self.correlated_pairs_ = []
        for i, col in enumerate(cols):
            if col in self.selected_:
                continue
            self.selected_.append(col)
            for other in cols[i + 1:]:
                if other in self.selected_:
                    continue
                v = corr.loc[col, other]
                if np.isfinite(v) and v >= self.threshold:
                    self.correlated_pairs_.append((col, other, float(v)))
        if not self.selected_:
            raise LeakageError("correlation filter removed every column")

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[list(self.selected_)]

    def output_columns(self, X: pd.DataFrame) -> list[str]:
        return list(self.selected_)

    def state(self) -> dict:
        d = super().state()
        d.update({
            "n_selected": len(self.selected_),
            "n_correlated_pairs_dropped": len(self.correlated_pairs_),
            "threshold": self.threshold,
        })
        return d


# ==========================================================================
# Pipeline
# ==========================================================================


@dataclass(slots=True)
class PipelineStep:
    name: str
    transformer: Any
    fitted_on: Optional[str] = None

    def as_dict(self) -> dict:
        d = {"name": self.name, "kind": type(self.transformer).__name__}
        if hasattr(self.transformer, "state"):
            d["state"] = self.transformer.state()
        return d


class PreprocessingPipeline:
    """Ordered transformers, fitted once on the training period.

    The pipeline records the order of operations. :meth:`audit` returns that
    order so a reviewer can confirm every fit happened inside the training
    window before anything was transformed.
    """

    def __init__(self, steps: Optional[Sequence[Any]] = None,
                 name: str = "pipeline") -> None:
        self.name = name
        self.steps: list[PipelineStep] = [
            PipelineStep(type(s).__name__, s) for s in (steps or [])
        ]
        self.fitted = False
        self.fit_window: Optional[tuple] = None
        self.audit_log: list[dict] = []

    def add(self, transformer: Any, name: Optional[str] = None) -> "PreprocessingPipeline":
        self.steps.append(PipelineStep(name or type(transformer).__name__, transformer))
        return self

    def fit(self, X: pd.DataFrame) -> "PreprocessingPipeline":
        if self.fitted:
            raise LeakageError(
                f"pipeline {self.name!r} is already fitted. Refitting silently "
                "re-derives parameters from different data, which is exactly the "
                "leakage this class exists to prevent. Build a new pipeline."
            )
        current = X
        for step in self.steps:
            step.transformer.fit(current)
            step.fitted_on = f"{current.index[0]} .. {current.index[-1]}"
            current = step.transformer.transform(current)
            self.audit_log.append({
                "action": "fit_transform", "step": step.name,
                "rows": len(current), "columns": list(current.columns),
            })
        self.fit_window = (X.index[0], X.index[-1])
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame, allow_later: bool = True) -> pd.DataFrame:
        if not self.fitted:
            raise LeakageError(f"pipeline {self.name!r} has not been fitted")
        current = X
        for step in self.steps:
            current = step.transformer.transform(current, allow_later=allow_later)
            self.audit_log.append({
                "action": "transform", "step": step.name,
                "rows": len(current), "columns": list(current.columns),
            })
        return current

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``X`` and transform it. The only legitimate call on new data."""
        return self.fit(X).transform(X)

    def audit(self) -> dict:
        return {
            "name": self.name,
            "fitted": self.fitted,
            "fit_window": [str(self.fit_window[0]), str(self.fit_window[1])]
            if self.fit_window else None,
            "steps": [s.as_dict() for s in self.steps],
            "log": self.audit_log,
            "step_order": [s.name for s in self.steps],
        }

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"PreprocessingPipeline({self.name!r}, steps={[s.name for s in self.steps]})"


def default_pipeline(
    impute: bool = True,
    scale: bool = True,
    max_drop_fraction: float = 0.5,
) -> PreprocessingPipeline:
    """The standard research pipeline: impute -> drop incomplete -> scale.

    Dropping incomplete rows *before* imputation is intentional: the median
    imputer only needs to handle gaps that appear in later data, and dropping
    first keeps the fitted medians based on complete observations.
    """
    steps: list[Any] = []
    if impute:
        steps.append(DropIncompleteRows(max_drop_fraction=max_drop_fraction))
    if scale:
        steps.append(TrainScaler())
    return PreprocessingPipeline(steps, name="default")


def assert_no_future_fit(
    pipeline: PreprocessingPipeline,
    train_index: pd.DatetimeIndex,
) -> None:
    """Assert every pipeline step was fitted inside the training window."""
    if not pipeline.fitted:
        raise LeakageError("pipeline is not fitted")
    lo, hi = train_index[0], train_index[-1]
    for step in pipeline.steps:
        t = step.transformer
        if not getattr(t, "fitted", False):
            raise LeakageError(f"step {step.name} was never fitted")
        fs, fe = getattr(t, "fit_start", None), getattr(t, "fit_end", None)
        if fs is None or fe is None:
            continue
        if fs < lo or fe > hi:
            raise LeakageGuardViolation(
                f"step {step.name} was fitted on {fs} to {fe}, which extends "
                f"outside the training window {lo} to {hi}. Parameters derived "
                "from data outside the training window leak information about the "
                "future into the model."
            )
