"""Model wrappers with versioning and reproducibility.

Every model carries a full experiment manifest, and the promotion ladder in
:mod:`quantforge.ml.registry` is enforced structurally: a trained model cannot
reach ``live_eligible`` without passing validation and paper trading.

Two things are deliberately absent:

* **No automatic deployment.** ``fit`` produces a model at stage ``TRAINED``.
  Nothing moves it forward on its own.
* **No confidence-as-certainty.** Probability outputs are labelled as model
  scores. Calibrated classifiers produce a well-ordered probability; they do
  not produce a guaranteed likelihood of a future price move, and the
  calibration report is designed to show that limitation.

Every estimator is wrapped rather than used directly, so the seed, the
version, and the training window travel with the fitted object.
"""
from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.enums import DriftType, ModelStage, ModelTask, TargetLabel
from quantforge.core.errors import (
    InsufficientDataError, ModelNotTrainedError, ModelPromotionError,
    OptionalDependencyError,
)
from quantforge.core.reproducibility import (
    ExperimentManifest, content_hash, set_seed,
)
from quantforge.core.timestamps import utcnow
from quantforge.ml.preprocess import PreprocessingPipeline

_SKLEARN_OK = True
try:  # pragma: no cover
    import sklearn
    from sklearn.ensemble import (
        GradientBoostingClassifier, GradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor,
    )
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score, brier_score_loss, confusion_matrix,
        mean_absolute_error, mean_squared_error, roc_auc_score, r2_score,
    )
    from sklearn.neural_network import MLPClassifier, MLPRegressor
except ImportError:  # pragma: no cover
    _SKLEARN_OK = False


def _require_sklearn() -> None:
    if not _SKLEARN_OK:
        raise OptionalDependencyError(
            "scikit-learn is required for this model family. Install it with "
            "`pip install scikit-learn`. The deterministic, dependency-free parts "
            "of the platform (returns, indicators, statistics, risk, backtesting) "
            "remain available without it."
        )


# ==========================================================================
# Metrics
# ==========================================================================


def classification_metrics(
    y_true: Sequence, y_pred_proba: Optional[np.ndarray],
    y_pred: Optional[np.ndarray], labels: Sequence[str],
) -> dict:
    """Accuracy, macro F1, confusion matrix, and Brier score.

    Accuracy alone is misleading on imbalanced labels, which is the normal case
    for a directional target: always predicting "flat" can score well while
    being useless. Macro F1 weights each class equally and exposes that.
    """
    _require_sklearn()
    yt = np.asarray(y_true)
    out: dict[str, Any] = {"n": int(len(yt))}
    if y_pred is not None and len(y_pred):
        out["accuracy"] = float(accuracy_score(yt, y_pred))
        cm = confusion_matrix(yt, y_pred, labels=list(labels))
        out["confusion_matrix"] = cm.tolist()
        out["per_class_support"] = {
            lab: int((yt == lab).sum()) for lab in labels
        }
        out["recall_per_class"] = {
            lab: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan")
            for i, lab in enumerate(labels)
        }
        prec, rec, f1, sup = _prf(yt, y_pred, list(labels))
        out["macro_f1"] = float(np.mean(f1)) if len(f1) else float("nan")
        out["per_class_f1"] = {lab: float(f1[i]) for i, lab in enumerate(labels)}
    if y_pred_proba is not None and len(y_pred_proba):
        p = np.asarray(y_pred_proba)
        if p.ndim == 2 and p.shape[1] == 2:
            out["brier_score"] = float(brier_score_loss((yt == labels[1]).astype(int),
                                                        p[:, 1]))
            try:
                out["roc_auc"] = float(roc_auc_score((yt == labels[1]).astype(int),
                                                      p[:, 1]))
            except ValueError:
                out["roc_auc"] = float("nan")
        out["mean_predicted_probability"] = float(np.mean(p))
        out["base_rate"] = float(np.mean(yt == labels[1]))
    out["note"] = (
        "Probabilities are model scores, not guaranteed likelihoods of a price "
        "move. Brier score and ROC AUC describe how well the model orders its "
        "own outputs; neither says the model is correct."
    )
    return out


def _prf(yt, yp, labels):
    tp = np.array([int(((yt == l) & (yp == l)).sum()) for l in labels], dtype=float)
    fp = np.array([int(((yt != l) & (yp == l)).sum()) for l in labels], dtype=float)
    fn = np.array([int(((yt == l) & (yp != l)).sum()) for l in labels], dtype=float)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1, tp + fn


def regression_metrics(y_true, y_pred) -> dict:
    _require_sklearn()
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) < 2:
        return {"n": int(len(yt)), "note": "need at least 2 observations"}
    resid = yt - yp
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    # Information coefficient: the rank correlation between prediction and
    # outcome. For a directional model this matters more than R^2, which
    # rewards magnitude predictions on a noisy target.
    ic = _rank_corr(yt, yp)
    return {
        "n": int(len(yt)),
        "mse": float(mean_squared_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r_squared": float(r2_score(yt, yp)) if ss_tot > 0 else float("nan"),
        "directional_accuracy": float(np.mean(np.sign(yt) == np.sign(yp))),
        "information_coefficient": ic,
        "ic_std_error": float((1 - ic**2) / np.sqrt(len(yt))) if np.isfinite(ic) else float("nan"),
        "residual_mean": float(resid.mean()),
        "residual_std": float(resid.std(ddof=1)),
        "note": (
            "R-squared on a noisy return target is usually small and its "
            "significance is not evidence of tradable edge. The information "
            "coefficient measures directional ordering, which is closer to what "
            "a strategy needs."
        ),
    }


def _rank_corr(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ==========================================================================
# Base
# ==========================================================================


@dataclass(slots=True)
class TrainedModel:
    """Base wrapper. Holds the estimator, the preprocessing, and provenance."""

    model_id: str
    model_version: str
    task: ModelTask
    estimator: Any = None
    pipeline: Optional[PreprocessingPipeline] = None
    manifest: Optional[ExperimentManifest] = None
    stage: ModelStage = ModelStage.TRAINED
    feature_names: tuple[str, ...] = ()
    label_classes: tuple[str, ...] = ()
    metrics: dict = field(default_factory=dict)
    created_at: dt.datetime = field(default_factory=utcnow)
    target_definition: str = ""
    is_trained: bool = False
    calibration: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "task": self.task.value,
            "stage": self.stage.value,
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "label_classes": list(self.label_classes),
            "metrics": self.metrics,
            "target_definition": self.target_definition,
            "created_at": self.created_at.isoformat(),
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "pipeline": self.pipeline.audit() if self.pipeline else None,
        }


class QuantModel:
    """Common fit/predict/serialisation surface over the estimator families."""

    task: ModelTask = ModelTask.CLASSIFICATION

    def __init__(self, model_id: str, model_version: str = "1.0.0",
                 pipeline: Optional[PreprocessingPipeline] = None,
                 random_seed: int = 42, **params: Any) -> None:
        _require_sklearn()
        self.model_id = model_id
        self.model_version = model_version
        self.pipeline = pipeline
        self.random_seed = random_seed
        self.params = params
        self.trained: Optional[TrainedModel] = None

    # -- lifecycle --------------------------------------------------------

    def _build(self) -> Any:
        raise NotImplementedError

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        start: Optional[dt.datetime] = None,
        end: Optional[dt.datetime] = None,
        target_definition: str = "",
        label_classes: Optional[Sequence[str]] = None,
    ) -> TrainedModel:
        set_seed(self.random_seed)
        if len(X) != len(y):
            raise InsufficientDataError(
                f"{len(X)} feature rows vs {len(y)} labels; they must align"
            )
        if len(X) < 30:
            raise InsufficientDataError(
                f"{len(X)} training rows is too few for a fitted model. 30 is the "
                "floor; below that any result is dominated by sampling noise."
            )
        Z = self.pipeline.fit_transform(X) if self.pipeline is not None else X
        Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        yv = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

        est = self._build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(Z.to_numpy(), yv)
        if hasattr(est, "random_state"):
            try:
                est.set_params(random_state=self.random_seed)
            except (ValueError, AttributeError):
                pass

        manifest = ExperimentManifest(
            experiment_type=f"model_{self.task.value}",
            dataset_version=content_hash(Z.to_numpy().tobytes()[:8192].hex()),
            feature_set_version=content_hash(list(Z.columns)),
            strategy_version=self.model_version,
            model_version=self.model_version,
            parameters={k: str(v) for k, v in self.params.items()},
            random_seed=self.random_seed,
            preprocessing=self.pipeline.audit() if self.pipeline else {},
            time_period={
                "start": str(X.index[0]), "end": str(X.index[-1]), "n_rows": len(X),
            },
            notes=f"model_id={self.model_id}",
        )
        self.trained = TrainedModel(
            model_id=self.model_id, model_version=self.model_version,
            task=self.task, estimator=est, pipeline=self.pipeline,
            manifest=manifest, stage=ModelStage.TRAINED,
            feature_names=tuple(Z.columns),
            label_classes=tuple(label_classes or np.unique(yv).tolist()),
            target_definition=target_definition, is_trained=True,
        )
        return self.trained

    # -- prediction -------------------------------------------------------

    def _check(self) -> TrainedModel:
        if self.trained is None or not self.trained.is_trained:
            raise ModelNotTrainedError(
                f"model {self.model_id!r} has no fitted estimator. Training does "
                "not confer a live deployment; see ModelRegistry for the promotion "
                "ladder."
            )
        return self.trained

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        t = self._check()
        Z = self.pipeline.transform(X) if self.pipeline is not None else X
        Z = Z.reindex(columns=list(t.feature_names), fill_value=np.nan)
        return Z.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} does not produce class probabilities. "
            "Interpreting its point predictions as probabilities would be wrong."
        )

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        raise NotImplementedError

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Importance with an explicit association-not-causation banner."""
        t = self._check()
        imp = None
        if hasattr(t.estimator, "feature_importances_"):
            imp = np.asarray(t.estimator.feature_importances_)
        elif hasattr(t.estimator, "coef_"):
            coef = np.atleast_2d(np.asarray(t.estimator.coef_))
            imp = np.abs(coef).mean(axis=0)
        if imp is None:
            return pd.DataFrame(columns=["feature", "importance"])
        df = pd.DataFrame({
            "feature": list(t.feature_names)[: len(imp)],
            "importance": imp[: len(t.feature_names)],
        }).sort_values("importance", ascending=False)
        df["interpretation"] = (
            "predictive association within the training sample; not a causal "
            "effect and not a guarantee of future relationship"
        )
        return df.head(top_n).reset_index(drop=True)

    def fingerprint(self) -> str:
        t = self._check()
        return content_hash({
            "model_id": self.model_id, "version": self.model_version,
            "params": {k: str(v) for k, v in self.params.items()},
            "seed": self.random_seed, "features": list(t.feature_names),
            "manifest": t.manifest.compute_hash() if t.manifest else None,
        })


# ==========================================================================
# Classification
# ==========================================================================


class BinaryClassifier(QuantModel):
    task = ModelTask.CLASSIFICATION

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._check().estimator.predict(self._transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        t = self._check()
        if not hasattr(t.estimator, "predict_proba"):
            raise NotImplementedError(
                f"{type(self.estimator).__name__} has no predict_proba. A hard "
                "class label is not a probability; do not treat 0/1 as confidence."
            )
        return t.estimator.predict_proba(self._transform(X))

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        t = self._check()
        Z = self._transform(X)
        yv = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        pred = t.estimator.predict(Z)
        proba = (t.estimator.predict_proba(Z)
                 if hasattr(t.estimator, "predict_proba") else None)
        return classification_metrics(yv, proba, pred, t.label_classes)

    def out_of_sample_lift(self, X: pd.DataFrame, y: pd.Series,
                           buckets: int = 5) -> pd.DataFrame:
        """Observed outcome rate by predicted-probability bucket.

        The most honest single check on a probability model: if the top bucket
        of predicted probabilities does not actually have a higher outcome rate
        than the bottom bucket, the probabilities are not usable as scores. This
        is measured out of sample and is a *calibration* statement, not a claim
        about future outcomes.
        """
        t = self._check()
        if not hasattr(t.estimator, "predict_proba"):
            return pd.DataFrame()
        Z = self._transform(X)
        p = t.estimator.predict_proba(Z)
        classes = list(t.label_classes)
        pos = classes.index("up") if "up" in classes else 1
        score = p[:, pos] if p.ndim == 2 and p.shape[1] > pos else p.ravel()
        yv = (y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y))
        target = (yv == "up").astype(float) if "up" in classes else yv.astype(float)
        frame = pd.DataFrame({"score": score, "target": target})
        frame["bucket"] = pd.qcut(frame["score"], buckets, labels=False, duplicates="drop")
        out = frame.groupby("bucket").agg(
            n=("target", "size"),
            mean_predicted=("score", "mean"),
            observed_rate=("target", "mean"),
        ).reset_index()
        out["lift"] = out["observed_rate"] - out["mean_predicted"]
        out["note"] = (
            "A well-ordered model shows observed_rate rising monotonically with "
            "mean_predicted. Non-monotonicity means the score cannot be used to "
            "size positions."
        )
        return out


class LogisticRegressionModel(BinaryClassifier):
    """L2 logistic regression. The interpretable baseline.

    Included deliberately: a tree ensemble that cannot beat a regularised
    linear model on the same features is not earning its complexity or its
    opacity.
    """

    def __init__(self, model_id: str, C: float = 1.0, max_iter: int = 1000,
                 class_weight: Optional[str] = None, **kw) -> None:
        super().__init__(model_id, C=C, max_iter=max_iter,
                         class_weight=class_weight, **kw)

    def _build(self):
        return LogisticRegression(
            C=self.params.get("C", 1.0), max_iter=self.params.get("max_iter", 1000),
            class_weight=self.params.get("class_weight"), random_state=self.random_seed,
        )


class RandomForestClassifierModel(BinaryClassifier):
    def __init__(self, model_id: str, n_estimators: int = 200, max_depth=None,
                 min_samples_leaf: int = 5, max_features: str = "sqrt",
                 class_weight: Optional[str] = "balanced", **kw) -> None:
        super().__init__(model_id, n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, max_features=max_features,
                         class_weight=class_weight, **kw)

    def _build(self):
        return RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth"),
            min_samples_leaf=self.params.get("min_samples_leaf", 5),
            max_features=self.params.get("max_features", "sqrt"),
            class_weight=self.params.get("class_weight"),
            random_state=self.random_seed, n_jobs=1,
        )


class GradientBoostingClassifierModel(BinaryClassifier):
    def __init__(self, model_id: str, n_estimators: int = 200,
                 learning_rate: float = 0.05, max_depth: int = 3,
                 subsample: float = 0.9, min_samples_leaf: int = 20, **kw) -> None:
        super().__init__(model_id, n_estimators=n_estimators,
                         learning_rate=learning_rate, max_depth=max_depth,
                         subsample=subsample, min_samples_leaf=min_samples_leaf, **kw)

    def _build(self):
        return GradientBoostingClassifier(
            n_estimators=self.params.get("n_estimators", 200),
            learning_rate=self.params.get("learning_rate", 0.05),
            max_depth=self.params.get("max_depth", 3),
            subsample=self.params.get("subsample", 0.9),
            min_samples_leaf=self.params.get("min_samples_leaf", 20),
            random_state=self.random_seed,
        )


# ==========================================================================
# Regression
# ==========================================================================


class Regressor(QuantModel):
    task = ModelTask.REGRESSION

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._check().estimator.predict(self._transform(X))

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        t = self._check()
        yv = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        return regression_metrics(yv, t.estimator.predict(self._transform(X)))

    def quantile_predictions(self, X: pd.DataFrame, quantiles=(0.1, 0.5, 0.9)):
        """Quantile predictions, where the estimator supports them.

        Useful for a strategy that needs a distribution rather than a point
        estimate. Absent for estimators that only produce conditional means —
        and a conditional mean presented as a forecast distribution would be
        misleading.
        """
        t = self._check()
        if not hasattr(t.estimator, "quantiles"):
            raise NotImplementedError(
                f"{type(self.estimator).__name__} produces only a conditional mean, "
                "not quantiles. A point forecast is not a distribution."
            )
        Z = self._transform(X)
        return {q: t.estimator.predict(X, quantiles=[q]) for q in quantiles}


class LinearRegressionModel(Regressor):
    def __init__(self, model_id: str, alpha: float = 1.0, **kw) -> None:
        super().__init__(model_id, alpha=alpha, **kw)

    def _build(self):
        return Ridge(alpha=self.params.get("alpha", 1.0), random_state=None)


class RandomForestRegressorModel(Regressor):
    def __init__(self, model_id: str, n_estimators: int = 200, max_depth=None,
                 min_samples_leaf: int = 5, **kw) -> None:
        super().__init__(model_id, n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, **kw)

    def _build(self):
        return RandomForestRegressor(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth"),
            min_samples_leaf=self.params.get("min_samples_leaf", 5),
            random_state=self.random_seed, n_jobs=1,
        )


class GradientBoostingRegressorModel(Regressor):
    def __init__(self, model_id: str, n_estimators: int = 200,
                 learning_rate: float = 0.05, max_depth: int = 3,
                 subsample: float = 0.9, **kw) -> None:
        super().__init__(model_id, n_estimators=n_estimators,
                         learning_rate=learning_rate, max_depth=max_depth,
                         subsample=subsample, **kw)

    def _build(self):
        return GradientBoostingRegressor(
            n_estimators=self.params.get("n_estimators", 200),
            learning_rate=self.params.get("learning_rate", 0.05),
            max_depth=self.params.get("max_depth", 3),
            subsample=self.params.get("subsample", 0.9),
            random_state=self.random_seed,
        )


class NeuralNetworkModel(Regressor):
    """Optional MLP with early stopping.

    Not a default. A network is a high-capacity model with many local optima,
    and on tabular financial data a regularised linear model or a shallow
    ensemble is frequently the better choice. ``comparison_report`` exists to
    make that check explicit rather than assumed.
    """

    def __init__(self, model_id: str, hidden_layer_sizes=(64, 32),
                 alpha: float = 1e-3, max_iter: int = 400,
                 early_stopping: bool = True, validation_fraction: float = 0.15,
                 learning_rate_init: float = 1e-3, **kw) -> None:
        super().__init__(model_id, hidden_layer_sizes=hidden_layer_sizes,
                         alpha=alpha, max_iter=max_iter,
                         early_stopping=early_stopping,
                         validation_fraction=validation_fraction,
                         learning_rate_init=learning_rate_init, **kw)

    def _build(self):
        return MLPRegressor(
            hidden_layer_sizes=self.params.get("hidden_layer_sizes", (64, 32)),
            alpha=self.params.get("alpha", 1e-3),
            max_iter=self.params.get("max_iter", 400),
            early_stopping=self.params.get("early_stopping", True),
            validation_fraction=self.params.get("validation_fraction", 0.15),
            learning_rate_init=self.params.get("learning_rate_init", 1e-3),
            random_state=self.random_seed,
        )

    def training_curve(self) -> pd.DataFrame:
        t = self._check()
        if not hasattr(t.estimator, "loss_curve_"):
            return pd.DataFrame()
        return pd.DataFrame({
            "epoch": np.arange(1, len(t.estimator.loss_curve_) + 1),
            "loss": t.estimator.loss_curve_,
        })


# ==========================================================================
# Unsupervised
# ==========================================================================


@dataclass(slots=True)
class ClusteringModel:
    model_id: str
    model_version: str
    labels_: np.ndarray
    cluster_centers_: Optional[np.ndarray] = None
    inertia_: Optional[float] = None
    n_clusters: int = 0
    feature_names: tuple[str, ...] = ()
    fit_start: Optional[pd.Timestamp] = None
    fit_end: Optional[pd.Timestamp] = None
    method: str = "kmeans"
    stage: ModelStage = ModelStage.TRAINED
    noise_fraction: float = 0.0
    is_trained: bool = True
    metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id, "model_version": self.model_version,
            "method": self.method, "n_clusters": self.n_clusters,
            "noise_fraction": self.noise_fraction,
            "fit_window": [str(self.fit_start), str(self.fit_end)] if self.fit_start else None,
            "features": list(self.feature_names),
            "stage": self.stage.value,
            "note": (
                "Cluster assignments are fitted on a specific window. Re-using a "
                "historical assignment for a later date assumes the grouping is "
                "unchanged, which must be verified rather than assumed."
            ),
        }


def kmeans_cluster(
    X: pd.DataFrame, n_clusters: int = 5, model_id: str = "kmeans",
    model_version: str = "1.0.0", seed: int = 42, max_iter: int = 300,
) -> ClusteringModel:
    """K-means with a silhouette-based diagnostic for the chosen ``k``.

    The silhouette score is reported but the cluster count is not chosen
    automatically: the best silhouette is a description of the data, not a
    statement that the grouping is meaningful.
    """
    _require_sklearn()
    if len(X) < n_clusters:
        raise InsufficientDataError(
            f"k-means needs at least as many observations as clusters: "
            f"{len(X)} vs {n_clusters}"
        )
    km = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=seed,
                n_init=10)
    labels = km.fit_predict(X.to_numpy())
    sil = _silhouette(X.to_numpy(), labels, n_clusters)
    return ClusteringModel(
        model_id=model_id, model_version=model_version, labels_=labels,
        cluster_centers_=km.cluster_centers_, inertia_=float(km.inertia_),
        n_clusters=n_clusters, feature_names=tuple(X.columns),
        fit_start=X.index[0], fit_end=X.index[-1], method="kmeans",
        metrics={"silhouette": sil,
                 "note": "silhouette describes separation in feature space; it "
                         "does not establish that clusters correspond to "
                         "economically distinct regimes"},
    )


def hierarchical_cluster(
    X: pd.DataFrame, n_clusters: int = 5, linkage: str = "ward",
    model_id: str = "hierarchical", model_version: str = "1.0.0",
    metric: str = "euclidean",
) -> ClusteringModel:
    _require_sklearn()
    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage,
                                 metric=metric if linkage != "ward" else "euclidean")
    labels = ac.fit_predict(X.to_numpy())
    return ClusteringModel(
        model_id=model_id, model_version=model_version, labels_=labels,
        n_clusters=n_clusters, feature_names=tuple(X.columns),
        fit_start=X.index[0], fit_end=X.index[-1], method=f"hierarchical_{linkage}",
        metrics={"note": "the full dendrogram is the object of interest; a flat "
                         "cut at n_clusters is a presentation choice"},
    )


def dbscan_anomaly(
    X: pd.DataFrame, eps: float = 0.5, min_samples: int = 5,
    model_id: str = "dbscan", model_version: str = "1.0.0",
) -> ClusteringModel:
    """Density-based clustering, with noise points treated as anomalies.

    ``eps`` is scale-dependent, so it is only meaningful after standardisation.
    Running DBSCAN on raw price-scale features will either label everything as
    noise or nothing, which looks like a result but is a units error.
    """
    _require_sklearn()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X.to_numpy())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise = float(np.mean(labels == -1))
    return ClusteringModel(
        model_id=model_id, model_version=model_version, labels_=labels,
        n_clusters=n_clusters, feature_names=tuple(X.columns),
        fit_start=X.index[0], fit_end=X.index[-1], method="dbscan",
        noise_fraction=noise,
        metrics={"n_noise_points": int(np.sum(labels == -1)),
                 "note": "points labelled -1 are unclustered. A high noise fraction "
                         "usually means eps is mis-set for the feature scale, not "
                         "that the data is anomalous"},
    )


def _silhouette(X: np.ndarray, labels: np.ndarray, k: int) -> float:
    if k < 2 or len(set(labels)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")
