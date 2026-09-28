"""Model registry and the promotion ladder.

```
EXPERIMENT -> TRAINED -> VALIDATED -> APPROVED -> PAPER_TRADING -> LIVE_ELIGIBLE
```

Training produces a model at ``TRAINED`` and nothing more. Each promotion has
a documented precondition, and :meth:`ModelRegistry.promote` enforces them, so
"the backtest looked good so it goes live" is not a reachable state.

Live eligibility additionally requires, per blueprint Section 82:
a signed authorisation record, a locked strategy version, a locked model
version, and a documented paper-trading period. Those are checked here as
*prerequisites* — the human authorisation itself is the
:mod:`quantforge.execution.live_gate` module's job, so no model code can grant
itself permission to trade.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.enums import DriftType, ModelStage, ModelTask
from quantforge.core.errors import ModelPromotionError, ModelVersionError
from quantforge.core.timestamps import utcnow
from quantforge.core.reproducibility import ExperimentManifest, content_hash

#: Ordered promotion ladder. Index i must precede index i+1.
PROMOTION_ORDER: tuple[ModelStage, ...] = (
    ModelStage.EXPERIMENT,
    ModelStage.TRAINED,
    ModelStage.VALIDATED,
    ModelStage.APPROVED,
    ModelStage.PAPER_TRADING,
    ModelStage.LIVE_ELIGIBLE,
    ModelStage.RETIRED,
)


@dataclass(slots=True)
class ValidationRecord:
    """The evidence supporting a promotion. Empty metrics cannot be promoted."""

    performed_at: dt.datetime
    performed_by: str
    oos_metrics: dict = field(default_factory=dict)
    in_sample_metrics: dict = field(default_factory=dict)
    dataset_version: str = ""
    period_start: Optional[pd.Timestamp] = None
    period_end: Optional[pd.Timestamp] = None
    n_oos_observations: int = 0
    passed: bool = False
    checks: dict = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "performed_at": self.performed_at.isoformat(),
            "performed_by": self.performed_by,
            "oos_metrics": self.oos_metrics,
            "in_sample_metrics": self.in_sample_metrics,
            "dataset_version": self.dataset_version,
            "period": [str(self.period_start) if self.period_start is not None else None,
                       str(self.period_end) if self.period_end is not None else None],
            "n_oos_observations": self.n_oos_observations,
            "passed": self.passed,
            "checks": self.checks,
            "notes": self.notes,
        }


@dataclass(slots=True)
class RegistryEntry:
    model_id: str
    model_version: str
    task: ModelTask
    stage: ModelStage
    created_at: dt.datetime
    manifest: Optional[ExperimentManifest] = None
    fingerprint: str = ""
    model: Any = None
    validations: list[ValidationRecord] = field(default_factory=list)
    paper_trading_log: list[dict] = field(default_factory=list)
    drift_events: list[dict] = field(default_factory=list)
    locked: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[dt.datetime] = None
    live_authorisation_id: Optional[str] = None
    retired_at: Optional[dt.datetime] = None
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "task": self.task.value,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "fingerprint": self.fingerprint,
            "locked": self.locked,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "live_authorisation_id": self.live_authorisation_id,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "n_validations": len(self.validations),
            "latest_validation": (
                self.validations[-1].as_dict() if self.validations else None
            ),
            "paper_trading_sessions": len(self.paper_trading_log),
            "drift_events": len(self.drift_events),
            "notes": self.notes,
        }


# ==========================================================================
# Validation thresholds
# ==========================================================================


@dataclass(slots=True)
class ValidationCriteria:
    """Minimum evidence to accept a model.

    The defaults are deliberately unachievable on random data, which is the
    point. ``min_oos_observations`` matters as much as the metrics: a Sharpe of
    3 over 40 observations is noise.
    """

    min_oos_observations: int = 100
    min_oos_sharpe: float = 0.5
    min_oos_profit_factor: float = 1.05
    max_oos_drawdown: float = 0.35
    max_is_oos_sharpe_degradation: float = 0.6
    min_trades: int = 20
    require_monotonic_calibration: bool = True

    def as_dict(self) -> dict:
        return {
            "min_oos_observations": self.min_oos_observations,
            "min_oos_sharpe": self.min_oos_sharpe,
            "min_oos_profit_factor": self.min_oos_profit_factor,
            "max_oos_drawdown": self.max_oos_drawdown,
            "max_is_oos_sharpe_degradation": self.max_is_oos_sharpe_degradation,
            "min_trades": self.min_trades,
            "require_monotonic_calibration": self.require_monotonic_calibration,
        }


def evaluate_validation(
    oos: dict,
    in_sample: Optional[dict] = None,
    criteria: Optional[ValidationCriteria] = None,
    calibration_is_monotonic: Optional[bool] = None,
) -> tuple[bool, dict]:
    """Apply the criteria. Returns ``(passed, checks)``.

    Every check is recorded whether it passed or failed, so a rejected
    promotion shows exactly which threshold was missed rather than a bare
    ``False``.
    """
    c = criteria or ValidationCriteria()
    checks: dict[str, dict] = {}

    def chk(name: str, ok: bool, observed: Any, required: Any) -> None:
        checks[name] = {"passed": bool(ok), "observed": observed, "required": required}

    n = oos.get("n_trades", oos.get("n", 0)) or 0
    chk("min_oos_observations", n >= c.min_oos_observations, n, c.min_oos_observations)

    sharpe = oos.get("sharpe", oos.get("oos_sharpe", float("nan")))
    chk("min_oos_sharpe", np.isfinite(sharpe) and sharpe >= c.min_oos_sharpe,
        sharpe, c.min_oos_sharpe)

    pf = oos.get("profit_factor", float("nan"))
    # A strategy with no losing trades has an undefined profit factor, which is
    # not evidence of quality. It is treated as a failed check.
    pf_ok = np.isfinite(pf) and pf >= c.min_oos_profit_factor
    chk("min_oos_profit_factor", pf_ok, pf, c.min_oos_profit_factor)

    dd = oos.get("max_drawdown", float("nan"))
    chk("max_oos_drawdown", np.isfinite(dd) and dd <= c.max_oos_drawdown,
        dd, c.max_oos_drawdown)

    if in_sample and in_sample.get("sharpe") is not None:
        is_sharpe = in_sample.get("sharpe", float("nan"))
        if np.isfinite(is_sharpe) and is_sharpe > 0 and np.isfinite(sharpe):
            degradation = (is_sharpe - sharpe) / is_sharpe
            chk("max_sharpe_degradation", degradation <= c.max_is_oos_sharpe_degradation,
                degradation, c.max_is_oos_sharpe_degradation)
        else:
            chk("max_sharpe_degradation", True, None,
                c.max_is_oos_sharpe_degradation)

    if c.require_monotonic_calibration:
        if calibration_is_monotonic is None:
            chk("calibration_monotonic", False, "not measured",
                "measured out of sample")
        else:
            chk("calibration_monotonic", bool(calibration_is_monotonic),
                calibration_is_monotonic, True)

    passed = all(v["passed"] for v in checks.values())
    return passed, checks


# ==========================================================================
# Registry
# ==========================================================================


class ModelRegistry:
    """Versioned model store with enforced promotion.

    A production model is never replaced in place. ``register`` with an
    existing ``(model_id, model_version)`` and a different fingerprint raises,
    because a silently-changed model invalidates every backtest that referenced
    it.
    """

    def __init__(self, criteria: Optional[ValidationCriteria] = None) -> None:
        self.criteria = criteria or ValidationCriteria()
        self._entries: dict[tuple[str, str], RegistryEntry] = {}
        self._order: list[tuple[str, str]] = []
        self.audit_hook: Optional[callable] = None

    # -- registration -----------------------------------------------------

    def register(
        self,
        model: Any,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        stage: ModelStage = ModelStage.TRAINED,
        notes: str = "",
    ) -> RegistryEntry:
        mid = model_id or getattr(model, "model_id", None)
        mver = model_version or getattr(model, "model_version", "1.0.0")
        if not mid:
            raise ModelVersionError("a model_id is required for registration")
        trained = getattr(model, "trained", None)
        task = getattr(model, "task", ModelTask.CLASSIFICATION)
        manifest = trained.manifest if trained is not None else None
        fingerprint = model.fingerprint() if hasattr(model, "fingerprint") else \
            content_hash({"model_id": mid, "version": mver})

        key = (mid, mver)
        if key in self._entries:
            existing = self._entries[key]
            if existing.fingerprint != fingerprint:
                raise ModelVersionError(
                    f"{mid}@{mver} is already registered with a different "
                    "fingerprint. A registered version is immutable; create a new "
                    "version instead so that earlier results remain reproducible."
                )
            return existing

        entry = RegistryEntry(
            model_id=mid, model_version=mver, task=task, stage=stage,
            created_at=utcnow(), manifest=manifest, fingerprint=fingerprint,
            model=model, notes=notes,
        )
        self._entries[key] = entry
        self._order.append(key)
        if self.audit_hook:
            self.audit_hook({"action": "model_registered", **entry.as_dict()})
        return entry

    def get(self, model_id: str, model_version: Optional[str] = None) -> RegistryEntry:
        if model_version is None:
            versions = [k for k in self._order if k[0] == model_id]
            if not versions:
                raise ModelVersionError(f"no registered model {model_id!r}")
            model_version = versions[-1][1]
        key = (model_id, model_version)
        if key not in self._entries:
            raise ModelVersionError(
                f"model {model_id}@{model_version} is not registered; have "
                f"{[k for k in self._order if k[0] == model_id]}"
            )
        return self._entries[key]

    def all(self) -> list[RegistryEntry]:
        return [self._entries[k] for k in self._order]

    def versions(self, model_id: str) -> list[str]:
        return [k[1] for k in self._order if k[0] == model_id]

    def by_stage(self, stage: ModelStage) -> list[RegistryEntry]:
        return [e for e in self.all() if e.stage == stage]

    def live_eligible(self) -> list[RegistryEntry]:
        return self.by_stage(ModelStage.LIVE_ELIGIBLE)

    # -- promotion --------------------------------------------------------

    def promote(
        self,
        model_id: str,
        target: ModelStage,
        model_version: Optional[str] = None,
        actor: str = "system",
        validation: Optional[ValidationRecord] = None,
        **kwargs: Any,
    ) -> RegistryEntry:
        """Advance a model one stage, enforcing preconditions."""
        e = self.get(model_id, model_version)
        # A lock freezes the model's *definition*, not its lifecycle position.
        # Locking is a prerequisite for LIVE_ELIGIBLE, so refusing that final
        # promotion because the model is locked would make live eligibility
        # unreachable. Retiring is always permitted for the same reason.
        if e.locked and target not in (ModelStage.RETIRED, ModelStage.LIVE_ELIGIBLE):
            raise ModelPromotionError(
                f"{model_id}@{e.model_version} is locked for production; its version "
                "cannot be changed. Register a new version and validate it."
            )
        if target not in PROMOTION_ORDER:
            raise ModelPromotionError(f"unknown stage {target}")
        current_idx = PROMOTION_ORDER.index(e.stage)
        target_idx = PROMOTION_ORDER.index(target)
        if target_idx != current_idx + 1:
            raise ModelPromotionError(
                f"cannot promote {model_id}@{e.model_version} from {e.stage.value} "
                f"to {target.value}: a model advances one stage at a time through "
                f"{' -> '.join(s.value for s in PROMOTION_ORDER)}. Skipping "
                "validation or approval is precisely the failure mode this ladder "
                "exists to prevent."
            )

        if target == ModelStage.VALIDATED:
            if validation is None:
                raise ModelPromotionError(
                    "promotion to VALIDATED requires a ValidationRecord. Training "
                    "is not validation; a model must be evaluated on data it never "
                    "saw before it can be considered validated."
                )
            e.validations.append(validation)
            if not validation.passed:
                raise ModelPromotionError(
                    f"validation did not pass: "
                    f"{[k for k, v in validation.checks.items() if not v['passed']]}"
                )
        elif target == ModelStage.APPROVED:
            if not any(v.passed for v in e.validations):
                raise ModelPromotionError(
                    "promotion to APPROVED requires a passed validation record"
                )
            e.approved_by = kwargs.get("approved_by", actor)
            e.approved_at = utcnow()
        elif target == ModelStage.PAPER_TRADING:
            if not e.approved_by:
                raise ModelPromotionError(
                    "promotion to PAPER_TRADING requires an explicit approval"
                )
        elif target == ModelStage.LIVE_ELIGIBLE:
            missing = self._live_prerequisites(e)
            if missing:
                raise ModelPromotionError(
                    f"{model_id}@{e.model_version} is not live-eligible: {missing}. "
                    "Live trading additionally requires a signed authorisation "
                    "record from quantforge.execution.live_gate; this registry "
                    "cannot grant trading permission."
                )
            e.live_authorisation_id = kwargs.get("authorisation_id")

        e.stage = target
        if target == ModelStage.RETIRED:
            e.retired_at = utcnow()
        if self.audit_hook:
            self.audit_hook({
                "action": "model_promoted", "from": None, "to": target.value,
                "model_id": model_id, "model_version": e.model_version,
                "actor": actor, "at": utcnow().isoformat(),
            })
        return e

    def _live_prerequisites(self, e: RegistryEntry) -> list[str]:
        missing: list[str] = []
        if not any(v.passed for v in e.validations):
            missing.append("no passed validation")
        if not e.approved_by:
            missing.append("no recorded approval")
        if not e.paper_trading_log:
            missing.append("no paper-trading record")
        if not e.locked:
            missing.append("version is not locked")
        if not any(s.get("authorised") for s in e.paper_trading_log):
            missing.append("paper trading was never authorised")
        return missing

    def record_paper_trading(
        self, model_id: str, session: dict, model_version: Optional[str] = None
    ) -> RegistryEntry:
        e = self.get(model_id, model_version)
        e.paper_trading_log.append({**session, "at": utcnow().isoformat()})
        if self.audit_hook:
            self.audit_hook({
                "action": "paper_trading_recorded", "model_id": model_id,
                "model_version": e.model_version, "session": session,
            })
        return e

    def lock(self, model_id: str, model_version: Optional[str] = None) -> RegistryEntry:
        """Freeze this version. Required before live eligibility."""
        e = self.get(model_id, model_version)
        if e.stage not in (ModelStage.APPROVED, ModelStage.PAPER_TRADING,
                           ModelStage.LIVE_ELIGIBLE):
            raise ModelPromotionError(
                f"a model must be approved before it is locked; {model_id}@"
                f"{e.model_version} is at {e.stage.value}"
            )
        e.locked = True
        if self.audit_hook:
            self.audit_hook({"action": "model_locked", "model_id": model_id,
                             "model_version": e.model_version})
        return e

    def retire(self, model_id: str, model_version: Optional[str] = None,
               reason: str = "") -> RegistryEntry:
        e = self.get(model_id, model_version)
        e.stage = ModelStage.RETIRED
        e.retired_at = utcnow()
        e.notes = (e.notes + " | " if e.notes else "") + f"retired: {reason}"
        if self.audit_hook:
            self.audit_hook({"action": "model_retired", "model_id": model_id,
                             "model_version": e.model_version, "reason": reason})
        return e

    # -- drift ------------------------------------------------------------

    def record_drift(self, model_id: str, drift_type: DriftType, detail: dict,
                     model_version: Optional[str] = None) -> RegistryEntry:
        """Log a drift event. Never triggers an automatic retrain or deploy.

        Blueprint Section 71 is explicit: detection raises an alert for a human.
        Auto-retraining on drift tends to lock in whatever regime caused the
        drift, which is the opposite of what is wanted.
        """
        e = self.get(model_id, model_version)
        e.drift_events.append({
            "drift_type": drift_type.value, "detail": detail,
            "at": utcnow().isoformat(),
            "action_taken": "alerted; no automatic retrain or deployment",
        })
        if self.audit_hook:
            self.audit_hook({"action": "drift_detected", "model_id": model_id,
                             "model_version": e.model_version,
                             "drift_type": drift_type.value, "detail": detail})
        return e

    def as_dict(self) -> dict:
        return {
            "criteria": self.criteria.as_dict(),
            "n_models": len(self._entries),
            "ladder": [s.value for s in PROMOTION_ORDER],
            "models": [e.as_dict() for e in self.all()],
            "counts_by_stage": {
                s.value: len(self.by_stage(s)) for s in PROMOTION_ORDER
            },
        }


# ==========================================================================
# Drift detection
# ==========================================================================


@dataclass(slots=True)
class DriftResult:
    drift_type: DriftType
    score: float
    threshold: float
    drifted: bool
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "drift_type": self.drift_type.value, "score": self.score,
            "threshold": self.threshold, "drifted": self.drifted, "detail": self.detail,
        }


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    """PSI between a reference and a current sample.

    Conventionally: < 0.1 no meaningful shift, 0.1-0.25 moderate, > 0.25
    significant. The reference must be the *training* distribution; comparing
    two out-of-sample windows measures regime change, not model degradation.
    """
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if len(ref) < bins or len(cur) < 2:
        return float("nan")
    edges = np.unique(np.quantile(ref.to_numpy(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = np.histogram(ref.to_numpy(), bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur.to_numpy(), bins=edges)[0].astype(float)
    ref_p = ref_counts / max(ref_counts.sum(), 1)
    cur_p = cur_counts / max(cur_counts.sum(), 1)
    # Floor both to avoid a log(0) on an empty bin.
    eps = 1e-6
    ref_p = np.clip(ref_p, eps, None)
    cur_p = np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def detect_feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.25
) -> list[DriftResult]:
    """Per-column PSI between a training reference and a live sample."""
    out: list[DriftResult] = []
    for col in reference.columns:
        if col not in current.columns:
            continue
        psi = population_stability_index(reference[col], current[col])
        out.append(DriftResult(
            drift_type=DriftType.FEATURE_DRIFT, score=psi, threshold=threshold,
            drifted=bool(np.isfinite(psi) and psi > threshold),
            detail={"column": col,
                    "reference_median": float(pd.Series(reference[col]).median()),
                    "current_median": float(pd.Series(current[col]).median())},
        ))
    return out


def detect_prediction_drift(
    reference_scores: np.ndarray, current_scores: np.ndarray,
    threshold: float = 0.25,
) -> DriftResult:
    """PSI on the model's own output distribution.

    A large shift means the model is being asked to score inputs unlike
    anything it was trained on. That is a reason to investigate, not a reason
    to assume the model has become better or worse.
    """
    psi = population_stability_index(
        pd.Series(np.asarray(reference_scores).ravel()),
        pd.Series(np.asarray(current_scores).ravel()),
    )
    return DriftResult(
        drift_type=DriftType.PREDICTION_DRIFT, score=psi, threshold=threshold,
        drifted=bool(np.isfinite(psi) and psi > threshold),
        detail={"reference_mean": float(np.mean(reference_scores)),
                "current_mean": float(np.mean(current_scores))},
    )


def detect_calibration_drift(
    reference: pd.DataFrame, current: pd.DataFrame,
    probability_column: str, target_column: str,
    threshold: float = 0.05,
) -> DriftResult:
    """Brier-score deterioration between two out-of-sample windows."""
    def brier(frame):
        p = frame[probability_column].to_numpy(dtype=float)
        y = frame[target_column].to_numpy(dtype=float)
        return float(np.mean((p - y) ** 2))

    r, c = brier(reference), brier(current)
    delta = c - r
    return DriftResult(
        drift_type=DriftType.CALIBRATION_DRIFT, score=delta, threshold=threshold,
        drifted=bool(delta > threshold),
        detail={"reference_brier": r, "current_brier": c,
                "interpretation": "a rising Brier score means predicted "
                                  "probabilities are less accurate, not that the "
                                  "model is losing money"},
    )
