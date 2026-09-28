"""Latency and signal logging.

Latency is recorded as distributions, not averages, because the tail is the
operational risk. A pipeline with a 5ms mean and a 2s p99 is a system that
will occasionally miss a trading window, and an average hides that entirely.

The signal log answers "why was this order placed?" for every signal: the
feature values, the model prediction and version, the regime, the risk verdict,
and the resulting order. Without it, post-trade analysis is guesswork.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.enums import OrderSide, RiskLevel
from quantforge.core.timestamps import utcnow


# ==========================================================================
# Latency
# ==========================================================================


@dataclass(slots=True)
class LatencySample:
    stage: str
    seconds: float
    at: Optional[dt.datetime] = None
    tag: str = ""

    def as_dict(self) -> dict:
        return {"stage": self.stage, "seconds": self.seconds,
                "at": self.at.isoformat() if self.at else None, "tag": self.tag}


class LatencyTracker:
    """Records per-stage latencies and reports distributions.

    Stages follow the decision chain so the breakdown is directly comparable
    against the order's information cutoff: ``market_data`` -> ``feature`` ->
    ``signal`` -> ``risk`` -> ``order_submission`` -> ``broker_response`` ->
    ``execution``.
    """

    STAGES = (
        "market_data", "feature", "model", "signal", "risk",
        "order_submission", "broker_response", "execution",
    )

    def __init__(self, sample_rate: float = 1.0, max_samples: int = 100_000) -> None:
        self.samples: list[LatencySample] = []
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self._t0: Optional[float] = None
        self._marks: dict[str, float] = {}
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(0)

    # -- timing -----------------------------------------------------------

    def now(self) -> float:
        import time

        return time.perf_counter()

    def start(self, stage: str) -> None:
        self._marks[stage] = self.now()

    def record(self, stage: str, seconds: float, at: Optional[dt.datetime] = None,
               tag: str = "") -> None:
        if len(self.samples) >= self.max_samples:
            return
        if self.sample_rate < 1.0 and self._rng.random() > self.sample_rate:
            return
        self.samples.append(LatencySample(stage, float(seconds), at, tag))

    def measure(self, stage: str, tag: str = "") -> float:
        """Record the time since :meth:`start` for ``stage``."""
        t0 = self._marks.pop(stage, None)
        if t0 is None:
            return float("nan")
        elapsed = self.now() - t0
        self.record(stage, elapsed, tag=tag)
        return elapsed

    def mark(self, stage: str, when: Optional[dt.datetime] = None,
             tag: str = "") -> None:
        """Record a stage boundary.

        Convenience for replay-based paper trading, where elapsed time is
        simulated rather than measured. It records zero so the stage is
        *present* in the distribution; a caller measuring real latency should
        use :meth:`measure` instead.
        """
        self.record(stage, 0.0, at=when, tag=tag)

    # -- reporting --------------------------------------------------------

    def frame(self) -> pd.DataFrame:
        if not self.samples:
            return pd.DataFrame(columns=["stage", "seconds", "at", "tag"])
        return pd.DataFrame([s.as_dict() for s in self.samples])

    def summary(self) -> dict:
        """Full distribution per stage.

        The mean is included for continuity but the percentiles are what
        matter operationally: p99 is the number that determines whether a
        trading window is missed.
        """
        df = self.frame()
        if df.empty:
            return {"n_samples": 0, "stages": {},
                    "note": "no latency samples were recorded"}
        out: dict[str, Any] = {}
        for stage, group in df.groupby("stage"):
            s = group["seconds"].to_numpy(dtype=float)
            finite = s[np.isfinite(s)]
            if finite.size == 0:
                continue
            out[stage] = {
                "n": int(finite.size),
                "mean_s": float(np.mean(finite)),
                "median_s": float(np.median(finite)),
                "p90_s": float(np.quantile(finite, 0.90)),
                "p95_s": float(np.quantile(finite, 0.95)),
                "p99_s": float(np.quantile(finite, 0.99)),
                "max_s": float(np.max(finite)),
                "stdev_s": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            }
        return {
            "n_samples": len(df),
            "stages": out,
            "note": (
                "Percentiles are what matter; a low mean with a high p99 is a "
                "system that will occasionally miss a trading window. In "
                "replay-based paper trading these are simulated and will read "
                "near zero."
            ),
        }

    def slowest(self, n: int = 10) -> pd.DataFrame:
        df = self.frame()
        if df.empty:
            return df
        return df.nlargest(n, "seconds").reset_index(drop=True)

    def reset(self) -> None:
        self.samples.clear()
        self._marks.clear()


class DecisionChainTimer:
    """Measures the full chain for one decision, in real time.

    Produces the :class:`~quantforge.core.timestamps.DecisionChain` needed to
    prove no stage used information from a later stage.
    """

    def __init__(self) -> None:
        self._stages: dict[str, float] = {}
        self._start = LatencyTracker()

    def stage(self, name: str):
        return _StageTimer(self, name)

    def chain(self) -> dict:
        return {
            "stage_seconds": dict(self._stages),
            "total_seconds": sum(self._stages.values()),
        }


class _StageTimer:
    def __init__(self, parent: DecisionChainTimer, name: str) -> None:
        self.parent = parent
        self.name = name

    def __enter__(self):
        import time

        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time

        self.parent._stages[self.name] = time.perf_counter() - self._t0
        self.parent._start.record(self.name, self.parent._stages[self.name])


# ==========================================================================
# Signal log
# ==========================================================================


@dataclass(slots=True)
class SignalRecord:
    signal_id: str
    timestamp: dt.datetime
    instrument_id: str
    strategy_id: str
    strategy_version: str
    signal: str
    feature_values: dict = field(default_factory=dict)
    indicator_values: dict = field(default_factory=dict)
    model_prediction: Optional[float] = None
    model_version: Optional[str] = None
    regime: Optional[str] = None
    risk_level: str = RiskLevel.GREEN.value
    risk_approved: Optional[bool] = None
    position_before: float = 0.0
    order_id: Optional[str] = None
    decision: str = ""

    def as_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "instrument_id": self.instrument_id,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "signal": self.signal,
            "feature_values": self.feature_values,
            "indicator_values": self.indicator_values,
            "model_prediction": self.model_prediction,
            "model_version": self.model_version,
            "regime": self.regime,
            "risk_level": self.risk_level,
            "risk_approved": self.risk_approved,
            "position_before": self.position_before,
            "order_id": self.order_id,
            "decision": self.decision,
        }


class SignalLogger:
    """Append-only record of every signal, with its inputs and outcome.

    Writes JSONL so the log is greppable, appendable without a database, and
    survives a crash mid-write (a truncated final line is recoverable, whereas
    a half-written database transaction may not be).
    """

    def __init__(self, path: Optional[str | Path] = None,
                 keep_in_memory: bool = True, max_records: int = 100_000) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep_in_memory = keep_in_memory
        self.max_records = max_records
        self._records: list[SignalRecord] = []
        self._seq = 0
        self._lock = threading.Lock()

    def record(
        self,
        instrument_id: str,
        strategy_id: str,
        strategy_version: str,
        signal: str,
        timestamp: Optional[dt.datetime] = None,
        feature_values: Optional[dict] = None,
        indicator_values: Optional[dict] = None,
        model_prediction: Optional[float] = None,
        model_version: Optional[str] = None,
        regime: Optional[str] = None,
        risk_level: str = RiskLevel.GREEN.value,
        risk_approved: Optional[bool] = None,
        position_before: float = 0.0,
        order_id: Optional[str] = None,
        decision: str = "",
    ) -> SignalRecord:
        import uuid

        self._seq += 1
        rec = SignalRecord(
            signal_id=uuid.uuid4().hex[:16],
            timestamp=timestamp or utcnow(),
            instrument_id=instrument_id,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            signal=signal,
            feature_values=feature_values or {},
            indicator_values=indicator_values or {},
            model_prediction=model_prediction,
            model_version=model_version,
            regime=regime,
            risk_level=risk_level,
            risk_approved=risk_approved,
            position_before=position_before,
            order_id=order_id,
            decision=decision,
        )
        if self.keep_in_memory:
            self._records.append(rec)
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records:]
        if self.path is not None:
            with self._lock:
                with self.path.open("a") as fh:
                    fh.write(json.dumps(rec.as_dict(), default=str) + "\n")
        return rec

    def records(self) -> list[dict]:
        return [r.as_dict() for r in self._records]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records())

    def for_instrument(self, instrument_id: str) -> list[dict]:
        return [r.as_dict() for r in self._records
                if r.instrument_id == instrument_id]

    def signal_counts(self) -> dict:
        counts: dict[str, int] = {}
        for r in self._records:
            counts[r.signal] = counts.get(r.signal, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._records)


# ==========================================================================
# Execution log
# ==========================================================================


class ExecutionLogger:
    """Full order lifecycle log: signal -> risk -> order -> broker -> fill.

    Satisfies blueprint Section 94: every live order's complete provenance,
    reconstructable afterwards.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = []
        self._lock = threading.Lock()

    def log(
        self,
        order_id: str,
        stage: str,
        instrument_id: str = "",
        strategy_id: Optional[str] = None,
        model_version: Optional[str] = None,
        signal: Optional[str] = None,
        feature_values: Optional[dict] = None,
        risk_state: Optional[dict] = None,
        portfolio_state: Optional[dict] = None,
        order: Optional[dict] = None,
        broker_response: Optional[dict] = None,
        fill: Optional[dict] = None,
        slippage: Optional[float] = None,
        latency: Optional[dict] = None,
        final_position: Optional[dict] = None,
        note: str = "",
    ) -> dict:
        entry = {
            "at": utcnow().isoformat(),
            "order_id": order_id,
            "stage": stage,
            "instrument_id": instrument_id,
            "strategy_id": strategy_id,
            "model_version": model_version,
            "signal": signal,
            "feature_values": feature_values or {},
            "risk_state": risk_state,
            "portfolio_state": portfolio_state,
            "order": order,
            "broker_response": broker_response,
            "fill": fill,
            "slippage": slippage,
            "latency": latency,
            "final_position": final_position,
            "note": note,
        }
        self._entries.append(entry)
        if self.path is not None:
            with self._lock:
                with self.path.open("a") as fh:
                    fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    def entries(self) -> list[dict]:
        return list(self._entries)

    def for_order(self, order_id: str) -> list[dict]:
        """Every stage for one order, in order. The reconstruction view."""
        return [e for e in self._entries if e["order_id"] == order_id]

    def frame(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame()
        return pd.DataFrame([{
            "at": e["at"], "order_id": e["order_id"], "stage": e["stage"],
            "instrument_id": e["instrument_id"], "strategy_id": e["strategy_id"],
            "note": e["note"],
        } for e in self._entries])

    def __len__(self) -> int:
        return len(self._entries)


# ==========================================================================
# Alerts
# ==========================================================================


@dataclass(slots=True)
class Alert:
    alert_id: str
    severity: str            # info | warning | critical
    category: str
    message: str
    at: dt.datetime = field(default_factory=utcnow)
    context: dict = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "alert_id": self.alert_id, "severity": self.severity,
            "category": self.category, "message": self.message,
            "at": self.at.isoformat(), "context": self.context,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
        }


class AlertManager:
    """Deduplicated alerting for operational and safety events.

    A flapping condition (a feed flickering in and out) can emit thousands of
    alerts and train the on-call to ignore them. Alerts are therefore keyed by
    ``(category, subject)`` and repeated firings update a counter rather than
    creating new entries.
    """

    CATEGORIES = {
        "broker_disconnection", "data_feed_failure", "stale_data",
        "reconciliation_failure", "order_rejection", "risk_limit_breach",
        "unexpected_position", "model_failure", "strategy_crash",
        "database_failure", "excessive_latency", "kill_switch",
        "order_state_unknown", "duplicate_order_blocked",
    }

    def __init__(self, webhook: Optional[Any] = None) -> None:
        self._alerts: dict[str, Alert] = {}
        self._counts: dict[str, int] = {}
        self.webhook = webhook

    def raise_alert(
        self, category: str, message: str, severity: str = "warning",
        subject: Optional[str] = None, **context: Any,
    ) -> Alert:
        import uuid

        if category not in self.CATEGORIES:
            # An unknown category is a bug in the caller, but dropping the
            # alert entirely would be worse; it is recorded under 'other'.
            category = "other"
        key = f"{category}:{subject or ''}"
        if key in self._alerts and not self._alerts[key].acknowledged:
            self._counts[key] = self._counts.get(key, 1) + 1
            alert = self._alerts[key]
            alert.context["occurrences"] = self._counts[key]
            return alert
        alert = Alert(
            alert_id=uuid.uuid4().hex[:12], severity=severity, category=category,
            message=message, context=context,
        )
        self._alerts[key] = alert
        self._counts[key] = 1
        if self.webhook is not None:
            try:
                self.webhook(alert.as_dict())
            except Exception:
                # A failing webhook must not prevent the alert from being
                # recorded locally.
                pass
        return alert

    def acknowledge(self, category: str, subject: Optional[str] = None,
                    by: str = "operator") -> Optional[Alert]:
        key = f"{category}:{subject or ''}"
        alert = self._alerts.get(key)
        if alert is not None:
            alert.acknowledged = True
            alert.acknowledged_by = by
        return alert

    def active(self, min_severity: str = "info") -> list[dict]:
        order = {"info": 0, "warning": 1, "critical": 2}
        floor = order.get(min_severity, 0)
        return [a.as_dict() for a in self._alerts.values()
                if not a.acknowledged and order.get(a.severity, 0) >= floor]

    def counts(self) -> dict:
        return dict(self._counts)

    def summary(self) -> str:
        if not self._alerts:
            return "No alerts."
        lines = [f"{len(self._alerts)} distinct alert(s):"]
        for key, count in sorted(self._counts.items()):
            a = self._alerts[key]
            flag = "ACK" if a.acknowledged else "OPEN"
            lines.append(f"  [{a.severity.upper():8s}] {flag} {key} x{count}: {a.message}")
        return "\n".join(lines)
