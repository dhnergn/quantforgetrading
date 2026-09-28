"""Point-in-time feature store.

This is the single most important module for look-ahead prevention. Everything
else can be wrong; if this returns a value that did not exist at the decision
time, the backtest is fiction.

Two invariants:

**Retrospective availability.** Every value carries ``available_at``. A query
for decision time ``T`` returns only values with ``available_at <= T``. A
feature computed from a bar that closed at ``T`` is *not* available at ``T`` —
the close is known only after the session ends. Callers express intent by
passing an explicit decision timestamp that already accounts for this.

**Provenance.** Every value records its source, calculation method, and
feature-set version. A feature without provenance cannot be audited, and an
unauditable feature cannot be trusted in a validation result.

The store deliberately does not cache "the latest value" without its
``available_at``. A naive ``feature_store.get(instrument, name)`` returning the
most recent row is the exact pattern that makes a backtest leak.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.errors import FeatureError, FeatureUnavailableError
from quantforge.core.reproducibility import content_hash
from quantforge.core.timestamps import UTC, ensure_aware, utcnow


# --------------------------------------------------------------------------
# Feature identity and values
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Definition of a feature, independent of any particular value."""

    name: str
    category: str                       # price | technical | fundamental | macro | sentiment | regime
    source: str                         # where the inputs came from
    calculation_method: str             # formula or reference
    inputs: tuple[str, ...] = ()
    window: Optional[int] = None
    publication_time: str = "session_close"   # when the value first exists
    availability_lag: dt.timedelta = dt.timedelta(0)
    units: str = ""
    dtype: str = "float64"
    version: str = "1.0.0"
    higher_is_better: Optional[bool] = None
    #: Set when the feature's value at bar T is only known after bar T+1.
    requires_next_bar: bool = False
    notes: str = ""

    def fingerprint(self) -> str:
        return content_hash({
            "name": self.name, "category": self.category, "source": self.source,
            "calculation_method": self.calculation_method, "inputs": list(self.inputs),
            "window": self.window, "publication_time": self.publication_time,
            "availability_lag": self.availability_lag, "version": self.version,
        })

    def as_dict(self) -> dict:
        return {
            "name": self.name, "category": self.category, "source": self.source,
            "calculation_method": self.calculation_method, "inputs": list(self.inputs),
            "window": self.window, "publication_time": self.publication_time,
            "availability_lag_seconds": self.availability_lag.total_seconds(),
            "units": self.units, "version": self.version,
            "requires_next_bar": self.requires_next_bar, "notes": self.notes,
            "fingerprint": self.fingerprint(),
        }


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """One observation of one feature for one instrument."""

    instrument_id: str
    feature_name: str
    timestamp: dt.datetime          # the bar/event the value describes
    available_at: dt.datetime       # when it could first have been used
    value: float
    feature_version: str = "1.0.0"
    source: str = ""
    quality: str = "ok"

    def as_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "feature_name": self.feature_name,
            "timestamp": self.timestamp.isoformat(),
            "available_at": self.available_at.isoformat(),
            "value": self.value,
            "feature_version": self.feature_version,
            "source": self.source,
            "quality": self.quality,
        }


@dataclass(slots=True)
class FeatureSet:
    """A versioned collection of feature definitions."""

    version: str
    specs: dict[str, FeatureSpec] = field(default_factory=dict)
    created_at: dt.datetime = field(default_factory=utcnow)
    description: str = ""

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.name in self.specs:
            existing = self.specs[spec.name]
            if existing.fingerprint() != spec.fingerprint():
                raise FeatureError(
                    f"feature {spec.name!r} is already registered in set "
                    f"{self.version} with a different definition. Create a new "
                    "feature-set version instead of redefining a feature "
                    "in place; a redefined feature makes historical results "
                    "irreproducible."
                )
            return existing
        self.specs[spec.name] = spec
        return spec

    def register_many(self, specs: Iterable[FeatureSpec]) -> None:
        for s in specs:
            self.register(s)

    def get(self, name: str) -> FeatureSpec:
        try:
            return self.specs[name]
        except KeyError:
            raise KeyError(
                f"feature {name!r} not in feature set {self.version}; "
                f"available: {sorted(self.specs)}"
            ) from None

    def by_category(self, category: str) -> list[FeatureSpec]:
        return [s for s in self.specs.values() if s.category == category]

    def names(self) -> list[str]:
        return sorted(self.specs)

    def fingerprint(self) -> str:
        return content_hash({n: s.fingerprint() for n, s in sorted(self.specs.items())})

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, name: object) -> bool:
        return name in self.specs

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "fingerprint": self.fingerprint(),
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "features": {n: s.as_dict() for n, s in sorted(self.specs.items())},
        }


# --------------------------------------------------------------------------
# Availability inference
# --------------------------------------------------------------------------


def infer_available_at(
    bar_timestamp: dt.datetime,
    publication_time: str = "session_close",
    lag: dt.timedelta = dt.timedelta(0),
    bar_duration: Optional[dt.timedelta] = None,
) -> dt.datetime:
    """When a value computed from a bar first becomes usable.

    The critical case is ``session_close``. A close price for a session that
    ends at 20:00 UTC is not available at 20:00 — it becomes available at
    20:00 *plus* the time to receive and process it. Defaulting this to the bar
    timestamp is the most common and most damaging shortcut in backtesting: it
    lets a strategy act on a close it could not have known.

    ``bar_duration`` matters for intraday data, where the close of a 5-minute
    bar at 14:35 is available at 14:35 only in the sense that the bar *ended*
    then; a decision made at 14:35 cannot use it. The duration is added so the
    caller gets the boundary right.
    """
    t = ensure_aware(bar_timestamp)
    if publication_time in ("session_close", "bar_close", "end_of_bar"):
        base = t + (bar_duration or dt.timedelta(0))
    elif publication_time in ("immediate", "bar_open", "start_of_bar"):
        base = t
    elif publication_time == "next_session_open":
        base = t + dt.timedelta(days=1)
    else:
        base = t
    return base + lag


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


class PointInTimeFeatureStore:
    """In-memory store with strict temporal retrieval.

    Backed by a single tidy frame indexed by
    ``(instrument_id, feature_name, available_at)`` so that a point-in-time
    query is a range scan on the availability column — the only column that
    determines leakage.
    """

    def __init__(self, feature_set: Optional[FeatureSet] = None) -> None:
        # `feature_set or FeatureSet(...)` would be wrong: FeatureSet defines
        # __len__, so a freshly-built (but valid, still-to-be-populated) set
        # is falsy and gets replaced by an empty placeholder. Specs registered
        # afterwards then land in an object the store does not hold, and every
        # write fails with "feature not in feature set".
        self.feature_set = feature_set if feature_set is not None else FeatureSet(
            version="empty"
        )
        self._rows: list[FeatureValue] = []
        self._frame: Optional[pd.DataFrame] = None
        self._dirty = True

    # -- writes -----------------------------------------------------------

    def write(
        self,
        instrument_id: str,
        feature_name: str,
        timestamps: pd.DatetimeIndex,
        values: Sequence[float],
        available_at: Optional[pd.DatetimeIndex | Sequence] = None,
        quality: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
    ) -> int:
        """Write a feature series.

        ``available_at`` defaults to :func:`infer_available_at` on the spec,
        which is the safe path. Passing it explicitly is required only when the
        caller has better information (e.g. a vendor's stated publication lag).
        """
        spec = self.feature_set.get(feature_name)
        if len(timestamps) != len(values):
            raise FeatureError(
                f"{feature_name}: {len(timestamps)} timestamps vs {len(values)} values"
            )
        if available_at is None:
            avail = [infer_available_at(ts, spec.publication_time,
                                        spec.availability_lag) for ts in timestamps]
        elif isinstance(available_at, pd.DatetimeIndex):
            if len(available_at) != len(timestamps):
                raise FeatureError("available_at length mismatch")
            avail = list(available_at)
        else:
            avail = list(available_at)
            if len(avail) != len(timestamps):
                raise FeatureError("available_at length mismatch")

        if quality is None:
            quals = ["ok"] * len(values)
        elif isinstance(quality, str):
            quals = [quality] * len(values)
        else:
            quals = list(quality)

        src = source or spec.source
        for ts, av, val, q in zip(timestamps, avail, values, quals):
            if not np.isfinite(val):
                # A NaN is not a value. Recording it as available would let a
                # downstream imputation treat "not computable" as "observed
                # zero", which is precisely the substitution the missing-data
                # engine exists to prevent.
                q = "missing" if q == "ok" else q
                continue
            self._rows.append(
                FeatureValue(
                    instrument_id=instrument_id,
                    feature_name=feature_name,
                    timestamp=ensure_aware(ts),
                    available_at=ensure_aware(av),
                    value=float(val),
                    feature_version=spec.version,
                    source=src,
                    quality=q,
                )
            )
        self._dirty = True
        return len(self._rows)

    def write_frame(
        self,
        instrument_id: str,
        features: pd.DataFrame,
        available_at: Optional[pd.Series] = None,
    ) -> int:
        """Write a wide frame of features for one instrument."""
        n = 0
        for col in features.columns:
            if col not in self.feature_set:
                continue
            av = available_at if available_at is not None else None
            n += self.write(instrument_id, col, features.index,
                            features[col].to_numpy(), available_at=av)
        return n

    def write_value(self, value: FeatureValue) -> None:
        self._rows.append(value)
        self._dirty = True

    # -- reads ------------------------------------------------------------

    def _materialise(self) -> pd.DataFrame:
        if not self._dirty and self._frame is not None:
            return self._frame
        if not self._rows:
            self._frame = pd.DataFrame(
                columns=["instrument_id", "feature_name", "timestamp",
                         "available_at", "value", "feature_version", "source", "quality"]
            ).set_index(["instrument_id", "feature_name", "available_at"])
            self._dirty = False
            return self._frame
        df = pd.DataFrame([r.as_dict() for r in self._rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
        df = df.sort_values(["instrument_id", "feature_name", "available_at"],
                            kind="stable")
        df = df.set_index(["instrument_id", "feature_name", "available_at"])
        self._frame = df
        self._dirty = False
        return df

    def get_point_in_time(
        self,
        instrument_id: str,
        decision_time: dt.datetime,
        feature_names: Optional[Sequence[str]] = None,
        default: Optional[float] = None,
        strict: bool = False,
    ) -> dict[str, FeatureValue | float | None]:
        """The most recent value of each feature available at ``decision_time``.

        This is the query a research process must use. It deliberately does not
        accept a "give me the latest value" mode; the absence of that mode is
        the point.
        """
        T = ensure_aware(decision_time)
        df = self._materialise()
        out: dict[str, FeatureValue | float | None] = {}
        names = list(feature_names) if feature_names else self.feature_set.names()
        for name in names:
            if name not in self.feature_set:
                out[name] = default
                continue
            try:
                sub = df.loc[(instrument_id, name)]
            except KeyError:
                # No rows at all for this (instrument, feature). In strict mode
                # this must raise, otherwise a caller doing structural typing on
                # the result gets None and dereferences it.
                if strict:
                    raise FeatureUnavailableError(
                        f"{instrument_id}/{name}: no observations in the store at "
                        f"all, so nothing is available at {T}"
                    )
                out[name] = default
                continue
            if isinstance(sub, pd.Series):
                sub = sub.to_frame().T
            eligible = sub[sub.index <= T]
            if eligible.empty:
                if strict:
                    raise FeatureUnavailableError(
                        f"{instrument_id}/{name}: no value available at {T}. "
                        "The feature has not been computed for any bar at or "
                        "before this decision time."
                    )
                out[name] = default
                continue
            row = eligible.iloc[-1]
            if strict:
                # `available_at` is an index level, not a column, so it must be
                # read from the index rather than by label.
                out[name] = FeatureValue(
                    instrument_id=instrument_id, feature_name=name,
                    timestamp=row["timestamp"].to_pydatetime(),
                    available_at=eligible.index[-1].to_pydatetime(),
                    value=float(row["value"]), feature_version=row["feature_version"],
                    source=row["source"], quality=row["quality"],
                )
            else:
                out[name] = float(row["value"])
        return out

    def get_vector(
        self,
        instrument_ids: Sequence[str],
        decision_time: dt.datetime,
        feature_names: Sequence[str],
        missing_policy: str = "nan",
    ) -> pd.DataFrame:
        """Wide cross-section of features for many instruments at one instant."""
        if missing_policy not in ("nan", "raise", "drop"):
            raise FeatureError(f"unknown missing_policy: {missing_policy!r}")
        rows = {}
        for iid in instrument_ids:
            vals = self.get_point_in_time(iid, decision_time, feature_names)
            if missing_policy == "raise":
                absent = [k for k, v in vals.items() if v is None]
                if absent:
                    raise FeatureUnavailableError(
                        f"{iid} at {decision_time}: missing {absent}. A model must "
                        "not silently treat an unavailable feature as zero."
                    )
            rows[iid] = {k: (np.nan if v is None else v) for k, v in vals.items()}
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "instrument_id"
        if missing_policy == "drop":
            df = df.dropna(how="any")
        return df

    def get_history(
        self,
        instrument_id: str,
        feature_name: str,
        start: Optional[dt.datetime] = None,
        end: Optional[dt.datetime] = None,
    ) -> pd.Series:
        """Full series of a feature, for display and for label construction.

        ``end`` filters on ``available_at``, not on ``timestamp``, so the
        returned series is what was actually knowable over that window.
        """
        df = self._materialise()
        try:
            sub = df.loc[(instrument_id, feature_name)]
        except KeyError:
            return pd.Series(dtype="float64", name=feature_name)
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        sub = sub.sort_index()
        if end is not None:
            sub = sub[sub.index <= ensure_aware(end)]
        if start is not None:
            sub = sub[sub.index >= ensure_aware(start)]
        out = pd.Series(sub["value"].to_numpy(), index=sub["timestamp"], name=feature_name)
        return out

    def availability_time(
        self, instrument_id: str, feature_name: str
    ) -> Optional[dt.datetime]:
        """When a feature was first available. Used to test leakage."""
        df = self._materialise()
        try:
            sub = df.loc[(instrument_id, feature_name)]
        except KeyError:
            return None
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        if sub.empty:
            return None
        return sub.index.min().to_pydatetime()

    # -- auditing ---------------------------------------------------------

    def audit_leakage(
        self, instrument_id: str, feature_name: str,
        as_of: dt.datetime,
    ) -> dict:
        """Report what a given decision time would have exposed.

        Used by the leakage test suite and by the Data Quality dashboard. It
        returns the winning row's ``available_at`` so a test can assert the
        relationship explicitly instead of trusting the query.
        """
        T = ensure_aware(as_of)
        vals = self.get_point_in_time(instrument_id, T, [feature_name], strict=True)
        v: FeatureValue = vals[feature_name]  # type: ignore[assignment]
        return {
            "instrument_id": instrument_id,
            "feature_name": feature_name,
            "decision_time": T.isoformat(),
            "available_at": v.available_at.isoformat(),
            "value_timestamp": v.timestamp.isoformat(),
            "value": v.value,
            "no_leakage": v.available_at <= T,
        }

    def as_frame(self) -> pd.DataFrame:
        return self._materialise().reset_index()

    def instruments(self) -> list[str]:
        df = self._materialise()
        return sorted(df.index.get_level_values(0).unique().tolist()) if len(df) else []

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return (
            f"PointInTimeFeatureStore(set={self.feature_set.version!r}, "
            f"features={len(self.feature_set)}, values={len(self._rows)})"
        )


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------


@dataclass(slots=True)
class LeakageReport:
    clean: bool
    findings: list[dict] = field(default_factory=list)
    n_checks: int = 0

    def add(self, finding: dict) -> None:
        self.clean = False
        self.findings.append(finding)

    def summary(self) -> str:
        if self.clean:
            return f"Leakage audit: clean across {self.n_checks} checks."
        lines = [f"Leakage audit: {len(self.findings)} of {self.n_checks} checks FAILED"]
        for f in self.findings[:20]:
            lines.append(f"  - {f.get('check')}: {f.get('detail')}")
        lines.append(
            "A backtest with this leakage produces numbers that look plausible "
            "and are worthless. Fix the pipeline before interpreting any result."
        )
        return "\n".join(lines)


def assert_no_leakage(
    store: PointInTimeFeatureStore,
    instrument_ids: Sequence[str],
    feature_names: Sequence[str],
    decision_times: Sequence[dt.datetime],
    max_lead: dt.timedelta = dt.timedelta(0),
) -> LeakageReport:
    """Assert every queried value was available by its decision time.

    This is the machine-checkable form of blueprint Section 6's requirement
    that the backtest engine enforce temporal causality. Raises nothing; it
    reports, so a test suite can assert ``report.clean`` and a run can print
    the findings.
    """
    report = LeakageReport(clean=True)
    for iid in instrument_ids:
        for fname in feature_names:
            for T in decision_times:
                report.n_checks += 1
                try:
                    audit = store.audit_leakage(iid, fname, T)
                except FeatureUnavailableError:
                    continue
                lead = ensure_aware(T) - ensure_aware(dt.datetime.fromisoformat(
                    audit["available_at"]))
                if audit["available_at"] > ensure_aware(T).isoformat() or lead < -max_lead:
                    report.add({
                        "check": "point_in_time_retrieval",
                        "instrument_id": iid,
                        "feature": fname,
                        "decision_time": ensure_aware(T).isoformat(),
                        "available_at": audit["available_at"],
                        "detail": (
                            f"feature {fname} was available at "
                            f"{audit['available_at']} but was returned for a decision "
                            f"at {ensure_aware(T).isoformat()}"
                        ),
                    })
    return report


def assert_temporally_causal(
    info_cutoff: dt.datetime,
    prediction: Optional[dt.datetime] = None,
    decision: Optional[dt.datetime] = None,
    order: Optional[dt.datetime] = None,
    execution: Optional[dt.datetime] = None,
) -> None:
    """Enforce the monotonic decision chain. Raises on violation."""
    from quantforge.core.timestamps import DecisionChain

    DecisionChain(
        information_cutoff=info_cutoff, prediction=prediction,
        decision=decision, order=order, execution=execution,
    ).validate()
