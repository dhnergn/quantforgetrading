"""Explicit temporal semantics for every observation.

The single most important architectural guarantee in QuantForge is that an
event time, a publication time, and an availability time are never conflated.
This module makes that structural rather than a matter of discipline.

Three distinct questions are modelled:

``event_time``
    When the thing described by the record actually happened.

``publication_time``
    When the record was first made public by its source. A quarterly filing
    published 45 days after period end carries both times.

``availability_time``
    When *this platform* could first have used the record. For a real-time
    signal this is the decision time itself. For backfilled history it is the
    time the data landed in the store, which may be years later.

Using a backfilled record's *current* value at a historical decision time is
look-ahead bias even though nothing about the record changed. ``available_at``
is the field that prevents it, and every point-in-time query in the platform
filters on it.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import Optional

import pandas as pd

UTC = dt.timezone.utc


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def utcnow() -> dt.datetime:
    """Timezone-aware current UTC time.

    ``datetime.utcnow`` is deprecated from Python 3.12 and returns a naive
    value, which invites exactly the naive/aware comparison bugs this module
    exists to prevent.
    """
    return dt.datetime.now(UTC)


def ensure_aware(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
    """Coerce a datetime to timezone-aware UTC.

    Naive datetimes are *assumed* to be UTC, which is the convention used by
    the raw storage layer. The assumption is recorded rather than hidden.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_utc(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
    return ensure_aware(value)


def localize(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Attach an exchange timezone to a naive index, then convert to UTC.

    Session boundaries (09:30 open, 16:00 close) are defined in *local*
    exchange time, so the localizing step must happen before the UTC shift.
    """
    if index.tz is not None:
        raise ValueError(f"index already tz-aware: {index.tz}")
    return index.tz_localize(tz).tz_convert(UTC)


def exchange_session_bounds(
    ts: dt.datetime, tz: str, open_hhmm: str = "09:30", close_hhmm: str = "16:00"
) -> tuple[dt.datetime, dt.datetime]:
    """Return the UTC open/close datetimes for the exchange day containing ``ts``.

    Handles DST correctly because the local wall-clock bounds are resolved
    against the tz database for the specific date, not a fixed UTC offset.
    """
    local = ensure_aware(ts).astimezone(dt.timezone.utc).astimezone(
        _zone(tz)
    )
    day = local.date()
    o_h, o_m = (int(x) for x in open_hhmm.split(":"))
    c_h, c_m = (int(x) for x in close_hhmm.split(":"))
    open_local = dt.datetime.combine(day, dt.time(o_h, o_m), tzinfo=_zone(tz))
    close_local = dt.datetime.combine(day, dt.time(c_h, c_m), tzinfo=_zone(tz))
    return open_local.astimezone(UTC), close_local.astimezone(UTC)


_ZONE_CACHE: dict[str, dt.tzinfo] = {}


def _zone(tz: str) -> dt.tzinfo:
    cached = _ZONE_CACHE.get(tz)
    if cached is None:
        cached = _load_zone(tz)
        _ZONE_CACHE[tz] = cached
    return cached


def _load_zone(tz: str) -> dt.tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(tz)
    except Exception:  # pragma: no cover - fallback for exotic tz names
        try:
            import pytz

            return pytz.timezone(tz)
        except Exception:
            raise ValueError(f"unknown timezone: {tz!r}") from None


# --------------------------------------------------------------------------
# The temporal envelope
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalEnvelope:
    """The full set of clocks attached to one observation.

    Not every source populates every field. ``None`` means "this source does
    not distinguish", which is different from "the same as event_time" and is
    treated as such by the point-in-time layer.
    """

    event_time: dt.datetime
    publication_time: Optional[dt.datetime] = None
    effective_time: Optional[dt.datetime] = None
    ingestion_time: Optional[dt.datetime] = None
    exchange_time: Optional[dt.datetime] = None
    local_time: Optional[dt.datetime] = None
    timezone: str = "UTC"
    source: Optional[str] = None
    revision: int = 0
    ingested_as_of: Optional[dt.datetime] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_time", ensure_aware(self.event_time))
        for name in (
            "publication_time",
            "effective_time",
            "ingestion_time",
            "exchange_time",
            "local_time",
            "ingested_as_of",
        ):
            object.__setattr__(self, name, ensure_aware(getattr(self, name)))

        pt = self.publication_time
        if pt is not None and pt < self.event_time:
            raise ValueError(
                f"publication_time {pt} precedes event_time {self.event_time}; "
                "a record cannot be published before the event it describes"
            )
        it = self.ingestion_time
        if it is not None and pt is not None and it < pt:
            raise ValueError(
                f"ingestion_time {it} precedes publication_time {pt}; "
                "ingestion cannot precede publication"
            )

    # -- derived clocks ---------------------------------------------------

    @property
    def effective_or_event(self) -> dt.datetime:
        return self.effective_time or self.event_time

    @property
    def publication_or_event(self) -> dt.datetime:
        return self.publication_time or self.event_time

    @property
    def available_at(self) -> dt.datetime:
        """The earliest instant this record could have influenced a decision.

        For forward-only research the binding constraint is publication: a
        number computed today and backfilled into 2005 is not a number anyone
        could have used in 2005, but it is still the best estimate of what was
        published then. This platform's default is conservative — it honours
        ``ingested_as_of`` when the source tracks it, so a backfilled vintage
        is not silently treated as contemporaneously available.
        """
        candidates = [self.publication_or_event]
        if self.ingested_as_of is not None:
            candidates.append(self.ingested_as_of)
        return max(candidates)

    @property
    def feed_latency(self) -> Optional[dt.timedelta]:
        if self.exchange_time is None or self.ingestion_time is None:
            return None
        return self.ingestion_time - self.exchange_time

    @property
    def publication_lag(self) -> Optional[dt.timedelta]:
        if self.publication_time is None:
            return None
        return self.publication_time - self.event_time

    def shift(self, delta: dt.timedelta) -> "TemporalEnvelope":
        """Shift all populated clocks uniformly. Used for horizon tests."""
        fields = {}
        for name in (
            "event_time",
            "publication_time",
            "effective_time",
            "ingestion_time",
            "exchange_time",
            "local_time",
            "ingested_as_of",
        ):
            value = getattr(self, name)
            fields[name] = value + delta if value is not None else None
        return replace(self, **fields)

    def describe(self) -> dict:
        """Audit-friendly rendering for the signal/execution logs."""
        return {
            "event_time": _iso(self.event_time),
            "publication_time": _iso(self.publication_time),
            "effective_time": _iso(self.effective_time),
            "ingestion_time": _iso(self.ingestion_time),
            "exchange_time": _iso(self.exchange_time),
            "available_at": _iso(self.available_at),
            "timezone": self.timezone,
            "source": self.source,
            "revision": self.revision,
        }


def _iso(value: Optional[dt.datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


# --------------------------------------------------------------------------
# Decision-chain clocks
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionChain:
    """The five clocks a backtest tick must account for, in order.

    Temporal causality in the backtest engine is enforced as a monotonic
    invariant: ``information_cutoff <= prediction <= decision <= order <=
    execution``. A violation raises rather than warns, because a violation
    means the backtest is invalid.
    """

    information_cutoff: dt.datetime
    prediction: Optional[dt.datetime] = None
    decision: Optional[dt.datetime] = None
    order: Optional[dt.datetime] = None
    execution: Optional[dt.datetime] = None

    def validate(self) -> None:
        seq = [
            ("information_cutoff", self.information_cutoff),
            ("prediction", self.prediction),
            ("decision", self.decision),
            ("order", self.order),
            ("execution", self.execution),
        ]
        previous_name, previous = seq[0]
        for name, value in seq[1:]:
            if value is None:
                continue
            if value < previous:
                raise TemporalCausalityError(
                    f"{name} ({_iso(value)}) is earlier than {previous_name} "
                    f"({_iso(previous)}); the decision chain is not monotonic"
                )
            previous_name, previous = name, value

    @property
    def has_execution(self) -> bool:
        return self.execution is not None

    def latencies(self) -> dict:
        """Stage latencies in seconds for the monitoring layer."""
        out: dict[str, float] = {}
        if self.prediction is not None:
            out["model_latency_s"] = (self.prediction - self.information_cutoff).total_seconds()
        if self.decision is not None:
            base = self.prediction or self.information_cutoff
            out["signal_latency_s"] = (self.decision - base).total_seconds()
        if self.order is not None:
            base = self.decision or self.prediction or self.information_cutoff
            out["risk_latency_s"] = (self.order - base).total_seconds()
        if self.execution is not None:
            base = self.order or self.decision or self.prediction or self.information_cutoff
            out["execution_latency_s"] = (self.execution - base).total_seconds()
        return out


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class TemporalCausalityError(ValueError):
    """Raised when a decision chain or dataset violates temporal ordering.

    This is intentionally a hard failure. A backtest that trips it produces
    numbers that look plausible and are worthless, which is the single most
    expensive class of bug in quantitative research.
    """


class LookAheadBiasError(TemporalCausalityError):
    """Raised when a feature or query would read data unavailable at decision time."""
