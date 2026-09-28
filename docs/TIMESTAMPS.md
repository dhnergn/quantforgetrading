# Timestamps

## The problem

A single generic timestamp cannot express four different facts, and conflating
them is the root of most look-ahead bias:

| Fact | Question | Example |
|---|---|---|
| `event_time` | When did it happen? | Q3 ended 30 September |
| `publication_time` | When did it become public? | Filed 1 November |
| `available_at` | When could a system have used it? | Lands in our store 3 May **next year** |
| `ingestion_time` | When did *we* receive it? | Vendor feed, 1 November 09:04 |

A Q3 2023 income statement has a `period_end` in 2023, a `publication_date` in
November 2023, and — in a database loaded today — an `ingested_as_of` in 2024.
Using it to make a decision in October 2023 is look-ahead bias even though
nothing about the record changed.

---

## `TemporalEnvelope`

```python
env = TemporalEnvelope(
    event_time=datetime(2023, 9, 30),
    publication_time=datetime(2023, 11, 1),
    ingestion_time=datetime(2024, 5, 3),
    ingested_as_of=datetime(2024, 5, 3),
    source="SEC EDGAR",
    revision=1,
)
env.available_at          # 2024-05-03 — the conservative answer
env.publication_lag       # 32 days
env.feed_latency          # time from exchange to our store
```

### Construction validates

```python
TemporalEnvelope(event_time=d(2024,1,10), publication_time=d(2024,1,1))
# ValueError: publication_time 2024-01-01 precedes event_time 2024-01-10;
#   a record cannot be published before the event it describes

TemporalEnvelope(..., ingestion_time=before_publication)
# ValueError: ingestion cannot precede publication
```

### `available_at` is the conservative answer

```python
available_at = max(publication_time, ingested_as_of or publication_time)
```

For forward-only research the binding constraint is publication. But a record
backfilled into 2024 was not contemporaneously available in 2023, and the
platform does not pretend otherwise. When a source tracks vintages, supply
`ingested_as_of` and the bound tightens.

---

## The `DecisionChain`

```
information_cutoff ──► prediction ──► decision ──► order ──► execution
```

Every backtest tick and every live order passes through it, and
`validate()` raises on any non-monotonic sequence.

```python
DecisionChain(
    information_cutoff=t0,
    prediction=t0 + 1s, decision=t0 + 3s,
    order=t0 + 4s, execution=t0 + 6s,
).latencies()
# {"model_latency_s": 1.0, "signal_latency_s": 2.0, "execution_latency_s": 2.0}
```

A violation aborts the run. A backtest that trips it produces plausible-looking
numbers describing a system that could not have traded, which is the most
expensive class of bug in the discipline.

For an `Order`, `information_cutoff` defaults to the **decision** time, not the
creation time: the cutoff is when the inputs were available, which is at or
before the decision. Defaulting it to a later creation time would make every
order that decides and then constructs look like a causality violation.

---

## Naive datetimes

```python
ensure_aware(dt.datetime(2024, 1, 1))   # assumed UTC, tzinfo attached
```

`datetime.utcnow` is deprecated from Python 3.12 and returns a naive value,
which invites exactly the naive/aware comparison bugs this module exists to
prevent. `utcnow()` is the only accessor.

---

## Time zones and DST

Session boundaries are **local wall-clock times**, so they must be localised
before shifting to UTC.

```python
session_bounds_utc("XNYS", date(2024, 7, 1))   # 13:30 - 20:00 UTC
session_bounds_utc("XNYS", date(2024, 1, 15))  # 14:30 - 21:00 UTC
```

The one-hour difference is daylight saving, resolved per date by the tz
database. `exchange_session_bounds` localises to the exchange zone, builds the
wall-clock bounds for that specific date, then converts. A system that hard-codes
a UTC offset is wrong for half the year.

### Bar labels

```python
bar_timestamps_to_utc(local_naive_index, "XTKS")
```

`localize`, not `tz_convert`. A bar labelled 09:30 in Tokyo is 09:30 *Tokyo
time*; converting requires knowing the DST rule for that date. Converting an
already-UTC index would double-shift it, so passing a tz-aware index raises.

---

## Calendars

```python
SESSION_SPECS        # 23 exchanges, with pre/post and early-close support
build_calendar("XNYS", start, end, holidays=us_equity_holidays(2024))
cal.is_session(date(2024, 7, 4))     # False - Independence Day
cal.previous_session(d), cal.next_session(d)
cal.session_bounds(d)                # DST-correct UTC bounds
```

Early closes are supported because a 13:00 close looks exactly like a data
outage when the code does not know about it.

`us_equity_holidays` is **approximate** and documented as such. A production
deployment must load the exchange's authoritative calendar, because a
provisional or emergency closure will silently corrupt backtests otherwise.

---

## Staleness

A feed's age is measured against the **simulated clock** in replay:

```python
PaperTradingConfig(simulated_clock=True)
```

Without it, a 2020 bar is decades stale by wall time, the staleness kill switch
trips on bar one, and the session trades nothing. A test that "passes" by
producing no trades is not a passing test.

---

## Availability by feature type

| Feature type | Available from |
|---|---|
| Daily close, indicators | session close + processing allowance |
| Intraday bar features | bar end + processing |
| Quarterly fundamentals | `publication_date` |
| Macro releases | release timestamp |
| Corporate actions | **announcement** date |
| Analyst revisions | revision timestamp |
| News sentiment | publication time, not ingestion |
| Index constituents | effective date |

Every builder returns values **and** the availability boundary. A builder that
returns values without availability information cannot be used point-in-time,
so the type makes the omission visible at the call site.

---

## Tests

`tests/test_leakage.py::TestTemporalEnvelope` and `::TestDecisionChain`:

- publication before event raises
- ingestion before publication raises
- `available_at` honours `ingested_as_of`
- `available_at` falls back to publication
- feed latency and publication lag compute correctly
- naive datetimes are coerced
- monotonic chains validate
- every non-monotonic ordering raises
- latency breakdown sums correctly
