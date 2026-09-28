# Data Quality

## Philosophy: flag, never silently fix

A research platform that quietly repairs data produces a clean backtest of a
dataset that never existed, and the researcher has no way to know. Every check
here **reports**; the caller decides.

The one exception is a defect severe enough that continuing would corrupt
downstream state, which raises.

| Severity | Meaning |
|---|---|
| `error` | The data is structurally wrong; continuing produces nonsense |
| `warning` | Suspicious; a human should look before trusting it |
| `info` | Expected condition worth recording (a holiday, a coverage statistic) |

---

## Validators

### OHLCV

```python
report = validate_ohlcv(data, "AAPL_1d", expected_freq="1D", calendar=cal)
print(report.report())
```

**Errors:** non-positive prices, `high < low`, high/low not bracketing
open/close, negative volume, duplicate timestamps, unsorted index, missing
columns.

**Warnings:** missing values above threshold, extreme moves, stale prices
(10+ identical closes), zero volume.

**Info:** coverage, date range, inferred frequency, missing sessions.

The extreme-move check catches unadjusted splits specifically:

```
[WARNING] extreme_move: 1 bars with |return| > 35%. Check for unadjusted
         splits: a 2-for-1 split prints as exactly -50%.
```

---

### Fundamentals

```python
report = validate_fundamentals(fundamentals)
```

The checks that matter most are temporal:

- a record published before its period ended is impossible
- missing publication dates make a record unusable in any point-in-time query
- long publication lags are reported with a note about short-horizon use
- negative revenue or equity is flagged; negative earnings is legitimate

---

### Sentiment

```python
report = validate_sentiment(news)
```

- ingestion before publication (a feed cannot deliver information before it
  exists)
- future-dated timestamps, indicating a clock or timezone error
- **ingestion delay** over an hour, which must exclude or lag-adjust the item
  in a historical model

---

### Macro

```python
report = validate_macro(macro)
```

- a release published before the period it describes
- revisions present but no revision history (originals unrecoverable)
- future-dated releases

---

## Anomaly classification

```python
class DataQuality(str, Enum):
    OK = "ok"
    DATA_QUALITY_ANOMALY = "data_quality_anomaly"
    MARKET_EVENT = "market_event"
    POTENTIAL_OUTLIER = "potential_outlier"
```

**Anomalies are never automatically deleted.** A 40% gap may be a feed error or
the single most informative observation in the sample. Only a human with source
access can tell them apart.

Outlier detection is for flagging, not filtering:

```python
statistics.outlier_mask(x)        # Tukey fences
statistics.mad_outlier_mask(x)     # robust to heavy tails
statistics.iqr_bounds(x)          # bounds for inspection
```

MAD is preferable to IQR for financial data because it does not break down on
the very outliers it is looking for.

---

## `DataQualityReport`

```python
report.passed          # no errors
report.errors          # blocking
report.warnings
report.as_dict()       # machine-readable
report.report()        # formatted
report.report(raise_on_error=True)   # blocks research
```

Example output:

```
========================================================================
DATA QUALITY REPORT — synthetic_ohlcv
========================================================================
Observations: 1,500
Errors: 0   Warnings: 0

COVERAGE
  volume_coverage              100.0% of bars with volume
  date_range                   2018-01-01 to 2023-09-29
  n_bars                       1500
  inferred_freq                1 days

FINDINGS
  [INFO   ] calendar_coverage: 598 fewer bars than the calendar implies
           (1500 present vs 2098 expected). Confirm these are holidays,
           not missing data.
========================================================================
Note: this report describes the data. It does not establish that any
strategy built on it will work.
```

That last line is the point of the whole exercise.

---

## Calendars

```python
build_calendar("XNYS", start, end, holidays=us_equity_holidays(2024))
cal.is_session(date(2024, 7, 4))       # False
cal.previous_session(d), cal.next_session(d)
cal.session_bounds(d)                  # DST-correct UTC bounds
cal.filter_sessions(index)
cal.is_market_open(when)
cal.next_open(when)
```

`us_equity_holidays` is **approximate** and documented as such. A production
deployment must load the exchange's authoritative calendar, because a
provisional or emergency closure (a hurricane) will silently corrupt backtests
otherwise.

---

## Where quality checks run

| Point | What is checked |
|---|---|
| Before research | Full `DataQualityReport` |
| On ingestion | Schema, ranges, duplicate detection |
| Corporate-action validation | Duplicate actions, implausible dividend jumps |
| Before a backtest | Index alignment, required columns |
| Before live trading | Reconciliation, feed-vs-broker price comparison |
| Continuously | Staleness, sequence gaps, feed latency |

---

## Known gaps

- No cross-field consistency between adjusted and unadjusted series.
- No check that the vendor's split factors match an independent source.
- No detection of a vendor silently changing its adjustment methodology.
- No survivorship check on a supplied universe (the registry supports
  point-in-time membership, but a third-party universe file is taken at face
  value).
- No corporate-action detection for spin-offs and rights offerings, whose price
  effects depend on terms that require a bespoke adjustment.
