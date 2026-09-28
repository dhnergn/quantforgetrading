# Feature Store

## The contract

`get_features(asset, decision_time)` must return **only** features that were
actually available at that time. Every value carries `available_at`, and every
query filters on it.

```python
store.get_point_in_time("EQ001", decision_time, ["close", "rsi_14"])
# {"close": 184.20, "rsi_14": 61.4}
```

The store deliberately provides **no "give me the latest value" mode**. The
absence of that method is the point: a `get(instrument, feature)` returning the
most recent row is the exact pattern that makes a backtest leak.

---

## Value model

```python
FeatureValue(
    instrument_id, feature_name,
    timestamp,        # the bar/event the value describes
    available_at,     # when it could first have been used
    value,
    feature_version, source, quality,
)
```

`timestamp` answers "what does this describe"; `available_at` answers "when
could I have used it". Conflating them is the bug.

`NaN` is never stored. A NaN is not a value, and recording it as available would
let a downstream imputation treat "not computable" as "observed zero" — exactly
the substitution the missing-data engine exists to prevent.

---

## `FeatureSpec`

Every feature declares how it is produced:

```python
FeatureSpec(
    name="rsi_14",
    category="technical",              # price | technical | fundamental |
                                      # macro | sentiment | regime
    source="price_data",
    calculation_method="RSI(14), Wilder smoothing",
    inputs=("close",),
    window=14,
    publication_time="session_close",
    availability_lag=timedelta(0),
    version="1.0.0",
    notes="... an overbought reading does not mean reversal is due",
)
```

`fingerprint()` hashes the definition, so two specs with different methods are
detected as different features even under the same name.

---

## Feature sets are immutable

```python
fs.register(FeatureSpec(name="rsi_14", ..., calculation_method="DIFFERENT"))
# FeatureError: feature 'rsi_14' is already registered in set mvp-1.0.0 with a
# different definition. Create a new feature-set version instead of redefining
# a feature in place; a redefined feature makes historical results
# impossible to reproduce.
```

Re-registering an *identical* spec is a no-op, so a builder can run
idempotently.

`fingerprint()` over the whole set gives a content hash usable as
`feature_set_version` in an experiment manifest.

---

## Point-in-time retrieval

```python
store.get_point_in_time("EQ001", T, ["close"])              # values or defaults
store.get_point_in_time("EQ001", T, ["close"], strict=True) # raises if missing
store.get_vector(instruments, T, features)                  # wide cross-section
store.get_history("EQ001", "rsi_14", start, end)            # full series
```

`get_history` filters on `available_at`, not `timestamp`, so the returned
series is what was actually knowable over the window.

`missing_policy` for cross-sections:

| Policy | Behaviour |
|---|---|
| `"nan"` | Missing features become NaN |
| `"raise"` | `FeatureUnavailableError` — a model must not treat unavailable as zero |
| `"drop"` | Rows with any missing feature are dropped |

---

## Leakage auditing

```python
store.audit_leakage("EQ001", "rsi_14", T)
# {"available_at": "...", "value_timestamp": "...", "value": 61.4,
#  "no_leakage": True}

report = assert_no_leakage(store, instruments, features, decision_times)
report.clean      # bool
report.summary()  # human-readable
```

`assert_no_leakage` raises nothing and reports everything, so a test suite can
assert `report.clean` and a run can print the findings.

A deliberately injected future value is never returned:

```python
store.write("EQ001", "close", [bar], [999.0],
            available_at=[bar + timedelta(days=365)])
store.audit_leakage("EQ001", "close", bar + 21h)["value"]   # not 999.0
```

---

## Availability inference

```python
infer_available_at(bar_timestamp, publication_time="session_close", lag=...)
```

| `publication_time` | Available from |
|---|---|
| `session_close` / `bar_close` | bar end + processing + `lag` |
| `immediate` / `bar_open` | the timestamp itself |
| `next_session_open` | + 1 day |

`session_close` is the important one. A close for a session ending at 20:00 UTC
is not available at 20:00; it is available afterwards. Defaulting availability
to the bar timestamp is the most common and most damaging shortcut in
backtesting.

For intraday data, `bar_duration` is added so the boundary is right.

---

## The MVP feature set

45 columns, all close-derived and all available at the session close plus a
processing allowance:

- **price:** open, high, low, close, volume, hl_range, true_range, gap
- **returns:** simple, log, return_lag_1/2/5, cumulative_return_20/60/252
- **moving averages:** sma_10/20/50/200, ema_12/26, wma_20
- **momentum:** rsi_14, roc_10, adx_14, macd, macd_signal, macd_hist
- **volatility:** atr_14, atr_pct, bb_upper/lower/width/pct_b,
  realized_vol_20/60
- **volume:** rel_volume_20, volume_sma_20, obv, cmf_20, vwap_20,
  vwap_distance

Each spec carries a `notes` field stating the caveat — for example
`cumulative_return_252` notes that 12-month momentum is one of the most
widely replicated anomalies and correspondingly one of the most arbitraged
away.

---

## Lags

Explicitly lagged features are first-class:

```python
"return_lag_1", "return_lag_5", "cumulative_return_20"
```

A model using `return_lag_1` is deliberately not using today's move. This
makes the lag visible in the feature name rather than in a parameter.

---

## Training frames

```python
frame = training_frame(store, "EQ001",
                       feature_names=["close", "rsi_14", "sma_20"],
                       prices=close, horizon=1)
```

Joins features on their own availability and builds the label from a *forward*
return. If dropping missing values would remove more than half the rows, it
raises rather than quietly training on a small non-representative subsample:

```python
# FeatureError: dropping missing values removed 63% of rows (200 -> 74).
# Choose imputation explicitly rather than letting an incomplete feature set
# silently shrink the sample.
```

---

## Storage notes

The store is in-memory and rebuilds a sorted frame lazily. For production:

- **Hot:** Redis or similar for low-latency recent features.
- **Cold:** Parquet on object storage, partitioned by instrument and date.
- **Versioning:** invalidation creates a new feature-set version rather than
  overwriting, so a historical result remains reproducible.
- **Corruption recovery:** clear the cache and recompute from raw data; a
  last-known-good version is preserved.
