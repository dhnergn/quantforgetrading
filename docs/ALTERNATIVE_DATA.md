# Fundamental, Macro, and Alternative Data

Every dataset in this platform has a way to look correct while leaking the
future. These are the four:

| Trap | Looks like | Actually is |
|---|---|---|
| Using the latest fundamentals revision | "the current value of net income" | a number that did not exist yet |
| Using the final revised macro figure | "US GDP for 2024 Q2" | the number published in 2026 |
| Timestamping news by ingestion | "an article about the 15th" | an article that reached us earlier |
| Reading a regime label in its own bar | "the market was high-volatility" | a label computed using that bar's close |

Every store here makes those four impossible rather than merely discouraged.

**No vendor connector is included.** The layer is complete; the feed is the
operator's.

---

## Fundamentals

```python
from quantforge.data.fundamentals import FundamentalStore, FundamentalObservation

store = FundamentalStore()
store.add(FundamentalObservation(
    "EQ001", "net_income", 1.0e9, "2024-Q3",
    period_end=datetime(2024, 9, 30),
    publication_date=datetime(2024, 11, 1),
    source="my_vendor",
))

store.get_value("EQ001", "net_income", "2024-10-01")   # None - not published yet
store.get_value("EQ001", "net_income", "2024-12-01")   # 1.0e9
```

### Three dates, not one

| Field | Question |
|---|---|
| `period_end` | when did the period the figure describes end? |
| `publication_date` | when was it released? |
| `ingested_as_of` | when did *we* get it? |

`available_at` is `max(publication_date, ingested_as_of)`. The second term
matters: a figure backfilled into the database in 2024 was not available to a
decision made in 2023, and the store does not pretend otherwise.

### Revisions are immutable and both are kept

```python
store.add(obs)                                     # revision 0, published Nov
store.add(obs_restated)                            # revision 1, published Mar

store.get_value("EQ001", "net_income", "2024-12-01")   # revision 0
store.get_value("EQ001", "net_income", "2025-04-01")   # revision 1
```

A query picks the newest value **available at that time**, not the newest
filed. Re-adding a revision with different contents raises, because
overwriting history destroys the ability to reproduce a result made against the
earlier value.

`store.audit_timing()` reports long publication lags and restatements. It flags;
it does not correct.

### Metrics

Thirty named metrics across valuation, liquidity, leverage, stability,
profitability, growth, size, and the three statements. Each carries its
**unit**, because storing a yield as 0.15 in one place and 15 in another
produces silently different screens:

```python
FUNDAMENTAL_METRICS["dividend_yield"]   # {"unit": "fraction"}
FUNDAMENTAL_METRICS["pe_ratio"]         # {"unit": "ratio"}
```

Derived: `altman_z_score` (2.6 model, non-manufacturers), the 1968
manufacturing variant, `free_cash_flow` (sign-agnostic on capex),
`piotroski_f_score` (0-9, requires `total_assets` for the accruals test).

---

## Macro

```python
from quantforge.data.alternative import MacroStore, MacroObservation

store.add(MacroObservation("GDP", period, 3.1, publication_date=..., 
                           scheduled_release=...))
store.value_at("GDP", "2024-09-01")     # the vintage published then
```

### Vintage-aware

Macro is revised, sometimes years later. `value_at` returns the figure as it
was **known then**. Using the final revised number is the most flattering form
of macro look-ahead, and it is the one every naive macro backtest does.

### The release calendar matters

`scheduled_release` is when the number was **scheduled**. The information that
it was coming then was public weeks earlier, so a strategy that acts on GDP in
the week before the release is trading on the schedule. A backtest that ignores
the schedule systematically overstates macro strategies.

```python
store.audit_calendar_coverage()
# {"severity": "warning", "series_id": "GDP",
#  "detail": "N/M observations have no scheduled release date. A strategy using
#             this series may appear to trade on data before it was public, and
#             there is no way to check."}
```

A macro strategy with no release calendar cannot be evaluated honestly.

---

## News and sentiment

```python
from quantforge.data.alternative import NewsStore, NewsItem

store.add(NewsItem("n1", ("EQ001",), published_at=..., sentiment=0.6,
                   sentiment_source="dictionary", relevance=0.8))
store.daily_sentiment("EQ001", start, end)
```

| | |
|---|---|
| day | sentiment | n_articles | std |
|---|---|---|---|
| 2024-03-01 | 0.40 | 2 | 0.28 |
| 2024-03-02 | -0.40 | 1 | 0.00 |

`available_at` is **publication**, not ingestion. A vendor that timestamps by
ingestion silently moves every article earlier in the sample, which flatters
any strategy trading on news.

**The article count is reported beside the mean**, because a mean of 0.9 from
one article is not a signal and a mean of 0.0 from forty is conflicting news.
The mean alone hides that difference completely.

A sentiment score requires both a bounded range (`[-1, 1]`) and a **named
method**. A score with no provenance cannot be compared across items or
reproduced.

**Sentiment is a weak signal.** Social and news sentiment is measured on
platforms with selection bias, bot activity, and changing user bases. News
sentiment reflects the tone of coverage, which responds to events rather than
predicting them. Treat these as features to validate, not information to
trust.

---

## Analyst revisions

```python
from quantforge.data.alternative import aggregate_revisions

aggregate_revisions(revisions, as_of, lookback_days=90, min_brokers=2)
```

`min_brokers` exists because a single analyst's upgrade is an opinion, not
consensus. Below the threshold, `consensus` is `None` and the reason is stated:

```python
# "only 1 broker(s) contributed, below the required 5; not consensus"
```

The **revision** carries more information than the level: a raise from an
already-expensive rating is a different event from an upgrade. Analyst
revisions are widely followed and correspondingly arbitraged, so the historical
edge attributed to them has largely been competed away.

---

## What is verified vs. asserted

| Claim | How |
|---|---|
| `available_at` is publication, not period end | `test_available_at_is_publication_not_period_end` |
| A query before the revision returns the original | `test_query_before_the_revision_returns_the_original` |
| Revisions are immutable | `test_revisions_are_immutable` |
| Macro honours the vintage | `test_vintage_is_respected` |
| A missing release calendar is flagged | `test_release_calendar_absence_is_flagged` |
| News availability is publication, not ingestion | `test_available_at_is_publication` |
| Sentiment needs a method | `test_sentiment_requires_a_method` |
| The article count is reported | `test_daily_sentiment_reports_the_article_count` |
| Consensus needs a broker minimum | `test_consensus_requires_a_broker_minimum` |
| Regime labels are never knowable in their own bar | `test_available_at_is_strictly_after_the_label_bar` |
| Volatility thresholds are expanding | `test_volatility_reference_is_expanding` |
| Warmup is unlabelled, not guessed | `test_warmup_is_unlabelled_not_guessed` |
