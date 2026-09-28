# Corporate Actions

## Raw data is never modified

Every adjustment is a factor applied at read time. The raw observation remains
exactly as ingested, so any adjustment can be audited, corrected, or redone when
a vendor revises history.

```python
res = apply_adjustments(raw_prices, actions)
res.raw              # unchanged, byte-for-byte
res.adjusted_price   # split-restated
res.adjusted_total_return   # split + dividend credited
res.describe()       # full account of what was applied
```

`res.describe()` always states the distinction:

> `adjusted_price` restates history for splits only, so it remains comparable
> to a traded price. `adjusted_total_return` additionally credits cash
> dividends, so it is a total-return series and is NOT a price series. Never
> use the total-return series as a tradable price; never use the split-adjusted
> series to measure total return.

Confusing the two is a common and expensive error.

---

## The split-factor direction

The factor applies to bars **strictly before** the ex-date. This is the
direction that makes the series continuous, and getting it backwards is the
single most common corporate-action bug.

2-for-1 split, raw price stepping 100 → 50:

| bar | raw | factor | adjusted |
|---|---|---|---|
| pre-ex-date | 100 | 2.0 | 50 |
| ex-date | 50 | 1.0 | 50 |

One pre-split share became two post-split shares, each trading at half the old
price, so the *old* bar is the one that must be restated.

Applying the factor from the ex-date forward instead divides the already-halved
price by two again, producing a **−75% move at every split**. This was a real
bug in this codebase, caught by `test_split_adjusted_returns_contain_no_jump`.

```python
split_factor_series(actions, index)   # factor 2.0 for index < ex_date
```

---

## Dividends

Per-share dividend on each ex-date bar, expressed in the same basis as the
price series.

When a split occurs *after* a dividend's ex-date, the quoted amount is on the
pre-split basis and must be divided by the factor:

| dividend ex-date | split ex-date | quoted | stored |
|---|---|---|---|
| after the split | before | $1.00 (post-split) | $1.00 |
| before the split | after | $1.00 (pre-split) | $0.50 |

Omitting this division overstates every post-split dividend by the split ratio
and manufactures a large fake return on the ex-date.

```python
total_return = total_return_from_raw(prices, actions)
dividend_cash = dividend_cashflow(actions, index, position_size)
```

`dividend_cashflow` multiplies per-share dividend by shares held. Omitting it
biases every dividend-paying strategy downward by roughly its yield, and the
bias is largest for the high-yield names a value strategy would hold.

---

## Announcement timing

A split is known at its **announcement**, not its ex-date. The register is
point-in-time aware:

```python
register.known_at("X", datetime(2024, 5, 1))    # []    - not announced
register.known_at("X", datetime(2024, 5, 20))   # [split] - announced 5/15
```

Using an action before its announcement lets a backtest restate history with
knowledge the market did not have. `available_at` uses
`announced_date or ex_date`.

An action with `announced_date > ex_date` is rejected at construction, as is a
non-positive split factor or a negative dividend.

---

## Revision handling

Registering a second action for the same `(instrument, ex_date, type)`
**supersedes** the earlier record rather than being applied on top of it:

```python
register.add(action_revision)   # replaces, does not compound
```

Applying a revision on top of the original double-counts the adjustment, which
is far harder to detect than a missing one.

---

## Validation

```python
validate_actions(actions)
# ['X: dividend jumped 12.4x on 2024-03-15 (0.25 -> 3.10). Verify this is not
#  a special dividend mislabelled as a regular one or a currency error.']

reconcile_price_jump(prices, threshold=0.4)
# [{'timestamp': ..., 'return': -0.5, 'direction': 'down',
#   'suggestion': 'check for a split/reverse-split; a -50% move with a
#                  matching 2-for-1 corporate action is a data artefact,
#                  not a loss'}]
```

These **flag, never correct**. A genuine 50% crash and an unadjusted 2-for-1
split are identical in the price data. Only a human with source access can tell
them apart, and a platform that silently picks one is making an unrecorded
decision.

---

## Survivorship and delistings

`InstrumentRow` retains:

```python
delisted_date, delisting_reason, status
```

`InstrumentRegistry.universe_at(date, include_delisted=True)` returns every
instrument alive on a date, including those that failed later. A universe built
only from today's survivors is the most common and most severe form of
look-ahead bias in equity research.

```python
registry.universe_at(date(2008, 6, 1))   # includes firms that failed in 2009
```

Symbol changes are modelled too, since a ticker can be reassigned:

```python
registry.resolve("XYZ", as_of=date(2015, 1, 1))   # the entity that owned it then
```

Ambiguity (a ticker reused across two non-overlapping entities) resolves by
interval; simultaneous overlap raises, because that is a data error.

---

## Supported action types

`SPLIT`, `REVERSE_SPLIT`, `CASH_DIVIDEND`, `SPECIAL_DIVIDEND`,
`STOCK_DIVIDEND`, `RIGHTS_OFFERING`, `SPINOFF`, `Merger`, `TICKER_CHANGE`,
`DELISTING`, `NAME_CHANGE`.

Splits and cash dividends are fully implemented. Spinoffs and rights
offerings are recorded in the register but require a custom adjustment
function, since their price effect depends on the specific terms.
