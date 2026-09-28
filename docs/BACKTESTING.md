# Backtesting

## The event-driven loop

```
bar T   features available at T+close ─► strategy ─► signal
        risk engine check ─► order queued
bar T+1 order fills at the open, plus impact, spread, commission
        portfolio marked to market
        stop and exit rules evaluated against T+1's range
```

The **one-bar gap** is the central design decision. A signal computed from bar
`T`'s close cannot be filled at bar `T`'s close, because that close is only
known once the bar has ended.

```python
assert o.filled_bar > o.signal_bar      # every signal-driven order
```

---

## Order lifecycle in a backtest

Orders carry a full decision chain, so a reader can verify which bars a trade
depended on:

```python
OrderRecord(
    order_id, client_order_id, instrument_id, side, quantity, order_type,
    status, requested_bar, filled_bar, signal_bar,
    requested_price, fill_price, fees, slippage,
    reason, risk_level, risk_violations, fill_reason,
)
```

Statuses: `filled`, `rejected`, `cancelled`, `partially_filled`, `accepted`.

Rejections record the risk verdict, so a run with many rejections is visible
rather than silently producing fewer trades.

---

## Fill simulation

| Situation | Fill |
|---|---|
| Market order, next-bar open | open + impact |
| Market order with large participation | open + `k·√participation·range` |
| Limit inside the bar range | the limit price |
| Limit above the market (buy) | immediately, at the better price |
| Limit outside the bar range | no fill |
| Stop within the bar | the stop price |
| Stop gapped through | the **open**, not the stop |
| Order above the participation cap | partial fill, or rejection |
| Volume unknown | **no fill** |

Refusing to fill on unknown volume is deliberate: assuming infinite liquidity
is how backtests acquire unachievable returns.

```python
simulate_fill_price(side, open, high, low, close, qty, volume,
                     liquidity, order_type="stop", stop_price=95.0)
# bar gapped through: {"fill_price": 88.0, "gap_adjustment": 7.0,
#                      "reason": "stop gapped through: filled at the open ..."}
```

---

## Costs

Three components, all reported, none defaulting to zero:

```python
CostModel(
    commission=CommissionSchedule(per_share=0.005, regulatory_bps=0.0002),
    slippage=SlippageModelSpec(model=SlippageModel.PERCENTAGE,
                               pct=0.0005, half_spread_bps=2.5),
)
```

- **commission** — flat, per-share, percentage, or tiered; exchange and
  regulatory fees; per-share fees are not charged on FX or futures
- **spread** — a baseline on every fill; the *half* spread is the standard
  convention against a mid reference, and using a full spread per side doubles
  the modelled cost
- **market impact** — square-root in participation, which is the standard
  empirical shape; a linear model badly overstates small orders

```python
buy_price, est = cost_model.apply(100, 50.0, "long")
sell_price, est = cost_model.apply(100, 50.0, "short")
buy_price > 50.0 > sell_price     # always against the trader
```

A zero-cost configuration is loudly flagged:

```python
cost_model.is_free       # True
cost_model.as_dict()["warning"]
# "COSTS ARE ZERO. Any result from this configuration is an upper bound that
#  no real execution could achieve."
```

---

## Portfolio accounting

One implementation serves backtesting and paper trading, so they cannot drift.

- Realised P&L is booked at fill; unrealised is marked and is not realised.
- **Cash is explicit.** A position's notional is not equity. Leverage and idle
  cash are visible rather than assumed away.
- **Fees reduce cash**, never cost basis. Deducting them from the entry price
  makes average cost disagree with actual cash spent.
- **Partial scale-outs** do not book a trade, and the average entry price is
  preserved.
- **Reversals** split: the closed portion books a trade, the residual opens a
  fresh position at the fill price. Using the full order size would
  triple-count the shares just closed.
- **Unpriced holdings are not valued at zero.** Valuing them at zero fabricates
  a total loss and can trip a drawdown kill switch on a data gap.

```python
pf.close_all(prices, when="end")      # flattens; returns the trades booked
pf.returns()                          # index-agnostic; reconciles with equity
```

---

## Preconditions the engine refuses

### A missing feature

```python
eng = BacktestEngine(sma_crossover_strategy(5, 10), config)  # needs sma_5
eng.run(ohlcv, features_without_sma_5)
# BacktestError: strategy references features absent from the feature frame:
#   ['sma_5']. Without them every condition evaluates False and the backtest
#   would report a flat result rather than failing.
```

A zero-signal backtest is the most dangerous failure mode, because it looks
like a valid result.

### Non-overlapping indices

```python
eng.run(price_frame, feature_frame)   # indices differ
# BacktestError: price and feature indices do not overlap; the strategy would
#   have no causal inputs
```

### Empty input

```python
eng.run(empty_frame, empty_features)   # BacktestError: no bars to simulate
```

---

## Reproducibility

Every result carries a manifest and a config hash:

```python
result.manifest.compute_hash()    # stable over all inputs
result.manifest.to_dict()         # includes transaction costs, slippage,
                                  # universe, periods, risk config, seed
```

Excluded from the hash: wall-clock timestamps and absolute paths, which would
make every run unique and defeat the purpose.

```python
ExperimentManifest.verify(path)   # recompute and compare
```

---

## Reading a result

```
BACKTEST cc2e37257853
strategy:        sma_cross@1.0.0
config hash:     469b1068e4a596fa
initial capital: 100,000.00
final equity:    96,988.97
orders placed:   6 (6 filled)
trades closed:   3
total fees:      9.00
```

Then read:

- **the trade count** — 3 trades is not evidence of anything
- **the in-sample vs out-of-sample gap** — this run's Sharpe is from one period
- **the cost sensitivity** — does the edge survive a conservative assumption
- **the overfitting flags** — run `validate_overfitting` before believing it

Every summary ends with:

> A profitable result under these assumptions is evidence about the tested
> historical period only. It is not evidence of future profitability, and it
> does not account for market impact beyond the configured model.

---

## Known limitations

- Single-instrument per engine; multi-asset runs orchestrate several engines
  against a shared portfolio and risk engine.
- Intraday realism is basic: no queue position, order-book depth, or adverse
  selection.
- Position-level margin and borrow costs are not modelled; shorting assumes
  availability.
- No tax modelling.
- Dividends credit cash on the ex-date rather than the payment date, which is
  optimistic for a falling market.
