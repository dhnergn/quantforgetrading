# Execution

## The broker abstraction

`BrokerAdapter` is the only broker interface the platform knows. No
broker-specific behaviour is permitted above it.

```python
class BrokerAdapter(abc.ABC):
    mode: ExecutionMode
    name: str

    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def get_account(self) -> Account: ...
    def get_positions(self) -> list[BrokerPosition]: ...
    def get_order(self, client_order_id) -> Optional[dict]: ...
    def get_orders(self, status=None) -> list[dict]: ...
    def get_fills(self, since=None) -> list[Fill]: ...
    def submit_order(self, order) -> OrderAck: ...
    def cancel_order(self, client_order_id) -> bool: ...
    def replace_order(self, order, new_quantity, new_limit_price=None) -> OrderAck: ...
    def reconcile_unknown_orders(self, order_ids) -> dict: ...
```

**No broker adapter ships with the platform.** `PaperBrokerAdapter` is
provided and functional for backtest and paper-trading realism. Connecting a
real venue is left to the operator; see [LIVE_TRADING.md](LIVE_TRADING.md).

---

## Three kinds of communication

| Kind | Semantics |
|---|---|
| **Query** | Returns data or raises |
| **Command** | Returns an *acknowledgement*, which is not a fill |
| **Event** | Arrives asynchronously, reconciled into the order book |

An `OrderAck` with `accepted=True` means the broker took responsibility for the
order. The order is then live: it may fill partially, fill fully, or be rejected
later. Treating an acknowledgement as a fill is the second most common cause of
a position mismatch after double submission.

---

## Order state machine

```
CREATED ──► SUBMITTED ──► ACCEPTED ──► PARTIALLY_FILLED ──► FILLED
                │             │              │  ▲
                │             │              │  └── (repeatable)
                ├─────────────┴──────────────┴──► CANCELLED / REJECTED / EXPIRED
                │
                └────────────────────────────────► UNKNOWN
                                                       │
                       only after a broker query ─────┴──► ACCEPTED / FILLED /
                                                              CANCELLED / REJECTED
```

`UNKNOWN` is a first-class state and the most important transition rule in the
platform:

```python
sm.transition(OrderStatus.UNKNOWN, reason="submit timeout")
sm.transition(OrderStatus.SUBMITTED)
# ExecutionStateError: illegal transition unknown -> submitted
```

**The response to a submission timeout is reconciliation, never a retry.**

### Why this matters

```
submit ──► timeout ──► ??? ──► resubmit ──► double fill
```

A system that treats a timeout as a failure and resubmits will eventually hold
twice the intended position. `UNKNOWN` makes that state unrepresentable as a
success, and the state machine refuses to move from it back to `SUBMITTED`.

### Other enforced rules

- terminal states are final (a filled order cannot later appear unfilled)
- an overfill raises: a broker overfill is a reconciliation failure
- a "partial" fill covering the whole quantity is promoted to `FILLED`
- `information_cutoff` defaults to the **decision** time, not creation time

---

## Idempotency

Every order carries a deterministic key derived from the logical decision:

```python
make_idempotency_key(strategy_id, portfolio_id, instrument_id,
                     side, quantity, when)
```

Deterministic rather than random, so a retry from a timeout handler, a process
restart, or a duplicate webhook produces the same key and is recognisable as
the same order.

Three layers of protection:

1. **In-memory** `OrderBook.check_duplicate` returns the existing order and
   increments `duplicates_blocked`.
2. **Broker-side** a real venue rejects a duplicate client order id.
3. **Durable** `OrderRepository.by_idempotency_key` survives a restart:

```python
with session_scope(url) as s:
    existing = OrderRepository(s).by_idempotency_key("k1")
    if existing is not None:
        # do not submit
```

---

## Reconciliation

```python
report = PositionReconciler().reconcile_portfolio(portfolio, broker.get_positions())
report.matched            # False
report.state              # TradingState.REQUIRES_RECONCILIATION
```

**The broker is the source of truth for quantities**, because the broker's
records are what a counterparty will settle against. The alternative — trusting
internal state because it is the one the strategy reasoned about — is how a
platform ends up confident about a position it does not hold.

### Classification

| Situation | Severity | Meaning |
|---|---|---|
| Both hold a position, quantities differ | critical | A partial fill is unaccounted for on one side |
| Only internal holds it | critical | A fill was recorded that never happened |
| Only the broker holds it | critical | A manual trade, or a lost acknowledgement |
| Quantities agree, entry prices differ | warning | Usually a partial-fill accounting difference |

### Corrections are explicit

```python
rec = PositionReconciler()
report = rec.reconcile_portfolio(portfolio, broker_positions)
# internal state is UNTOUCHED
rec.apply_corrections(portfolio, report, actor="risk_manager")
```

Automatic correction is off by default. A quantity difference has a *cause* —
a missed fill, a manual trade, a corporate action — and overwriting the
internal record destroys the evidence of why they diverged. Every correction
records the previous value.

### Unmapped broker states escalate

```python
# base.py reconcile_unknown_orders
if state not in recognised:
    out[cid] = {"resolved": False, "action": "escalate",
                "detail": "the broker reported status 'X', which this platform "
                          "does not map to a known state. It is not safe to "
                          "assume the order is live or that it is not."}
```

Defaulting an unrecognised status to `ACCEPTED` would convert an unknown
outcome into a live order — the double-fill failure the whole path exists to
prevent.

---

## Paper broker

Deliberately pessimistic:

| Behaviour | Modelling |
|---|---|
| Market order | Next bar's open plus impact |
| Limit above market | Fills immediately as marketable (as a real venue behaves) |
| Limit outside the range | Does not fill |
| Order above participation cap | Partial fill, or rejection if disabled |
| Stop gapped through | Fills at the open, not the stop |
| Submission failure | Configurable reject or timeout, to exercise the safety paths |

```python
broker = PaperBrokerAdapter(
    initial_equity=100_000.0,
    fill_probability=0.95,          # limit orders are not certain
    participation_cap=0.05,
    timeout_probability=0.0,        # set to 1.0 to test reconciliation
    reject_probability=0.0,
)
```

---

## Paper trading

Uses the **same** strategy, feature, risk, portfolio, and order engines as live
trading. Only the execution adapter differs. Blueprint Section 58 requires this,
and the only way to guarantee it is a single implementation; a separate paper
path would drift, and the drift would surface at the worst moment.

The loop per bar:

```
market data ─► strategy ─► risk engine ─► order ─► adapter fill ─►
portfolio update ─► signal log ─► latency log ─► reconciliation
```

Every stage the live path has is present, including the uncomfortable ones:
staleness detection, kill switches, risk vetoes, and reconciliation.

Replay drives the staleness check from the **bar clock**, not wall time:

```python
PaperTradingConfig(simulated_clock=True)
```

Without it, a 2020 bar is decades stale by the wall clock and the session
trips the staleness switch on bar one and trades nothing.

---

## Latency

Recorded as **distributions**, not averages. A pipeline with a 5ms mean and a 2s
p99 will occasionally miss a trading window, and an average hides that.

```python
tracker.summary()["stages"]["execution"]
# {"n": 500, "median_s": 0.182, "p95_s": 0.68, "p99_s": 0.747, "max_s": 0.83}
```

Stages follow the decision chain: `market_data` → `feature` → `model` →
`signal` → `risk` → `order_submission` → `broker_response` → `execution`.

---

## Signal and execution logs

`SignalLogger` records every signal with its feature values, model prediction
and version, regime, risk verdict, and resulting order, so "why was this order
placed?" is answerable afterwards.

`ExecutionLogger` records the full order lifecycle: signal → risk → order →
broker response → fill → final position.

```python
executor.for_order(order_id)   # every stage, in order
```

Both write JSONL, which is greppable, appendable without a database, and
crash-tolerant: a truncated final line is recoverable where a half-written
database transaction may not be.

---

## Alerts

Deduplicated by `(category, subject)`, so a flickering feed produces one alert
with an occurrence count rather than thousands:

```python
for _ in range(5):
    alerts.raise_alert("stale_data", "feed silent", "critical", subject="AAPL")
alerts.counts()   # {"stale_data:AAPL": 5} - one distinct alert
```

Categories: `broker_disconnection`, `data_feed_failure`, `stale_data`,
`reconciliation_failure`, `order_rejection`, `risk_limit_breach`,
`unexpected_position`, `model_failure`, `strategy_crash`, `database_failure`,
`excessive_latency`, `kill_switch`, `order_state_unknown`,
`duplicate_order_blocked`.
