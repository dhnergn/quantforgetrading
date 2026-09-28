# Risk Engine

The risk engine has authority over the strategy engine. It returns a verdict,
not advice, and a strategy has no mechanism to override it.

```python
verdict = risk_engine.check_order(portfolio, "AAPL", "long", 500, 150.0)
if not verdict.approved:
    order.status = OrderStatus.REJECTED
    order.reject_reason = verdict.reason
```

## Check order

Limits are evaluated in a fixed order, cheapest and most binding first, so a
rejection reports the reason that matters rather than the first to trip:

1. per-order notional
2. single-name concentration
3. sector concentration
4. asset-class concentration
5. gross exposure
6. net exposure
7. leverage
8. projected drawdown
9. daily loss
10. portfolio volatility
11. VaR / CVaR

Every limit is checked against the **post-fill** state. A limit that an order
cannot breach is not a limit.

## Verdict

```python
RiskVerdict(
    approved=False,
    risk_level=RiskLevel.RED,
    violations=[{limit_type, limit, observed, instrument_id, detail}],
    warnings=[{metric, observed, limit, detail}],
    utilisation={position_pct, gross_exposure_pct, net_exposure_pct,
                 leverage, drawdown_pct, daily_loss_pct},
    reason="2 risk limit(s) breached: order notional 20,000 exceeds 10.0% ..."
)
```

A warning band at 80% of each limit gives an advance signal before a breach:

```python
if verdict.warnings:
    risk_level = RiskLevel.YELLOW
```

## Clamping

`clamp_quantity` binary-searches for the largest size that passes every limit:

```python
qty = risk_engine.clamp_quantity(portfolio, "AAPL", "long", 5000, 150.0)
```

It returns 95% of the boundary, not the boundary itself. Binary search finds
the largest *passing* size, which sits infinitesimally below the limit; since
the order is sized at bar `T`'s close and filled at bar `T+1`'s open, any
adverse gap then pushes it over and the order is rejected every time. Sizing to
95% of the boundary trades a sliver of position size for orders that fill.

## Limits are validated at construction

```python
RiskLimits(max_position_pct=0.0).validate()
# ValueError: max_position_pct = 0.0 is outside (0, 1]. A limit of zero or
# negative value would reject every order.
```

---

## Position sizing

Five methods, each with stated assumptions. All are bounded by the risk engine
afterwards.

| Method | Assumption | Notes |
|---|---|---|
| `FIXED_FRACTIONAL` | None | Risk capped by construction; survives being wrong |
| `VOLATILITY_ATR` | Volatility proxies return dispersion | Fails on asymmetric loss distributions |
| `VOLATILITY_HV` | Same | Scale-free |
| `KELLY` | Estimated edge is near-true | Most sensitive to input error |
| `RISK_BASED` | Stop distance is meaningful | Connects size to an invalidation point |

```python
res = sizer.size(equity=100_000, price=50.0, atr=2.0)
res.quantity, res.risk_amount, res.capped_by, res.notes
```

### Kelly is bounded for a reason

Full Kelly maximises expected growth and maximises drawdown simultaneously;
drawdown at half-Kelly is roughly half, and at quarter-Kelly roughly a quarter.
The default is `kelly_fraction=0.25` with `max_kelly_weight=0.10`, because the
error in the win-probability estimate exceeds the arithmetic's precision.

A non-positive Kelly fraction takes **no position**:

```python
res = sizer.kelly(100_000, 50, win_probability=0.3, payoff_ratio=0.5)
res.quantity == 0.0
res.capped_by == "negative_kelly"
```

Missing or invalid inputs fall back to fixed-fractional **visibly**, with a
note recorded, rather than silently extrapolating.

---

## Stops

`StopType`: `PERCENTAGE`, `ATR`, `TRAILING_ATR`, `TRAILING_PERCENT`,
`SUPPORT`, `TIME`, `VOLATILITY`, `NONE`.

ATR is the default because a percentage stop ignores the instrument's own
volatility: 5% is a very different proposition on a 1%-daily-volatility mega-cap
and a 6%-volatility small cap.

**A stop is a trigger, not a fill price.** The engine models the gap between
trigger and achievable fill:

```python
if side == "long" and bar["low"] <= stop_price:
    fill = simulate_fill_price(..., order_type="stop", stop_price=stop_price)
    # If the bar gapped through, fill is the OPEN, not the stop.
```

Stop losses are also an **exit rule** in the strategy, so an exit is
`signal → risk → order → fill` like any other. A stop evaluated against the
bar close and filled at that close is look-ahead bias.

---

## Kill switches

```python
ks = KillSwitch(KillSwitchConfig(max_daily_loss_pct=0.03, max_drawdown_pct=0.15))
ks.evaluate(equity=..., daily_start_equity=..., peak_equity=...)
ks.trading_state()   # NORMAL | HALTED_KILL_SWITCH | ...
```

**Scopes:** `GLOBAL`, `PORTFOLIO`, `STRATEGY`, `BROKER`.

**Automatic triggers:** daily loss, drawdown, stale data, execution latency,
consecutive order rejections, reconciliation failures, portfolio volatility.

**Fail-closed.** `is_engaged()` with no context returns `True` for any active
switch. The previous behaviour required a context match, which meant a
portfolio switch engaged for `PF1` left the platform reporting `NORMAL` to a
caller that had not yet identified its portfolio — a halted portfolio looked
tradable.

**No auto-release.** An automatically-engaged switch stays engaged until a
human clears it. Silently resuming after a kill switch is how a system re-enters
a market that has just rejected it several times.

**No flapping.** An identical repeated trigger updates a counter rather than
creating duplicate events, which would flood the audit log.

```python
for _ in range(5):
    ks.evaluate(equity=90_000, daily_start_equity=100_000, portfolio_id="P")
len(ks.active_events)   # 1
```

---

## VaR and CVaR

Four methods, each reporting **confidence, horizon, method, and dataset**:

```python
v = historical_var(returns, confidence=0.95, horizon_periods=1, dataset="oos")
v.value, v.confidence, v.horizon_periods, v.method, v.dataset, v.caveat
```

| Method | Assumption | Limitation |
|---|---|---|
| `HISTORICAL` | None | Depends entirely on the sample's tail |
| `PARAMETRIC` | Normal or Student-t | Normal understates fat tails |
| `MONTE_CARLO` | The generating distribution | Simulation adds no tail information |

Properties enforced and tested:

- losses are positive, always
- ES ≥ VaR by construction
- VaR increases with confidence
- Student-t exceeds normal at high confidence for fat-tailed input
- multi-period scaling is `sqrt(horizon)`, with the assumption stated
- insufficient samples return `NaN` with the count, not a confident number
- constant series returns `NaN` with an explanation

---

## Risk-adjusted ratios

Undefined is `NaN`, never zero:

```python
sharpe_ratio(returns)   # 0 volatility -> NaN + "Zero dispersion is not
                        #                     evidence of a riskless strategy"
sortino_ratio(returns)  # no MAR breach   -> NaN + explanation
profit_factor(pnls)     # no losers       -> NaN + "a small-sample artefact"
calmar_ratio(equity)    # no drawdown     -> NaN
```

The Sharpe guard uses a scale-relative epsilon, not exact zero: repeated
summation of a constant leaves residue around `1e-18`, and dividing by that
overflows to `1e16` rather than triggering a zero check.

---
