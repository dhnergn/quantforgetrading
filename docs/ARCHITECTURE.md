# Architecture

## Domain boundaries

Each domain is a package with a narrow public surface. The boundaries are
enforced by import direction, not by convention.

```
            ┌────────────┐
            │    API     │  HTTP surface, auth, rate limits
            └─────┬──────┘
      ┌───────────┼───────────────┬──────────────┐
      ▼           ▼               ▼              ▼
┌──────────┐ ┌──────────┐  ┌────────────┐  ┌────────────┐
│ STRATEGY │ │   ML     │  │ VALIDATION │  │ EXECUTION  │
└────┬─────┘ └────┬─────┘  └─────┬──────┘  └─────┬──────┘
     │            │              │               │
     └──────┬─────┴──────────────┘               │
            ▼                                    ▼
     ┌─────────────┐                     ┌──────────────┐
     │  FEATURES   │◀────────────────────│   BROKER     │
     └──────┬──────┘   (point-in-time)   └──────────────┘
            │
            ▼
     ┌─────────────┐        ┌──────────┐        ┌──────────┐
     │    DATA     │───────▶│   QUANT  │◀───────│  BACKTEST │
     └─────────────┘        └──────────┘        └──────────┘
            ▲
            │
     ┌─────────────┐
     │    RISK     │  consulted by backtest and execution, never by strategy
     └─────────────┘
```

**Allowed directions.** The data layer knows nothing about strategies. The
feature store is the only path from data to signals, and it will not return a
value that was unavailable at the requested time. The risk engine is consulted
by the backtest and execution layers; strategies can request a size but cannot
obtain one that bypasses a limit.

**Forbidden.** The execution layer never reads raw data. The strategy layer
never calls a broker. The feature store never applies a transformation fitted
outside its declared training window.

---

## Component map

### `core/` — foundations

| Module | Responsibility |
|---|---|
| `timestamps.py` | `TemporalEnvelope`, `DecisionChain`, DST-correct session bounds |
| `instrument.py` | Multi-asset instrument abstraction, point-in-time registry, universes |
| `config.py` | Sectioned configuration with range validation at construction |
| `reproducibility.py` | Canonical hashing, experiment manifests, `set_seed` |
| `errors.py` | Typed hierarchy distinguishing "skip this row" from "abort the run" |
| `enums.py` | Every shared enumeration |

### `data/` — ingestion and integrity

| Module | Responsibility |
|---|---|
| `corporate_actions.py` | Splits, dividends, ticker changes; factors applied at read time |
| `quality.py` | Validators producing `DataQualityReport`; trading calendars |
| `calendars.py` | Multi-exchange session specs, holidays, timezone conversion |
| `simulator.py` | GARCH + Student-t synthetic market data |

### `quant/` — the quantitative library

| Module | Responsibility |
|---|---|
| `indicators.py` | 23 versioned indicators with self-verified metadata |
| `returns.py` | Simple/log/cumulative/total returns with an explicit price basis |
| `statistics.py` | Dispersion, association, shape, conditioning, outlier flags |
| `regression.py` | QR-based OLS, regularised variants, factor and cross-sectional |
| `risk_metrics.py` | VaR, ES/CVaR, drawdown, risk-adjusted ratios |

### `features/` — point-in-time features

| Module | Responsibility |
|---|---|
| `store.py` | `FeatureSet`, `PointInTimeFeatureStore`, `assert_no_leakage` |
| `engineering.py` | Feature construction with explicit availability boundaries |
| `frame.py` | Convenience frame builder for backtests |

### `backtest/` — simulation

| Module | Responsibility |
|---|---|
| `engine.py` | The event-driven loop and decision chain |
| `portfolio.py` | Cash, positions, trades, returns |
| `costs.py` | Commission schedules and slippage models |
| `liquidity.py` | Participation caps, partial fills, gap-aware stop fills |
| `sizing.py` | Five sizing methods with provenance and clamps |

### `validation/` — falsification

| Module | Responsibility |
|---|---|
| `splits.py` | Temporal splits, walk-forward, Monte Carlo, grid, sensitivity, regime |
| `metrics.py` | Full metric set with NaN for undefined |
| `overfitting.py` | Deflated Sharpe, reality check, diagnostic flags |

### `ml/` — models

| Module | Responsibility |
|---|---|
| `models.py` | Logreg, RF, GB, MLP, clustering, with importance and calibration |
| `preprocess.py` | Leakage-guarded transformers and pipelines |
| `registry.py` | Enforced promotion ladder and drift detection |

### `risk/` — authority

| Module | Responsibility |
|---|---|
| `limits.py` | Limit evaluation with a fixed check order |
| `kill_switch.py` | Scoped switches, fail-closed, no auto-release |

### `execution/` — orders and adapters

| Module | Responsibility |
|---|---|
| `brokers/base.py` | The `BrokerAdapter` interface |
| `brokers/paper.py` | Pessimistic simulated adapter |
| `orders.py` | Order model, state machine, idempotency |
| `reconciliation.py` | Internal vs broker comparison |
| `live_gate.py` | Signed authorisation enforcement |
| `paper_trading.py` | Bar-by-bar session reusing the live engines |

---

## The decision chain

Every backtest tick and every live order passes through the same five clocks:

```
information_cutoff  the instant the decision's inputs became available
        │
        ▼
prediction          a model's output timestamp
        │
        ▼
decision            when the strategy chose
        │
        ▼
order               when the order was submitted
        │
        ▼
execution           when it was filled
```

`DecisionChain.validate()` raises on any non-monotonic sequence. A violation
means the backtest is invalid, so it aborts rather than warns.

The engine also separates a **signal timestamp** from a **fill timestamp**. A
signal computed from bar `T`'s close cannot fill at bar `T`'s close, because
that close is only known once the bar ends. Orders fill on `T+1`, at the open
plus modelled impact.

---

## Data flow with an example

For a daily-bar US equity on 2024-06-17:

| Stage | Clock | Value |
|---|---|---|
| Bar closes | 20:00 UTC | `close = 184.20` |
| Feature availability | 21:00 UTC | `rsi_14` computed, now queryable |
| Signal | 21:00 UTC | `sma_20 > sma_50` and a fresh cross |
| Risk check | 21:00 UTC | position 12%, limit 20% → approved |
| Order | 21:00 UTC | submitted with a deterministic idempotency key |
| Fill | 2024-06-18 13:30 UTC | next session's open, plus impact and spread |

A query for `rsi_14` at 20:30 UTC on 2024-06-17 returns **2024-06-14's**
value, not 2024-06-17's. That is the guarantee, and it is tested.

---

## Reproducibility

`ExperimentManifest` records everything needed to reproduce a run:

```
experiment_type, dataset_version, feature_set_version, strategy_version,
model_version, parameters, random_seed, transaction_costs, slippage,
universe, time_period, risk_configuration, preprocessing
```

`compute_hash()` produces a stable `config_hash` over these, excluding
wall-clock timestamps and absolute paths. The same configuration against the
same immutable dataset reproduces the same result; the backtest result object
carries the hash, so a run can be located by what it was rather than when it
happened.

---

## Extension points

| To add | Implement | Register |
|---|---|---|
| A data vendor | A loader producing normalized OHLCV + a corporate-action feed | Not needed; the data layer is source-agnostic |
| A broker | `BrokerAdapter` subclass | `ExecutionConfig.broker_adapter` |
| A feature | `FeatureSpec` + a series | `FeatureSet.register` |
| An indicator | `Indicator` subclass with `METADATA` | `indicators.INDICATORS` |
| A strategy | `Strategy` with a declarative `entry_rule` | `StrategyRegistry` |
| A model | `QuantModel` subclass | `ModelRegistry` |

New asset classes need a `ContractSpec` and a session spec, not changes to the
backtest engine. The engine is asset-agnostic by construction.
