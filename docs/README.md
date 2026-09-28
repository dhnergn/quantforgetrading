# QuantForge AI — Documentation Index

**Read [RISK_DISCLOSURE.md](RISK_DISCLOSURE.md) before using any output.**

No strategy, indicator, or model here produces guaranteed profits. A
profitable backtest is evidence about the tested historical period under the
stated assumptions, not a prediction of future returns.

---

## Foundations

| Document | Covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Domain boundaries, component map, decision chain, extension points |
| [TIMESTAMPS.md](TIMESTAMPS.md) | `TemporalEnvelope`, availability, DST, calendars, staleness |
| [LOOKAHEAD_BIAS.md](LOOKAHEAD_BIAS.md) | The five sources, the five defences, the machine check |
| [DATA_LEAKAGE.md](DATA_LEAKAGE.md) | Stateful preprocessing, PCA, imputation, splits, embargo |
| [DATA_QUALITY.md](DATA_QUALITY.md) | Validators, coverage, anomaly classification |

## Data

| Document | Covers |
|---|---|
| [CORPORATE_ACTIONS.md](CORPORATE_ACTIONS.md) | Splits, dividends, announcements, survivorship |
| [FEATURE_STORE.md](FEATURE_STORE.md) | Point-in-time storage, availability, feature sets |

## Research

| Document | Covers |
|---|---|
| [BACKTESTING.md](BACKTESTING.md) | The event loop, fills, costs, portfolio accounting |
| [VALIDATION.md](VALIDATION.md) | Splits, walk-forward, Monte Carlo, sensitivity, regimes |
| [OVERFITTING.md](OVERFITTING.md) | Deflated Sharpe, reality check, diagnostic flags |
| [MACHINE_LEARNING.md](MACHINE_LEARNING.md) | Model families, calibration, importance, limitations |
| [MODEL_REGISTRY.md](MODEL_REGISTRY.md) | The promotion ladder and drift |
| [STATISTICS.md](STATISTICS.md) | Degenerate-input behaviour, conditioning |

## Risk and execution

| Document | Covers |
|---|---|
| [RISK_ENGINE.md](RISK_ENGINE.md) | Limits, sizing, stops, kill switches, VaR/ES |
| [EXECUTION.md](EXECUTION.md) | Order state machine, idempotency, reconciliation, adapters |
| [LIVE_TRADING.md](LIVE_TRADING.md) | The authorisation gate and operator procedure |
| [SECURITY.md](SECURITY.md) | Auth, RBAC, audit, sandbox, injection defences |
| [TESTING.md](TESTING.md) | Test strategy, coverage, and the bugs the suite caught |

---

## Not yet written

The blueprint (sections 9-48 of `../BLUEPRINT.md`) specifies these domains;
the current implementation covers the MVP path end to end. These are the
planned documents for modules not yet built:

`MARKET_DATA.md`, `TECHNICAL_INDICATORS.md`, `FUNDAMENTALS.md`,
`MACRO_DATA.md`, `SENTIMENT.md`, `REGIME_ENGINE.md`, `REGRESSION.md`,
`MODEL_VERSIONING.md`, `STRATEGY_ENGINE.md`, `SURVIVORSHIP_BIAS.md`,
`TRANSACTION_COSTS.md`, `SLIPPAGE.md`, `LIQUIDITY.md`, `POSITION_SIZING.md`,
`VAR_CVAR.md`, `PORTFOLIO_CONSTRUCTION.md`, `WALK_FORWARD.md`,
`MONTE_CARLO.md`, `ROBUSTNESS.md`, `SENSITIVITY.md`, `PAPER_TRADING.md`,
`BROKER_API.md`, `POSITION_RECONCILIATION.md`, `KILL_SWITCH.md`,
`API_SECURITY.md`, `STRATEGY_SANDBOX.md`, `AUDIT_LOGGING.md`,
`OBSERVABILITY.md`, `DISASTER_RECOVERY.md`, `DEPLOYMENT.md`.

The topics are covered substantively in the documents above even where a
dedicated file is pending — for example stop and position-sizing behaviour is
in [RISK_ENGINE.md](RISK_ENGINE.md), and the broker interface and
reconciliation are in [EXECUTION.md](EXECUTION.md).

---

## What is implemented vs. planned

### Implemented and tested (228 tests)

- Instrument abstraction across 7 asset classes with contract specs
- `TemporalEnvelope`, `DecisionChain`, multi-exchange calendars with DST
- Corporate actions, quality validators, GARCH simulator
- 23 versioned indicators with self-verified metadata
- Returns, statistics, regression, risk metrics
- Point-in-time feature store with leakage auditing
- Declarative versioned strategies and conditions
- Event-driven backtest with costs, slippage, liquidity, sizing
- IS/OOS splits, walk-forward, Monte Carlo, grid, sensitivity, overfitting
- Logistic regression, random forest, gradient boosting, MLP, clustering
- Leakage-guarded preprocessing pipelines
- Model registry with an enforced promotion ladder
- Risk limits engine, kill switches
- Order state machine, idempotency, paper broker, reconciliation
- Live-trading authorisation gate
- Latency, signal and execution logging, alerts
- RBAC, password hashing, tokens, rate limits, audit chain, sandbox
- SQLAlchemy persistence with durable order dedupe
- FastAPI with authn/authz, BOLA prevention, job queue

### Not implemented

- Data vendor connectors (point-in-time fundamentals, macro vintages, news)
- Broker adapters beyond the simulated paper broker
- The web dashboards listed in blueprint Section 90
- Multi-asset portfolio construction and risk parity
- Options, futures roll strategies, and other asset-specific analytics
