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
| [PLATFORM_SUPPORT.md](PLATFORM_SUPPORT.md) | **Host support: Linux, macOS, Windows; mobile as client** |
| [PORTFOLIO_CONSTRUCTION.md](PORTFOLIO_CONSTRUCTION.md) | Multi-asset allocation, risk parity, risk attribution |

---

## Coverage of the blueprint

`../BLUEPRINT.md` specifies more domains than this implementation covers. Rather
than ship 31 placeholder files, the topics are listed here with where the real
coverage lives, so a reader is not sent to a missing file or misled into
thinking a topic is undocumented.

| Blueprint domain | Where it is covered | Dedicated doc |
|---|---|---|
| Market data, calendars, quality | `data/` | [DATA_QUALITY.md](DATA_QUALITY.md) |
| Technical indicators | `quant/indicators.py` | this table |
| Corporate actions, survivorship | `data/corporate_actions.py` | [CORPORATE_ACTIONS.md](CORPORATE_ACTIONS.md) |
| Feature store, availability | `features/` | [FEATURE_STORE.md](FEATURE_STORE.md) |
| Fundamentals, macro, sentiment | `core/timestamps.md` availability model | [TIMESTAMPS.md](TIMESTAMPS.md) |
| Regime detection | `ml/` clustering + `validation/splits.regime_breakdown` | this table |
| Regression, factor, cross-sectional | `quant/regression.py` | this table |
| Strategy engine | `strategy/` | this table |
| Transaction costs, slippage, liquidity | `backtest/costs.py`, `liquidity.py` | [BACKTESTING.md](BACKTESTING.md) |
| Position sizing | `backtest/sizing.py` | [RISK_ENGINE.md](RISK_ENGINE.md) |
| VaR, CVaR, drawdown | `quant/risk_metrics.py` | [RISK_ENGINE.md](RISK_ENGINE.md) |
| Portfolio construction, risk parity | `portfolio/construction.py` | [PORTFOLIO_CONSTRUCTION.md](PORTFOLIO_CONSTRUCTION.md) |
| Walk-forward, Monte Carlo, robustness, sensitivity | `validation/splits.py` | [VALIDATION.md](VALIDATION.md) |
| Overfitting, multiple testing | `validation/overfitting.py` | [OVERFITTING.md](OVERFITTING.md) |
| ML models, calibration, drift | `ml/` | [MACHINE_LEARNING.md](MACHINE_LEARNING.md) |
| Model versioning, promotion | `ml/registry.py` | [MODEL_REGISTRY.md](MODEL_REGISTRY.md) |
| Risk limits, kill switches | `risk/` | [RISK_ENGINE.md](RISK_ENGINE.md) |
| Paper trading | `execution/paper_trading.py` | [EXECUTION.md](EXECUTION.md) |
| Broker API, reconciliation | `execution/` | [EXECUTION.md](EXECUTION.md) |
| Live-trading gate | `execution/live_gate.py` | [LIVE_TRADING.md](LIVE_TRADING.md) |
| API security, RBAC, audit, sandbox | `security/` | [SECURITY.md](SECURITY.md) |
| Observability, latency, alerts | `monitoring/` | [EXECUTION.md](EXECUTION.md) |
| Cross-platform support | `core/platform.py` | [PLATFORM_SUPPORT.md](PLATFORM_SUPPORT.md) |

**Not implemented**, and stated rather than implied:

- **Data vendor connectors.** No live fundamental, macro, or news feed. The
  simulator produces synthetic data, and point-in-time fundamentals need a
  licensed source with real publication timestamps.
- **Broker adapters.** Only the simulated paper broker. Wiring a real venue is
  left to the operator, deliberately — see [LIVE_TRADING.md](LIVE_TRADING.md).
- **Web dashboards.** The FastAPI surface and PWA assets exist; no UI is built.
- **Options, futures roll, and asset-specific analytics.** The instrument layer
  supports the asset classes, but there is no chain analytics or roll logic.
- **Intraday execution realism.** No queue position, order-book depth, or
  adverse selection.

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
