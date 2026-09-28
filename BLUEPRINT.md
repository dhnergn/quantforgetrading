# QuantForge AI — Technical Blueprint

**Version:** 1.0 (Blueprint)
**Status:** Architecture Definition — No application code written
**Date:** 2026-09-28

> **Disclaimer:** QuantForge AI is a research-grade quantitative platform. No strategy, indicator, or ML model guarantees profits or predicts markets with certainty. A profitable historical backtest is evidence about the tested historical assumptions, not proof of future profitability. Live trading is disabled by default and requires explicit authorization.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Recommended Technology Stack](#2-recommended-technology-stack)
3. [Multi-Asset Data Architecture](#3-multi-asset-data-architecture)
4. [Historical Database Architecture](#4-historical-database-architecture)
5. [Real-Time Feed Architecture](#5-real-time-feed-architecture)
6. [Timestamp Architecture](#6-timestamp-architecture)
7. [Corporate-Action Architecture](#7-corporate-action-architecture)
8. [Point-in-Time Data Architecture](#8-point-in-time-data-architecture)
9. [Feature-Store Architecture](#9-feature-store-architecture)
10. [Technical-Indicator Architecture](#10-technical-indicator-architecture)
11. [Fundamental-Data Architecture](#11-fundamental-data-architecture)
12. [Macro-Data Architecture](#12-macro-data-architecture)
13. [News-Sentiment Architecture](#13-news-sentiment-architecture)
14. [Social-Sentiment Architecture](#14-social-sentiment-architecture)
15. [Analyst-Revision Architecture](#15-analyst-revision-architecture)
16. [Market-Regime Architecture](#16-market-regime-architecture)
17. [Statistical Engine](#17-statistical-engine)
18. [Regression Architecture](#18-regression-architecture)
19. [ML Architecture](#19-ml-architecture)
20. [Random-Forest Architecture](#20-random-forest-architecture)
21. [Gradient-Boosting Architecture](#21-gradient-boosting-architecture)
22. [Neural-Network Architecture](#22-neural-network-architecture)
23. [Clustering Architecture](#23-clustering-architecture)
24. [Dimensionality-Reduction Architecture](#24-dimensionality-reduction-architecture)
25. [Anomaly-Detection Architecture](#25-anomaly-detection-architecture)
26. [Strategy Engine](#26-strategy-engine)
27. [Backtesting Engine](#27-backtesting-engine)
28. [Look-Ahead-Bias Prevention](#28-look-ahead-bias-prevention)
29. [Data-Leakage Prevention](#29-data-leakage-prevention)
30. [Survivorship-Bias Prevention](#30-survivorship-bias-prevention)
31. [Transaction-Cost Model](#31-transaction-cost-model)
32. [Slippage Model](#32-slippage-model)
33. [Liquidity Model](#33-liquidity-model)
34. [Position-Sizing Architecture](#34-position-sizing-architecture)
35. [Stop Architecture](#35-stop-architecture)
36. [VaR/CVaR Architecture](#36-varcvar-architecture)
37. [Risk-Parity Architecture](#37-risk-parity-architecture)
38. [Portfolio Constraints](#38-portfolio-constraints)
39. [In-Sample/Out-of-Sample Architecture](#39-in-sampleout-of-sample-architecture)
40. [Walk-Forward Architecture](#40-walk-forward-architecture)
41. [Monte Carlo Architecture](#41-monte-carlo-architecture)
42. [Robustness Testing](#42-robustness-testing)
43. [Sensitivity Analysis](#43-sensitivity-analysis)
44. [Overfitting Controls](#44-overfitting-controls)
45. [Feature-Store Design](#45-feature-store-design)
46. [Model Registry](#46-model-registry)
47. [Model Versioning](#47-model-versioning)
48. [Paper-Trading Architecture](#48-paper-trading-architecture)
49. [Broker Abstraction](#49-broker-abstraction)
50. [Order Architecture](#50-order-architecture)
51. [Position Reconciliation](#51-position-reconciliation)
52. [Execution Confirmation](#52-execution-confirmation)
53. [Latency Monitoring](#53-latency-monitoring)
54. [Signal Logging](#54-signal-logging)
55. [Error-Alert Architecture](#55-error-alert-architecture)
56. [Portfolio Dashboard](#56-portfolio-dashboard)
57. [API Security](#57-api-security)
58. [Rate Limiting](#58-rate-limiting)
59. [Authentication](#59-authentication)
60. [Authorization](#60-authorization)
61. [Secret Management](#61-secret-management)
62. [Strategy Sandbox](#62-strategy-sandbox)
63. [Audit Trails](#63-audit-trails)
64. [Emergency Shutdown Architecture](#64-emergency-shutdown-architecture)
65. [Database Schema](#65-database-schema)
66. [API Architecture](#66-api-architecture)
67. [Testing Architecture](#67-testing-architecture)
68. [Observability](#68-observability)
69. [Disaster Recovery](#69-disaster-recovery)
70. [Deployment Architecture](#70-deployment-architecture)
71. [MVP Scope](#71-mvp-scope)
72. [Future Architecture](#72-future-architecture)
73. [Quantitative Risks](#73-quantitative-risks)
74. [ML Risks](#74-ml-risks)
75. [Security Risks](#75-security-risks)
76. [Operational Risks](#76-operational-risks)
77. [Open Decisions Requiring Approval](#77-open-decisions-requiring-approval)

---

## 1. System Architecture

### 1.1 Guiding Principles

1. **Temporal causality is a first-class invariant.** No historical decision may use information that was not available at that decision's `information_cutoff_time`. This is enforced at the data layer, not merely documented.
2. **The risk engine has authority over the strategy engine.** Strategy code can never override risk controls. The risk service is the single authority for pre-trade checks, position limits, and kill switches.
3. **Raw data is immutable.** Raw observations are never overwritten. Adjusted/derived data is stored separately with full provenance.
4. **Everything is versioned.** Datasets, features, indicators, models, strategies, parameters, and configurations are versioned and reproducible.
5. **Everything is auditable.** Every state change (strategy, model, order, risk limit, credential, kill switch) is recorded in an append-only, tamper-evident audit log.
6. **Live trading is opt-in and gated.** It requires explicit authorization, a checklist, and independent risk controls. It is never automatically enabled.
7. **Separation of domains.** DATA, FEATURES, RESEARCH, MODELS, STRATEGIES, BACKTESTING, VALIDATION, RISK, PAPER TRADING, EXECUTION, and MONITORING are separate bounded contexts.

### 1.2 Architectural Style

**Modular monolith with event-driven internals, extractable to microservices.**

Rationale: A research platform needs tight iteration speed, strong consistency within domains, and the ability to scale heavy workloads (backtesting, ML training) independently. We begin as a modular monolith with strict domain boundaries and an internal event bus, then extract services (Backtesting, ML, Execution) as they grow. This avoids premature distributed-system complexity while preserving a clean migration path.

### 1.3 Domain Decomposition

```
┌─────────────────────────────────────────────────────────────────────┐
│                        API Gateway (FastAPI)                        │
│         AuthN/AuthZ · Rate Limiting · Request Validation            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Identity │ │  Market  │ │  Feature │ │ Research │ │   ML     │  │
│  │ & Access │ │  Data    │ │  Store   │ │  Engine  │ │  Engine  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Strategy │ │Backtest  │ │  Risk    │ │ Portfolio│ │ Paper    │  │
│  │  Engine  │ │  Engine  │ │  Engine  │ │  Engine  │ │ Trading  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Execution │ │Monitoring│ │  Audit   │ │Notification│ │  Data   │  │
│  │  Engine  │ │& Alerting│ │  Service │ │  Service  │ │ Quality │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                    Event Bus (Redis Streams → Kafka)                │
├─────────────────────────────────────────────────────────────────────┤
│  PostgreSQL+TimescaleDB │ Redis │ Object Storage (S3) │ Job Queue  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Service Responsibilities

| Service | Responsibility |
|---|---|
| **Identity & Access** | Authentication, RBAC, API keys, session management |
| **Market Data** | Ingestion, normalization, validation, storage, corporate actions |
| **Feature Store** | Point-in-time feature computation, versioning, retrieval |
| **Research Engine** | Indicators, statistics, regression, regime detection |
| **ML Engine** | Training, inference, model registry, drift monitoring |
| **Strategy Engine** | Strategy evaluation, signal generation, conditions |
| **Backtest Engine** | Event-driven historical simulation (job-based) |
| **Risk Engine** | Pre-trade checks, position sizing validation, limits, kill switches |
| **Portfolio Engine** | Allocation, rebalancing, exposure tracking |
| **Paper Trading** | Real-time simulation using live engines + simulated execution |
| **Execution Engine** | Broker adapters, order management, reconciliation |
| **Monitoring & Alerting** | Health checks, thresholds, alert routing |
| **Audit Service** | Append-only, hash-chained audit log |
| **Notification Service** | Email, webhook, in-app alerts |
| **Data Quality** | Validation rules, quality reports |

### 1.5 Event-Driven Communication

- **Internal event bus** for asynchronous, decoupled communication between domains.
- **MVP:** Redis Streams (lightweight, sufficient throughput).
- **Scale path:** Apache Kafka (durable log, replay, higher throughput).
- **Event schema:** CloudEvents-compatible envelope with `event_id`, `event_type`, `source`, `time`, `subject`, `data` (versioned payload).
- **Key event types:** `market_data.received`, `feature.computed`, `signal.generated`, `order.created`, `order.filled`, `risk.limit_breached`, `kill_switch.activated`, `model.deployed`, `reconciliation.failed`.

### 1.6 Request-Response vs. Event-Driven

- **Synchronous (request/response):** Queries (market data, features, portfolio state), authentication, configuration reads.
- **Asynchronous (jobs + events):** Backtests, ML training, large data imports, Monte Carlo simulations, walk-forward analysis. These enqueue jobs and emit progress/completion events.

---

## 2. Recommended Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.12+ | Quant ecosystem (numpy, pandas, polars, scikit-learn, statsmodels) |
| **API Framework** | FastAPI | Async, OpenAPI, Pydantic validation, dependency injection |
| **Task Queue** | Celery + Redis (MVP) → Celery + Kafka (scale) | Distributed job execution |
| **Relational DB** | PostgreSQL 16+ | ACID, JSONB, mature ecosystem |
| **Time-Series DB** | TimescaleDB (PostgreSQL extension) | Hypertables, continuous aggregates, compression |
| **Cache / Real-time** | Redis 7+ | Low-latency cache, pub/sub, streams, rate limiting |
| **Analytics DB** (optional) | ClickHouse | Columnar analytics for large research queries |
| **Object Storage** | S3-compatible (MinIO for self-hosted) | Datasets, model artifacts, backtest results |
| **ML Frameworks** | scikit-learn, XGBoost, LightGBM, PyTorch (optional) | Mature, versioned, reproducible |
| **ML Registry** | MLflow | Experiment tracking, model versioning, lineage |
| **Feature Store** | Custom (built on PostgreSQL + Redis) | Point-in-time correctness is domain-specific |
| **Frontend** | React 18 + TypeScript + Vite | Type-safe, component-based |
| **Charts** | Lightweight-charts / Plotly | Financial charting |
| **Auth** | OAuth2 / OIDC (Keycloak or Auth0) | Standards-based identity |
| **Secrets** | HashiCorp Vault (or cloud KMS) | Dynamic secrets, rotation, audit |
| **Observability** | OpenTelemetry, Prometheus, Grafana, Loki | Metrics, traces, logs |
| **Containerization** | Docker, Docker Compose (dev) | Reproducible environments |
| **Orchestration** | Kubernetes (prod) | Scaling, self-healing, rolling deploys |
| **CI/CD** | GitHub Actions | Testing, security scanning, deployment |
| **Infrastructure as Code** | Terraform | Reproducible infrastructure |
| **Testing** | pytest, hypothesis, Great Expectations | Unit, property-based, data validation |

### 2.1 Dependency Isolation

- Each domain has its own Python package with explicit dependencies.
- Domain interfaces defined via abstract base classes / protocols.
- No circular dependencies between domains; communication via events and well-defined APIs.

---

## 3. Multi-Asset Data Architecture

### 3.1 Instrument Abstraction

Every tradable instrument is represented by a unified `Instrument` record:

```python
@dataclass(frozen=True)
class Instrument:
    instrument_id: str          # UUID, immutable
    symbol: str                # Exchange-specific ticker
    asset_class: AssetClass     # EQUITY, ETF, INDEX, BOND, COMMODITY, FX, SECTOR
    exchange: str              # Exchange identifier
    currency: str              # ISO 4217
    country: str               # ISO 3166
    sector: Optional[str]      # Sector classification
    industry: Optional[str]    # Industry classification
    tick_size: Decimal         # Minimum price increment
    lot_size: int              # Minimum quantity
    trading_calendar: str      # Calendar identifier
    contract_specs: Optional[ContractSpecs]  # Futures/options specifics
    is_active: bool
    metadata: Dict[str, Any]   # Asset-class-specific extensions
```

### 3.2 Asset-Class-Specific Handling

| Asset Class | Key Considerations |
|---|---|
| **Equities** | Dividends, splits, corporate actions, delistings |
| **ETFs** | NAV, distributions, premium/discount |
| **Indices** | Point-in-time constituents, not directly tradable |
| **Bonds** | Yield, duration, coupon, credit spread, maturity |
| **Commodities** | Contract months, roll schedule, term structure, open interest |
| **FX** | Pip size, rollover, cross-currency exposure |
| **Sector Instruments** | Sector classification, membership changes |

### 3.3 Asset-Class Capability Matrix

Each asset class declares which features/indicators are applicable. The system never assumes equity-specific indicators are meaningful for bonds or commodities.

```python
class AssetClassCapabilities:
    supports_dividends: bool
    supports_splits: bool
    supports_yield: bool
    supports_duration: bool
    supports_open_interest: bool
    supports_term_structure: bool
    applicable_indicators: Set[str]
```

### 3.4 Symbol & Instrument History

- `instrument_history` table tracks symbol changes, ticker changes, and instrument lifecycle.
- Every historical observation references the `instrument_id` valid at that time.
- Symbol-to-instrument resolution is point-in-time aware.

---

## 4. Historical Database Architecture

### 4.1 Data Layer Separation

```
RAW DATA → NORMALIZED DATA → ADJUSTED DATA → DERIVED DATA → FEATURE DATA
```

| Layer | Description | Mutability |
|---|---|---|
| **Raw** | As-received from vendor/exchange | **Immutable, append-only** |
| **Normalized** | Cleaned, deduplicated, timezone-aligned, validated | Append-only, versioned |
| **Adjusted** | Corporate-action-adjusted prices (separate factor tables) | Derived, reproducible |
| **Derived** | Indicators, statistics, intermediate calculations | Derived, versioned |
| **Feature** | ML-ready features with full provenance | Derived, versioned |

**Critical rule:** Raw historical observations are never overwritten with adjusted values. Adjustment factors are stored separately and applied deterministically.

### 4.2 Storage Strategy

| Data Type | Storage | Partitioning |
|---|---|---|
| Daily OHLCV | TimescaleDB hypertable | By month |
| Intraday bars | TimescaleDB hypertable | By day |
| Tick data | TimescaleDB hypertable (compressed) | By day |
| Bid/Ask | TimescaleDB hypertable | By day |
| Fundamentals | PostgreSQL | By fiscal period |
| Macro data | PostgreSQL | By effective period |
| Sentiment | PostgreSQL | By publication time |
| Features | PostgreSQL + Redis cache | By computation batch |
| Model artifacts | Object Storage (S3) | By model version |
| Backtest results | Object Storage + PostgreSQL metadata | By run |

### 4.3 Data Retention

- **Raw tick data:** Configurable (default 90 days hot, archive to cold storage).
- **Intraday bars:** Configurable (default 2 years hot, archive).
- **Daily bars:** Indefinite.
- **Fundamentals/macro/sentiment:** Indefinite (point-in-time records).

### 4.4 Data Versioning

- Every dataset has a `dataset_version` (immutable snapshot identifier).
- Experiments reference a specific `dataset_version`.
- The same `dataset_version` always produces the same data.

---

## 5. Real-Time Feed Architecture

### 5.1 Feed Abstraction

```python
class DataFeedAdapter(ABC):
    @abstractmethod
    async def connect(self) -> None: ...
    @abstractmethod
    async def subscribe(self, instruments: List[str]) -> None: ...
    @abstractmethod
    async def stream(self) -> AsyncIterator[MarketDataEvent]: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    def get_health(self) -> FeedHealth: ...
```

### 5.2 Feed Types

| Type | Latency | Use Case |
|---|---|---|
| **Real-time** | < 100ms | Live trading, paper trading |
| **Near-real-time** | 1–15 min | Research, monitoring |
| **Historical batch** | N/A | Backtesting, research |

### 5.3 Feed Health Monitoring

Every feed tracks:
- `event_timestamp` (when the event occurred at source)
- `ingestion_timestamp` (when received by our system)
- `feed_latency` = ingestion_timestamp - event_timestamp
- `sequence_number` (where available)
- Message gaps, duplicates, out-of-order detection

### 5.4 Stale Data Protection

```python
class StaleDataGuard:
    def __init__(self, max_staleness_seconds: float):
        self.max_staleness = max_staleness_seconds

    def is_stale(self, event: MarketDataEvent) -> bool:
        return (now() - event.event_timestamp).total_seconds() > self.max_staleness

    def on_stale_data(self, instrument_id: str) -> None:
        # Block new live orders unless explicit strategy policy allows
        self.risk_engine.block_new_orders(instrument_id, reason="STALE_DATA")
        self.alerting.raise(AlertLevel.WARNING, f"Stale data: {instrument_id}")
```

### 5.5 Duplicate & Out-of-Order Handling

- **Idempotency:** Every market data event has a unique `event_id`; duplicates are detected and dropped.
- **Sequence gaps:** Detected via sequence numbers; gaps trigger reconciliation with the vendor.
- **Out-of-order:** Buffered and reordered within a configurable window; late arrivals beyond the window are logged and processed with a `late_arrival` flag.

---

## 6. Timestamp Architecture

### 6.1 Timestamp Fields

Every observation carries explicit temporal semantics:

| Field | Meaning |
|---|---|
| `event_timestamp` | When the event occurred at the source (exchange) |
| `publication_timestamp` | When the information became publicly available |
| `effective_timestamp` | When the value takes effect (e.g., macro release effective period) |
| `ingestion_timestamp` | When our system received the data |
| `availability_timestamp` | When the data was available for use in decisions |
| `exchange_timestamp` | Exchange-local timestamp |
| `local_timestamp` | System-local timestamp |
| `timezone` | IANA timezone identifier (e.g., `America/New_York`) |

### 6.2 Storage Rules

- **All timestamps stored as UTC** (`timestamptz` in PostgreSQL).
- **Exchange timezone** recorded alongside for display and calendar logic.
- **Daylight saving time** handled via IANA timezone database (not manual offset math).
- **Never use a single generic timestamp** for fundamentally different events.

### 6.3 Temporal Semantics by Data Type

| Data Type | Primary Timestamp | Availability Rule |
|---|---|---|
| OHLCV bars | `event_timestamp` (bar close) | Bar close + publication delay |
| Tick data | `event_timestamp` | Event time + feed latency |
| Fundamentals | `publication_date` | Publication date (not period end) |
| Macro releases | `publication_timestamp` | Release time (not effective period) |
| News sentiment | `publication_time` | Publication time |
| Social sentiment | `post_timestamp` | Post time + ingestion delay |
| Analyst revisions | `revision_time` | Revision time |

### 6.4 Decision-Time Architecture

Every prediction/decision chain records:

```
information_cutoff_time → prediction_time → decision_time → order_time → execution_time
```

- `information_cutoff_time`: The latest data timestamp used.
- `prediction_time`: When the model/strategy generated the prediction.
- `decision_time`: When the trading decision was made.
- `order_time`: When the order was submitted.
- `execution_time`: When the order was filled.

---

## 7. Corporate-Action Architecture

### 7.1 Supported Corporate Actions

| Action | Handling |
|---|---|
| **Stock split** | Adjustment factor applied to historical prices |
| **Reverse split** | Adjustment factor applied |
| **Dividend** | Cash dividend adjustment factor |
| **Special dividend** | Adjustment factor applied |
| **Rights offering** | Adjustment factor where supported |
| **Ticker change** | Instrument history table update |
| **Merger** | Instrument lifecycle event, position conversion |
| **Spin-off** | Instrument lifecycle event, position adjustment |
| **Delisting** | Instrument marked inactive, final settlement |

### 7.2 Adjustment Factor Model

```python
@dataclass(frozen=True)
class AdjustmentFactor:
    instrument_id: str
    factor_date: datetime          # Effective date
    factor_type: str               # SPLIT, DIVIDEND, etc.
    factor_value: Decimal          # Multiplicative factor
    raw_value: Decimal             # Original value before adjustment
    source: str                    # Data source
    recorded_at: datetime          # When recorded in our system
```

### 7.3 Price Representations

| Representation | Description | Use Case |
|---|---|---|
| `raw_price` | As-traded price | Corporate action analysis, audit |
| `adjusted_price` | Back-adjusted for splits/dividends | Return calculation, backtesting |
| `total_return_price` | Adjusted for splits + dividends reinvested | Performance measurement |

**Rule:** Every strategy must declare which price representation it uses. The system records this in the strategy configuration.

### 7.4 Point-in-Time Adjustment

- Adjustment factors are applied based on the `factor_date` relative to the observation date.
- Historical backtests use only adjustment factors known at the backtest date.
- Raw prices are never modified.

---

## 8. Point-in-Time Data Architecture

### 8.1 Core Principle

**A backtest at time T may only use information with `availability_timestamp <= T`.**

This is enforced at the data access layer, not left to developer discipline.

### 8.2 Point-in-Time Query Pattern

```python
def get_point_in_time(
    table: str,
    as_of: datetime,
    filters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Returns only rows where:
      effective_timestamp <= as_of
      AND (expiration_timestamp IS NULL OR expiration_timestamp > as_of)
      AND availability_timestamp <= as_of
    """
```

### 8.3 Temporal Join Pattern

```python
def temporal_join(
    left: pd.DataFrame,      # Decisions at time T
    right: pd.DataFrame,     # Data with availability timestamps
    on: str,
    tolerance: timedelta = timedelta(0)
) -> pd.DataFrame:
    """
    For each decision at time T, joins only right-side rows
    where availability_timestamp <= T.
    """
```

### 8.4 Bitemporal Modeling

Fundamental and macro data use bitemporal modeling:

| Field | Description |
|---|---|
| `valid_from` / `valid_to` | When the value was true in the real world |
| `available_from` / `available_to` | When the value was available in our system |

This allows reconstruction of both "what was true" and "what was known."

---

## 9. Feature-Store Architecture

### 9.1 Feature Definition

Every feature is a versioned, self-describing computation:

```python
@dataclass(frozen=True)
class FeatureDefinition:
    feature_name: str
    feature_version: str
    calculation_method: str          # Human-readable formula/description
    source: str                      # Data source
    calculation_window: int          # Lookback period
    publication_time: datetime       # When the source data was published
    availability_time: datetime      # When the feature is available
    lag: int                         # Number of periods lagged
    asset_class: Set[AssetClass]     # Applicable asset classes
    parameters: Dict[str, Any]      # Feature-specific parameters
```

### 9.2 Feature Record

```python
@dataclass(frozen=True)
class Feature:
    feature_name: str
    feature_version: str
    instrument_id: str
    timestamp: datetime               # Feature timestamp (end of calculation window)
    availability_timestamp: datetime # When available for decisions
    value: float
    source: str
    calculation_method: str
    parameters: Dict[str, Any]
    dataset_version: str
```

### 9.3 Point-in-Time Retrieval

```python
class FeatureStore:
    def get_features(
        self,
        instrument_id: str,
        decision_time: datetime,
        feature_names: Optional[List[str]] = None
    ) -> FeatureVector:
        """
        Returns only features where availability_timestamp <= decision_time.
        Enforces temporal causality at the storage layer.
        """
```

### 9.4 Feature Categories

| Category | Examples |
|---|---|
| **Price-based** | Returns, gaps, high-low range, true range |
| **Technical** | SMA, EMA, RSI, MACD, Bollinger, ATR |
| **Volume** | Relative volume, OBV, VWAP distance, CMF |
| **Volatility** | Rolling vol, ATR%, Bollinger width, vol percentile |
| **Fundamental** | P/E, P/B, ROE, debt/equity, Altman Z |
| **Macro** | Interest rates, inflation, GDP growth, unemployment |
| **Sentiment** | News sentiment, social sentiment, analyst revisions |
| **Regime** | Bull/bear/sideways, high/low volatility |
| **Statistical** | Z-scores, percentiles, rolling correlations |

### 9.5 Feature Versioning

- Changing a feature calculation creates a new `feature_version`.
- Experiments pin to a specific `feature_version`.
- Old versions remain available for reproducibility.

---

## 10. Technical-Indicator Architecture

### 10.1 Indicator Interface

Every indicator is a pure, versioned function with full metadata:

```python
@dataclass(frozen=True)
class IndicatorMetadata:
    name: str
    version: str
    formula: str                     # Mathematical formula
    inputs: List[str]                # Required input fields
    lookback_period: int             # Minimum periods for valid output
    warmup_period: int               # Periods before first valid value
    missing_data_behavior: str       # "propagate_nan", "skip", "fill"
    time_alignment: str              # "left", "right", "center"
    nan_behavior: str                # "propagate", "fill_zero", "interpolate"
    precision: int                   # Decimal places
```

### 10.2 Indicator Library

#### Moving Averages
| Indicator | Formula | Lookback |
|---|---|---|
| SMA | `SMA(n) = (1/n) * Σ Close[i]` | n |
| EMA | `EMA[i] = α * Close[i] + (1-α) * EMA[i-1]`, `α = 2/(n+1)` | n |
| WMA | `WMA(n) = (Σ w[i] * Close[i]) / Σ w[i]`, linear weights | n |
| Adaptive MA | Kaufman AMA, efficiency-ratio based | variable |

#### Momentum
| Indicator | Formula | Lookback |
|---|---|---|
| RSI | `RSI = 100 - 100/(1 + RS)`, `RS = avg_gain/avg_loss` | 14 (default) |
| MACD | `MACD = EMA(12) - EMA(26)`, `Signal = EMA(9, MACD)` | 12, 26, 9 |
| Stochastic | `%K = 100 * (C - LL) / (HH - LL)`, `%D = SMA(%K, 3)` | 14, 3 |
| ROC | `ROC = (Close[i] / Close[i-n] - 1) * 100` | n |

#### Volatility
| Indicator | Formula | Lookback |
|---|---|---|
| Bollinger Bands | `Mid = SMA(n)`, `Upper = Mid + k*σ`, `Lower = Mid - k*σ` | 20, 2 |
| ATR | `ATR = TR smoothed (Wilder or EMA)` | 14 (default) |
| Std Dev | `σ = sqrt(Σ(x - μ)² / (n-1))` | n |
| Keltner Channels | `Mid = EMA(n)`, `Upper = EMA + k*ATR`, `Lower = EMA - k*ATR` | 20, 2 |

#### Volume
| Indicator | Formula | Lookback |
|---|---|---|
| OBV | Cumulative volume signed by price direction | n/a |
| VWAP | `VWAP = Σ(Price * Volume) / Σ Volume` | n |
| CMF | `CMF = Σ(((C-L)-(H-C))/(H-L) * V) / Σ V` | 20 (default) |

### 10.3 Indicator Registry

- All indicators registered in a central `IndicatorRegistry`.
- Each indicator has a unique `(name, version)` pair.
- Reference values from published sources are stored for validation.

### 10.4 Numerical Precision

- All calculations use `float64` (or `Decimal` for monetary values).
- NaN propagation is explicit and configurable.
- Warm-up periods produce NaN (not zero) until sufficient data exists.

---

## 11. Fundamental-Data Architecture

### 11.1 Fundamental Observation

Every fundamental data point carries full provenance:

```python
@dataclass(frozen=True)
class FundamentalObservation:
    instrument_id: str
    fiscal_period: str               # e.g., "2024-Q3"
    period_end: datetime             # Fiscal period end date
    publication_date: datetime       # When results were published
    availability_timestamp: datetime # When available for decisions
    source: str                      # Data vendor
    revision: int                    # Revision number (0 = original)
    field_name: str                  # e.g., "net_income", "total_debt"
    value: Decimal
    currency: str
```

### 11.2 Supported Fundamental Metrics

| Category | Metrics |
|---|---|
| **Valuation** | P/E, P/B, EV/EBITDA, PEG, Dividend Yield, FCF Yield |
| **Liquidity** | Current Ratio |
| **Leverage** | Debt-to-Equity, Interest Coverage |
| **Stability** | Altman Z-Score |
| **Profitability** | ROE, ROA, Operating Margin, Net Margin |
| **Growth** | Earnings Growth, OCF Growth, FCF Growth |
| **Cash Generation** | Operating Cash Flow, Free Cash Flow, Cash Conversion, FCF Growth |

### 11.3 Point-in-Time Fundamental Access

- Fundamentals are available only after `publication_date`.
- Revised financials create new records with incremented `revision`.
- Backtests use only the revision available at the decision time.
- **Never** use today's revised financials as if they were known historically.

### 11.4 Fundamental Data Schema

```sql
CREATE TABLE fundamental_observations (
    instrument_id UUID REFERENCES instruments(id),
    fiscal_period VARCHAR(10) NOT NULL,
    period_end DATE NOT NULL,
    publication_date DATE NOT NULL,
    availability_timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(100) NOT NULL,
    revision INT NOT NULL DEFAULT 0,
    field_name VARCHAR(100) NOT NULL,
    value NUMERIC,
    currency CHAR(3),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (instrument_id, fiscal_period, field_name, revision)
);
```

---

## 12. Macro-Data Architecture

### 12.1 Macro Variable Record

```python
@dataclass(frozen=True)
class MacroObservation:
    variable_name: str               # e.g., "US_CPI_YOY", "FED_FUNDS_RATE"
    source: str                      # e.g., "BLS", "FRED"
    publication_timestamp: datetime   # When the release occurred
    effective_period: str            # e.g., "2024-08" (the month the data describes)
    value: Decimal
    revision: int                    # Revision number
    previous_value: Optional[Decimal]
    currency: Optional[str]
    country: str
    metadata: Dict[str, Any]
```

### 12.2 Supported Macro Variables

| Category | Variables |
|---|---|
| **Interest Rates** | Policy rate, 10Y yield, 2Y yield, yield curve slope |
| **Inflation** | CPI (headline/core), PPI, PCE |
| **Growth** | GDP (quarterly), industrial production, retail sales |
| **Labor** | Unemployment rate, non-farm payrolls, jobless claims |
| **Currency** | DXY, major FX rates |
| **Commodity** | Oil, gold, copper prices |

### 12.3 Revision Handling

- Macro data is frequently revised. Each revision is a new record.
- Backtests use only the value available at the decision time (first release or the then-current revision).
- **Never** use the current revised GDP value as though that exact value was known historically.

### 12.4 Point-in-Time Macro Access

```python
def get_macro_at(
    variable_name: str,
    as_of: datetime,
    country: str = "US"
) -> Optional[Decimal]:
    """
    Returns the value of the macro variable as known at `as_of`.
    Uses the first release or the then-current revision.
    """
```

---

## 13. News-Sentiment Architecture

### 13.1 News Record

```python
@dataclass(frozen=True)
class NewsItem:
    article_id: str                  # Unique identifier
    source: str                      # News source
    publication_time: datetime        # When published
    ingestion_time: datetime         # When received by our system
    headline: str
    content_reference: str           # URL or content hash (not full text unless licensed)
    entities: List[str]              # Mentioned instruments/companies
    sentiment: Optional[float]      # -1.0 to 1.0
    sentiment_confidence: float      # 0.0 to 1.0
    sentiment_model_version: str
    language: str
```

### 13.2 Sentiment Processing Pipeline

```
Raw News → Deduplication → Entity Extraction → Sentiment Scoring → Aggregation → Feature Store
```

- **Entity extraction:** Map news to instruments using entity linking.
- **Sentiment scoring:** Versioned NLP model (configurable provider).
- **Aggregation:** Rolling sentiment windows (e.g., 1-day, 5-day average sentiment per instrument).

### 13.3 Temporal Correctness

- Sentiment is available only after `publication_time`.
- Ingestion latency is tracked and recorded.
- **Never** allow ingestion latency to create accidental future leakage.

### 13.4 Licensing & Compliance

- News content storage respects licensing agreements.
- Only store what is legally permitted (headlines, references, or full text if licensed).
- Attribution and source tracking for all news items.

---

## 14. Social-Sentiment Architecture

### 14.1 Social Sentiment Record

```python
@dataclass(frozen=True)
class SocialSentimentObservation:
    platform: str                    # e.g., "twitter", "reddit", "stocktwits"
    post_timestamp: datetime         # When the post was made
    ingestion_time: datetime         # When received
    entity: str                      # Instrument/company
    sentiment: Optional[float]      # -1.0 to 1.0
    volume: int                      # Post volume in window
    engagement: Optional[int]        # Likes, shares, comments
    bot_quality_signal: Optional[float]  # 0.0 (human) to 1.0 (likely bot)
    sentiment_model_version: str
```

### 14.2 Aggregation

- Social sentiment is aggregated into rolling windows per instrument.
- Bot-quality signals are used to weight or filter observations.
- Volume and engagement metrics provide context.

### 14.3 Caveats

- Social sentiment is **not** ground truth.
- Bot manipulation, coordinated campaigns, and platform bias are explicitly tracked.
- Social sentiment features include confidence intervals and quality flags.

---

## 15. Analyst-Revision Architecture

### 15.1 Revision Record

```python
@dataclass(frozen=True)
class AnalystRevision:
    instrument_id: str
    revision_type: str               # EPS_ESTIMATE, REVENUE_ESTIMATE, TARGET_PRICE, RATING
    previous_value: Optional[Decimal]
    new_value: Decimal
    revision_time: datetime         # When the revision was made public
    analyst_source: str              # Analyst or firm
    availability_timestamp: datetime
```

### 15.2 Supported Revision Types

| Type | Description |
|---|---|
| **EPS estimates** | Consensus EPS estimate changes |
| **Revenue estimates** | Consensus revenue estimate changes |
| **Target price** | Analyst target price revisions (where permitted) |
| **Rating changes** | Buy/Hold/Sell rating changes (where permitted) |

### 15.3 Point-in-Time Access

- Revisions are available only after `revision_time`.
- Historical models see revisions only after they became available.
- Revision momentum features (e.g., 30-day net revisions) are computed with strict temporal alignment.

---

## 16. Market-Regime Architecture

### 16.1 Regime Labels

| Regime | Description |
|---|---|
| `BULL` | Sustained upward trend |
| `BEAR` | Sustained downward trend |
| `SIDEWAYS` | No clear trend |
| `HIGH_VOLATILITY` | Elevated volatility regime |
| `LOW_VOLATILITY` | Suppressed volatility regime |

### 16.2 Regime Label Record

```python
@dataclass(frozen=True)
class RegimeLabel:
    regime: Regime
    method: str                      # "rule_based" or "ml_based"
    parameters: Dict[str, Any]       # Method-specific parameters
    timestamp: datetime               # When the regime was determined
    model_version: Optional[str]     # ML model version (if ml_based)
    confidence: Optional[float]      # Classification confidence
```

### 16.3 Rule-Based Regime Detection

Deterministic rules using:
- **Trend:** Price relative to moving average, moving average slope
- **Volatility:** Realized volatility percentiles, ATR
- **Returns:** Rolling return thresholds
- **Breadth:** Advance-decline ratios (where available)

```python
class RuleBasedRegimeDetector:
    def detect(self, market_state: MarketState) -> RegimeLabel:
        # Example rules (configurable parameters):
        # BULL: price > SMA(200) AND SMA(50) > SMA(200)
        # BEAR: price < SMA(200) AND SMA(50) < SMA(200)
        # HIGH_VOLATILITY: realized_vol > 75th percentile
        # LOW_VOLATILITY: realized_vol < 25th percentile
        # SIDEWAYS: otherwise
```

### 16.4 ML-Based Regime Classification

- Separate ML model trained on labeled regime data.
- Uses features: trend, volatility, breadth, macro variables, correlations, momentum.
- Model version recorded with every prediction.
- **Never** uses future information to determine a historical regime label.

### 16.5 Temporal Correctness

- Regime labels are computed using only data available at the label timestamp.
- Regime features are lagged appropriately for use in strategies.

---

## 17. Statistical Engine

### 17.1 Descriptive Statistics

| Statistic | Function | Edge Case Handling |
|---|---|---|
| Mean | `mean(x)` | NaN propagation, empty series → NaN |
| Median | `median(x)` | Empty series → NaN |
| Variance | `var(x, ddof=1)` | n < 2 → NaN |
| Std Dev | `std(x, ddof=1)` | n < 2 → NaN |
| Downside Deviation | `DD(x, threshold)` | No downside obs → 0 |
| Skewness | `skew(x)` | n < 3 → NaN |
| Kurtosis | `kurtosis(x)` | n < 4 → NaN |
| Percentile | `percentile(x, q)` | Linear interpolation |
| Z-score | `z = (x - μ) / σ` | σ = 0 → NaN |

### 17.2 Covariance & Correlation

| Function | Description |
|---|---|
| Covariance | `cov(x, y, ddof=1)` |
| Pearson correlation | `corr(x, y)` |
| Rolling correlation | Windowed correlation |
| Beta | `β = cov(r_asset, r_market) / var(r_market)` |

### 17.3 Edge Case Handling

- **Small samples:** Return NaN with a warning flag when insufficient data.
- **Missing observations:** Pairwise deletion with explicit tracking.
- **Constant series:** σ = 0 → z-score = NaN (not zero).
- **Numerical instability:** Use numerically stable algorithms (Welford for variance).

### 17.4 Outlier Detection

- Z-score method (|z| > threshold)
- IQR method (outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR])
- Modified Z-score (using median/MAD)
- **Outliers are flagged, never automatically deleted.**

---

## 18. Regression Architecture

### 18.1 Supported Regression Types

| Type | Implementation | Use Case |
|---|---|---|
| **Linear** | OLS | Factor models, return prediction |
| **Multiple** | OLS with multiple predictors | Multi-factor models |
| **Rolling** | Windowed OLS | Time-varying beta |
| **Regularized** | Ridge, Lasso, ElasticNet | High-dimensional features |
| **Factor** | OLS on factor returns | Factor exposure analysis |
| **Cross-sectional** | Per-period OLS across assets | Cross-sectional return prediction |

### 18.2 Regression Output

```python
@dataclass(frozen=True)
class RegressionResult:
    coefficients: Dict[str, Decimal]
    standard_errors: Dict[str, Decimal]
    r_squared: float
    adjusted_r_squared: float
    residuals: List[float]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]]
    f_statistic: float
    f_p_value: float
    n_observations: int
```

### 18.3 Temporal Correctness

- Rolling regressions use only data within the lookback window.
- Cross-sectional regressions use only data available at the period end.
- **Correlation and regression coefficients are not proof of causality.**

---

## 19. ML Architecture

### 19.1 ML Pipeline

```
Data → Feature Engineering → Train/Val/Test Split → Preprocessing (fit on train only) → Model Training → Validation → Test → Calibration → Registration
```

### 19.2 Experiment Record

Every ML experiment records:

```python
@dataclass(frozen=True)
class MLExperiment:
    experiment_id: str
    dataset_version: str
    feature_set_version: str
    feature_definition: str
    target_definition: str
    training_period: DateRange
    validation_period: DateRange
    test_period: DateRange
    preprocessing_version: str
    model_type: str
    hyperparameters: Dict[str, Any]
    random_seed: int
    model_version: str
    metrics: Dict[str, float]
    created_at: datetime
    config_hash: str                 # SHA-256 of full configuration
```

### 19.3 Reproducibility

- `random_seed` recorded and enforced.
- `config_hash` ensures the same configuration produces the same result.
- Model artifacts stored in object storage with version identifier.
- Training code version (git commit hash) recorded.

### 19.4 Model Types

| Task | Models |
|---|---|
| **Classification** | Logistic Regression, Random Forest, Gradient Boosting |
| **Regression** | Linear, Ridge, Lasso, Random Forest, Gradient Boosting, Neural Network |
| **Unsupervised** | K-Means, Hierarchical, DBSCAN, PCA, Isolation Forest |

---

## 20. Random-Forest Architecture

### 20.1 Configuration

```python
@dataclass
class RandomForestConfig:
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    oob_score: bool = True          # Out-of-bag evaluation
    random_seed: int = 42
    n_jobs: int = -1
```

### 20.2 Capabilities

- **Classification:** Probability outputs, class labels.
- **Regression:** Continuous value prediction.
- **Feature importance:** Gini importance + permutation importance.
- **OOB evaluation:** Out-of-bag error estimate (no separate validation set needed for initial assessment).

### 20.3 Temporal Correctness

- Training uses only data from the training period.
- Feature engineering uses only lagged features.
- Hyperparameter tuning uses only training/validation data.
- **Never** uses future observations during feature engineering or model selection.

---

## 21. Gradient-Boosting Architecture

### 21.1 Implementation Strategy

- **Pluggable backend:** XGBoost, LightGBM, or scikit-learn GradientBoosting.
- Architecture is not coupled to one library.
- Backend selected via configuration, recorded in experiment.

### 21.2 Configuration

```python
@dataclass
class GradientBoostingConfig:
    backend: str = "xgboost"        # "xgboost", "lightgbm", "sklearn"
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "binary:logistic"
    random_seed: int = 42
```

### 21.3 Capabilities

- Classification and regression.
- Feature importance (gain, cover, frequency).
- Probability outputs with optional calibration.
- Early stopping on validation set.

---

## 22. Neural-Network Architecture

### 22.1 Design Principles

- **Optional, not default.** Neural networks are one tool among many.
- **No assumption of superiority** over simpler models.
- **Resource limits** enforced (CPU/memory/time).
- **GPU support** where available, with CPU fallback.

### 22.2 Configuration

```python
@dataclass
class NeuralNetworkConfig:
    architecture: str = "mlp"       # "mlp", "lstm", "transformer"
    hidden_layers: List[int] = [64, 32]
    activation: str = "relu"
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    l2_regularization: float = 0.01
    random_seed: int = 42
    use_gpu: bool = False
    max_training_time_seconds: int = 3600
    max_memory_gb: float = 4.0
```

### 22.3 Training Safeguards

- **Train/validation/test separation** enforced.
- **Early stopping** on validation loss.
- **Regularization:** Dropout, L2, batch normalization.
- **Model versioning:** Every training run produces a new model version.
- **Random seeds** recorded and enforced.

### 22.4 Use Cases

- Return classification
- Return regression
- Volatility estimation
- Regime classification
- Feature representation (autoencoders)

---

## 23. Clustering Architecture

### 23.1 Supported Algorithms

| Algorithm | Implementation | Use Case |
|---|---|---|
| **K-Means** | scikit-learn | Asset grouping, regime discovery |
| **Hierarchical** | scipy/scikit-learn | Correlation clustering, dendrogram analysis |
| **DBSCAN** | scikit-learn | Density-based clustering, noise detection |

### 23.2 Temporal Awareness

- Cluster assignments are **time-aware** when used in historical simulations.
- Clustering is fitted only on the training period.
- Rolling clustering uses only data within the lookback window.
- Cluster labels are versioned and recorded.

### 23.3 Use Cases

- Asset grouping for diversification
- Regime discovery (unsupervised)
- Correlation clustering
- Portfolio diversification analysis

---

## 24. Dimensionality-Reduction Architecture

### 24.1 Supported Methods

| Method | Implementation | Use Case |
|---|---|---|
| **PCA** | scikit-learn | Feature compression, noise reduction |
| **ICA** | scikit-learn | Independent component analysis |
| **t-SNE** | scikit-learn | Visualization (non-linear) |
| **UMAP** | umap-learn | Visualization (non-linear, faster) |

### 24.2 Critical Rule: Temporal Correctness

**PCA must be fitted only on the training period during backtesting.**

```
Training period → Fit PCA → Transform training data
Validation period → Use training PCA → Transform validation data
Test period → Use training PCA → Transform test data
```

**Never** calculate PCA across the entire dataset and then backtest as though the components were historically known.

### 24.3 Output

- Transformed features with explained variance ratios.
- Component loadings for interpretability.
- Versioned and recorded in experiment configuration.

---

## 25. Anomaly-Detection Architecture

### 25.1 Anomaly Types

| Type | Description | Action |
|---|---|---|
| `DATA_QUALITY_ANOMALY` | Data error (bad price, timestamp issue) | Flag for review, exclude from training |
| `MARKET_EVENT` | Legitimate market event (crash, spike) | Keep, flag in analysis |
| `POTENTIAL_OUTLIER` | Statistical outlier, unclear cause | Flag, do not auto-delete |

### 25.2 Detection Methods

| Method | Application |
|---|---|
| **Z-score** | Univariate price/return outliers |
| **IQR** | Univariate outliers |
| **Isolation Forest** | Multivariate anomaly detection |
| **DBSCAN** | Density-based anomaly detection |
| **Rolling statistics** | Contextual anomalies (unusual vs recent history) |

### 25.3 Domains

Anomaly detection is applied to:
- Price, returns, volume, volatility
- Fundamental values
- Sentiment
- Trading behavior

### 25.4 Policy

- **Anomalous observations are never automatically deleted.**
- Anomalies are flagged with type, severity, and timestamp.
- Data quality anomalies may be excluded from training with explicit documentation.

---

## 26. Strategy Engine

### 26.1 Strategy Definition

Strategies are machine-readable, versioned definitions:

```python
@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    strategy_version: str
    name: str
    description: str
    asset_classes: Set[AssetClass]
    instruments: List[str]              # Universe
    conditions: List[Condition]        # Entry/exit conditions
    position_sizing: PositionSizingRule
    stop_rules: List[StopRule]
    risk_parameters: RiskParameters
    parameters: Dict[str, Any]
    config_hash: str
    created_at: datetime
    created_by: str
```

### 26.2 Strategy Pipeline

```
Signal → Position Sizing → Risk Engine → Portfolio Constraints → Order Engine
```

**The strategy must pass through every stage. No stage can be bypassed.**

### 26.3 Signal Types

| Signal Type | Description |
|---|---|
| **Crossover** | SMA/EMA/WMA/MACD crossovers |
| **Breakout** | Price, volatility, volume-confirmed, MA breakouts |
| **Mean reversion** | RSI extremes, Bollinger deviations, z-score, MA distance |
| **Candlestick** | Configurable deterministic patterns |
| **ML-based** | Model prediction thresholds |
| **Combined** | Multi-signal aggregation |

### 26.4 Condition System

Conditions are composable, deterministic expressions:

```python
@dataclass(frozen=True)
class Condition:
    indicator: str                   # e.g., "EMA", "RSI", "MACD"
    operator: str                    # ">", "<", "==", "crosses_above", "crosses_below"
    value: Union[float, str]        # Threshold or other indicator
    parameters: Dict[str, Any]       # Indicator parameters
    lookback: int                    # Lookback for crossover detection
```

### 26.5 Strategy Versioning

- Every strategy change creates a new `strategy_version`.
- Backtests pin to a specific `strategy_version`.
- Old versions remain available for reproducibility.

---

## 27. Backtesting Engine

### 27.1 Architecture

**Event-driven backtesting engine** processing bars sequentially:

```
Market Data → Feature Calculation → Signal → Risk Check → Order → Execution Simulation → Portfolio Update
```

### 27.2 Event Loop

```python
class BacktestEngine:
    def run(self, config: BacktestConfig) -> BacktestResult:
        for bar in data:
            # 1. Update market state
            self.market_state.update(bar)

            # 2. Calculate features (using only data up to bar.timestamp)
            features = self.feature_engine.calculate(
                self.market_state,
                as_of=bar.timestamp
            )

            # 3. Generate signals
            signals = self.strategy.evaluate(features, self.portfolio_state)

            # 4. Risk check
            approved_orders = self.risk_engine.check(signals, self.portfolio_state)

            # 5. Simulate execution
            fills = self.execution_simulator.execute(approved_orders, bar)

            # 6. Update portfolio
            self.portfolio_state.update(fills)

            # 7. Log everything
            self.logger.log(bar.timestamp, features, signals, approved_orders, fills)
```

### 27.3 Temporal Causality Enforcement

- Features are calculated using only data with `timestamp <= bar.timestamp`.
- The engine maintains a strict `current_time` and rejects any data with `timestamp > current_time`.
- Signal, decision, order, and execution timestamps are explicitly recorded and ordered.

### 27.4 Simulation Configuration

```python
@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: Decimal
    universe: List[str]
    start_date: datetime
    end_date: datetime
    timeframe: str                   # "1d", "1h", "15m"
    strategy_version: str
    parameters: Dict[str, Any]
    transaction_costs: TransactionCostConfig
    slippage: SlippageConfig
    liquidity: LiquidityConfig
    risk_config: RiskConfig
    benchmark: Optional[str]
    dataset_version: str
    random_seed: int
```

### 27.5 Reproducibility

- Same `BacktestConfig` + same `dataset_version` → same result.
- `config_hash` recorded for verification.
- All intermediate state logged for debugging.

---

## 28. Look-Ahead-Bias Prevention

### 28.1 Prevention Mechanisms

| Mechanism | Implementation |
|---|---|
| **Data layer enforcement** | Point-in-time queries filter by `availability_timestamp` |
| **Feature lag enforcement** | All features declare and enforce `lag` |
| **Temporal ordering** | Event loop processes bars strictly sequentially |
| **Timestamp validation** | Reject any data with `timestamp > current_time` |
| **Preprocessing isolation** | Fit on train only, transform on val/test |
| **Universe construction** | Point-in-time constituents only |

### 28.2 Prohibited Patterns

The system detects and prevents:
- Future closing prices entering earlier signals
- Future volume entering earlier features
- Future fundamentals entering historical models
- Revised financial statements before revision publication
- Future analyst revisions, sentiment, macro releases
- Future index constituents, sector classifications
- Future corporate actions, market-regime labels
- Future normalization parameters, feature scaling
- Future model training data

### 28.3 Detection

- **Automated tests** verify temporal causality in backtests.
- **Data lineage tracking** records the exact data used for each decision.
- **Audit logs** capture the `information_cutoff_time` for every prediction.

---

## 29. Data-Leakage Prevention

### 29.1 ML Pipeline Leakage Prevention

| Leakage Type | Prevention |
|---|---|
| **Target leakage** | Target excluded from features; explicit target definition |
| **Train/test contamination** | Strict temporal split; no overlapping windows |
| **Future normalization** | Scaler fit on train only |
| **Future imputation** | Imputation fit on train only |
| **Future feature selection** | Selection on train only |
| **Future PCA fitting** | PCA fit on train only |
| **Future clustering** | Clustering fit on train only |
| **Future hyperparameter tuning** | Tuning on validation only |
| **Future model selection** | Selection on validation only |

### 29.2 Preprocessing Pipeline

```
Training period → Fit scaler/imputer/selector → Transform training data
Validation period → Use training transformations → Transform validation data
Test period → Use training transformations → Transform test data
```

### 29.3 Enforcement

- The ML pipeline framework **structurally prevents** fitting on the full dataset.
- Transformations are objects with `fit()` and `transform()` methods; `fit()` can only be called on training data.
- Pipeline objects are serialized and versioned.

---

## 30. Survivorship-Bias Prevention

### 30.1 Universe Construction

Historical universes must include:
- Delisted companies
- Bankrupt companies
- Removed index constituents
- Historical ticker symbols
- Historical sector membership

### 30.2 Implementation

```python
def construct_historical_universe(
    as_of: datetime,
    index: Optional[str] = None
) -> List[Instrument]:
    """
    Returns all instruments that were in the universe at `as_of`,
    including those that have since been delisted.
    """
```

### 30.3 Data Requirements

- Delisting records with delisting date and reason.
- Historical index membership with effective dates.
- Historical ticker mappings.
- Historical sector/industry classifications.

### 30.4 Default Policy

- **Default:** Include delisted instruments where data is permitted.
- **Configurable:** Allow survivorship-biased universes for specific research purposes, with explicit warning.

---

## 31. Transaction-Cost Model

### 31.1 Cost Components

| Component | Model |
|---|---|
| **Commission** | Fixed, per-share, or percentage |
| **Exchange fees** | Fixed schedule |
| **Regulatory fees** | SEC, FINRA, etc. (where applicable) |
| **Bid/ask spread** | Spread cost at execution |
| **Market impact** | Volume-based impact model |
| **Slippage** | See Section 32 |

### 31.2 Fee Schedule Configuration

```python
@dataclass(frozen=True)
class TransactionCostConfig:
    commission_type: str             # "fixed", "per_share", "percentage"
    commission_value: Decimal
    exchange_fee: Decimal
    regulatory_fee: Decimal
    spread_cost_model: str          # "fixed", "percentage", "volatility_based"
    spread_cost_value: Decimal
    market_impact_model: str        # "square_root", "linear", "none"
    market_impact_coefficient: Decimal
    minimum_commission: Decimal
    maximum_commission: Optional[Decimal]
```

### 31.3 Fee Display

- All transaction cost assumptions are displayed in every backtest report.
- Fee schedules are versioned and recorded in the experiment configuration.

---

## 32. Slippage Model

### 32.1 Slippage Types

| Type | Model |
|---|---|
| **Fixed** | Constant slippage per trade |
| **Percentage** | Percentage of price |
| **Spread-based** | Fraction of bid/ask spread |
| **Volatility-based** | Function of recent volatility |
| **Volume-based** | Function of order size relative to volume |

### 32.2 Configuration

```python
@dataclass(frozen=True)
class SlippageConfig:
    model: str                       # "fixed", "percentage", "spread", "volatility", "volume"
    fixed_slippage: Optional[Decimal]
    percentage_slippage: Optional[Decimal]
    spread_fraction: Optional[Decimal]  # Fraction of spread (0.0 to 1.0)
    volatility_multiplier: Optional[Decimal]
    volume_participation_limit: Optional[Decimal]  # Max % of ADV
```

### 32.3 Execution Price Calculation

```
execution_price = market_price + slippage
slippage = f(model, market_price, spread, volatility, order_size, volume)
```

### 32.4 Conservative Default

- Default slippage is conservative (non-zero).
- **Never assume unlimited liquidity or perfect fills.**

---

## 33. Liquidity Model

### 33.1 Liquidity Metrics

| Metric | Description |
|---|---|
| **Average Daily Volume (ADV)** | Rolling average volume |
| **Relative volume** | Current volume / ADV |
| **Volume participation** | Order size / ADV |
| **Estimated market impact** | Based on participation and volatility |
| **Minimum liquidity** | Configurable threshold |

### 33.2 Liquidity Constraints

```python
@dataclass(frozen=True)
class LiquidityConfig:
    min_adv: Decimal                 # Minimum average daily volume
    max_participation: Decimal       # Max % of ADV per order (e.g., 0.10 = 10%)
    max_order_size: Optional[Decimal]
    min_liquidity_score: float       # 0.0 to 1.0
```

### 33.3 Order Sizing Impact

- Orders are capped at `max_participation * ADV`.
- Large orders may be split into child orders (configurable).
- Illiquid instruments are flagged or excluded.

---

## 34. Position-Sizing Architecture

### 34.1 Sizing Methods

| Method | Formula | Parameters |
|---|---|---|
| **Fixed Fractional** | `position_size = (risk_fraction * equity) / stop_distance` | `risk_fraction` |
| **Volatility-Based (ATR)** | `position_size = (target_risk * equity) / (ATR * multiplier)` | `target_risk`, `ATR_period` |
| **Target Volatility** | `position_size = (target_vol * equity) / (asset_vol * multiplier)` | `target_vol`, `vol_lookback` |
| **Full Kelly** | `f* = (p * b - q) / b` | `p` (win prob), `b` (win/loss ratio) |
| **Fractional Kelly** | `f = kelly_fraction * f*` | `kelly_fraction` (e.g., 0.25) |
| **Risk-Based** | `position_size = risk_capital / stop_distance` | `risk_capital`, `stop_distance` |

### 34.2 Kelly Criterion Safeguards

- Kelly sizing is **always bounded** by portfolio-level constraints.
- Fractional Kelly is the default (full Kelly is rarely appropriate).
- Kelly inputs (win probability, win/loss ratio) are estimated from historical data with confidence intervals.

### 34.3 Hard Limits

All sizing methods are subject to hard limits:
- Maximum position size (% of equity)
- Maximum position value (absolute)
- Maximum gross exposure
- Maximum net exposure
- Maximum leverage

**No sizing method can override hard risk limits.**

---

## 35. Stop Architecture

### 35.1 Stop Types

| Stop Type | Description |
|---|---|
| **Percentage stop** | Fixed percentage below entry |
| **ATR stop** | Multiple of ATR below entry |
| **Trailing stop** | Trails price by fixed amount or percentage |
| **Support-based stop** | Below identified support level |
| **Time stop** | Exit after N periods |
| **Volatility stop** | Exit if volatility exceeds threshold |
| **Strategy-defined exit** | Custom exit condition |

### 35.2 Stop Execution Modeling

Stop execution must model:
- **Gaps:** Price gaps through the stop level → execution at next available price.
- **Slippage:** Additional slippage on stop execution.
- **Liquidity:** Partial fills in illiquid conditions.
- **Never assume the exact stop price is always achieved.**

### 35.3 Stop Configuration

```python
@dataclass(frozen=True)
class StopRule:
    stop_type: str
    parameters: Dict[str, Any]
    trailing: bool
    activation_price: Optional[Decimal]
    time_stop_periods: Optional[int]
```

---

## 36. VaR/CVaR Architecture

### 36.1 Value at Risk (VaR)

| Method | Description |
|---|---|
| **Historical VaR** | Empirical quantile of historical returns |
| **Parametric VaR** | Mean - z * σ (normal assumption) |
| **Monte Carlo VaR** | Simulated return distribution quantile |

### 36.2 Conditional VaR (CVaR / Expected Shortfall)

| Method | Description |
|---|---|
| **Historical CVaR** | Mean of returns beyond VaR threshold |
| **Parametric CVaR** | Analytical formula (normal assumption) |
| **Monte Carlo CVaR** | Mean of simulated returns beyond VaR |

### 36.3 Configuration

```python
@dataclass(frozen=True)
class VaRConfig:
    method: str                      # "historical", "parametric", "monte_carlo"
    confidence: float                # e.g., 0.95, 0.99
    horizon: int                     # Days
    lookback_window: int             # Historical observations
    n_simulations: int               # Monte Carlo only
    random_seed: int
```

### 36.4 Output

Every VaR/CVaR calculation records:
- `confidence` level
- `horizon`
- `method`
- `dataset` (version and period)
- `value`

### 36.5 Insufficient Sample Handling

- When sample size is insufficient, return NaN with an explicit warning.
- **Never** silently return zero or a default value.

---

## 37. Risk-Parity Architecture

### 37.1 Risk-Parity Construction

```
Inputs: Covariance matrix Σ, target risk contribution per asset
Optimization: Find weights w such that each asset's risk contribution is equal
Risk contribution: RC_i = w_i * (Σw)_i / sqrt(w'Σw)
```

### 37.2 Covariance Matrix Regularization

- **Shrinkage:** Ledoit-Wolf shrinkage toward constant correlation matrix.
- **Eigenvalue clipping:** Floor eigenvalues to prevent singularity.
- **Factor model:** Use factor covariance when asset count is large.

### 37.3 Configuration

```python
@dataclass(frozen=True)
class RiskParityConfig:
    target_volatility: Optional[Decimal]
    max_iterations: int
    tolerance: float
    shrinkage_method: str            # "ledoit_wolf", "constant_correlation", "factor"
    factor_model: Optional[str]
```

### 37.4 Constraints

- Long-only or long-short (configurable).
- Position limits.
- Sector/asset-class limits.
- Leverage limits.

---

## 38. Portfolio Constraints

### 38.1 Constraint Types

| Constraint | Description |
|---|---|
| **Concentration limits** | Max single position (% of equity) |
| **Position limits** | Max number of positions |
| **Sector limits** | Max sector exposure (% of equity) |
| **Asset-class limits** | Max asset-class exposure (% of equity) |
| **Gross exposure** | Sum of absolute position values / equity |
| **Net exposure** | Sum of signed position values / equity |
| **Leverage limits** | Gross exposure / equity |
| **Maximum drawdown** | Halt trading if drawdown exceeds threshold |
| **Daily loss limits** | Halt trading if daily loss exceeds threshold |
| **Volatility limits** | Max portfolio volatility target |

### 38.2 Constraint Enforcement

- Constraints are checked **before** every order.
- The risk engine can **reject** any strategy-generated order.
- **Strategy code can never override risk controls.**

### 38.3 Constraint Configuration

```python
@dataclass(frozen=True)
class PortfolioConstraints:
    max_single_position_pct: Decimal
    max_positions: int
    max_sector_exposure_pct: Decimal
    max_asset_class_exposure_pct: Decimal
    max_gross_exposure: Decimal
    max_net_exposure: Decimal
    max_leverage: Decimal
    max_drawdown_pct: Decimal
    daily_loss_limit_pct: Decimal
    max_portfolio_volatility: Optional[Decimal]
```

---

## 39. In-Sample/Out-of-Sample Architecture

### 39.1 Data Splitting

```
TRAINING → VALIDATION → OUT-OF-SAMPLE TEST
```

| Period | Purpose | Rules |
|---|---|---|
| **Training** | Model fitting, feature engineering | All preprocessing fit here |
| **Validation** | Hyperparameter tuning, model selection | Never used for final evaluation |
| **Out-of-sample test** | Final performance assessment | **Untouched during model selection** |

### 39.2 Temporal Splitting

- Splits are **temporal** (not random).
- No overlapping windows between train/validation/test.
- Gap between periods to prevent leakage (configurable).

### 39.3 Contamination Prevention

- The final test dataset is **structurally isolated** during model selection.
- The ML pipeline framework prevents access to test data during training/validation.
- Experiment records document which data was used for what purpose.

---

## 40. Walk-Forward Architecture

### 40.1 Walk-Forward Process

```
Training Window → Validation → OOS Test → Move Forward → Retrain → Validate → Test → ...
```

### 40.2 Configuration

```python
@dataclass(frozen=True)
class WalkForwardConfig:
    training_window: int             # Periods per training window
    validation_window: int           # Periods per validation window
    test_window: int                 # Periods per test window
    step_size: int                   # Periods to move forward
    n_windows: int                   # Number of walk-forward windows
    refit_frequency: int             # Refit every N periods
```

### 40.3 Output

| Metric | Description |
|---|---|
| **Window-level results** | Performance for each walk-forward window |
| **Aggregate results** | Mean, median, std across windows |
| **Parameter stability** | Consistency of optimal parameters |
| **Drawdown stability** | Drawdown consistency across windows |
| **Performance dispersion** | Range of outcomes |
| **Failure windows** | Windows where strategy underperformed |

### 40.4 Reporting

- Walk-forward results are reported as distributions, not single values.
- Failure windows are explicitly identified and analyzed.

---

## 41. Monte Carlo Architecture

### 41.1 Simulation Types

| Type | Description |
|---|---|
| **Trade-sequence reshuffling** | Randomize trade order, recompute equity curve |
| **Bootstrap sampling** | Resample returns with replacement |
| **Return resampling** | Resample from historical return distribution |
| **Parameter perturbation** | Randomize parameters within ranges |
| **Slippage perturbation** | Randomize slippage assumptions |
| **Transaction-cost perturbation** | Randomize cost assumptions |

### 41.2 Configuration

```python
@dataclass(frozen=True)
class MonteCarloConfig:
    simulation_type: str
    n_simulations: int               # e.g., 10000
    parameter_ranges: Dict[str, Tuple[float, float]]
    slippage_range: Tuple[Decimal, Decimal]
    cost_range: Tuple[Decimal, Decimal]
    random_seed: int
    confidence_level: float
```

### 41.3 Output Distributions

Generate distributions for:
- Return, CAGR
- Maximum drawdown
- Sharpe, Sortino, Calmar
- Probability of loss
- Drawdown duration

### 41.4 Insufficient Sample Handling

- When `n_simulations` is insufficient, return NaN with warning.
- **Never** silently return a default value.

---

## 42. Robustness Testing

### 42.1 Stress Dimensions

| Dimension | Test |
|---|---|
| **Parameter changes** | Performance across parameter perturbations |
| **Increased transaction costs** | Performance with 2x, 5x costs |
| **Increased slippage** | Performance with 2x, 5x slippage |
| **Execution delays** | Performance with 1-bar, 2-bar delays |
| **Missing data** | Performance with randomly removed data |
| **Different markets** | Performance across markets |
| **Different timeframes** | Performance across timeframes |
| **Different market regimes** | Performance in bull/bear/sideways/high-vol/low-vol |
| **Different volatility environments** | Performance across volatility regimes |

### 42.2 Robustness Report

- Performance degradation under each stress dimension.
- Identification of fragile strategies.
- Pass/fail against configurable robustness criteria.

---

## 43. Sensitivity Analysis

### 43.1 Analysis Types

| Type | Description |
|---|---|
| **One-parameter sensitivity** | Vary one parameter, hold others constant |
| **Two-parameter sensitivity** | Heatmap of two parameters |
| **Parameter stability** | Identify stable performance regions |

### 43.2 Output

- Performance surface (return, Sharpe, drawdown) across parameter space.
- Identification of **stable regions** (broad plateaus).
- **Flagging** of strategies whose performance depends on extremely precise parameters.

### 43.3 Configuration

```python
@dataclass(frozen=True)
class SensitivityConfig:
    parameters: Dict[str, Tuple[float, float, float]]  # (min, max, step)
    metrics: List[str]                 # ["return", "sharpe", "drawdown"]
    stability_threshold: float         # Min stable region size
```

---

## 44. Overfitting Controls

### 44.1 Detection Metrics

| Metric | Threshold (configurable) | Flag |
|---|---|---|
| **Excessive parameters** | > N parameters | Complexity warning |
| **Very few trades** | < M trades | Statistical significance warning |
| **Extremely high in-sample Sharpe** | > threshold | Overfitting suspicion |
| **Large OOS degradation** | OOS Sharpe < X% of IS Sharpe | Overfitting flag |
| **Parameter instability** | High variance across windows | Robustness flag |
| **Narrow profitable regions** | Small stable parameter region | Overfitting flag |
| **Excessive strategy complexity** | > N conditions/indicators | Complexity warning |
| **High sensitivity to costs** | Performance drops > X% with 2x costs | Fragility flag |
| **High sensitivity to slippage** | Performance drops > X% with 2x slippage | Fragility flag |

### 44.2 Policy

- Overfitting flags are **warnings, not automatic rejections**.
- **Never** automatically declare a model robust.
- Overfitting assessment is reported alongside performance metrics.

---

## 45. Feature-Store Design

See Section 9 (Feature-Store Architecture) for the complete design. Key additional points:

### 45.1 Caching Strategy

- Features cached in Redis with TTL.
- Cache key includes `feature_version` and `dataset_version`.
- Cache invalidation on new data arrival.
- **Stale cache values never silently replace newer real-time information.**

### 45.2 Feature Lineage

- Every feature records its complete lineage: source data → transformations → parameters → version.
- Feature dependencies are tracked as a DAG.

---

## 46. Model Registry

### 46.1 Registry Workflow

```
EXPERIMENT → MODEL → VALIDATION → APPROVAL → PAPER TRADING → LIVE ELIGIBILITY
```

### 46.2 Model States

| State | Description |
|---|---|
| `EXPERIMENT` | Training in progress |
| `VALIDATION` | Under validation review |
| `APPROVED` | Approved for paper trading |
| `PAPER_TRADING` | Running in paper trading |
| `LIVE_ELIGIBLE` | Eligible for live trading (requires explicit authorization) |
| `DEPRECATED` | No longer in use |
| `ARCHIVED` | Retained for historical reference |

### 46.3 Model Record

```python
@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    model_version: str
    training_dataset: str            # dataset_version
    feature_set: str                 # feature_set_version
    training_period: DateRange
    validation_period: DateRange
    test_period: DateRange
    algorithm: str
    hyperparameters: Dict[str, Any]
    random_seed: int
    preprocessing: str               # preprocessing_version
    performance_metrics: Dict[str, float]
    created_at: datetime
    created_by: str
    state: ModelState
    approved_by: Optional[str]
    approved_at: Optional[datetime]
```

### 46.4 No Silent Replacement

- A model update **always** creates a new `model_version`.
- Production models are never silently replaced.
- Model state transitions are audited.

---

## 47. Model Versioning

### 47.1 Versioning Strategy

- **Semantic versioning:** `major.minor.patch`
  - `major`: Breaking change to features, target, or architecture
  - `minor`: New features or improved training
  - `patch`: Bug fixes, minor improvements
- Every version is immutable once created.
- Model artifacts stored in object storage with version identifier.

### 47.2 Version Lineage

- Parent version recorded (which version this was derived from).
- Training code version (git commit hash) recorded.
- Full configuration hash recorded.

### 47.3 Rollback

- Previous versions remain available for rollback.
- Rollback is an audited operation.

---

## 48. Paper-Trading Architecture

### 48.1 Core Principle

**Paper trading uses the same strategy engine, feature engine, risk engine, portfolio engine, and order engine as live trading. Only the execution adapter differs.**

### 48.2 Architecture

```
Live Market Data → Feature Engine → Strategy Engine → Risk Engine → Portfolio Engine → Order Engine → Simulated Execution Adapter → Paper Portfolio
```

### 48.3 Simulated Execution Adapter

```python
class SimulatedExecutionAdapter(ExecutionAdapter):
    def submit_order(self, order: Order) -> ExecutionResult:
        # Simulate fills based on market data
        # Model slippage, partial fills, liquidity constraints
        # Record all simulated executions
        ...

    def get_account(self) -> AccountState:
        # Return simulated account state
        ...

    def get_positions(self) -> List[Position]:
        # Return simulated positions
        ...
```

### 48.4 Paper Trading Features

- Real-time market data (or delayed data for testing).
- Realistic execution simulation (slippage, partial fills, liquidity).
- Full signal logging and audit trail.
- Performance comparison against backtest expectations.
- **No real money at risk.**

### 48.5 Paper Trading Configuration

```python
@dataclass(frozen=True)
class PaperTradingConfig:
    strategy_version: str
    model_version: Optional[str]
    initial_capital: Decimal
    instruments: List[str]
    risk_config: RiskConfig
    transaction_costs: TransactionCostConfig
    slippage: SlippageConfig
    liquidity: LiquidityConfig
    execution_model: str            # "realistic", "optimistic", "pessimistic"
```

---

## 49. Broker Abstraction

### 49.1 Broker Adapter Interface

```python
class BrokerAdapter(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def get_account(self) -> AccountState: ...

    @abstractmethod
    async def get_positions(self) -> List[Position]: ...

    @abstractmethod
    async def get_orders(self) -> List[Order]: ...

    @abstractmethod
    async def submit_order(self, order: Order) -> SubmissionResult: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> CancellationResult: ...

    @abstractmethod
    async def replace_order(self, order_id: str, new_order: Order) -> SubmissionResult: ...

    @abstractmethod
    async def get_market_data(self, instrument_id: str) -> MarketData: ...

    @abstractmethod
    def get_capabilities(self) -> BrokerCapabilities: ...
```

### 49.2 Broker Capabilities

```python
@dataclass(frozen=True)
class BrokerCapabilities:
    supported_order_types: Set[OrderType]
    supports_bracket_orders: bool
    supports_stop_limit: bool
    supports_trailing_stops: bool
    market_data_available: bool
    sandbox_available: bool
    max_orders_per_second: int
```

### 49.3 Implementation Rule

- **Never hard-code broker-specific logic throughout the application.**
- Broker-specific code lives only in the adapter implementation.
- The rest of the system depends only on the `BrokerAdapter` interface.

---

## 50. Order Architecture

### 50.1 Order Record

```python
@dataclass(frozen=True)
class Order:
    order_id: str                    # Internal UUID
    client_order_id: str             # Idempotency key
    strategy_id: str
    portfolio_id: str
    instrument_id: str
    side: Side                       # BUY, SELL
    quantity: Decimal
    order_type: OrderType            # MARKET, LIMIT, STOP, STOP_LIMIT, BRACKET
    limit_price: Optional[Decimal]
    stop_price: Optional[Decimal]
    trailing_amount: Optional[Decimal]
    time_in_force: TimeInForce       # DAY, GTC, IOC, FOK
    status: OrderStatus
    created_at: datetime
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    broker_order_id: Optional[str]
    idempotency_key: str
```

### 50.2 Order Types

| Type | Description |
|---|---|
| **Market** | Execute at best available price |
| **Limit** | Execute at specified price or better |
| **Stop** | Market order triggered at stop price |
| **Stop-limit** | Limit order triggered at stop price |
| **Bracket** | Entry + stop-loss + take-profit (where supported) |

### 50.3 Order State Machine

```
CREATED → SUBMITTED → ACCEPTED → PARTIALLY_FILLED → FILLED
                                    ↓
                              CANCELLED / REJECTED / EXPIRED
                                    ↓
                                UNKNOWN (requires reconciliation)
```

### 50.4 Order Status Tracking

Every status transition is recorded with timestamp and reason.

---

## 51. Position Reconciliation

### 51.1 Reconciliation Process

```
Internal Position State ↔ Broker Position State
         ↓
    Compare quantity, average entry, current price
         ↓
    If discrepancy → TRADING STATE = REQUIRES_RECONCILIATION
```

### 51.2 Reconciliation Record

```python
@dataclass(frozen=True)
class ReconciliationResult:
    instrument_id: str
    internal_quantity: Decimal
    broker_quantity: Decimal
    discrepancy: Decimal
    status: ReconciliationStatus     # MATCHED, DISCREPANCY, UNKNOWN
    resolved_at: Optional[datetime]
    resolution: Optional[str]
```

### 51.3 Discrepancy Handling

- Discrepancies trigger an alert.
- Trading may be halted for the affected instrument.
- Reconciliation is an audited operation.
- **Never assume internal state matches broker state without verification.**

---

## 52. Execution Confirmation

### 52.1 Fill Record

```python
@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    broker_order_id: str
    instrument_id: str
    side: Side
    quantity: Decimal
    price: Decimal
    commission: Decimal
    fees: Decimal
    slippage: Decimal
    execution_time: datetime
    latency_ms: int
```

### 52.2 Confirmation Tracking

- **Never assume order submitted = order filled.**
- Track: `CREATED → SUBMITTED → ACCEPTED → PARTIALLY_FILLED → FILLED`
- Unknown states trigger reconciliation.
- Every fill records slippage and latency.

---

## 53. Latency Monitoring

### 53.1 Latency Stages

| Stage | Measurement |
|---|---|
| `market_data_latency` | Feed event time → ingestion time |
| `feature_latency` | Data available → feature computed |
| `model_latency` | Features available → prediction generated |
| `signal_latency` | Prediction → signal generated |
| `risk_latency` | Signal → risk check complete |
| `order_submission_latency` | Order created → submitted to broker |
| `broker_response_latency` | Order submitted → broker response |
| `execution_latency` | Order submitted → fill confirmed |

### 53.2 Reporting

- Latency distributions (p50, p90, p99, max), not just averages.
- Alerts on excessive latency.
- Latency recorded per order for post-trade analysis.

---

## 54. Signal Logging

### 54.1 Signal Record

Every signal is logged with complete context:

```python
@dataclass(frozen=True)
class SignalRecord:
    signal_id: str
    timestamp: datetime
    instrument_id: str
    strategy_id: str
    strategy_version: str
    signal: str                      # "LONG", "SHORT", "EXIT", etc.
    feature_values: Dict[str, float]
    indicator_values: Dict[str, float]
    model_prediction: Optional[float]
    model_version: Optional[str]
    regime: Optional[str]
    risk_state: str
    position: Decimal
    decision: str                    # "ENTER", "EXIT", "HOLD", etc.
    metadata: Dict[str, Any]
```

### 54.2 Purpose

- Complete post-trade analysis.
- Strategy debugging.
- Regulatory compliance.
- Performance attribution.

---

## 55. Error-Alert Architecture

### 55.1 Alert Categories

| Category | Examples |
|---|---|
| **Broker** | Disconnection, rejection, timeout |
| **Data feed** | Failure, staleness, gaps |
| **Reconciliation** | Position mismatch, order mismatch |
| **Risk** | Limit breach, kill switch activation |
| **Model** | Failure, drift, excessive latency |
| **Strategy** | Crash, unexpected behavior |
| **System** | Database failure, cache failure, queue failure |
| **Security** | Unauthorized access, credential exposure |

### 55.2 Alert Levels

| Level | Description | Response |
|---|---|---|
| `INFO` | Informational | Log only |
| `WARNING` | Potential issue | Log + notify |
| `ERROR` | Operational issue | Log + notify + investigate |
| `CRITICAL` | Immediate threat | Log + notify + automatic action (e.g., kill switch) |

### 55.3 Alert Routing

- In-app notifications
- Email
- Webhook (configurable)
- SMS (for critical alerts)

### 55.4 Alert Configuration

```python
@dataclass(frozen=True)
class AlertRule:
    category: str
    condition: str
    threshold: float
    level: AlertLevel
    channels: List[str]
    cooldown_seconds: int
    auto_action: Optional[str]       # e.g., "kill_switch"
```

---

## 56. Portfolio Dashboard

### 56.1 Dashboard Components

| Component | Data |
|---|---|
| **Equity curve** | Overlaid with benchmark |
| **Daily P&L** | Bar chart with cumulative |
| **Unrealized P&L** | Per position |
| **Realized P&L** | Per trade, cumulative |
| **Exposure** | Gross, net, by asset class |
| **Positions** | Table with entry, current, P&L |
| **Sector exposure** | Pie/bar chart |
| **Asset-class exposure** | Pie/bar chart |
| **Correlations** | Heatmap |
| **Portfolio volatility** | Rolling |
| **VaR / CVaR** | Current values with confidence |
| **Drawdown** | Underwater chart |
| **Sharpe / Sortino / Calmar** | Summary statistics |
| **Beta** | Rolling vs benchmark |

### 56.2 Data Sources

- Real-time portfolio state from the portfolio engine.
- Historical performance from the backtest engine.
- Risk metrics from the risk engine.

---

## 57. API Security

### 57.1 Security Controls

| Control | Implementation |
|---|---|
| **Authentication** | OAuth2 / OIDC |
| **Authorization** | RBAC with fine-grained permissions |
| **Input validation** | Pydantic models, strict schemas |
| **Rate limiting** | Token bucket per user/endpoint |
| **API versioning** | URL-based (`/v1/`, `/v2/`) |
| **Secure headers** | HSTS, CSP, X-Frame-Options, etc. |
| **CORS** | Restricted to known origins |
| **CSRF protection** | Token-based (where applicable) |
| **Request-size limits** | Configured per endpoint |
| **Audit logging** | All API calls logged |

### 57.2 Threat Prevention

| Threat | Prevention |
|---|---|
| **BOLA** | Object-level authorization checks on every request |
| **IDOR** | UUID-based resource IDs, ownership verification |
| **SQL injection** | Parameterized queries (ORM), input validation |
| **XSS** | Output encoding, CSP headers |
| **SSRF** | URL allowlists, network restrictions |
| **Authentication bypass** | Token validation, session management |
| **Privilege escalation** | RBAC, permission checks on every operation |

### 57.3 Security Headers

```
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: camera=(), microphone=(), geolocation=()
```

---

## 58. Rate Limiting

### 58.1 Rate Limit Categories

| Category | Limit | Scope |
|---|---|---|
| **Authentication** | 10 req/min | Per IP |
| **Market data queries** | 100 req/min | Per user |
| **Backtests** | 10 concurrent | Per user |
| **ML training** | 5 concurrent | Per user |
| **Model inference** | 1000 req/min | Per user |
| **Broker APIs** | Per broker limits | Per broker connection |
| **Order submission** | 10 req/min | Per portfolio |
| **Order cancellation** | 20 req/min | Per portfolio |
| **Strategy creation** | 50 req/min | Per user |
| **General API** | 1000 req/min | Per user |

### 58.2 Implementation

- Token bucket algorithm.
- Redis-backed distributed rate limiting.
- Rate limit headers in responses (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`).
- 429 status code with `Retry-After` header when exceeded.

### 58.3 Research-Heavy Operations

- Backtests, ML training, Monte Carlo, walk-forward → **job queue** (not blocking API requests).
- API returns `job_id` immediately; client polls or receives webhook on completion.

---

## 59. Authentication

### 59.1 Authentication Methods

| Method | Use Case |
|---|---|
| **OAuth2 / OIDC** | Web application, SSO |
| **API keys** | Programmatic access, integrations |
| **MFA** | Required for privileged operations |

### 59.2 Token Management

- Short-lived access tokens (15 min).
- Refresh tokens with rotation.
- Token revocation on logout or suspicious activity.
- Secure token storage (HttpOnly, Secure, SameSite cookies).

### 59.3 Session Management

- Session timeout (configurable).
- Concurrent session limits.
- Session invalidation on password change.

---

## 60. Authorization

### 60.1 Roles

| Role | Description |
|---|---|
| **Researcher** | Research and backtesting |
| **Quant Developer** | Strategy and feature development |
| **Portfolio Manager** | Portfolio management and authorized trading |
| **Risk Manager** | Risk limits and risk monitoring |
| **Administrator** | Platform administration |
| **Security Administrator** | Security monitoring and incident response |

### 60.2 Permissions

| Permission | Description |
|---|---|
| `CAN_RESEARCH` | Access research tools |
| `CAN_BACKTEST` | Create and run backtests |
| `CAN_TRAIN_MODEL` | Train ML models |
| `CAN_PAPER_TRADE` | Run paper trading |
| `CAN_CONFIGURE_RISK` | Modify risk limits |
| `CAN_CONNECT_BROKER` | Connect broker accounts |
| `CAN_ENABLE_LIVE_TRADING` | Enable live trading |
| `CAN_SUBMIT_LIVE_ORDER` | Submit live orders |
| `CAN_CANCEL_LIVE_ORDER` | Cancel live orders |
| `CAN_ACTIVATE_KILL_SWITCH` | Activate kill switches |

### 60.3 Permission Enforcement

- Every API endpoint checks permissions.
- Every service operation checks permissions.
- Permission checks are audited.
- **Principle of least privilege** by default.

---

## 61. Secret Management

### 61.1 Secret Storage

- **HashiCorp Vault** (or cloud KMS) for all secrets.
- Broker API keys, database credentials, API keys, certificates.
- **Never** store secrets in code, configuration files, environment variables (except for non-sensitive config), or logs.

### 61.2 Secret Access

- Services authenticate to Vault via short-lived tokens.
- Dynamic secrets with automatic rotation.
- Secret access is audited.

### 61.3 Secret Exposure Prevention

Secrets are **never** exposed to:
- Frontend
- Logs
- Browser local storage
- Client-side JavaScript
- Model prompts
- Strategy sandbox

### 61.4 Broker Credential Security

- Broker API keys stored in Vault.
- Prefer broker permissions that **cannot withdraw funds**.
- Separate API keys for trading vs. market data.
- Regular credential rotation.

---

## 62. Strategy Sandbox

### 62.1 Isolation Requirements

Custom strategies execute in an isolated environment with:

| Resource | Limit |
|---|---|
| **CPU** | Configurable (e.g., 1 core) |
| **Memory** | Configurable (e.g., 512 MB) |
| **Time** | Configurable timeout (e.g., 30 seconds) |
| **Network** | Restricted (no external access) |
| **Filesystem** | Read-only except for designated output directory |
| **System calls** | Restricted (seccomp/AppArmor) |

### 62.2 Prohibited Access

A custom strategy **must not** access:
- Broker credentials
- Other users' data
- Database credentials
- Internal services
- Operating-system secrets
- Network resources (except explicitly allowed data feeds)

### 62.3 Implementation

- Container-based isolation (Docker with resource limits).
- Seccomp profile for syscall filtering.
- Network namespace isolation.
- Read-only filesystem.
- Resource limits enforced by container runtime.

### 62.4 Strategy API

Custom strategies interact with the platform through a restricted, well-defined API:

```python
class StrategyContext:
    def get_features(self, instrument_id: str) -> Dict[str, float]: ...
    def get_indicator(self, name: str, **params) -> float: ...
    def get_regime(self) -> str: ...
    def get_position(self, instrument_id: str) -> Decimal: ...
    def submit_order(self, order: OrderRequest) -> OrderResult: ...
    # No access to credentials, database, or internal services
```

---

## 63. Audit Trails

### 63.1 Audited Events

| Category | Events |
|---|---|
| **Strategy** | Creation, modification, deletion, version changes |
| **Model** | Training, deployment, version changes, approval |
| **Feature** | Definition changes, version changes |
| **Backtest** | Creation, execution, results |
| **Dataset** | Version creation, data changes |
| **Broker** | Connection, disconnection, credential changes |
| **Risk** | Limit changes, breach events |
| **Paper trading** | Activation, deactivation, configuration changes |
| **Live trading** | Activation, deactivation, authorization |
| **Orders** | Creation, submission, cancellation, fills |
| **Positions** | Changes, reconciliation events |
| **Kill switch** | Activation, deactivation |
| **User** | Login, logout, permission changes |
| **API** | All API calls (who, what, when, result) |

### 63.2 Audit Log Properties

- **Append-only:** Records cannot be modified or deleted.
- **Tamper-evident:** Hash-chained (each record includes hash of previous record).
- **Timestamped:** UTC timestamp with microsecond precision.
- **Attributed:** User ID, session ID, IP address.
- **Complete:** Before/after values for changes.

### 63.3 Audit Log Schema

```sql
CREATE TABLE audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id UUID,
    session_id VARCHAR(100),
    ip_address INET,
    resource_type VARCHAR(100),
    resource_id VARCHAR(100),
    action VARCHAR(100),
    before_value JSONB,
    after_value JSONB,
    metadata JSONB,
    previous_hash VARCHAR(64),
    record_hash VARCHAR(64) NOT NULL
);
```

---

## 64. Emergency Shutdown Architecture

### 64.1 Kill Switch Levels

| Level | Scope |
|---|---|
| **GLOBAL KILL SWITCH** | Halt all trading across all portfolios |
| **PORTFOLIO KILL SWITCH** | Halt all trading for a specific portfolio |
| **STRATEGY KILL SWITCH** | Halt a specific strategy |
| **BROKER CONNECTION KILL SWITCH** | Disconnect a specific broker |

### 64.2 Automatic Shutdown Triggers

| Trigger | Threshold | Action |
|---|---|---|
| **Maximum daily loss** | Configurable % | Halt trading, alert |
| **Maximum drawdown** | Configurable % | Halt trading, alert |
| **Broker reconciliation failure** | Any failure | Halt affected portfolio |
| **Stale market data** | Beyond threshold | Block new orders |
| **Excessive execution latency** | Beyond threshold | Halt trading |
| **Unexpected position** | Any mismatch | Halt affected portfolio |
| **Risk engine failure** | Any failure | Halt all trading |
| **Model service failure** | Any failure | Halt affected strategies |
| **Data corruption** | Any detection | Halt all trading |
| **Repeated order rejection** | N rejections in M minutes | Halt affected strategy |
| **Connectivity uncertainty** | Broker timeout | Halt affected portfolio |

### 64.3 Kill Switch Configuration

```python
@dataclass(frozen=True)
class KillSwitchConfig:
    max_daily_loss_pct: Decimal
    max_drawdown_pct: Decimal
    stale_data_threshold_seconds: float
    max_execution_latency_ms: int
    max_order_rejections: int
    order_rejection_window_minutes: int
    auto_halt_on_risk_engine_failure: bool
    auto_halt_on_model_failure: bool
    auto_halt_on_data_corruption: bool
```

### 64.4 Kill Switch Activation

- Activation is an audited operation.
- All open orders are cancelled (configurable).
- All positions are flattened (configurable, requires authorization).
- Alert is sent to all relevant parties.
- Deactivation requires explicit authorization and investigation.

---

## 65. Database Schema

### 65.1 Schema Overview

```
quantforge/
├── identity/          # Users, roles, permissions, sessions, API keys
├── instruments/       # Instruments, instrument history, corporate actions
├── market_data/       # Raw, normalized, adjusted price data (TimescaleDB)
├── fundamentals/      # Fundamental observations
├── macro/             # Macro economic observations
├── sentiment/         # News, social, analyst revisions
├── features/          # Feature definitions, feature values
├── research/          # Indicator definitions, regime labels
├── ml/                # Experiments, models, training runs
├── strategies/        # Strategy definitions, versions
├── backtests/         # Backtest configurations, results
├── risk/              # Risk configurations, limit breaches, kill switch events
├── portfolio/         # Portfolio state, positions, trades
├── paper_trading/     # Paper trading accounts, state
├── execution/         # Orders, fills, broker connections
├── audit/             # Audit log (append-only, hash-chained)
├── monitoring/        # Health checks, alerts, metrics
└── data_quality/      # Validation rules, quality reports
```

### 65.2 Key Tables

#### Instruments
```sql
CREATE TABLE instruments (
    instrument_id UUID PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    asset_class VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    currency CHAR(3) NOT NULL,
    country CHAR(2) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    tick_size NUMERIC(20, 8) NOT NULL,
    lot_size INT NOT NULL DEFAULT 1,
    trading_calendar VARCHAR(50) NOT NULL,
    contract_specs JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### Market Data (TimescaleDB Hypertable)
```sql
CREATE TABLE market_data_daily (
    instrument_id UUID REFERENCES instruments(id),
    event_timestamp TIMESTAMPTZ NOT NULL,
    open_price NUMERIC(20, 8),
    high_price NUMERIC(20, 8),
    low_price NUMERIC(20, 8),
    close_price NUMERIC(20, 8),
    volume BIGINT,
    adjusted_close NUMERIC(20, 8),
    source VARCHAR(100),
    ingestion_timestamp TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (instrument_id, event_timestamp)
);
SELECT create_hypertable('market_data_daily', 'event_timestamp');
```

#### Orders
```sql
CREATE TABLE orders (
    order_id UUID PRIMARY KEY,
    client_order_id VARCHAR(100) UNIQUE NOT NULL,
    strategy_id UUID REFERENCES strategies(strategy_id),
    portfolio_id UUID REFERENCES portfolios(portfolio_id),
    instrument_id UUID REFERENCES instruments(id),
    side VARCHAR(10) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    limit_price NUMERIC(20, 8),
    stop_price NUMERIC(20, 8),
    time_in_force VARCHAR(10) DEFAULT 'DAY',
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    filled_at TIMESTAMPTZ,
    broker_order_id VARCHAR(100),
    idempotency_key VARCHAR(100) UNIQUE NOT NULL
);
```

#### Audit Log
```sql
CREATE TABLE audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id UUID,
    session_id VARCHAR(100),
    ip_address INET,
    resource_type VARCHAR(100),
    resource_id VARCHAR(100),
    action VARCHAR(100),
    before_value JSONB,
    after_value JSONB,
    metadata JSONB,
    previous_hash VARCHAR(64),
    record_hash VARCHAR(64) NOT NULL
);
```

---

## 66. API Architecture

### 66.1 API Style

- **RESTful** for CRUD operations.
- **WebSocket** for real-time data (market data, portfolio updates, alerts).
- **Async job pattern** for long-running operations (backtests, ML training).

### 66.2 API Versioning

- URL-based versioning: `/api/v1/`, `/api/v2/`.
- Breaking changes require a new version.
- Deprecation notices with sunset dates.

### 66.3 API Endpoints (Selected)

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/instruments` | GET | List instruments |
| `/api/v1/market-data/{symbol}` | GET | Get market data |
| `/api/v1/features/{symbol}` | GET | Get features (point-in-time) |
| `/api/v1/strategies` | POST | Create strategy |
| `/api/v1/backtests` | POST | Submit backtest job |
| `/api/v1/backtests/{id}` | GET | Get backtest result |
| `/api/v1/models` | POST | Train model |
| `/api/v1/models/{id}` | GET | Get model details |
| `/api/v1/portfolio` | GET | Get portfolio state |
| `/api/v1/orders` | POST | Submit order |
| `/api/v1/risk/limits` | GET/PUT | Get/update risk limits |
| `/api/v1/kill-switch` | POST | Activate kill switch |
| `/api/v1/audit` | GET | Query audit log |

### 66.4 Request/Response Format

- JSON for request/response bodies.
- Standard error format: `{ "error": { "code": "...", "message": "...", "details": {...} } }`.
- Pagination for list endpoints.
- Filtering and sorting via query parameters.

### 66.5 WebSocket Events

| Event | Description |
|---|---|
| `market_data` | Real-time price updates |
| `portfolio_update` | Position/P&L changes |
| `order_update` | Order status changes |
| `alert` | Alert notifications |
| `job_progress` | Backtest/ML job progress |

---

## 67. Testing Architecture

### 67.1 Test Categories

| Category | Tools | Coverage |
|---|---|---|
| **Unit tests** | pytest | Indicators, returns, statistics, risk metrics, sizing, stops, conditions |
| **Data tests** | Great Expectations | Missing data, corporate actions, timestamps, adjusted prices, point-in-time fundamentals |
| **ML tests** | pytest + custom | Leakage, train/test separation, preprocessing, feature-store retrieval, reproducibility, serialization, versioning |
| **Backtest tests** | pytest + custom | Look-ahead bias, survivorship bias, transaction costs, slippage, liquidity, corporate actions |
| **Security tests** | pytest + OWASP ZAP | Authentication, authorization, BOLA, API security, secret exposure, sandbox escape, webhook forgery, replay attacks |
| **Live execution tests** | Broker sandbox | Order submission, cancellation, reconciliation (sandbox only) |
| **Integration tests** | pytest | End-to-end workflows |
| **Property-based tests** | hypothesis | Statistical properties, edge cases |

### 67.2 Critical Edge Cases (from Section 98)

| Edge Case | Test |
|---|---|
| Missing candle | Verify NaN propagation, no crash |
| Duplicate candle | Verify deduplication |
| Incorrect timestamp | Verify rejection |
| Corporate action | Verify adjustment correctness |
| Delisted company | Verify universe inclusion |
| Fundamental restatement | Verify point-in-time access |
| Sentiment arriving late | Verify temporal alignment |
| Analyst revision arriving late | Verify temporal alignment |
| Macro revision | Verify point-in-time access |
| Stale data | Verify order blocking |
| Broker timeout | Verify reconciliation, no duplicate orders |
| Unknown order state | Verify reconciliation |
| Partial fill | Verify position tracking |
| Duplicate webhook | Verify idempotency |
| Position mismatch | Verify reconciliation alert |
| Strategy crash | Verify graceful degradation |
| Model crash | Verify fallback behavior |
| Risk engine fail | Verify trading halt |
| Kill switch activation | Verify immediate halt |
| Maximum loss breach | Verify automatic halt |
| Excessive leverage | Verify order rejection |
| Excessive concentration | Verify order rejection |
| Singular covariance matrix | Verify regularization |
| Insufficient Monte Carlo samples | Verify NaN with warning |
| Insufficient training data | Verify graceful handling |
| Extreme outliers | Verify flagging, no auto-deletion |

### 67.3 Test Data

- Synthetic data generators for testing.
- Reference datasets with known outcomes.
- Property-based test generators.

### 67.4 CI/CD Integration

- All tests run on every pull request.
- Security scanning (SAST, DAST, dependency scanning).
- Code coverage reporting.
- Performance regression testing.

---

## 68. Observability

### 68.1 Metrics (Prometheus)

| Category | Metrics |
|---|---|
| **Application** | Request rate, error rate, latency (p50, p90, p99) |
| **Backtest** | Queue depth, execution time, success rate |
| **ML** | Training time, inference latency, model drift metrics |
| **Market Data** | Feed latency, message rate, gap count |
| **Execution** | Order latency, fill rate, slippage |
| **Risk** | Limit utilization, breach count, kill switch events |
| **Infrastructure** | CPU, memory, disk, network |

### 68.2 Logging (Structured)

- JSON-formatted logs.
- Correlation IDs for request tracing.
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
- Sensitive data redaction.

### 68.3 Tracing (OpenTelemetry)

- Distributed tracing across services.
- Span per operation (data fetch, feature calc, model inference, order submission).
- Trace context propagation via headers.

### 68.4 Dashboards (Grafana)

| Dashboard | Content |
|---|---|
| **System Health** | Service status, resource utilization, error rates |
| **Trading** | Positions, P&L, exposure, risk metrics |
| **Market Data** | Feed health, latency, gaps |
| **ML Operations** | Model performance, drift, training jobs |
| **Backtest Operations** | Queue status, execution time, results summary |

### 68.5 Alerting

- Prometheus Alertmanager for metric-based alerts.
- PagerDuty/Opsgenie integration for on-call routing.
- Alert routing by severity and category.

---

## 69. Disaster Recovery

### 69.1 Failure Scenarios & Recovery

| Scenario | Recovery Behavior |
|---|---|
| **Database failure** | Failover to replica; queue jobs for retry; alert |
| **Broker outage** | Halt affected portfolio; queue orders for retry; alert |
| **Market-data outage** | Block new orders; use last known prices with staleness flag; alert |
| **Application restart** | Recover state from database; reconcile positions; resume jobs |
| **Network partition** | Halt trading; reconcile on recovery; alert |
| **Duplicate messages** | Idempotent processing; deduplicate by event ID |
| **Lost order response** | Reconcile with broker; never auto-resubmit |
| **Corrupted feature store** | Recompute from source data; alert; halt affected strategies |
| **Model service failure** | Halt affected strategies; fallback to simpler model or no-model mode |

### 69.2 Recovery Principles

- **No accidental duplicate live orders** during recovery.
- **State reconciliation** before resuming trading.
- **Audit trail** of all recovery actions.
- **Graceful degradation** (reduce functionality rather than fail completely).

### 69.3 Backup Strategy

| Data | Backup Frequency | Retention |
|---|---|---|
| PostgreSQL | Continuous (WAL) + daily full | 30 days |
| Redis | Every 15 minutes | 7 days |
| Object Storage | Versioned | Indefinite |
| Audit Log | Real-time replication | Indefinite |

### 69.4 Recovery Time Objectives

| Component | RTO | RPO |
|---|---|---|
| API | 5 minutes | 0 |
| Market Data | 1 minute | 0 |
| Trading Engine | 1 minute | 0 |
| Database | 5 minutes | 1 minute |
| Backtest Engine | 30 minutes | N/A |

---

## 70. Deployment Architecture

### 70.1 Environments

| Environment | Purpose |
|---|---|
| **Development** | Local development (Docker Compose) |
| **Staging** | Pre-production testing |
| **Production** | Live system |

### 70.2 Production Architecture

```
                    ┌─────────────┐
                    │   CDN/WAF   │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  API Gateway │
                    │  (Ingress)   │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │ API     │       │ WebSocket│       │ Worker  │
   │ Service │       │ Service  │       │ Nodes   │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │PostgreSQL│       │  Redis  │       │   S3    │
   │+Timescale│       │         │       │         │
   └─────────┘       └─────────┘       └─────────┘
```

### 70.3 Container Orchestration (Kubernetes)

- **Deployments:** API service, WebSocket service, worker nodes.
- **StatefulSets:** PostgreSQL, Redis.
- **Horizontal Pod Autoscaling:** Based on CPU/memory/custom metrics.
- **Rolling updates:** Zero-downtime deployments.
- **Resource limits:** CPU/memory requests and limits for all pods.
- **Liveness/readiness probes:** For all services.
- **Pod disruption budgets:** Ensure availability during maintenance.

### 70.4 CI/CD Pipeline

```
Code → Lint → Unit Tests → Integration Tests → Security Scan → Build → Deploy to Staging → E2E Tests → Deploy to Production
```

### 70.5 Infrastructure as Code

- Terraform for all infrastructure.
- Helm charts for Kubernetes deployments.
- Environment-specific variable files.

---

## 71. MVP Scope

### 71.1 MVP Components

The MVP implements the core pipeline:

```
Historical market data → Data validation → Adjusted prices → Lagged features →
SMA / EMA / WMA → RSI / MACD → Bollinger Bands → ATR → Volume features →
Basic fundamentals → Return engine → Risk metrics → Strategy builder →
Backtesting → Transaction costs → Slippage → Position sizing → Stops →
In-sample / out-of-sample → Walk-forward → Paper trading
```

### 71.2 MVP Feature Checklist

| Component | MVP Scope |
|---|---|
| **Data** | Historical daily OHLCV, data validation, corporate actions (splits/dividends), adjusted prices |
| **Features** | Lagged prices/returns, SMA/EMA/WMA, RSI/MACD, Bollinger Bands, ATR, volume features |
| **Fundamentals** | Basic metrics (P/E, P/B, ROE, debt/equity) with point-in-time access |
| **Returns** | Simple, log, cumulative, annualized, rolling, excess |
| **Risk Metrics** | Sharpe, Sortino, Calmar, max drawdown, volatility, VaR (historical) |
| **Strategy** | Crossover, breakout, mean reversion conditions; fixed fractional + ATR sizing; percentage + ATR stops |
| **Backtesting** | Event-driven engine, transaction costs, slippage, basic liquidity |
| **Validation** | In-sample/out-of-sample split, walk-forward |
| **Paper Trading** | Simulated execution with realistic fills |
| **Security** | Authentication, RBAC, API security, audit logging |
| **Monitoring** | Basic health checks, alerting |

### 71.3 MVP Exclusions (Deferred)

- Real-time feeds (use historical data for backtesting)
- Broker integration (paper trading only)
- Live trading (disabled)
- Macro data, sentiment, regime detection
- ML models (random forest, gradient boosting, neural networks)
- Clustering, PCA, anomaly detection
- Monte Carlo, robustness testing (basic versions only)

---

## 72. Future Architecture

### 72.1 Post-MVP Components

| Phase | Components |
|---|---|
| **Phase 5** | Macro data, news sentiment, social sentiment, analyst revisions |
| **Phase 6** | Market regime detection (rule-based + ML) |
| **Phase 7** | Statistical engine (regression, correlation, beta) |
| **Phase 8** | ML engine (random forest, gradient boosting, neural networks, clustering, PCA, anomaly detection) |
| **Phase 9** | Advanced strategy conditions (candlestick patterns, ML-driven signals) |
| **Phase 10** | Advanced backtesting (Monte Carlo, robustness, sensitivity) |
| **Phase 11** | Advanced risk (risk parity, CVaR, optimization-based allocation) |
| **Phase 12** | Real-time feeds |
| **Phase 13** | Broker integration (sandbox) |
| **Phase 14** | Live trading (with full safety controls) |
| **Phase 15** | Production hardening (observability, DR, security) |

### 72.2 Scalability Path

| Component | MVP | Scale |
|---|---|---|
| Event bus | Redis Streams | Kafka |
| Database | Single PostgreSQL | Read replicas + sharding |
| Cache | Redis single | Redis Cluster |
| Workers | Celery + Redis | Celery + Kafka + Kubernetes Jobs |
| ML | scikit-learn | Distributed training (Ray/Spark) |
| Feature Store | PostgreSQL | Dedicated feature store (Feast) |

---

## 73. Quantitative Risks

| Risk | Description | Mitigation |
|---|---|---|
| **Overfitting** | Strategy performs well in-sample but fails OOS | Walk-forward, OOS validation, overfitting controls |
| **Look-ahead bias** | Future information leaks into historical decisions | Point-in-time data architecture, temporal enforcement |
| **Survivorship bias** | Only surviving companies in historical universe | Include delisted instruments |
| **Data snooping** | Testing multiple strategies, selecting best | Multiple hypothesis correction, OOS validation |
| **Regime change** | Market conditions change, strategy fails | Regime detection, regime-specific validation |
| **Non-stationarity** | Statistical properties change over time | Rolling validation, walk-forward |
| **Transaction cost underestimation** | Real costs exceed backtest assumptions | Conservative cost modeling, sensitivity analysis |
| **Liquidity risk** | Cannot execute at assumed prices | Liquidity model, participation limits |
| **Model risk** | ML model makes incorrect predictions | Model validation, drift monitoring, fallback |
| **Crowding** | Many quant funds using similar signals | Diversification, capacity analysis |

---

## 74. ML Risks

| Risk | Description | Mitigation |
|---|---|---|
| **Target leakage** | Target variable leaks into features | Explicit target definition, feature audit |
| **Train/test contamination** | Overlapping data between splits | Strict temporal splits |
| **Future preprocessing** | Scaler/PCA fit on full dataset | Fit on train only |
| **Overfitting** | Model memorizes training data | Regularization, OOS validation, early stopping |
| **Model drift** | Model performance degrades over time | Drift monitoring, retraining triggers |
| **Reproducibility failure** | Cannot reproduce model results | Version control, config hashing, seed recording |
| **Calibration failure** | Predicted probabilities don't match reality | Calibration (Platt/isotonic), reliability diagrams |
| **Adversarial manipulation** | Market actors exploit model predictions | Monitoring, robustness testing |
| **Insufficient training data** | Model trained on too little data | Data requirements, minimum sample thresholds |
| **Feature instability** | Feature distributions change | Feature monitoring, drift detection |

---

## 75. Security Risks

| Risk | Description | Mitigation |
|---|---|---|
| **Credential theft** | Broker API keys exposed | Vault storage, no frontend exposure, rotation |
| **Unauthorized access** | Attacker gains system access | Authentication, MFA, RBAC, least privilege |
| **BOLA/IDOR** | Accessing other users' data | Object-level authorization, ownership verification |
| **Injection attacks** | SQL/XSS/SSRF | Parameterized queries, input validation, output encoding |
| **Sandbox escape** | Custom strategy accesses restricted resources | Container isolation, seccomp, network restrictions |
| **Replay attacks** | Old requests replayed | Nonces, timestamps, idempotency keys |
| **Webhook forgery** | Fake webhook triggers action | Signature verification, HMAC |
| **Insider threat** | Authorized user misuses system | Audit logging, separation of duties, monitoring |
| **Supply chain** | Compromised dependencies | Dependency scanning, SBOM, signed images |
| **Data breach** | Sensitive data exposed | Encryption at rest/transit, access controls, monitoring |

---

## 76. Operational Risks

| Risk | Description | Mitigation |
|---|---|---|
| **System downtime** | Platform unavailable | High availability, failover, monitoring |
| **Data loss** | Historical data corrupted or lost | Backups, replication, immutable storage |
| **Broker outage** | Cannot execute orders | Multiple brokers, graceful degradation |
| **Market data outage** | No price data | Multiple vendors, staleness detection |
| **Latency degradation** | Slow execution | Monitoring, alerting, performance optimization |
| **Human error** | Misconfiguration, wrong parameters | Validation, approval workflows, audit trails |
| **Regulatory change** | New regulations affect trading | Compliance monitoring, configurable rules |
| **Capacity limits** | System overloaded | Autoscaling, rate limiting, resource limits |
| **Key person dependency** | Critical knowledge with one person | Documentation, cross-training, runbooks |
| **Vendor dependency** | Critical vendor becomes unavailable | Multi-vendor strategy, abstraction layers |

---

## 77. Open Decisions Requiring Approval

The following decisions require explicit approval before implementation:

| # | Decision | Options | Impact |
|---|---|---|---|
| 1 | **Cloud provider** | AWS / GCP / Azure / Self-hosted | Infrastructure, compliance, cost |
| 2 | **Primary data vendor** | Bloomberg / Refinitiv / Polygon / Alpha Vantage / Other | Data quality, cost, coverage |
| 3 | **Broker integration target** | Interactive Brokers / Alpaca / Other / None (MVP) | Execution capabilities, regulatory |
| 4 | **Authentication provider** | Keycloak / Auth0 / Okta / Custom | User management, SSO, cost |
| 5 | **Secrets management** | HashiCorp Vault / AWS Secrets Manager / Azure Key Vault | Security, operational complexity |
| 6 | **ML framework preference** | scikit-learn + XGBoost / LightGBM / PyTorch | Model capabilities, team expertise |
| 7 | **Feature store build vs. buy** | Custom / Feast / Tecton / SageMaker | Development time, flexibility |
| 8 | **Deployment target** | Kubernetes / ECS / VM-based / Serverless | Scalability, operational complexity |
| 9 | **Database hosting** | Self-managed / RDS / Cloud SQL / Aurora | Operational burden, cost |
| 10 | **Real-time feed provider** | Direct exchange / Vendor / WebSocket | Latency, cost, reliability |
| 11 | **News/sentiment provider** | Bloomberg / RavenPack / Other / Build | Cost, coverage, licensing |
| 12 | **Regulatory jurisdiction** | US / EU / Other / Multiple | Compliance requirements, data handling |
| 13 | **Multi-tenancy model** | Single-tenant / Multi-tenant / Hybrid | Isolation, cost, complexity |
| 14 | **Open-source vs. proprietary** | Open-source core / Proprietary / Hybrid | Community, support, licensing |
| 15 | **Live trading timeline** | MVP includes / Post-MVP / Undefined | Resource allocation, regulatory |
| 16 | **Custom strategy support** | MVP includes / Post-MVP / Never | Security complexity, user demand |
| 17 | **Asset class priority** | Equities first / Multi-asset from start | Development scope, data availability |
| 18 | **Backtest engine build vs. buy** | Custom / Vectorbt / Backtrader / Other | Development time, flexibility |
| 19 | **Frontend framework** | React / Vue / Angular / Other | Team expertise, ecosystem |
| 20 | **Disaster recovery tier** | Active-passive / Active-active / Basic backup | Cost, RTO/RPO requirements |

---

## Appendix A: Configuration Hash

Every experiment, backtest, and model training run computes a SHA-256 hash of its complete configuration:

```python
def compute_config_hash(config: Dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
```

This hash is recorded and used to verify reproducibility.

## Appendix B: Glossary

| Term | Definition |
|---|---|
| **BOLA** | Broken Object Level Authorization |
| **CVaR** | Conditional Value at Risk (Expected Shortfall) |
| **OOS** | Out-of-Sample |
| **VaR** | Value at Risk |
| **ADV** | Average Daily Volume |
| **ATR** | Average True Range |
| **CAGR** | Compound Annual Growth Rate |
| **RSI** | Relative Strength Index |
| **MACD** | Moving Average Convergence Divergence |
| **OBV** | On-Balance Volume |
| **VWAP** | Volume-Weighted Average Price |
| **CMF** | Chaikin Money Flow |
| **PCA** | Principal Component Analysis |
| **SHAP** | SHapley Additive exPlanations |
| **RBAC** | Role-Based Access Control |
| **MFA** | Multi-Factor Authentication |
| **RTO** | Recovery Time Objective |
| **RPO** | Recovery Point Objective |

---

*End of Blueprint. No application code has been written. No real broker connections have been made. No live trading has been enabled. Awaiting further instructions and approval on open decisions before proceeding to implementation.*
