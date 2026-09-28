"""Enumerations shared across QuantForge domains.

These enums are deliberately explicit. Implicit string values are used as the
persistence representation so that the stored value stays stable even if the
Python identifier is renamed.
"""
from __future__ import annotations

import enum

# --------------------------------------------------------------------------
# Asset classification
# --------------------------------------------------------------------------


class AssetClass(str, enum.Enum):
    """Supported asset classes. No asset is assumed to behave like an equity."""

    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    SECTOR = "sector"
    FUTURE = "future"
    OPTION = "option"
    OTHER = "other"


class InstrumentStatus(str, enum.Enum):
    ACTIVE = "active"
    DELISTED = "delisted"
    SUSPENDED = "suspended"
    BANKRUPT = "bankrupt"
    HISTORICAL = "historical"


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------


class Timeframe(str, enum.Enum):
    TICK = "tick"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class DataQuality(str, enum.Enum):
    """Classification of an anomaly.

    Anomalies are never silently deleted. Each is classified so the researcher
    can decide explicitly whether to include, exclude, or investigate it.
    """

    OK = "ok"
    DATA_QUALITY_ANOMALY = "data_quality_anomaly"
    MARKET_EVENT = "market_event"
    POTENTIAL_OUTLIER = "potential_outlier"


# --------------------------------------------------------------------------
# Corporate actions
# --------------------------------------------------------------------------


class CorporateActionType(str, enum.Enum):
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    CASH_DIVIDEND = "cash_dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    RIGHTS_OFFERING = "rights_offering"
    SPINOFF = "spinoff"
    MERGER = "merger"
    TICKER_CHANGE = "ticker_change"
    DELISTING = "delisting"
    NAME_CHANGE = "name_change"


# --------------------------------------------------------------------------
# Fundamentals / macro
# --------------------------------------------------------------------------


class FiscalPeriod(str, enum.Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    FY = "FY"
    TTM = "TTM"
    ANNUAL = "annual"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"
    QUARTERLY = "quarterly"


class RevisionKind(str, enum.Enum):
    INITIAL = "initial"
    RESTATEMENT = "restatement"
    REVISION = "revision"


# --------------------------------------------------------------------------
# Sentiment
# --------------------------------------------------------------------------


class SentimentLabel(str, enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SentimentSource(str, enum.Enum):
    NEWS = "news"
    SOCIAL = "social"
    ANALYST = "analyst"
    SURVEY = "survey"
    DERIVED = "derived"


class SocialPlatform(str, enum.Enum):
    REDDIT = "reddit"
    X = "x"
    STOCKTWITS = "stocktwits"
    SEEK = "seek"
    HACKER_NEWS = "hackernews"
    OTHER = "other"


class EstimateType(str, enum.Enum):
    EPS = "eps"
    REVENUE = "revenue"
    EBITDA = "ebitda"
    TARGET_PRICE = "target_price"
    RATING = "rating"
    OTHER = "other"


# --------------------------------------------------------------------------
# Regime
# --------------------------------------------------------------------------


class MarketRegime(str, enum.Enum):
    """Market condition labels.

    These are descriptive classifications of historical conditions, not
    forecasts. A label describes what already happened.
    """

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class RegimeMethod(str, enum.Enum):
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    ENSEMBLE = "ensemble"


# --------------------------------------------------------------------------
# Signals / strategies
# --------------------------------------------------------------------------


class SignalType(str, enum.Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class TargetLabel(str, enum.Enum):
    """Classification target labels. Thresholds must be fixed before evaluation."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class StrategyKind(str, enum.Enum):
    CROSSOVER = "crossover"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    ML_DRIVEN = "ml_driven"
    MULTI_FACTOR = "multi_factor"
    COMPOSITE = "composite"


class SizingMethod(str, enum.Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_ATR = "volatility_atr"
    VOLATILITY_HV = "volatility_hv"
    KELLY = "kelly"
    RISK_BASED = "risk_based"
    EQUAL_WEIGHT = "equal_weight"


class StopType(str, enum.Enum):
    NONE = "none"
    PERCENTAGE = "percentage"
    ATR = "atr"
    TRAILING_PERCENT = "trailing_percent"
    TRAILING_ATR = "trailing_atr"
    SUPPORT = "support"
    TIME = "time"
    VOLATILITY = "volatility"


# --------------------------------------------------------------------------
# Orders / execution
# --------------------------------------------------------------------------


class OrderSide(str, enum.Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    BRACKET = "bracket"


class OrderStatus(str, enum.Enum):
    """Full order lifecycle.

    ``UNKNOWN`` is a first-class state. An order in ``UNKNOWN`` requires
    reconciliation with the broker; it must never be re-submitted blindly.
    """

    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class TimeInForce(str, enum.Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"


class TradingState(str, enum.Enum):
    """System-wide trading state. Anything other than NORMAL blocks new orders."""

    NORMAL = "normal"
    HALTED_DATA_STALE = "halted_data_stale"
    HALTED_KILL_SWITCH = "halted_kill_switch"
    HALTED_RECONCILIATION = "halted_reconciliation"
    HALTED_BROKER_DOWN = "halted_broker_down"
    HALTED_RISK_BREACH = "halted_risk_breach"
    REQUIRES_RECONCILIATION = "requires_reconciliation"


class RiskLevel(str, enum.Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------


class RiskLimitType(str, enum.Enum):
    POSITION_CONCENTRATION = "position_concentration"
    SECTOR_CONCENTRATION = "sector_concentration"
    ASSET_CLASS_CONCENTRATION = "asset_class_concentration"
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"
    LEVERAGE = "leverage"
    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS = "daily_loss"
    VOLATILITY = "volatility"
    VAR_LIMIT = "var_limit"
    CVAR_LIMIT = "cvar_limit"
    ORDER_NOTIONAL = "order_notional"
    LIQUIDITY_PARTICIPATION = "liquidity_participation"


class VaRMethod(str, enum.Enum):
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class PortfolioConstruction(str, enum.Enum):
    EQUAL_WEIGHT = "equal_weight"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    RISK_PARITY = "risk_parity"
    FIXED_FRACTIONAL = "fixed_fractional"
    OPTIMIZATION = "optimization"
    SIGNAL_WEIGHTED = "signal_weighted"


# --------------------------------------------------------------------------
# Model lifecycle
# --------------------------------------------------------------------------


class ModelStage(str, enum.Enum):
    """Model registry promotion ladder.

    A model never skips a stage. Training does not confer live eligibility.
    """

    EXPERIMENT = "experiment"
    TRAINED = "trained"
    VALIDATED = "validated"
    APPROVED = "approved"
    PAPER_TRADING = "paper_trading"
    LIVE_ELIGIBLE = "live_eligible"
    RETIRED = "retired"


class ModelTask(str, enum.Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"


class DriftType(str, enum.Enum):
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CALIBRATION_DRIFT = "calibration_drift"
    PERFORMANCE_DECAY = "performance_decay"
    REGIME_SHIFT = "regime_shift"


# --------------------------------------------------------------------------
# Authorization
# --------------------------------------------------------------------------


class Role(str, enum.Enum):
    RESEARCHER = "researcher"
    QUANT_DEVELOPER = "quant_developer"
    PORTFOLIO_MANAGER = "portfolio_manager"
    RISK_MANAGER = "risk_manager"
    ADMINISTRATOR = "administrator"
    SECURITY_ADMINISTRATOR = "security_administrator"
    READONLY = "readonly"


class Permission(str, enum.Enum):
    CAN_RESEARCH = "CAN_RESEARCH"
    CAN_BACKTEST = "CAN_BACKTEST"
    CAN_TRAIN_MODEL = "CAN_TRAIN_MODEL"
    CAN_PAPER_TRADE = "CAN_PAPER_TRADE"
    CAN_CONFIGURE_RISK = "CAN_CONFIGURE_RISK"
    CAN_CONNECT_BROKER = "CAN_CONNECT_BROKER"
    CAN_ENABLE_LIVE_TRADING = "CAN_ENABLE_LIVE_TRADING"
    CAN_SUBMIT_LIVE_ORDER = "CAN_SUBMIT_LIVE_ORDER"
    CAN_CANCEL_LIVE_ORDER = "CAN_CANCEL_LIVE_ORDER"
    CAN_ACTIVATE_KILL_SWITCH = "CAN_ACTIVATE_KILL_SWITCH"
    CAN_VIEW_RISK = "CAN_VIEW_RISK"
    CAN_ADMIN_PLATFORM = "CAN_ADMIN_PLATFORM"
    CAN_VIEW_AUDIT = "CAN_VIEW_AUDIT"
    CAN_MANAGE_SECRETS = "CAN_MANAGE_SECRETS"


class AuditAction(str, enum.Enum):
    STRATEGY_CREATED = "strategy_created"
    STRATEGY_UPDATED = "strategy_updated"
    STRATEGY_DELETED = "strategy_deleted"
    MODEL_TRAINED = "model_trained"
    MODEL_PROMOTED = "model_promoted"
    MODEL_RETIRED = "model_retired"
    FEATURE_SET_REGISTERED = "feature_set_registered"
    BACKTEST_CREATED = "backtest_created"
    DATASET_VERSION_CREATED = "dataset_version_created"
    BROKER_CONNECTED = "broker_connected"
    BROKER_DISCONNECTED = "broker_disconnected"
    CREDENTIAL_CHANGED = "credential_changed"
    RISK_LIMITS_CHANGED = "risk_limits_changed"
    PAPER_TRADING_ACTIVATED = "paper_trading_activated"
    PAPER_TRADING_STOPPED = "paper_trading_stopped"
    LIVE_TRADING_ACTIVATED = "live_trading_activated"
    LIVE_TRADING_DEACTIVATED = "live_trading_deactivated"
    ORDER_CREATED = "order_created"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    POSITION_CHANGED = "position_changed"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    KILL_SWITCH_DEACTIVATED = "kill_switch_deactivated"
    RECONCILIATION_FAILED = "reconciliation_failed"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    AUTH_FAILURE = "auth_failure"
    ACCESS_DENIED = "access_denied"
    CONFIG_CHANGED = "config_changed"


# --------------------------------------------------------------------------
# Execution environment
# --------------------------------------------------------------------------


class ExecutionMode(str, enum.Enum):
    """Execution environment.

    ``LIVE`` exists in the type system because the platform must be able to
    represent it, but it is unreachable without passing the explicit
    authorization gate in :mod:`quantforge.execution.live_gate`.
    """

    BACKTEST = "backtest"
    PAPER = "paper"
    SANDBOX = "sandbox"
    LIVE = "live"


class SlippageModel(str, enum.Enum):
    NONE = "none"
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    SPREAD_BASED = "spread_based"
    VOLATILITY_BASED = "volatility_based"
    VOLUME_BASED = "volume_based"
    SQUARE_ROOT = "square_root"


class KillSwitchScope(str, enum.Enum):
    GLOBAL = "global"
    PORTFOLIO = "portfolio"
    STRATEGY = "strategy"
    BROKER = "broker"
