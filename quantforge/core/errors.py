"""Platform-wide error taxonomy.

Errors are typed so callers can distinguish "this is a data integrity problem,
skip the observation" from "this is a causality violation, abort the run".
That distinction matters: silently skipping a corrupt bar is defensible,
silently continuing past a look-ahead violation is not.
"""
from __future__ import annotations

from typing import Optional


class QuantForgeError(Exception):
    """Root of the platform error hierarchy."""

    code = "QF_ERROR"


# -- data -------------------------------------------------------------------


class DataError(QuantForgeError):
    code = "DATA_ERROR"


class DataQualityError(DataError):
    code = "DATA_QUALITY_ERROR"


class MissingDataError(DataError):
    code = "MISSING_DATA"


class CorporateActionError(DataError):
    code = "CORPORATE_ACTION_ERROR"


class TimestampError(DataError):
    code = "TIMESTAMP_ERROR"


class CalibrationError(DataError):
    """Insufficient or inconsistent instrument metadata."""

    code = "CALIBRATION_ERROR"


# -- features / research ----------------------------------------------------


class FeatureError(QuantForgeError):
    code = "FEATURE_ERROR"


class FeatureUnavailableError(FeatureError):
    """Feature has no value at the requested decision time."""

    code = "FEATURE_UNAVAILABLE"


class WarmupError(FeatureError):
    """Indicator has not accumulated enough history."""

    code = "WARMUP_ERROR"


class LeakageError(QuantForgeError):
    """A preprocessing step was fitted on data it should not have seen.

    The exact class of bug the blueprint's Section 7 exists to prevent.
    """

    code = "LEAKAGE_ERROR"


class LeakageGuardViolation(LeakageError):
    code = "LEAKAGE_GUARD_VIOLATION"


class ImputationError(QuantForgeError):
    code = "IMPUTATION_ERROR"


# -- models -----------------------------------------------------------------


class ModelError(QuantForgeError):
    code = "MODEL_ERROR"


class ModelNotTrainedError(ModelError):
    code = "MODEL_NOT_TRAINED"


class ModelVersionError(ModelError):
    code = "MODEL_VERSION_ERROR"


class ModelPromotionError(ModelError):
    """Attempted to skip a stage in the promotion ladder."""

    code = "MODEL_PROMOTION_ERROR"


class InsufficientDataError(ModelError):
    code = "INSUFFICIENT_DATA"


class OptionalDependencyError(ModelError):
    """A model family requiring an uninstalled optional dependency was requested."""

    code = "OPTIONAL_DEPENDENCY_ERROR"


# -- strategies / backtest --------------------------------------------------


class StrategyError(QuantForgeError):
    code = "STRATEGY_ERROR"


class BacktestError(QuantForgeError):
    code = "BACKTEST_ERROR"


class ExecutionSimulationError(BacktestError):
    code = "EXECUTION_SIMULATION_ERROR"


class ValidationError(QuantForgeError):
    code = "VALIDATION_ERROR"


# -- risk -------------------------------------------------------------------


class RiskError(QuantForgeError):
    code = "RISK_ERROR"


class RiskLimitBreached(RiskError):
    """A risk limit was violated. Carries the limit, the limit, and the observation."""

    code = "RISK_LIMIT_BREACHED"

    def __init__(
        self,
        limit_type: str,
        limit: float,
        observed: float,
        message: Optional[str] = None,
    ) -> None:
        self.limit_type = limit_type
        self.limit = limit
        self.observed = observed
        detail = message or (
            f"{limit_type}: observed {observed:.6g} exceeds limit {limit:.6g}"
        )
        super().__init__(detail)


class KillSwitchEngaged(RiskError):
    code = "KILL_SWITCH_ENGAGED"


# -- execution --------------------------------------------------------------


class BrokerError(QuantForgeError):
    code = "BROKER_ERROR"


class BrokerConnectionError(BrokerError):
    code = "BROKER_CONNECTION_ERROR"


class BrokerTimeoutError(BrokerError):
    """Submission outcome is unknown. Never safe to retry blindly."""

    code = "BROKER_TIMEOUT_ERROR"


class OrderRejected(BrokerError):
    code = "ORDER_REJECTED"


class ReconciliationError(QuantForgeError):
    code = "RECONCILIATION_ERROR"


class DuplicateOrderError(QuantForgeError):
    code = "DUPLICATE_ORDER_ERROR"


# -- live trading safety ----------------------------------------------------


class LiveTradingNotAuthorized(QuantForgeError):
    """Live execution attempted without passing the explicit authorization gate."""

    code = "LIVE_TRADING_NOT_AUTHORIZED"


class StaleDataError(QuantForgeError):
    code = "STALE_DATA_ERROR"


class ExecutionStateError(QuantForgeError):
    code = "EXECUTION_STATE_ERROR"


# -- security ---------------------------------------------------------------


class AuthenticationError(QuantForgeError):
    code = "AUTHENTICATION_ERROR"


class AuthorizationError(QuantForgeError):
    code = "AUTHORIZATION_ERROR"


class RateLimitExceeded(QuantForgeError):
    def __init__(self, resource: str, retry_after: float) -> None:
        self.resource = resource
        self.retry_after = retry_after
        super().__init__(
            f"rate limit exceeded for {resource}; retry after {retry_after:.1f}s"
        )


class SandboxViolation(QuantForgeError):
    code = "SANDBOX_VIOLATION"


class SecretAccessError(QuantForgeError):
    code = "SECRET_ACCESS_ERROR"


class AuditError(QuantForgeError):
    code = "AUDIT_ERROR"


# -- configuration ----------------------------------------------------------


class ConfigurationError(QuantForgeError):
    code = "CONFIGURATION_ERROR"


class UnsupportedAssetClass(QuantForgeError):
    code = "UNSUPPORTED_ASSET_CLASS"


# -- portfolio construction -------------------------------------------------


class PortfolioConstructionError(QuantForgeError):
    """An allocation could not be produced, or would be meaningless.

    Raised rather than returning a degraded result. A covariance that is not
    positive definite has no risk-parity solution; substituting the
    equal-weight answer for it would report a portfolio the caller did not ask
    for and that no one evaluated.
    """

    code = "PORTFOLIO_CONSTRUCTION_ERROR"
