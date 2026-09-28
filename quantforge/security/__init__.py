"""Security: access control, auditing, and the strategy sandbox."""
from quantforge.security.access import (
    LIVE_TRADING_ROLES, ROLE_PERMISSIONS, PasswordHasher, RateLimit,
    RateLimitResult, RateLimiter, TokenPayload, TokenService, User,
    require_live_trading_authority, require_permission,
)
from quantforge.security.audit import AuditEntry, AuditLog, StrategySandbox

__all__ = [
    "AuditEntry", "AuditLog", "LIVE_TRADING_ROLES", "PasswordHasher",
    "ROLE_PERMISSIONS", "RateLimit", "RateLimitResult", "RateLimiter",
    "StrategySandbox", "TokenPayload", "TokenService", "User",
    "require_live_trading_authority", "require_permission",
]
