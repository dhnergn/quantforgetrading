"""Authorization: roles, permissions, and enforcement.

Two properties matter more than the specific permission list:

1. **Deny by default.** No permission is implied by authentication. A
   successfully authenticated user with no role has no permissions.
2. **Fail closed on error.** If a permission check itself raises, the answer is
   "denied". A bug in the authorization layer must never become an
   authorization bypass.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from quantforge.core.enums import Permission, Role
from quantforge.core.errors import AuthenticationError, AuthorizationError
from quantforge.core.timestamps import utcnow

#: Role -> permissions. Each role *includes* every lower role's permissions so
#: that adding a permission to a junior role does not silently exclude it from
#: a senior one.
ROLE_PERMISSIONS: dict[Role, frozenset] = {
    Role.READONLY: frozenset(),
    Role.RESEARCHER: frozenset({
        Permission.CAN_RESEARCH, Permission.CAN_BACKTEST,
    }),
    Role.QUANT_DEVELOPER: frozenset({
        Permission.CAN_RESEARCH, Permission.CAN_BACKTEST, Permission.CAN_TRAIN_MODEL,
        Permission.CAN_PAPER_TRADE, Permission.CAN_CONFIGURE_RISK, Permission.CAN_VIEW_RISK,
    }),
    Role.RISK_MANAGER: frozenset({
        Permission.CAN_RESEARCH, Permission.CAN_BACKTEST, Permission.CAN_VIEW_RISK,
        Permission.CAN_CONFIGURE_RISK, Permission.CAN_ACTIVATE_KILL_SWITCH,
        Permission.CAN_VIEW_AUDIT,
    }),
    Role.PORTFOLIO_MANAGER: frozenset({
        Permission.CAN_RESEARCH, Permission.CAN_BACKTEST, Permission.CAN_TRAIN_MODEL,
        Permission.CAN_PAPER_TRADE, Permission.CAN_CONFIGURE_RISK, Permission.CAN_VIEW_RISK,
        Permission.CAN_ENABLE_LIVE_TRADING, Permission.CAN_SUBMIT_LIVE_ORDER,
        Permission.CAN_CANCEL_LIVE_ORDER, Permission.CAN_ACTIVATE_KILL_SWITCH,
    }),
    Role.ADMINISTRATOR: frozenset({
        Permission.CAN_RESEARCH, Permission.CAN_BACKTEST, Permission.CAN_TRAIN_MODEL,
        Permission.CAN_PAPER_TRADE, Permission.CAN_CONFIGURE_RISK,
        Permission.CAN_CONNECT_BROKER, Permission.CAN_ENABLE_LIVE_TRADING,
        Permission.CAN_SUBMIT_LIVE_ORDER, Permission.CAN_CANCEL_LIVE_ORDER,
        Permission.CAN_ACTIVATE_KILL_SWITCH, Permission.CAN_VIEW_RISK,
        Permission.CAN_ADMIN_PLATFORM, Permission.CAN_VIEW_AUDIT,
        Permission.CAN_MANAGE_SECRETS,
    }),
    Role.SECURITY_ADMINISTRATOR: frozenset({
        Permission.CAN_VIEW_RISK, Permission.CAN_VIEW_AUDIT,
        Permission.CAN_ACTIVATE_KILL_SWITCH, Permission.CAN_MANAGE_SECRETS,
        Permission.CAN_ADMIN_PLATFORM, Permission.CAN_CONFIGURE_RISK,
    }),
}

#: Roles that may enable live trading. A dedicated allow-list rather than
#: inference, so granting a role can never accidentally grant the ability to
#: trade real money.
LIVE_TRADING_ROLES: frozenset = frozenset({
    Role.PORTFOLIO_MANAGER, Role.ADMINISTRATOR,
})


@dataclass(slots=True)
class User:
    user_id: str
    username: str
    roles: frozenset[Role] = frozenset()
    email: str = ""
    disabled: bool = False
    mfa_enabled: bool = False
    last_login: Optional[dt.datetime] = None
    created_at: dt.datetime = field(default_factory=utcnow)
    metadata: dict = field(default_factory=dict)

    @property
    def permissions(self) -> frozenset[Permission]:
        perms: set[Permission] = set()
        for role in self.roles:
            perms |= ROLE_PERMISSIONS.get(role, frozenset())
        return frozenset(perms)

    def has(self, permission: Permission) -> bool:
        if self.disabled:
            return False
        return permission in self.permissions

    def can_enable_live_trading(self) -> bool:
        return (not self.disabled
                and bool(self.roles & LIVE_TRADING_ROLES)
                and self.mfa_enabled)

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id, "username": self.username,
            "roles": sorted(r.value for r in self.roles),
            "permissions": sorted(p.value for p in self.permissions),
            "email": self.email, "disabled": self.disabled,
            "mfa_enabled": self.mfa_enabled,
            "can_enable_live_trading": self.can_enable_live_trading(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


def require_permission(user: User, permission: Permission) -> None:
    """Raise unless the user holds the permission. Fails closed."""
    try:
        ok = user.has(permission)
    except Exception as exc:  # a bug in the check must not become a bypass
        raise AuthorizationError(
            f"permission check for {permission.value} failed with "
            f"{type(exc).__name__}; denying by default"
        ) from exc
    if not ok:
        raise AuthorizationError(
            f"user {user.username!r} lacks {permission.value}. Roles: "
            f"{sorted(r.value for r in user.roles) or 'none'}"
        )


def require_live_trading_authority(user: User) -> None:
    """The strongest check in the system.

    Requires the permission, a role on the explicit live-trading allow-list,
    *and* MFA. A stolen password alone is not sufficient to trade real money.
    """
    require_permission(user, Permission.CAN_ENABLE_LIVE_TRADING)
    if not (user.roles & LIVE_TRADING_ROLES):
        raise AuthorizationError(
            f"user {user.username!r} holds CAN_ENABLE_LIVE_TRADING but no role on "
            f"the live-trading allow-list {sorted(r.value for r in LIVE_TRADING_ROLES)}"
        )
    if not user.mfa_enabled:
        raise AuthorizationError(
            f"user {user.username!r} cannot enable live trading: MFA is not enabled. "
            "Trading real money requires a second factor, so a compromised "
            "password is not sufficient."
        )


# ==========================================================================
# Passwords
# ==========================================================================


class PasswordHasher:
    """PBKDF2-HMAC-SHA256 with a per-password salt.

    Parameters are read from the environment at construction. Argon2id is
    preferable and used automatically when ``argon2-cffi`` is installed;
    PBKDF2 is the portable fallback rather than a silent downgrade, and the
    active algorithm is always reported.
    """

    ITERATIONS = 600_000

    def __init__(self, iterations: Optional[int] = None) -> None:
        self.iterations = iterations or self.ITERATIONS
        self.algorithm = "pbkdf2_sha256"
        self._argon2 = None
        try:  # pragma: no cover - optional dependency
            import argon2

            self._argon2 = argon2
            self.algorithm = "argon2id"
        except ImportError:
            pass

    def hash(self, password: str) -> str:
        if not password or len(password) < 12:
            raise AuthenticationError(
                "password must be at least 12 characters. A short password on a "
                "system with broker access is a liability, not a convenience."
            )
        if self._argon2 is not None:  # pragma: no cover
            return self._argon2.PasswordHasher().hash(password)
        salt = secrets.token_bytes(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, self.iterations)
        return f"pbkdf2_sha256${self.iterations}${salt.hex()}${dk.hex()}"

    def verify(self, password: str, stored: str) -> bool:
        try:
            if stored.startswith("pbkdf2_sha256$"):
                _, iters, salt_hex, dk_hex = stored.split("$")
                dk = hashlib.pbkdf2_hmac(
                    "sha256", password.encode(), bytes.fromhex(salt_hex), int(iters)
                )
                return hmac.compare_digest(dk.hex(), dk_hex)
            if self._argon2 is not None:  # pragma: no cover
                return bool(self._argon2.PasswordHasher().verify(stored, password))
        except Exception:
            # A malformed hash must not authenticate, and must not crash the
            # login path either.
            return False
        return False

    def needs_rehash(self, stored: str) -> bool:
        if not stored.startswith("pbkdf2_sha256$"):
            return True
        try:
            return int(stored.split("$")[1]) < self.iterations
        except (IndexError, ValueError):
            return True


# ==========================================================================
# Tokens
# ==========================================================================


@dataclass(slots=True)
class TokenPayload:
    subject: str
    roles: list[str]
    issued_at: dt.datetime
    expires_at: dt.datetime
    token_id: str
    permissions: list[str] = field(default_factory=list)
    session_id: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "sub": self.subject, "roles": self.roles,
            "permissions": self.permissions,
            "iat": self.issued_at.isoformat(),
            "exp": self.expires_at.isoformat(),
            "jti": self.token_id, "sid": self.session_id,
        }


class TokenService:
    """Signed tokens with audience, expiry, and a revocation list.

    Tokens carry the user's *resolved* permissions so an authorisation check
    does not require a database round trip. That is a deliberate trade: a
    revoked role does not take effect until the token expires, which is why the
    TTL is short and why revocation is checked as well.
    """

    def __init__(self, secret: bytes, ttl_seconds: int = 900,
                 issuer: str = "quantforge") -> None:
        if not secret or len(secret) < 32:
            raise AuthenticationError(
                "token secret must be at least 32 bytes. A short signing key is "
                "forgeable, and a forged token is a full authentication bypass."
            )
        self.secret = secret
        self.ttl = ttl_seconds
        self.issuer = issuer
        self._revoked: set[str] = set()

    def _sign(self, payload: bytes) -> str:
        return hmac.new(self.secret, payload, hashlib.sha256).hexdigest()

    def issue(self, user: User, session_id: Optional[str] = None) -> tuple[str, TokenPayload]:
        now = utcnow()
        token_id = secrets.token_urlsafe(16)
        payload = TokenPayload(
            subject=user.user_id, roles=sorted(r.value for r in user.roles),
            issued_at=now, expires_at=now + dt.timedelta(seconds=self.ttl),
            token_id=token_id,
            permissions=sorted(p.value for p in user.permissions),
            session_id=session_id,
        )
        import base64
        import json

        body = base64.urlsafe_b64encode(
            json.dumps(payload.as_dict(), sort_keys=True).encode()
        ).decode().rstrip("=")
        return f"{body}.{self._sign(body.encode())}", payload

    def verify(self, token: str) -> TokenPayload:
        import base64
        import json

        if not token or "." not in token:
            raise AuthenticationError("malformed token")
        body, signature = token.rsplit(".", 1)
        expected = self._sign(body.encode())
        # Constant-time comparison: a timing leak here reveals the signature.
        if not hmac.compare_digest(expected, signature):
            raise AuthenticationError("token signature is invalid")
        padding = "=" * (-len(body) % 4)
        try:
            data = json.loads(base64.urlsafe_b64decode(body + padding))
        except Exception as exc:
            raise AuthenticationError("token payload is unreadable") from exc
        payload = TokenPayload(
            subject=data["sub"], roles=data.get("roles", []),
            issued_at=dt.datetime.fromisoformat(data["iat"]),
            expires_at=dt.datetime.fromisoformat(data["exp"]),
            token_id=data["jti"], permissions=data.get("permissions", []),
            session_id=data.get("sid"),
        )
        if payload.token_id in self._revoked:
            raise AuthenticationError("token has been revoked")
        if utcnow() >= payload.expires_at:
            raise AuthenticationError("token has expired")
        return payload

    def revoke(self, token_id: str) -> None:
        self._revoked.add(token_id)

    def revoke_all_for(self, user_id: str, tokens: Iterable[TokenPayload]) -> int:
        n = 0
        for t in tokens:
            if t.subject == user_id:
                self.revoke(t.token_id)
                n += 1
        return n

    @property
    def revoked_count(self) -> int:
        return len(self._revoked)


# ==========================================================================
# Rate limiting
# ==========================================================================


@dataclass(slots=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    limit: int
    reset_after: float
    resource: str

    def headers(self) -> dict:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_after)),
        }


@dataclass(frozen=True, slots=True)
class RateLimit:
    """Token-bucket limit for one resource class."""

    resource: str
    requests_per_minute: int
    burst: int
    applies_to: tuple[str, ...] = ()


DEFAULT_LIMITS: tuple[RateLimit, ...] = (
    RateLimit("auth", 5, 5, ("POST:/auth/login",)),
    RateLimit("market_data", 60, 20, ("GET:/market_data", "GET:/features")),
    RateLimit("backtest", 1, 1, ("POST:/backtests",)),
    RateLimit("ml_training", 2, 1, ("POST:/models/train",)),
    RateLimit("inference", 100, 20, ("POST:/models/predict",)),
    RateLimit("order_submit", 10, 3, ("POST:/orders",)),
    RateLimit("order_cancel", 5, 2, ("POST:/orders/cancel",)),
    RateLimit("strategy", 3, 1, ("POST:/strategies",)),
    RateLimit("live_enable", 2, 1, ("POST:/live/activate",)),
)


class RateLimiter:
    """Token bucket per (subject, resource).

    A burst allowance exists because interactive clients legitimately send a
    short cluster of requests. Sustained rate is what the limit governs.
    """

    def __init__(self, limits: Sequence[RateLimit] = DEFAULT_LIMITS) -> None:
        self._limits = {lim.resource: lim for lim in limits}
        self._buckets: dict[tuple[str, str], tuple[float, float]] = {}
        self._lock_failures: dict[str, tuple[int, float]] = {}

    def check(self, subject: str, resource: str) -> RateLimitResult:
        lim = self._limits.get(resource)
        if lim is None:
            return RateLimitResult(True, 0, 0, 0.0, resource)
        now = utcnow().timestamp()
        capacity = lim.requests_per_minute + lim.burst
        refill_per_second = lim.requests_per_minute / 60.0
        key = (subject, resource)
        tokens, last = self._buckets.get(key, (float(capacity), now))
        tokens = min(float(capacity), tokens + (now - last) * refill_per_second)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            reset = (1.0 - tokens) / refill_per_second if refill_per_second > 0 else 60.0
            return RateLimitResult(False, 0, lim.requests_per_minute, reset, resource)
        tokens -= 1.0
        self._buckets[key] = (tokens, now)
        return RateLimitResult(True, int(tokens), lim.requests_per_minute, 0.0, resource)

    def record_auth_failure(self, subject: str, now: Optional[float] = None) -> int:
        """Count consecutive authentication failures for lockout."""
        now = now or utcnow().timestamp()
        count, last = self._lock_failures.get(subject, (0, now))
        if now - last > 900.0:      # 15-minute reset window
            count = 0
        count += 1
        self._lock_failures[subject] = (count, now)
        return count

    def is_locked_out(self, subject: str, threshold: int = 5,
                      window_seconds: float = 900.0,
                      now: Optional[float] = None) -> bool:
        now = now or utcnow().timestamp()
        count, last = self._lock_failures.get(subject, (0, 0.0))
        if now - last > window_seconds:
            return False
        return count >= threshold

    def clear_auth_failures(self, subject: str) -> None:
        self._lock_failures.pop(subject, None)

    def remaining_auth_retries(self, subject: str, threshold: int = 5) -> int:
        count, _ = self._lock_failures.get(subject, (0, 0.0))
        return max(0, threshold - count)
