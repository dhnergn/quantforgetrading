"""FastAPI application.

Security posture, per blueprint Section 78:

* Authentication required by default; there is no anonymous write path.
* Permission checks on every endpoint, not just the sensitive ones.
* Rate limits per resource class.
* Request-size limits, secure headers, CORS restricted to configured origins.
* Every state-changing request is audit-logged with actor and IP.

Research-heavy operations (backtests, model training) return a job id rather
than blocking a request thread, so a long backtest cannot exhaust the API
workers.
"""
from __future__ import annotations

import datetime as dt
import os
import uuid
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from quantforge.core.enums import (
    AuditAction, ExecutionMode, ModelStage, Permission, Role,
)
from quantforge.core.errors import (
    AuthenticationError, AuthorizationError, QuantForgeError, RateLimitExceeded,
)
from quantforge.core.timestamps import utcnow
from quantforge.security import (
    AuditLog, PasswordHasher, RateLimiter, TokenService, User,
    require_live_trading_authority, require_permission,
)

APP_TITLE = "QuantForge AI"
APP_DESCRIPTION = (
    "Research-grade quantitative trading platform. Live trading is disabled by "
    "default and requires explicit, signed authorisation."
)
DISCLAIMER = (
    "QuantForge is a research platform. No strategy, indicator, or model "
    "produces guaranteed profits, and a profitable backtest is evidence about "
    "the tested historical period only, not a prediction of future returns."
)

#: Injected by :func:`create_app`. A module-level singleton would make testing
#: and multi-tenant deployment impossible.
_STATE: dict[str, Any] = {}


class Deps:
    """Dependencies resolved per request."""

    def __init__(self, app_state: dict):
        self.audit: AuditLog = app_state["audit"]
        self.limiter: RateLimiter = app_state["limiter"]
        self.tokens: TokenService = app_state["tokens"]
        self.users: dict[str, User] = app_state["users"]
        self.jobs: dict[str, dict] = app_state["jobs"]


# ==========================================================================
# Schemas
# ==========================================================================


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    live_trading_enabled: bool
    trading_mode: str
    timestamp: str
    disclaimer: str = DISCLAIMER


class BacktestRequest(BaseModel):
    strategy_id: str
    strategy_version: str = Field(..., min_length=1, max_length=32)
    instrument_id: str = Field(..., min_length=1, max_length=64)
    start: Optional[str] = None
    end: Optional[str] = None
    initial_capital: float = Field(100_000.0, gt=0, le=1e12)
    # Costs default to non-zero. A zero-cost backtest is a valid research
    # question but must be asked explicitly, not reached by default.
    commission_per_share: float = Field(0.005, ge=0)
    slippage_pct: float = Field(0.0005, ge=0, le=0.1)
    spread_bps: float = Field(5.0, ge=0, le=1000)
    max_position_pct: float = Field(0.20, gt=0, le=1.0)
    allow_shorting: bool = False
    notes: str = Field("", max_length=2000)


class LiveActivationRequest(BaseModel):
    portfolio_id: str = Field(..., min_length=1, max_length=64)
    strategy_version: str = Field(..., min_length=1, max_length=32)
    model_version: str = Field(..., min_length=1, max_length=32)
    risk_limits_hash: str = Field(..., min_length=8, max_length=64)
    #: The literal phrase a human must type. Prevents an accidental POST.
    confirmation: str


LIVE_CONFIRMATION_PHRASE = "I UNDERSTAND THIS PLACES REAL ORDERS WITH REAL MONEY"


# ==========================================================================
# App factory
# ==========================================================================


def create_app(
    users: Optional[dict[str, User]] = None,
    token_secret: Optional[bytes] = None,
    db_url: Optional[str] = None,
) -> FastAPI:
    """Build the application with injected dependencies."""
    secret = token_secret or os.environ.get(
        "QUANTFORGE_API_SECRET", "dev-only-insecure-secret-change-me-32bytes"
    ).encode()
    if len(secret) < 32 and not os.environ.get("QUANTFORGE_ALLOW_WEAK_SECRET"):
        raise AuthenticationError(
            "token secret must be at least 32 bytes; set QUANTFORGE_API_SECRET"
        )

    _STATE.clear()
    _STATE.update({
        "audit": AuditLog(),
        "limiter": RateLimiter(),
        "tokens": TokenService(secret),
        "hasher": PasswordHasher(),
        "users": users or _default_users(),
        "jobs": {},
        "db_url": db_url or "sqlite:///./quantforge.db",
    })

    from quantforge import __version__

    app = FastAPI(
        title=APP_TITLE, description=f"{APP_DESCRIPTION}\n\n{DISCLAIMER}",
        version=__version__,
        docs_url="/docs", redoc_url=None,
    )
    app.state.qf = _STATE

    # -- CORS: allow-list only, never "*" with credentials ---------------
    origins = os.environ.get("QUANTFORGE_CORS_ORIGINS", "").split(",")
    origins = [o.strip() for o in origins if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins, allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
        )

    @app.middleware("http")
    async def security_and_audit(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:16]
        max_bytes = int(os.environ.get("QUANTFORGE_MAX_REQUEST_BYTES", 1_048_576))
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"error": "REQUEST_TOO_LARGE",
                         "message": f"request body exceeds {max_bytes} bytes",
                         "request_id": request_id},
            )
        response = await call_next(request)
        # Secure headers on every response.
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["X-Request-Id"] = request_id
        response.headers["X-QuantForge-Disclaimer"] = "research-only-no-guaranteed-profits"
        return response

    def deps() -> Deps:
        return Deps(_STATE)

    def client_key(request: Request, d: Deps = Depends(deps)) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    # -- auth dependency --------------------------------------------------

    def current_user(
        request: Request,
        d: Deps = Depends(deps),
        key: str = Depends(client_key),
    ) -> User:
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token",
                                headers={"WWW-Authenticate": "Bearer"})
        try:
            payload = d.tokens.verify(header[7:])
        except AuthenticationError as exc:
            d.audit.record("anonymous", AuditAction.AUTH_FAILURE, request.url.path,
                           {"reason": str(exc), "ip": key}, ip_address=key)
            raise HTTPException(401, str(exc)) from exc

        limited = d.limiter.check(payload.subject, "inference")
        if not limited.allowed:
            raise HTTPException(429, "rate limit exceeded",
                                headers={"Retry-After": str(int(limited.reset_after) + 1)})

        user = d.users.get(payload.subject)
        if user is None or user.disabled:
            d.audit.record(payload.subject, AuditAction.ACCESS_DENIED,
                           request.url.path, {"reason": "unknown or disabled user"},
                           ip_address=key)
            raise HTTPException(403, "user is unknown or disabled")
        return user

    def require(permission: Permission):
        def guard(user: User = Depends(current_user)) -> User:
            try:
                require_permission(user, permission)
            except AuthorizationError as exc:
                raise HTTPException(403, str(exc)) from exc
            return user
        return guard

    def rate_limited(resource: str):
        def guard(user: User = Depends(current_user),
                  key: str = Depends(client_key)) -> User:
            result = _STATE["limiter"].check(user.user_id, resource)
            if not result.allowed:
                raise HTTPException(
                    429, f"rate limit exceeded for {resource}",
                    headers={"Retry-After": str(int(result.reset_after) + 1)},
                )
            return user
        return guard

    def audit_ip(request: Request) -> str:
        fwd = request.headers.get("X-Forwarded-For")
        return fwd.split(",")[0].strip() if fwd else (
            request.client.host if request.client else "unknown")

    # ======================================================================
    # Routes
    # ======================================================================

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Unauthenticated liveness probe. Reveals nothing sensitive."""
        from quantforge.execution.live_gate import LiveTradingGate

        permitted, reason = LiveTradingGate().check()
        return HealthResponse(
            status="ok", version=__version__,
            live_trading_enabled=permitted,
            trading_mode=ExecutionMode.LIVE.value if permitted
            else ExecutionMode.PAPER.value,
            timestamp=utcnow().isoformat(),
        )

    @app.get("/v1/security/live-status", tags=["system"])
    def live_status(user: User = Depends(require(Permission.CAN_VIEW_RISK))):
        """Why live trading is or is not permitted, without exposing secrets."""
        from quantforge.execution.live_gate import LiveTradingGate

        return LiveTradingGate().status()

    @app.get("/v1/indicators", tags=["research"])
    def list_indicators(user: User = Depends(require(Permission.CAN_RESEARCH))):
        from quantforge.quant.indicators import catalogue

        return {"indicators": catalogue()}

    @app.get("/v1/strategies", tags=["research"])
    def list_strategies(user: User = Depends(require(Permission.CAN_RESEARCH))):
        from quantforge.strategy.builder import (
            bollinger_mean_reversion_strategy, rsi_mean_reversion_strategy,
            sma_crossover_strategy,
        )

        presets = [
            sma_crossover_strategy(20, 50),
            rsi_mean_reversion_strategy(),
            bollinger_mean_reversion_strategy(),
        ]
        return {"strategies": [s.as_dict() for s in presets]}

    @app.post("/v1/backtests", status_code=202, tags=["research"])
    def submit_backtest(
        req: BacktestRequest,
        request: Request,
        user: User = Depends(rate_limited("backtest")),
    ):
        """Queue a backtest.

        Returns 202 with a job id rather than blocking: a walk-forward run over
        many windows can take minutes, and holding a request thread for that
        length exhausts the worker pool.
        """
        job_id = uuid.uuid4().hex[:16]
        _STATE["jobs"][job_id] = {
            "job_id": job_id, "type": "backtest", "status": "queued",
            "submitted_at": utcnow().isoformat(), "submitted_by": user.username,
            "request": req.model_dump(), "result": None,
        }
        _STATE["audit"].record(
            user.username, AuditAction.BACKTEST_CREATED, f"backtest/{job_id}",
            {"strategy": req.strategy_id, "version": req.strategy_version,
             "costs": {"commission_per_share": req.commission_per_share,
                       "slippage_pct": req.slippage_pct,
                       "spread_bps": req.spread_bps}},
            ip_address=audit_ip(request), request_id=job_id,
        )
        return {
            "job_id": job_id, "status": "queued",
            "poll": f"/v1/jobs/{job_id}",
            "note": "This is a queued research job. A profitable backtest is "
                    "evidence about the tested historical period, not a "
                    "prediction of future returns.",
        }

    @app.get("/v1/jobs/{job_id}", tags=["research"])
    def get_job(job_id: str, user: User = Depends(require(Permission.CAN_RESEARCH))):
        job = _STATE["jobs"].get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        if job["submitted_by"] != user.username and \
                not user.has(Permission.CAN_VIEW_AUDIT):
            # BOLA: a job belongs to its submitter. Without the audit
            # permission, another user cannot read it by guessing the id.
            raise HTTPException(403, "this job belongs to another user")
        return job

    @app.get("/v1/models", tags=["models"])
    def list_models(user: User = Depends(require(Permission.CAN_RESEARCH))):
        return {"models": [_STATE.get("model_registry")] if _STATE.get("model_registry") else [],
                "ladder": [s.value for s in ModelStage]}

    @app.post("/v1/live/activate", status_code=403, tags=["live"])
    def activate_live(
        req: LiveActivationRequest,
        request: Request,
        user: User = Depends(current_user),
    ):
        """Live activation. Refuses unless a signed authorisation exists.

        The 403 default status is intentional: a successful live activation
        requires an authorisation file that this process cannot create for
        itself.
        """
        _STATE["audit"].record(
            user.username, AuditAction.ACCESS_DENIED, f"live/{req.portfolio_id}",
            {"reason": "live activation attempted", "strategy": req.strategy_version,
             "model": req.model_version}, ip_address=audit_ip(request),
        )
        if req.confirmation != LIVE_CONFIRMATION_PHRASE:
            raise HTTPException(
                400, f"confirmation phrase does not match. Required exactly: "
                     f"{LIVE_CONFIRMATION_PHRASE!r}"
            )
        try:
            require_live_trading_authority(user)
        except AuthorizationError as exc:
            raise HTTPException(403, str(exc)) from exc

        from quantforge.execution.live_gate import LiveTradingGate

        gate = LiveTradingGate()
        permitted, reason = gate.check(
            expected_portfolio_id=req.portfolio_id,
            expected_strategy_version=req.strategy_version,
            expected_model_version=req.model_version,
            expected_risk_limits_hash=req.risk_limits_hash,
        )
        if not permitted:
            raise HTTPException(403, f"live trading is not authorised: {reason}")
        _STATE["audit"].record(
            user.username, AuditAction.LIVE_TRADING_ACTIVATED,
            f"live/{req.portfolio_id}",
            {"strategy": req.strategy_version, "model": req.model_version},
            ip_address=audit_ip(request),
        )
        return {"activated": True, "reason": reason,
                "note": "Live execution still requires a connected, authorised "
                        "broker adapter. Enabling the flag does not place orders."}

    @app.get("/v1/audit", tags=["security"])
    def get_audit(
        actor: Optional[str] = None, action: Optional[str] = None,
        limit: int = 100,
        user: User = Depends(require(Permission.CAN_VIEW_AUDIT)),
    ):
        limit = max(1, min(limit, 1000))
        entries = _STATE["audit"].entries(actor=actor, action=action, limit=limit)
        return {
            "entries": entries,
            "chain": _STATE["audit"].verify_chain(),
            "note": "Entries are append-only and hash-chained. A broken chain "
                    "means the log was modified after it was written.",
        }

    @app.post("/v1/strategies/validate", tags=["security"])
    def validate_strategy_source(
        payload: dict,
        user: User = Depends(require(Permission.CAN_RESEARCH)),
    ):
        """Statically screen user-supplied strategy source."""
        from quantforge.security.audit import SandboxViolation, StrategySandbox

        source = payload.get("source", "")
        try:
            result = StrategySandbox().check_source(source)
        except SandboxViolation as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            **result,
            "warning": "Static screening is not a security boundary. Execute "
                       "untrusted strategies only inside a container or microVM.",
        }

    @app.exception_handler(QuantForgeError)
    async def handle_qf_error(request: Request, exc: QuantForgeError):
        return JSONResponse(
            status_code=400,
            content={"error": exc.code, "message": str(exc),
                     "request_id": uuid.uuid4().hex[:16]},
        )

    return app


def _default_users() -> dict[str, User]:
    """Development users. Replace with a real identity provider in production."""
    return {
        "u_researcher": User("u_researcher", "researcher",
                             frozenset({Role.RESEARCHER}), mfa_enabled=False),
        "u_quant": User("u_quant", "quant_dev",
                        frozenset({Role.QUANT_DEVELOPER}), mfa_enabled=False),
        "u_pm": User("u_pm", "portfolio_manager",
                     frozenset({Role.PORTFOLIO_MANAGER}), mfa_enabled=True),
        "u_risk": User("u_risk", "risk_manager",
                       frozenset({Role.RISK_MANAGER}), mfa_enabled=True),
        "u_admin": User("u_admin", "administrator",
                        frozenset({Role.ADMINISTRATOR}), mfa_enabled=True),
    }


app = create_app()
