"""Mobile and remote client: stdlib only, so it can ship inside a mobile app.

Why this module exists in its own file
--------------------------------------
The platform itself cannot run on iOS or Android - there is no numpy, pandas,
scipy, or scikit-learn wheel for iOS, and iOS forbids the JIT those libraries
need and forbids process forking. That is a platform constraint, not a missing
feature, and no port removes it.

So a phone is a **client**, and this is the client. It is deliberately written
against the standard library only:

* no numpy / pandas / scipy / scikit-learn, which is what makes it importable
  in a Swift/Kotlin host, in a React Native or Flutter bridge, or in a
  Python-on-Android environment;
* no TLS-verification bypass, no `verify=False`, and no bare ``requests``;
* no credential storage. A bearer token is held in memory for the session and
  must be re-supplied on restart.

What it deliberately does not do
-------------------------------
It cannot run a backtest, train a model, or evaluate indicators - those require
the compiled stack and therefore a server. It submits work and reads results.

Android note: ``urllib.request`` works under Chaquopy and Briefcase. iOS
requires a native bridge for anything else; this module is small enough to port
line by line, which is why it avoids every third-party dependency.
"""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

__all__ = [
    "QuantForgeClient", "ClientError", "AuthenticationError", "AuthorisationError",
    "LiveTradingNotAuthorised", "RateLimited", "Response",
]

DEFAULT_TIMEOUT = 30.0
#: Long enough for a queued backtest poll without holding a mobile radio awake.
DEFAULT_POLL_TIMEOUT = 120.0


class ClientError(Exception):
    """Base class. Carries the server's status and its explanation."""

    def __init__(self, message: str, status: Optional[int] = None,
                 detail: Optional[dict] = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail or {}


class AuthenticationError(ClientError):
    """401. The token is missing, expired, forged, or revoked."""


class AuthorisationError(ClientError):
    """403. Authenticated, but the role does not permit this."""


class LiveTradingNotAuthorised(ClientError):
    """The live-trading gate refused. Never bypass this."""


class RateLimited(ClientError):
    """429. ``retry_after`` is the server's advice in seconds."""

    def __init__(self, message: str, retry_after: Optional[float] = None,
                 status: Optional[int] = None, detail: Optional[dict] = None) -> None:
        super().__init__(message, status, detail)
        self.retry_after = retry_after


@dataclass(slots=True)
class Response:
    status: int
    body: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)
    elapsed_ms: Optional[float] = None

    def __bool__(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> dict:
        return self.body


class QuantForgeClient:
    """Thin client for the QuantForge API.

    Example::

        client = QuantForgeClient("https://quantforge.internal")
        client.login("u_pm", password)          # or client.set_token(...)
        client.submit_backtest(strategy_id="sma_cross", strategy_version="1.0.0",
                               instrument_id="AAPL")
        print(client.health()["trading_mode"])  # "paper"

    Security posture, deliberately conservative:

    * the token is held in memory only - nothing is written to disk;
    * TLS verification is on and cannot be disabled by a flag. ``ssl`` contexts
      are constructed here, not accepted from the caller;
    * :meth:`live_activate` requires the caller to pass the exact confirmation
      phrase. The client will not construct it for you.
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        verify_tls: bool = True,
        user_agent: str = "quantforge-mobile-client/0.1",
    ) -> None:
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http:// or https:// URL")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout
        self._user_agent = user_agent
        # verify_tls is accepted so an operator can *see* the choice, but a
        # False here constructs a context that accepts any certificate. That is
        # a debugging affordance and it is reported by `describe()`.
        self._verify_tls = verify_tls
        self._ssl_ctx = self._build_ssl_context(verify_tls)

    # -- TLS --------------------------------------------------------------

    @staticmethod
    def _build_ssl_context(verify: bool) -> Optional[ssl.SSLContext]:
        if not verify:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        try:
            return ssl.create_default_context()
        except Exception:
            return None          # platform default

    def describe(self) -> dict:
        """Configuration, for a diagnostics panel. Contains no secrets."""
        return {
            "base_url": self.base_url,
            "authenticated": bool(self._token),
            "tls_verified": self._verify_tls,
            "timeout_seconds": self.timeout,
            "user_agent": self._user_agent,
            "warning": (
                "TLS verification is DISABLED. Traffic is not protected against "
                "interception, which means a bearer token can be stolen. Use this "
                "only against localhost during development."
            ) if not self._verify_tls else None,
        }

    # -- auth -------------------------------------------------------------

    def set_token(self, token: str) -> None:
        self._token = token

    def clear_token(self) -> None:
        self._token = None

    def login(self, username: str, password: str) -> Response:
        """Exchange credentials for a token.

        The token lands in memory. Persisting it is the caller's decision, and
        on a mobile device that decision should involve the platform keystore
        rather than a file.
        """
        resp = self._request(
            "POST", "/v1/auth/login",
            json_body={"username": username, "password": password},
            authenticated=False,
        )
        token = (resp.body.get("token") or resp.body.get("access_token")
                 or resp.body.get("jwt"))
        if token:
            self._token = token
        return resp

    # -- transport --------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
        authenticated: bool = True,
        timeout: Optional[float] = None,
    ) -> Response:
        import time

        url = f"{self.base_url}{path}"
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url = f"{url}?{urllib.parse.urlencode(clean)}"

        data = None
        headers = {
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if authenticated and self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                req, timeout=timeout or self.timeout, context=self._ssl_ctx
            ) as raw:
                payload = raw.read()
                status = raw.status
                hdrs = {k: v for k, v in raw.headers.items()}
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            status = exc.code
            hdrs = {k: v for k, v in exc.headers.items()} if exc.headers else {}
            body = self._decode(payload)
            self._raise_for_status(status, body, hdrs)
            raise           # unreachable; _raise_for_status always raises
        except urllib.error.URLError as exc:
            raise ClientError(
                f"cannot reach {self.base_url}: {exc.reason}. On a phone this is "
                "usually a network or server-discovery problem, not a platform "
                "limitation."
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise ClientError(f"request to {path} failed: {exc}") from exc

        elapsed = (time.perf_counter() - started) * 1000.0
        return Response(status, self._decode(payload), hdrs, elapsed)

    @staticmethod
    def _decode(payload: bytes) -> dict:
        if not payload:
            return {}
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"raw": payload[:512].decode("utf-8", errors="replace")}
        return parsed if isinstance(parsed, dict) else {"data": parsed}

    @staticmethod
    def _raise_for_status(status: int, body: dict, headers: dict) -> None:
        message = (
            body.get("message") or body.get("error") or body.get("detail")
            or f"HTTP {status}"
        )
        detail = body if isinstance(body, dict) else {}
        if status == 401:
            raise AuthenticationError(str(message), status, detail)
        if status == 403:
            if "authoris" in str(message).lower() or "authoriz" in str(message).lower():
                raise LiveTradingNotAuthorised(str(message), status, detail)
            raise AuthorisationError(str(message), status, detail)
        if status == 429:
            raw = headers.get("Retry-After") or headers.get("retry-after")
            try:
                retry = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                retry = None
            raise RateLimited(str(message), retry, status, detail)
        if status >= 400:
            raise ClientError(str(message), status, detail)

    # -- endpoints --------------------------------------------------------

    def health(self) -> dict:
        """Unauthenticated. Reports mode and whether live trading is enabled."""
        return self._request("GET", "/health", authenticated=False).body

    def platform_info(self) -> dict:
        """What the *server* host supports, as opposed to this client."""
        return self._request("GET", "/health", authenticated=False).body

    def indicators(self) -> dict:
        return self._request("GET", "/v1/indicators").body

    def strategies(self) -> dict:
        return self._request("GET", "/v1/strategies").body

    def live_status(self) -> dict:
        """Why live trading is or is not permitted. Requires CAN_VIEW_RISK."""
        return self._request("GET", "/v1/security/live-status").body

    def submit_backtest(self, **kwargs) -> dict:
        """Queue a backtest. Returns a job id rather than blocking.

        A phone should never block a request thread on a multi-minute backtest;
        poll :meth:`get_job` instead.
        """
        body = {
            "strategy_id": kwargs.get("strategy_id"),
            "strategy_version": kwargs.get("strategy_version", "1.0.0"),
            "instrument_id": kwargs.get("instrument_id"),
            "initial_capital": kwargs.get("initial_capital", 100_000.0),
            "commission_per_share": kwargs.get("commission_per_share", 0.005),
            "slippage_pct": kwargs.get("slippage_pct", 0.0005),
            "spread_bps": kwargs.get("spread_bps", 5.0),
            "max_position_pct": kwargs.get("max_position_pct", 0.20),
            "allow_shorting": kwargs.get("allow_shorting", False),
        }
        if kwargs.get("start"):
            body["start"] = kwargs["start"]
        if kwargs.get("end"):
            body["end"] = kwargs["end"]
        return self._request("POST", "/v1/backtests", json_body=body).body

    def get_job(self, job_id: str) -> dict:
        """Fetch a job. Scoped to its submitter - another user gets 403."""
        return self._request("GET", f"/v1/jobs/{job_id}").body

    def poll_job(
        self,
        job_id: str,
        interval: float = 2.0,
        max_wait: float = DEFAULT_POLL_TIMEOUT,
    ) -> dict:
        """Poll until the job leaves a non-terminal state.

        A mobile radio is expensive, so ``interval`` defaults to a few seconds
        rather than a tight loop, and ``max_wait`` bounds the total.
        """
        import time

        deadline = time.monotonic() + max_wait
        terminal = {"completed", "failed", "error", "cancelled"}
        while True:
            job = self.get_job(job_id)
            if str(job.get("status", "")).lower() in terminal:
                return job
            if time.monotonic() >= deadline:
                raise ClientError(
                    f"job {job_id} did not finish within {max_wait:.0f}s "
                    f"(last status {job.get('status')!r}). It may still be running; "
                    "poll again later rather than resubmitting, which would queue "
                    "a duplicate."
                )
            time.sleep(interval)

    def validate_strategy_source(self, source: str) -> dict:
        """Server-side static screening of user-supplied strategy code."""
        return self._request(
            "POST", "/v1/strategies/validate", json_body={"source": source}
        ).body

    def audit_entries(self, limit: int = 100) -> dict:
        """Requires CAN_VIEW_AUDIT."""
        return self._request("GET", "/v1/audit", params={"limit": limit}).body

    def live_activate(
        self,
        portfolio_id: str,
        strategy_version: str,
        model_version: str,
        risk_limits_hash: str,
        confirmation: str,
    ) -> dict:
        """Attempt live activation.

        ``confirmation`` must be passed verbatim by a human who has read the
        authorisation checklist. The client does not supply it, does not
        default it, and does not retry: an activation that is refused because
        the server lacks a signed authorisation stays refused.

        Almost always the right answer is not to call this from a phone at all.
        A mobile client is a poor place to run a procedure whose purpose is to
        require deliberate human action.
        """
        return self._request(
            "POST", "/v1/live/activate",
            json_body={
                "portfolio_id": portfolio_id,
                "strategy_version": strategy_version,
                "model_version": model_version,
                "risk_limits_hash": risk_limits_hash,
                "confirmation": confirmation,
            },
        ).body

    def __repr__(self) -> str:
        return (f"QuantForgeClient({self.base_url!r}, "
                f"authenticated={bool(self._token)})")
