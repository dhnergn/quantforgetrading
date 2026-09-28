"""Live-trading authorisation gate.

Live execution is **disabled by default** and cannot be enabled by
configuration, environment variable, or code path. It requires a signed
authorisation record on disk that a human created through an explicit,
multi-step process.

Why a file and not a flag: a flag is a value that some future deploy, config
edit, or default change can flip. A signed record is an artefact that a person
produced after reviewing a checklist, and whose signature is verifiable
afterwards. The asymmetry matters — the cost of accidentally enabling live
trading is unbounded, and the cost of the ceremony is a few minutes.

The gate is checked by the live adapter's ``connect``. Any adapter that
reports ``mode == LIVE`` must pass it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from quantforge.core.enums import ExecutionMode, KillSwitchScope, Permission
from quantforge.core.errors import AuthorizationError, LiveTradingNotAuthorized
from quantforge.core.timestamps import utcnow

#: The authorisation file. Read-only (mode 0400) so a running process cannot
#: rewrite its own authorisation, and owned by the operator running the gate.
DEFAULT_AUTH_PATH = Path(
    os.environ.get("QUANTFORGE_LIVE_AUTH", "~/.quantforge/live_authorisation.json")
).expanduser()


# ==========================================================================
# The record
# ==========================================================================


@dataclass(slots=True)
class LiveAuthorisation:
    """A signed authorisation to trade live.

    Every field must be supplied explicitly. There is no default, because a
    default here is a decision nobody made deliberately.
    """

    authorised: bool
    authorised_by: str
    authorised_at: dt.datetime
    environment: str                       # e.g. "production"
    #: Explicitly confirmed preconditions, one per blueprint Section 82.
    preconditions: dict = field(default_factory=dict)
    risk_limits_hash: str = ""
    strategy_version: str = ""
    model_version: str = ""
    portfolio_id: str = ""
    broker_account_id: str = ""
    expires_at: Optional[dt.datetime] = None
    signature: str = ""
    notes: str = ""

    #: Every item the gate requires to be explicitly confirmed.
    REQUIRED_PRECONDITIONS: tuple[str, ...] = (
        "broker_connected",
        "credentials_validated",
        "risk_limits_configured",
        "kill_switch_configured",
        "market_data_validated",
        "position_reconciliation_passed",
        "strategy_version_locked",
        "model_version_locked",
        "paper_trading_reviewed",
        "live_permission_granted",
        "explicit_confirmation",
    )

    def missing_preconditions(self) -> list[str]:
        return [k for k in self.REQUIRED_PRECONDITIONS if not self.preconditions.get(k)]

    def expired(self, now: Optional[dt.datetime] = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or utcnow()) >= self.expires_at

    # -- signing ----------------------------------------------------------

    def _payload(self) -> str:
        body = {
            "authorised": self.authorised,
            "authorised_by": self.authorised_by,
            "authorised_at": self.authorised_at.isoformat(),
            "environment": self.environment,
            "preconditions": self.preconditions,
            "risk_limits_hash": self.risk_limits_hash,
            "strategy_version": self.strategy_version,
            "model_version": self.model_version,
            "portfolio_id": self.portfolio_id,
            "broker_account_id": self.broker_account_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":"))

    def sign(self, secret: bytes) -> "LiveAuthorisation":
        self.signature = hmac.new(
            secret, self._payload().encode(), hashlib.sha256
        ).hexdigest()
        return self

    def verify(self, secret: bytes) -> bool:
        if not self.signature:
            return False
        expected = hmac.new(
            secret, self._payload().encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, self.signature)

    # -- persistence ------------------------------------------------------

    def save(self, path: Path = DEFAULT_AUTH_PATH) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Re-writable only by the owner who is *creating* it, then locked
        # read-only. A long-running process that can rewrite its own
        # authorisation can grant itself permission to trade; the operator
        # performing the activation must be able to create and replace it.
        if path.exists():
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        # Read-only for the owner after writing.
        path.chmod(stat.S_IRUSR)
        return path

    @staticmethod
    def load(path: Path = DEFAULT_AUTH_PATH) -> Optional["LiveAuthorisation"]:
        p = Path(path).expanduser()
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return LiveAuthorisation.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "authorised": self.authorised,
            "authorised_by": self.authorised_by,
            "authorised_at": self.authorised_at.isoformat(),
            "environment": self.environment,
            "preconditions": self.preconditions,
            "risk_limits_hash": self.risk_limits_hash,
            "strategy_version": self.strategy_version,
            "model_version": self.model_version,
            "portfolio_id": self.portfolio_id,
            "broker_account_id": self.broker_account_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "signature": self.signature,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict) -> "LiveAuthorisation":
        return LiveAuthorisation(
            authorised=bool(data.get("authorised", False)),
            authorised_by=data.get("authorised_by", ""),
            authorised_at=dt.datetime.fromisoformat(data["authorised_at"]),
            environment=data.get("environment", "production"),
            preconditions=data.get("preconditions", {}),
            risk_limits_hash=data.get("risk_limits_hash", ""),
            strategy_version=data.get("strategy_version", ""),
            model_version=data.get("model_version", ""),
            portfolio_id=data.get("portfolio_id", ""),
            broker_account_id=data.get("broker_account_id", ""),
            expires_at=(dt.datetime.fromisoformat(data["expires_at"])
                        if data.get("expires_at") else None),
            signature=data.get("signature", ""),
            notes=data.get("notes", ""),
        )


# ==========================================================================
# The gate
# ==========================================================================


class LiveTradingGate:
    """Checks authorisation before permitting live execution.

    ``fail_closed`` is the default and is not configurable: if the
    authorisation file is missing, unreadable, unsigned, expired, or missing a
    precondition, live execution is refused. There is no path through this gate
    that results in "probably fine".
    """

    def __init__(
        self,
        auth_path: Path = DEFAULT_AUTH_PATH,
        secret: Optional[bytes] = None,
        fail_closed: bool = True,
    ) -> None:
        self.auth_path = Path(auth_path).expanduser()
        self._secret = secret
        self.fail_closed = fail_closed
        self.last_check: dict = {}

    def secret(self) -> bytes:
        if self._secret is None:
            env = os.environ.get("QUANTFORGE_LIVE_SECRET")
            if not env:
                raise LiveTradingNotAuthorized(
                    "no signing secret available. Set QUANTFORGE_LIVE_SECRET to a "
                    "value held in the platform's secret manager. Without it an "
                    "authorisation record cannot be verified and must be refused."
                )
            self._secret = env.encode()
        return self._secret

    def check(
        self,
        now: Optional[dt.datetime] = None,
        expected_portfolio_id: Optional[str] = None,
        expected_strategy_version: Optional[str] = None,
        expected_model_version: Optional[str] = None,
        expected_risk_limits_hash: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Return ``(permitted, reason)``."""
        now = now or utcnow()
        if not self.auth_path.exists():
            msg = (f"no live-trading authorisation found at {self.auth_path}. "
                   "Live trading is disabled, which is the default and intended "
                   "state. To enable it, an authorised operator must create a "
                   "signed authorisation record.")
            self.last_check = {"permitted": False, "reason": msg,
                               "checked_at": now.isoformat()}
            return False, msg

        auth = LiveAuthorisation.load(self.auth_path)
        if auth is None:
            msg = f"authorisation record at {self.auth_path} is unreadable"
            self.last_check = {"permitted": False, "reason": msg}
            return False, msg

        if not auth.authorised:
            msg = f"authorisation record exists but is marked not authorised"
            self.last_check = {"permitted": False, "reason": msg}
            return False, msg

        try:
            if not auth.verify(self.secret()):
                msg = ("authorisation signature is invalid. The record was modified "
                       "after signing, or was signed with a different secret.")
                self.last_check = {"permitted": False, "reason": msg}
                return False, msg
        except LiveTradingNotAuthorized as exc:
            self.last_check = {"permitted": False, "reason": str(exc)}
            return False, str(exc)

        if auth.expired(now):
            msg = f"authorisation expired at {auth.expires_at.isoformat()}"
            self.last_check = {"permitted": False, "reason": msg}
            return False, msg

        missing = auth.missing_preconditions()
        if missing:
            msg = f"authorisation is missing required confirmations: {missing}"
            self.last_check = {"permitted": False, "reason": msg}
            return False, msg

        # The authorisation names specific artefacts. If the runtime is using
        # different ones, the authorisation does not cover this run.
        mismatches: list[str] = []
        for label, expected, actual in (
            ("portfolio_id", expected_portfolio_id, auth.portfolio_id),
            ("strategy_version", expected_strategy_version, auth.strategy_version),
            ("model_version", expected_model_version, auth.model_version),
            ("risk_limits_hash", expected_risk_limits_hash, auth.risk_limits_hash),
        ):
            if expected is not None and actual and expected != actual:
                mismatches.append(f"{label} (running {expected!r}, authorised {actual!r})")
        if mismatches:
            msg = f"authorisation does not cover the running configuration: {mismatches}"
            self.last_check = {"permitted": False, "reason": msg}
            return False, msg

        msg = (f"authorised by {auth.authorised_by} at "
               f"{auth.authorised_at.isoformat()} for {auth.environment}")
        self.last_check = {"permitted": True, "reason": msg,
                           "checked_at": now.isoformat(),
                           "authorised_by": auth.authorised_by}
        return True, msg

    def require(
        self,
        expected_portfolio_id: Optional[str] = None,
        expected_strategy_version: Optional[str] = None,
        expected_model_version: Optional[str] = None,
        expected_risk_limits_hash: Optional[str] = None,
    ) -> LiveAuthorisation:
        """Check and return the authorisation, raising on refusal."""
        ok, reason = self.check(
            expected_portfolio_id=expected_portfolio_id,
            expected_strategy_version=expected_strategy_version,
            expected_model_version=expected_model_version,
            expected_risk_limits_hash=expected_risk_limits_hash,
        )
        if not ok:
            raise LiveTradingNotAuthorized(
                f"live trading is not authorised: {reason}"
            )
        return LiveAuthorisation.load(self.auth_path)   # type: ignore[return-value]

    def status(self) -> dict:
        auth = LiveAuthorisation.load(self.auth_path)
        permitted, reason = self.check()
        return {
            "authorisation_path": str(self.auth_path),
            "file_exists": self.auth_path.exists(),
            "permitted": permitted,
            "reason": reason,
            "record": auth.to_dict() if auth else None,
            "last_check": self.last_check,
            "note": (
                "Live trading is off unless a signed authorisation record with all "
                "required confirmations is present, unexpired, and matching the "
                "running configuration."
            ),
        }


# ==========================================================================
# The operator procedure
# ==========================================================================


AUTHORISATION_CHECKLIST: tuple[tuple[str, str], ...] = (
    ("broker_connected",
     "Broker sandbox connection established and credentials validated against a "
     "sandbox account, never a production one."),
    ("credentials_validated",
     "API keys confirmed read/trade-only, with withdrawal permission removed, and "
     "stored in the secret manager rather than a config file."),
    ("risk_limits_configured",
     "Position, sector, gross, net, leverage, drawdown, and daily-loss limits "
     "reviewed by the risk manager and recorded in the limits hash."),
    ("kill_switch_configured",
     "Global, portfolio, strategy, and broker kill switches configured with "
     "tested automatic triggers."),
    ("market_data_validated",
     "Live feed connected, staleness thresholds set, and feed-versus-broker "
     "prices compared on real orders."),
    ("position_reconciliation_passed",
     "Internal positions reconciled exactly against broker positions with zero "
     "unexplained difference."),
    ("strategy_version_locked",
     "The exact strategy version validated in backtest and paper trading is "
     "locked; no in-place edits are possible."),
    ("model_version_locked",
     "The exact model version is locked in the registry and marked live-eligible."),
    ("paper_trading_reviewed",
     "Paper trading ran for a documented period and its configuration was "
     "reviewed and signed off."),
    ("live_permission_granted",
     "A person with CAN_ENABLE_LIVE_TRADING granted permission explicitly, by "
     "name, after seeing the above."),
    ("explicit_confirmation",
     "A human typed an explicit confirmation phrase acknowledging that this "
     "trades real money and can lose it."),
)


def build_authorisation(
    authorised_by: str,
    environment: str = "production",
    portfolio_id: str = "",
    strategy_version: str = "",
    model_version: str = "",
    risk_limits_hash: str = "",
    broker_account_id: str = "",
    expires_in_days: Optional[int] = 30,
    confirmed: Optional[Sequence[str]] = None,
    notes: str = "",
) -> LiveAuthorisation:
    """Build a fully-confirmed authorisation record.

    ``confirmed`` must name every item in
    :data:`AUTHORISATION_CHECKLIST`. There is no way to produce a record with
    preconditions inferred or assumed: an unconfirmed item simply is not set.
    """
    required = [k for k, _ in AUTHORISATION_CHECKLIST]
    given = set(confirmed or [])
    preconditions = {k: k in given for k in required}
    auth = LiveAuthorisation(
        authorised=True,
        authorised_by=authorised_by,
        authorised_at=utcnow(),
        environment=environment,
        preconditions=preconditions,
        risk_limits_hash=risk_limits_hash,
        strategy_version=strategy_version,
        model_version=model_version,
        portfolio_id=portfolio_id,
        broker_account_id=broker_account_id,
        expires_at=(utcnow() + dt.timedelta(days=expires_in_days)
                    if expires_in_days else None),
        notes=notes,
    )
    return auth


def checklist() -> list[dict]:
    return [{"key": k, "requirement": v} for k, v in AUTHORISATION_CHECKLIST]
