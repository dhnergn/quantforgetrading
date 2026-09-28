"""Platform configuration.

Design constraint: live trading cannot be turned on by editing a config file.

``ExecutionConfig.live_trading_enabled`` exists, but the execution layer
refuses to honour it unless a signed authorisation record has also been
written to disk. Configuration alone is therefore insufficient, which is the
intent of blueprint Section 82.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

from quantforge.core.enums import SlippageModel
from quantforge.core.errors import ConfigurationError

DEFAULT_ROOT = Path(os.environ.get("QUANTFORGE_HOME", Path.cwd() / ".quantforge"))


# --------------------------------------------------------------------------
# Sectioned config objects
# --------------------------------------------------------------------------


@dataclass
class DatabaseConfig:
    """SQLite by default so the platform runs with no external services.

    Set ``url`` to a PostgreSQL/TimescaleDB DSN for production. The ORM models
    use portable types only.
    """

    url: str = "sqlite:///./quantforge.db"
    echo: bool = False
    pool_size: int = 5
    pool_timeout: int = 30
    pool_recycle: int = 1800


@dataclass
class DataConfig:
    root: str = str(DEFAULT_ROOT / "data")
    raw_path: str = str(DEFAULT_ROOT / "data" / "raw")
    normalized_path: str = str(DEFAULT_ROOT / "data" / "normalized")
    adjusted_path: str = str(DEFAULT_ROOT / "data" / "adjusted")
    features_path: str = str(DEFAULT_ROOT / "data" / "features")
    cache_path: str = str(DEFAULT_ROOT / "data" / "cache")
    cache_ttl_seconds: int = 3600
    parquet_compression: str = "snappy"
    default_timeframe: str = "1d"
    default_source: str = "local"
    #: When False, stale cached values are never substituted for newer live data.
    allow_stale_cache_reads: bool = False


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    base_currency: str = "USD"
    default_timeframe: str = "1d"
    bars_per_year: int = 252
    #: Commission is modelled explicitly; 0 is not the same as "free".
    commission_per_share: float = 0.0
    commission_pct: float = 0.0
    min_commission: float = 0.0
    regulatory_fee_bps: float = 0.0
    assume_spread_bps: float = 5.0
    slippage_model: str = SlippageModel.PERCENTAGE.value
    slippage_pct: float = 0.001
    slippage_fixed: float = 0.0
    max_participation_pct: float = 5.0
    min_adv_to_trade: float = 0.0


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20
    max_sector_pct: float = 0.40
    max_asset_class_pct: float = 0.80
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.50
    max_leverage: float = 1.0
    max_drawdown_pct: float = 0.20
    max_daily_loss_pct: float = 0.03
    max_portfolio_volatility: float = 0.25
    var_limit_pct: float = 0.05
    cvar_limit_pct: float = 0.08
    var_confidence: float = 0.95
    var_horizon_days: int = 1
    position_sizing: str = "fixed_fractional"
    risk_fraction: float = 0.01
    #: Kelly is always fractional; full Kelly is a common way to blow up.
    kelly_fraction: float = 0.25
    max_kelly_weight: float = 0.10


@dataclass
class KillSwitchConfig:
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    stale_data_seconds: int = 300
    max_execution_latency_ms: float = 2000.0
    max_order_rejections: int = 5
    reconciliation_tolerance: float = 1e-6
    reconciliation_retries: int = 3
    auto_engage: bool = True


@dataclass
class ExecutionConfig:
    mode: str = "backtest"  # backtest | paper | sandbox | live
    live_trading_enabled: bool = False
    paper_trading: bool = True
    sandbox_only: bool = True
    broker_adapter: str = "paper"
    order_timeout_seconds: float = 30.0
    unknown_order_grace_seconds: float = 60.0
    reconcile_on_unknown: bool = True
    require_idempotency_key: bool = True
    submission_dedupe_window_seconds: float = 300.0


@dataclass
class SecurityConfig:
    auth_required: bool = True
    jwt_algorithm: str = "HS256"
    jwt_ttl_seconds: int = 900
    jwt_refresh_ttl_seconds: int = 604800
    password_hash: str = "argon2"
    rate_limit_enabled: bool = True
    cors_origins: list[str] = field(default_factory=list)
    max_request_bytes: int = 1_048_576
    audit_enabled: bool = True
    #: Secrets are read from these env var names, never from config files.
    secret_env_var: str = "QUANTFORGE_SECRET_KEY"


@dataclass
class SandboxConfig:
    """Limits for user-supplied strategy code.

    These are *platform-side* limits. A real deployment should additionally
    enforce them with container isolation; the in-process limit is a
    defence-in-depth measure, not a substitute.
    """

    enabled: bool = True
    cpu_seconds: float = 30.0
    wall_seconds: float = 60.0
    memory_mb: int = 512
    max_output_bytes: int = 1_048_576
    network_enabled: bool = False
    filesystem_write_enabled: bool = False
    blocked_modules: list[str] = field(
        default_factory=lambda: [
            "socket", "subprocess", "ctypes", "multiprocessing",
            "http", "urllib", "requests", "ftplib", "telnetlib", "smtplib",
            "sqlite3", "psycopg2", "sqlalchemy", "vault", "hvac", "pickle",
        ]
    )


@dataclass
class MLConfig:
    random_seed: int = 42
    enable_gpu: bool = False
    max_threads: int = 4
    max_train_rows: int = 5_000_000
    early_stopping_rounds: int = 25
    calibration_method: str = "isotonic"  # isotonic | sigmoid | none
    n_jobs: int = 1


@dataclass
class MonitoringConfig:
    latency_sample_rate: float = 1.0
    alert_on_p99_latency: bool = True
    metrics_path: str = str(DEFAULT_ROOT / "logs" / "metrics.jsonl")
    audit_path: str = str(DEFAULT_ROOT / "logs" / "audit.jsonl")
    signal_log_path: str = str(DEFAULT_ROOT / "logs" / "signals.jsonl")
    execution_log_path: str = str(DEFAULT_ROOT / "logs" / "executions.jsonl")


@dataclass
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    version: str = "v1"
    docs_enabled: bool = True


# --------------------------------------------------------------------------
# Root config
# --------------------------------------------------------------------------


@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    kill_switch: KillSwitchConfig = field(default_factory=KillSwitchConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: APIConfig = field(default_factory=APIConfig)
    debug: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Range-check every section. Bad config must fail at construction."""
        b = self.backtest
        if b.initial_capital <= 0:
            raise ConfigurationError("backtest.initial_capital must be > 0")
        if b.bars_per_year <= 0:
            raise ConfigurationError("backtest.bars_per_year must be > 0")
        if not 0 <= b.assume_spread_bps <= 1000:
            raise ConfigurationError("backtest.assume_spread_bps out of range")
        if b.max_participation_pct <= 0:
            raise ConfigurationError("backtest.max_participation_pct must be > 0")

        r = self.risk
        if not 0 < r.max_position_pct <= 1:
            raise ConfigurationError("risk.max_position_pct must be in (0, 1]")
        if r.max_gross_exposure <= 0:
            raise ConfigurationError("risk.max_gross_exposure must be > 0")
        if not 0.5 < r.var_confidence < 1.0:
            raise ConfigurationError("risk.var_confidence must be in (0.5, 1.0)")
        if not 0 < r.kelly_fraction <= 1:
            raise ConfigurationError("risk.kelly_fraction must be in (0, 1]")
        if r.max_kelly_weight <= 0 or r.max_kelly_weight > 1:
            raise ConfigurationError("risk.max_kelly_weight must be in (0, 1]")

        k = self.kill_switch
        if k.stale_data_seconds <= 0:
            raise ConfigurationError("kill_switch.stale_data_seconds must be > 0")
        if k.reconciliation_retries < 1:
            raise ConfigurationError("kill_switch.reconciliation_retries must be >= 1")

        e = self.execution
        if e.mode not in ("backtest", "paper", "sandbox", "live"):
            raise ConfigurationError(f"unknown execution mode: {e.mode}")
        if e.live_trading_enabled and not e.sandbox_only:
            raise ConfigurationError(
                "execution.live_trading_enabled requires an explicit authorisation "
                "record; setting it here is insufficient and sandbox_only must "
                "be cleared deliberately"
            )

        s = self.sandbox
        if s.memory_mb < 64:
            raise ConfigurationError("sandbox.memory_mb too small")
        if s.network_enabled:
            raise ConfigurationError(
                "sandbox.network_enabled must be False; user strategies are not "
                "permitted outbound network access"
            )

    # -- convenience accessors -------------------------------------------

    @property
    def data_root(self) -> Path:
        return Path(self.data.root)

    def ensure_directories(self) -> None:
        paths = [
            self.data.raw_path, self.data.normalized_path, self.data.adjusted_path,
            self.data.features_path, self.data.cache_path,
            Path(self.monitoring.metrics_path).parent,
            Path(self.monitoring.audit_path).parent,
        ]
        for p in paths:
            Path(p).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        from quantforge.core.reproducibility import content_hash

        return content_hash(self.to_dict())


# --------------------------------------------------------------------------
# Global accessor
# --------------------------------------------------------------------------

_config: Optional[Config] = None


def get_config() -> Config:
    """Process-wide config. Override with the ``QUANTFORGE_CONFIG`` env var."""
    global _config
    if _config is None:
        path = os.environ.get("QUANTFORGE_CONFIG")
        if path and Path(path).exists():
            _config = Config.from_file(path)
        else:
            _config = Config()
    return _config


def set_config(config: Config) -> None:
    global _config
    config.validate()
    _config = config


def reset_config() -> None:
    global _config
    _config = None


# -- (de)serialisation -------------------------------------------------------


def _build(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigurationError(
            f"{cls.__name__}: unknown configuration keys {sorted(unknown)}; "
            "refusing to silently ignore them"
        )
    return cls(**raw)


def config_to_dict(config: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(config):
        section = getattr(config, f.name)
        out[f.name] = asdict(section) if is_dataclass(section) else section
    return out


def _scalar(name: str, value: Any, declared: Any) -> Any:
    """Type-check a non-section config field.

    ``declared`` arrives as a string under ``from __future__ import annotations``
    for the same reason sections need ``_RESOLVE``. A declared type of
    ``bool`` is checked strictly, because ``bool`` subclasses ``int`` and a
    ``PositionLimit`` of ``1`` where ``0.2`` was documented is a far more
    dangerous mistake than a type error.
    """
    want = declared if isinstance(declared, type) else None
    if want is None:
        text = str(declared)
        want = {"bool": bool, "int": int, "float": float, "str": str}.get(text)
    if want is None:
        return value                      # unrecognised annotation: pass through
    if want is bool:
        ok = isinstance(value, bool)
    elif want is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        ok = isinstance(value, want)
    if not ok:
        raise ConfigurationError(
            f"configuration key {name!r} must be {want.__name__}, got "
            f"{type(value).__name__} ({value!r})"
        )
    return value


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build a :class:`Config` from a nested mapping.

    Not every field is a section: ``debug`` is a plain ``bool``. Treating it as
    a section crashed the loader on the first config file that contained it,
    which is the sort of failure that leaves a stale file in place for months
    because nobody can work out why loading "does nothing".

    A section given a non-mapping is also rejected rather than coerced, since
    ``_build`` would otherwise call ``set()`` on a scalar and report the
    resulting ``TypeError`` instead of naming the offending key.
    """
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"configuration must be a mapping of sections, got "
            f"{type(raw).__name__}"
        )
    kwargs: dict[str, Any] = {}
    for f in fields(Config):
        if f.name not in raw:
            continue
        declared = f.type
        section_cls = declared
        if isinstance(section_cls, str):  # postponed annotation
            section_cls = _RESOLVE.get(f.name)
        if section_cls is None or not is_dataclass(section_cls):
            # A scalar field: pass the value through, but check its type.
            # Without this, `debug: {typo: 1}` is accepted, the unknown-key
            # protection that sections get is bypassed, and a mapping ends up
            # living in a field documented as a bool. The ORIGINAL declared
            # type is passed, not the resolved-None above, or there is nothing
            # left to check against.
            kwargs[f.name] = _scalar(f.name, raw[f.name], declared)
            continue
        value = raw[f.name]
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ConfigurationError(
                f"configuration section {f.name!r} must be a mapping, got "
                f"{type(value).__name__}"
            )
        kwargs[f.name] = _build(section_cls, value)
    return Config(**kwargs)


_RESOLVE = {
    "database": DatabaseConfig,
    "data": DataConfig,
    "backtest": BacktestConfig,
    "risk": RiskConfig,
    "kill_switch": KillSwitchConfig,
    "execution": ExecutionConfig,
    "security": SecurityConfig,
    "sandbox": SandboxConfig,
    "ml": MLConfig,
    "monitoring": MonitoringConfig,
    "api": APIConfig,
}


def _load_raw(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    if p.suffix == ".json":
        import json

        return json.loads(text or "{}")
    raise ConfigurationError(f"unsupported config format: {p.suffix}")


Config.from_file = classmethod(  # type: ignore[attr-defined]
    lambda cls, path: cls(**{
        k: _build(_RESOLVE[k], v)
        for k, v in _load_raw(path).items()
        if k in _RESOLVE
    })
)  # type: ignore[attr-defined]
