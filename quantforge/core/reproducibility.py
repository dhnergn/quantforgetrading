"""Deterministic hashing and experiment reproducibility.

Section 86 of the blueprint requires that the same configuration against the
same immutable dataset reproduce the same result. That is only checkable if
every component contributes a stable content hash, so this module provides the
canonicalisation rules and the experiment manifest that ties them together.

Two things are deliberately *excluded* from the hash: wall-clock timestamps and
absolute filesystem paths. Including them would make every run unique and
defeat the purpose.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from quantforge.core.timestamps import utcnow

# Keys stripped before hashing because they vary run-to-run without affecting
# the result. Keeping this list explicit (rather than filtering by heuristic)
# keeps the hash auditable.
_VOLATILE_KEYS = frozenset(
    {
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "timestamp",
        "run_id",
        "execution_id",
        "path",
        "db_path",
        "outdir",
        "output_dir",
        "cache_dir",
        "host",
        "pid",
    }
)


def _normalise(value: Any) -> Any:
    """Reduce a value to a canonical, JSON-serialisable, order-stable form."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # -0.0 and 0.0 are the same quantity; 1.0 and 1 are not conflated
        # because ints and floats are distinct JSON types.
        return 0.0 if value == 0.0 else value
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (dt.timedelta, np.timedelta64)):
        return float(value / np.timedelta64(1, "s"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"__ndarray__": _normalise(value.tolist())}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__name__,
            **{k: _normalise(v) for k, v in asdict(value).items()},
        }
    if isinstance(value, dict):
        return {
            str(k): _normalise(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if str(k) not in _VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_normalise(v) for v in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return items
    if hasattr(value, "value") and hasattr(type(value), "__members__"):
        # Enum
        return f"{type(value).__name__}.{value.name}"
    if hasattr(value, "fingerprint") and callable(value.fingerprint):
        return value.fingerprint()
    return str(value)


def canonical_json(payload: Any) -> str:
    """Deterministic JSON rendering used as hash input."""
    return json.dumps(
        _normalise(payload), sort_keys=True, separators=(",", ":"), default=str
    )


def content_hash(payload: Any, length: int = 16) -> str:
    """Stable short hash of any structure."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:length]


def array_hash(array: np.ndarray, length: int = 16) -> str:
    """Hash a numeric array's values and shape (dtype-exact, shape-aware)."""
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:length]


def file_hash(path: str | os.PathLike, length: int = 16) -> str:
    """Streaming content hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def set_seed(seed: int) -> None:
    """Seed every RNG the platform may touch.

    Called by every training entry point. Order matters: numpy must be seeded
    before any legacy global state is used.
    """
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:  # optional, not fatal
        import os as _os

        _os.environ.setdefault("PYTHONHASHSEED", str(seed))
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------
# Experiment manifest
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ExperimentManifest:
    """Everything needed to reproduce a research run.

    This is the object a backtest, model-training run, or paper-trading session
    is identified by. It is serialised into the run's output directory and its
    ``config_hash`` is the reproducibility key.
    """

    experiment_type: str
    dataset_version: str
    feature_set_version: str
    strategy_version: str
    model_version: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 0
    transaction_costs: dict[str, Any] = field(default_factory=dict)
    slippage: dict[str, Any] = field(default_factory=dict)
    universe: dict[str, Any] = field(default_factory=dict)
    time_period: dict[str, Any] = field(default_factory=dict)
    risk_configuration: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def compute_hash(self) -> str:
        return content_hash(
            {
                "experiment_type": self.experiment_type,
                "dataset_version": self.dataset_version,
                "feature_set_version": self.feature_set_version,
                "strategy_version": self.strategy_version,
                "model_version": self.model_version,
                "parameters": self.parameters,
                "random_seed": self.random_seed,
                "transaction_costs": self.transaction_costs,
                "slippage": self.slippage,
                "universe": self.universe,
                "time_period": self.time_period,
                "risk_configuration": self.risk_configuration,
                "preprocessing": self.preprocessing,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.compute_hash(),
            "experiment_type": self.experiment_type,
            "dataset_version": self.dataset_version,
            "feature_set_version": self.feature_set_version,
            "strategy_version": self.strategy_version,
            "model_version": self.model_version,
            "parameters": self.parameters,
            "random_seed": self.random_seed,
            "transaction_costs": self.transaction_costs,
            "slippage": self.slippage,
            "universe": self.universe,
            "time_period": self.time_period,
            "risk_configuration": self.risk_configuration,
            "preprocessing": self.preprocessing,
            "environment": self.environment,
            "notes": self.notes,
            "created_at": utcnow().isoformat(),
        }

    def save(self, path: str | os.PathLike) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return str(out)

    @staticmethod
    def verify(path: str | os.PathLike) -> bool:
        """Recompute a stored hash and compare against the recorded one."""
        stored = json.loads(Path(path).read_text())
        recorded = stored.get("config_hash")
        rebuilt = ExperimentManifest(
            experiment_type=stored["experiment_type"],
            dataset_version=stored["dataset_version"],
            feature_set_version=stored["feature_set_version"],
            strategy_version=stored["strategy_version"],
            model_version=stored.get("model_version"),
            parameters=stored.get("parameters", {}),
            random_seed=stored.get("random_seed", 0),
            transaction_costs=stored.get("transaction_costs", {}),
            slippage=stored.get("slippage", {}),
            universe=stored.get("universe", {}),
            time_period=stored.get("time_period", {}),
            risk_configuration=stored.get("risk_configuration", {}),
            preprocessing=stored.get("preprocessing", {}),
        )
        return rebuilt.compute_hash() == recorded


def capture_environment() -> dict[str, Any]:
    """Record library versions so a drift in results can be attributed."""
    env: dict[str, Any] = {"python": os.sys.version.split()[0]}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "sqlalchemy", "pydantic"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            env[mod] = "absent"
    return env


def combine_hashes(parts: Iterable[tuple[str, str]], length: int = 16) -> str:
    """Hash an ordered sequence of ``(name, hash)`` pairs."""
    return content_hash(sorted(parts), length=length)


def rolling_hash(values: Sequence[float], length: int = 16) -> str:
    return array_hash(np.asarray(values, dtype="float64"), length=length)
