"""Base strategy interface."""
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

class SignalDirection(enum.Enum):
    LONG = "long"
    SHORT = "short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"

@dataclass
class StrategyConfig:
    name: str
    version: str = "1.0.0"
    asset_classes: List[str] = field(default_factory=lambda: ["equity", "etf"])
    instruments: Optional[List[str]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    position_sizing: Dict[str, Any] = field(default_factory=dict)
    stop_rules: Optional[Dict[str, Any]] = None
    risk_parameters: Optional[Dict[str, Any]] = None

@dataclass
class Signal:
    direction: SignalDirection
    instrument_id: str
    timestamp: datetime
    confidence: float
    features: Dict[str, float]
    indicators: Dict[str, float]
    strategy_name: str
    strategy_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class Strategy(ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, features: Dict[str, float], position: float = 0.0) -> Optional[Signal]:
        pass

    @abstractmethod
    def required_features(self) -> List[str]:
        pass

    def compute_config_hash(self) -> str:
        import hashlib, json
        canonical = json.dumps(self.config.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()
