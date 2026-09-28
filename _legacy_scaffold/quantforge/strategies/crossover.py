"""Crossover strategy — SMA/EMA crossovers with volume confirmation."""
from typing import List, Optional
from quantforge.strategies.base import Signal, SignalDirection, Strategy, StrategyConfig

class CrossoverConfig(StrategyConfig):
    fast_period: int = 12
    slow_period: int = 26
    ma_type: str = "ema"
    volume_confirm: bool = True
    volume_threshold: float = 1.2

class CrossoverStrategy(Strategy):
    def __init__(self, config: CrossoverConfig):
        self.fast_period = config.fast_period
        self.slow_period = config.slow_period
        self.ma_type = config.ma_type
        self.volume_confirm = config.volume_confirm
        self.volume_threshold = config.volume_threshold
        super().__init__(config)

    def _validate_config(self) -> None:
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        if self.ma_type not in ("sma", "ema", "wma"):
            raise ValueError(f"Invalid ma_type: {self.ma_type}")

    def required_features(self) -> List[str]:
        features = [f"{self.ma_type}_{self.fast_period}", f"{self.ma_type}_{self.slow_period}"]
        if self.volume_confirm:
            features.append(f"relative_volume_{self.slow_period}")
        return features

    def evaluate(self, features: Dict[str, float], position: float = 0.0) -> Optional[Signal]:
        fast_key = f"{self.ma_type}_{self.fast_period}"
        slow_key = f"{self.ma_type}_{self.slow_period}"
        fast_val = features.get(fast_key)
        slow_val = features.get(slow_key)
        if fast_val is None or slow_val is None:
            return None

        volume_ok = True
        if self.volume_confirm:
            rv = features.get(f"relative_volume_{self.slow_period}")
            volume_ok = rv is not None and rv > self.volume_threshold

        from datetime import datetime
        if fast_val > slow_val and volume_ok and position <= 0:
            return Signal(
                direction=SignalDirection.LONG,
                instrument_id="",
                timestamp=datetime.utcnow(),
                confidence=min(abs(fast_val - slow_val) / slow_val * 100, 1.0) if slow_val else 0.5,
                features=features,
                indicators={fast_key: fast_val, slow_key: slow_val},
                strategy_name=self.config.name,
                strategy_version=self.config.version,
            )
        elif fast_val < slow_val and position > 0:
            return Signal(
                direction=SignalDirection.EXIT_LONG,
                instrument_id="",
                timestamp=datetime.utcnow(),
                confidence=0.8,
                features=features,
                indicators={fast_key: fast_val, slow_key: slow_val},
                strategy_name=self.config.name,
                strategy_version=self.config.version,
            )
        return None
