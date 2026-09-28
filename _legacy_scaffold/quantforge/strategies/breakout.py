"""Breakout strategy — price breakouts with volume confirmation."""
from typing import List, Optional
from quantforge.strategies.base import Signal, SignalDirection, Strategy, StrategyConfig

class BreakoutConfig(StrategyConfig):
    entry_period: int = 20
    exit_period: int = 10
    volume_confirm: bool = True
    volume_threshold: float = 1.5
    atr_filter: bool = True

class BreakoutStrategy(Strategy):
    def __init__(self, config: BreakoutConfig):
        self.entry_period = config.entry_period
        self.exit_period = config.exit_period
        self.volume_confirm = config.volume_confirm
        self.volume_threshold = config.volume_threshold
        self.atr_filter = config.atr_filter
        super().__init__(config)

    def _validate_config(self) -> None:
        if self.entry_period <= 0 or self.exit_period <= 0:
            raise ValueError("Periods must be positive")

    def required_features(self) -> List[str]:
        features = ["sma_20", "sma_50"]
        if self.volume_confirm:
            features.append("relative_volume_20")
        if self.atr_filter:
            features.append("atr_14")
        return features

    def evaluate(self, features: Dict[str, float], position: float = 0.0) -> Optional[Signal]:
        from datetime import datetime
        sma_20 = features.get("sma_20")
        sma_50 = features.get("sma_50")
        if sma_20 is None or sma_50 is None:
            return None

        volume_ok = True
        if self.volume_confirm:
            rv = features.get("relative_volume_20")
            volume_ok = rv is not None and rv > self.volume_threshold

        atr_ok = True
        if self.atr_filter:
            atr = features.get("atr_14")
            atr_ok = atr is not None and atr > 0

        if sma_20 > sma_50 and volume_ok and atr_ok and position <= 0:
            return Signal(
                direction=SignalDirection.LONG,
                instrument_id="",
                timestamp=datetime.utcnow(),
                confidence=0.7,
                features=features,
                indicators={"sma_20": sma_20, "sma_50": sma_50},
                strategy_name=self.config.name,
                strategy_version=self.config.version,
            )
        if sma_20 < sma_50 and position > 0:
            return Signal(
                direction=SignalDirection.EXIT_LONG,
                instrument_id="",
                timestamp=datetime.utcnow(),
                confidence=0.7,
                features=features,
                indicators={"sma_20": sma_20, "sma_50": sma_50},
                strategy_name=self.config.name,
                strategy_version=self.config.version,
            )
        return None
