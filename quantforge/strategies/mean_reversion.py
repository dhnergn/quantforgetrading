"""Mean reversion strategy — RSI extremes and Bollinger Band deviations."""
from typing import List, Optional
from quantforge.strategies.base import Signal, SignalDirection, Strategy, StrategyConfig

class MeanReversionConfig(StrategyConfig):
    rsi_period: int = 14
    oversold_threshold: float = 30.0
    overbought_threshold: float = 70.0

class MeanReversionStrategy(Strategy):
    def __init__(self, config: MeanReversionConfig):
        self.rsi_period = config.rsi_period
        self.oversold = config.oversold_threshold
        self.overbought = config.overbought_threshold
        super().__init__(config)

    def _validate_config(self) -> None:
        if not (0 < self.oversold < self.overbought < 100):
            raise ValueError("RSI thresholds must satisfy: 0 < oversold < overbought < 100")

    def required_features(self) -> List[str]:
        return [f"rsi_{self.rsi_period}"]

    def evaluate(self, features: Dict[str, float], position: float = 0.0) -> Optional[Signal]:
        from datetime import datetime
        rsi_key = f"rsi_{self.rsi_period}"
        rsi_val = features.get(rsi_key)
        if rsi_val is None:
            return None

        if rsi_val < self.oversold and position <= 0:
            confidence = min((self.oversold - rsi_val) / self.oversold, 1.0)
            return Signal(
                direction=SignalDirection.LONG,
                instrument_id="",
                timestamp=datetime.utcnow(),
                confidence=confidence,
                features=features,
                indicators={rsi_key: rsi_val},
                strategy_name=self.config.name,
                strategy_version=self.config.version,
            )
        if rsi_val > self.overbought and position > 0:
            confidence = min((rsi_val - self.overbought) / (100 - self.overbought), 1.0)
            return Signal(
                direction=SignalDirection.EXIT_LONG,
                instrument_id="",
                timestamp=datetime.utcnow(),
                confidence=confidence,
                features=features,
                indicators={rsi_key: rsi_val},
                strategy_name=self.config.name,
                strategy_version=self.config.version,
            )
        return None
