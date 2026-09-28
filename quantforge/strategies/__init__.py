"""Strategy engine — deterministic signal generation from features."""
from quantforge.strategies.base import Strategy, StrategyConfig, Signal, SignalDirection
from quantforge.strategies.crossover import CrossoverStrategy, CrossoverConfig
from quantforge.strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from quantforge.strategies.breakout import BreakoutStrategy, BreakoutConfig

__all__ = [
    "Strategy", "StrategyConfig", "Signal", "SignalDirection",
    "CrossoverStrategy", "CrossoverConfig",
    "MeanReversionStrategy", "MeanReversionConfig",
    "BreakoutStrategy", "BreakoutConfig",
]
