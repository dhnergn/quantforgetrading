"""Unit tests for strategy engine — signal generation and conditions."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantforge.features.engine import FeatureEngine, FeatureVector
from quantforge.strategies.crossover import CrossoverConfig, CrossoverStrategy
from quantforge.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from quantforge.strategies.breakout import BreakoutConfig, BreakoutStrategy
from quantforge.strategies.base import SignalDirection


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_price = low + np.random.rand(n) * (high - low)
    volume = np.random.randint(1000, 10000, n)

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


@pytest.fixture
def feature_engine():
    return FeatureEngine()


class TestCrossoverStrategy:
    def test_config_validation(self):
        """Invalid config should raise ValueError."""
        with pytest.raises(ValueError):
            CrossoverStrategy(CrossoverConfig(name="test", fast_period=50, slow_period=20))

    def test_signal_generation(self, sample_data, feature_engine):
        """Strategy should generate signals."""
        strategy = CrossoverStrategy(CrossoverConfig(name="crossover", volume_confirm=False))
        features = feature_engine.compute_features(
            instrument_id="TEST",
            data=sample_data,
            as_of=sample_data.index[-1],
            feature_names=strategy.required_features(),
        )
        signal = strategy.evaluate(features, position=0.0)
        # Signal may be None (hold) or a Signal — both are valid
        if signal is not None:
            assert signal.direction in (SignalDirection.LONG, SignalDirection.EXIT_LONG)
            assert 0 <= signal.confidence <= 1.0

    def test_required_features(self):
        strategy = CrossoverStrategy(CrossoverConfig(name="test"))
        features = strategy.required_features()
        assert "ema_12" in features
        assert "ema_26" in features


class TestMeanReversionStrategy:
    def test_config_validation(self):
        with pytest.raises(ValueError):
            MeanReversionStrategy(MeanReversionConfig(name="test", oversold_threshold=80, overbought_threshold=20))

    def test_oversold_signal(self):
        """RSI below threshold should generate LONG signal."""
        features = FeatureVector(
            instrument_id="TEST",
            timestamp=datetime.utcnow(),
            features={"rsi_14": 25.0},
            definitions={},
        )
        strategy = MeanReversionStrategy(MeanReversionConfig(name="test"))
        signal = strategy.evaluate(features, position=0.0)
        assert signal is not None
        assert signal.direction == SignalDirection.LONG

    def test_overbought_exit(self):
        """RSI above threshold with position should generate EXIT."""
        features = FeatureVector(
            instrument_id="TEST",
            timestamp=datetime.utcnow(),
            features={"rsi_14": 75.0},
            definitions={},
        )
        strategy = MeanReversionStrategy(MeanReversionConfig(name="test"))
        signal = strategy.evaluate(features, position=100.0)
        assert signal is not None
        assert signal.direction == SignalDirection.EXIT_LONG


class TestBreakoutStrategy:
    def test_config_validation(self):
        with pytest.raises(ValueError):
            BreakoutStrategy(BreakoutConfig(name="test", entry_period=0))

    def test_required_features(self):
        strategy = BreakoutStrategy(BreakoutConfig(name="test"))
        features = strategy.required_features()
        assert "sma_20" in features
        assert "sma_50" in features
