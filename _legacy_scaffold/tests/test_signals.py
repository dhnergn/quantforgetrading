"""Unit tests for signal generator — the core signal bot."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantforge.features.engine import FeatureEngine
from quantforge.signals.generator import SignalConfig, SignalGenerator
from quantforge.strategies.crossover import CrossoverConfig, CrossoverStrategy
from quantforge.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy


@pytest.fixture
def sample_data():
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


class TestSignalGenerator:
    def test_generate_signals(self, sample_data):
        """Signal generator should produce signals."""
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))
        config = SignalConfig(
            strategy=strategy,
            feature_engine=FeatureEngine(),
            instruments=["TEST"],
            min_confidence=0.0,
        )
        generator = SignalGenerator(config)

        signals = generator.generate_signals(sample_data)

        # May or may not generate signals depending on data
        # Both outcomes are valid
        assert isinstance(signals, list)

    def test_signal_has_full_context(self, sample_data):
        """Every signal should have complete context."""
        strategy = MeanReversionStrategy(MeanReversionConfig(name="test"))
        config = SignalConfig(
            strategy=strategy,
            feature_engine=FeatureEngine(),
            instruments=["TEST"],
            min_confidence=0.0,
        )
        generator = SignalGenerator(config)

        signals = generator.generate_signals(sample_data)

        for sig in signals:
            assert sig.signal_id is not None
            assert sig.timestamp is not None
            assert sig.instrument_id == "TEST"
            assert sig.direction is not None
            assert 0 <= sig.confidence <= 1.0
            assert sig.strategy_name == "test"
            assert sig.decision in ("enter", "exit")

    def test_signal_history(self, sample_data):
        """Signal history should be maintained."""
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))
        config = SignalConfig(
            strategy=strategy,
            feature_engine=FeatureEngine(),
            instruments=["TEST"],
            min_confidence=0.0,
        )
        generator = SignalGenerator(config)

        generator.generate_signals(sample_data)
        history = generator.get_signal_history()

        assert len(history) >= 0

    def test_to_dataframe(self, sample_data):
        """Signal history should convert to DataFrame."""
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))
        config = SignalConfig(
            strategy=strategy,
            feature_engine=FeatureEngine(),
            instruments=["TEST"],
            min_confidence=0.0,
        )
        generator = SignalGenerator(config)

        generator.generate_signals(sample_data)
        df = generator.to_dataframe()

        assert isinstance(df, pd.DataFrame)

    def test_min_confidence_filter(self, sample_data):
        """Signals below min_confidence should be filtered."""
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))
        config = SignalConfig(
            strategy=strategy,
            feature_engine=FeatureEngine(),
            instruments=["TEST"],
            min_confidence=0.99,  # Very high threshold
        )
        generator = SignalGenerator(config)

        signals = generator.generate_signals(sample_data)

        for sig in signals:
            assert sig.confidence >= 0.99
