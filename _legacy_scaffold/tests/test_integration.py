"""Integration tests — end-to-end workflow from data to signals to backtest."""
import numpy as np
import pandas as pd
import pytest

from quantforge.backtest.engine import BacktestConfig, BacktestEngine
from quantforge.data.ingestion import DataIngestion
from quantforge.data.validation import DataValidator
from quantforge.features.engine import FeatureEngine
from quantforge.signals.generator import SignalConfig, SignalGenerator
from quantforge.strategies.crossover import CrossoverConfig, CrossoverStrategy
from quantforge.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy


class TestEndToEndWorkflow:
    """Test the complete pipeline: data → validation → features → signals → backtest."""

    @pytest.fixture
    def full_pipeline_data(self):
        """Generate data for the full pipeline."""
        np.random.seed(42)
        n = 300
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

    def test_full_pipeline(self, full_pipeline_data):
        """Test the complete workflow."""
        data = full_pipeline_data

        # Step 1: Ingest
        ingestion = DataIngestion()
        ingested = ingestion.ingest_dataframe(data, "TEST")
        assert len(ingested) == len(data)

        # Step 2: Validate
        validator = DataValidator()
        report = validator.validate(data, "TEST")
        assert report.total_rows == len(data)

        # Step 3: Generate signals
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))
        sig_config = SignalConfig(
            strategy=strategy,
            feature_engine=FeatureEngine(),
            instruments=["TEST"],
            min_confidence=0.0,
        )
        generator = SignalGenerator(sig_config)
        signals = generator.generate_signals(data)
        assert isinstance(signals, list)

        # Step 4: Backtest
        bt_config = BacktestConfig(
            strategy=strategy,
            data=data,
            instrument_ids=["TEST"],
            start_date=data.index[50],
            end_date=data.index[-1],
        )
        engine = BacktestEngine(bt_config)
        result = engine.run()

        assert result.total_return is not None
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0

    def test_multiple_strategies(self, full_pipeline_data):
        """Test that multiple strategies can run on the same data."""
        data = full_pipeline_data
        strategies = [
            CrossoverStrategy(CrossoverConfig(name="crossover", volume_confirm=False)),
            MeanReversionStrategy(MeanReversionConfig(name="mean_reversion")),
        ]

        for strategy in strategies:
            sig_config = SignalConfig(
                strategy=strategy,
                feature_engine=FeatureEngine(),
                instruments=["TEST"],
                min_confidence=0.0,
            )
            generator = SignalGenerator(sig_config)
            signals = generator.generate_signals(data)
            assert isinstance(signals, list)

    def test_reproducibility(self, full_pipeline_data):
        """Same config should produce same results."""
        data = full_pipeline_data
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))

        config = BacktestConfig(
            strategy=strategy,
            data=data,
            instrument_ids=["TEST"],
            start_date=data.index[50],
            end_date=data.index[-1],
            random_seed=42,
        )

        result1 = BacktestEngine(config).run()
        result2 = BacktestEngine(config).run()

        assert result1.total_return == pytest.approx(result2.total_return)
        assert result1.total_trades == result2.total_trades
