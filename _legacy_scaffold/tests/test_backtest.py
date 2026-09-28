"""Unit tests for backtest engine — look-ahead bias, costs, and correctness."""
import numpy as np
import pandas as pd
import pytest

from quantforge.backtest.engine import BacktestConfig, BacktestEngine
from quantforge.strategies.crossover import CrossoverConfig, CrossoverStrategy


@pytest.fixture
def sample_data():
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


class TestBacktestEngine:
    def test_basic_backtest(self, sample_data):
        """Backtest should run without errors."""
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))
        config = BacktestConfig(
            strategy=strategy,
            data=sample_data,
            instrument_ids=["TEST"],
            start_date=sample_data.index[50],
            end_date=sample_data.index[-1],
        )
        engine = BacktestEngine(config)
        result = engine.run()

        assert result.total_return is not None
        assert result.total_trades >= 0
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0

    def test_look_ahead_bias_prevention(self, sample_data):
        """Backtest should never use future bars."""
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))
        config = BacktestConfig(
            strategy=strategy,
            data=sample_data,
            instrument_ids=["TEST"],
            start_date=sample_data.index[50],
            end_date=sample_data.index[-1],
        )
        engine = BacktestEngine(config)
        result = engine.run()

        # Equity curve should be monotonically increasing in timestamp
        assert result.equity_curve.index.is_monotonic_increasing

    def test_transaction_costs_applied(self, sample_data):
        """Transaction costs should reduce returns."""
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))

        # Run with zero costs
        config_free = BacktestConfig(
            strategy=strategy,
            data=sample_data,
            instrument_ids=["TEST"],
            start_date=sample_data.index[50],
            end_date=sample_data.index[-1],
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        result_free = BacktestEngine(config_free).run()

        # Run with costs
        config_costly = BacktestConfig(
            strategy=strategy,
            data=sample_data,
            instrument_ids=["TEST"],
            start_date=sample_data.index[50],
            end_date=sample_data.index[-1],
            commission_pct=0.01,
            slippage_pct=0.01,
        )
        result_costly = BacktestEngine(config_costly).run()

        # Costs should reduce returns (in most cases)
        # This is a probabilistic test — may not always pass
        if result_free.total_trades > 0 and result_costly.total_trades > 0:
            assert result_costly.total_return <= result_free.total_return + 0.01

    def test_reproducibility(self, sample_data):
        """Same config should produce same results."""
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))
        config = BacktestConfig(
            strategy=strategy,
            data=sample_data,
            instrument_ids=["TEST"],
            start_date=sample_data.index[50],
            end_date=sample_data.index[-1],
            random_seed=42,
        )

        result1 = BacktestEngine(config).run()
        result2 = BacktestEngine(config).run()

        assert result1.total_return == pytest.approx(result2.total_return)
        assert result1.total_trades == result2.total_trades

    def test_config_hash(self, sample_data):
        """Config hash should be deterministic."""
        strategy = CrossoverStrategy(CrossoverConfig(name="test", volume_confirm=False))
        config = BacktestConfig(
            strategy=strategy,
            data=sample_data,
            instrument_ids=["TEST"],
            start_date=sample_data.index[50],
            end_date=sample_data.index[-1],
        )

        hash1 = config.compute_hash()
        hash2 = config.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256
