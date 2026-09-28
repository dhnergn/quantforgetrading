"""Pytest configuration and shared fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_data():
    """Generate realistic OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    # Generate realistic price series with trend and volatility
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_price = low + np.random.rand(n) * (high - low)
    volume = np.random.randint(100_000, 1_000_000, n)

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


@pytest.fixture
def multi_instrument_data():
    """Generate multi-instrument OHLCV data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    frames = []
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        returns = np.random.randn(n) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        high = close * (1 + np.abs(np.random.randn(n) * 0.01))
        low = close * (1 - np.abs(np.random.randn(n) * 0.01))
        open_price = low + np.random.rand(n) * (high - low)
        volume = np.random.randint(100_000, 1_000_000, n)

        df = pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=dates)
        df["instrument_id"] = symbol
        frames.append(df)

    combined = pd.concat(frames)
    combined.set_index(["instrument_id", combined.index], inplace=True)
    combined.sort_index(inplace=True)
    return combined
