"""Unit tests for data validation — OHLC relationships, timestamps, outliers."""
import numpy as np
import pandas as pd
import pytest

from quantforge.data.validation import DataValidator


@pytest.fixture
def valid_data():
    np.random.seed(42)
    n = 100
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


class TestDataValidator:
    def test_valid_data(self, valid_data):
        """Valid data should pass validation."""
        validator = DataValidator()
        report = validator.validate(valid_data, "TEST")

        # May have some warnings but should be mostly valid
        assert report.total_rows == len(valid_data)
        assert report.is_valid or len([i for i in report.issues if i.get("severity") != "warning"]) == 0

    def test_ohlc_violation(self):
        """OHLC violations should be detected."""
        data = pd.DataFrame({
            "open": [100.0],
            "high": [90.0],  # High < Low — violation
            "low": [95.0],
            "close": [98.0],
            "volume": [1000],
        }, index=pd.to_datetime(["2023-01-01"]))

        validator = DataValidator()
        report = validator.validate(data, "TEST")

        assert not report.is_valid
        assert any(issue["type"] == "ohlc_violation" for issue in report.issues)

    def test_negative_volume(self):
        """Negative volume should be detected."""
        data = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [-100],  # Negative volume
        }, index=pd.to_datetime(["2023-01-01"]))

        validator = DataValidator()
        report = validator.validate(data, "TEST")

        assert any(issue["type"] == "negative_volume" for issue in report.issues)

    def test_duplicate_timestamps(self):
        """Duplicate timestamps should be detected."""
        data = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [95.0, 96.0],
            "close": [102.0, 103.0],
            "volume": [1000, 2000],
        }, index=pd.to_datetime(["2023-01-01", "2023-01-01"]))  # Duplicate

        validator = DataValidator()
        report = validator.validate(data, "TEST")

        assert any(issue["type"] == "duplicate_timestamp" for issue in report.issues)

    def test_missing_data(self):
        """Missing data should be detected."""
        data = pd.DataFrame({
            "open": [100.0, np.nan],
            "high": [105.0, 106.0],
            "low": [95.0, 96.0],
            "close": [102.0, 103.0],
            "volume": [1000, 2000],
        }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))

        validator = DataValidator()
        report = validator.validate(data, "TEST")

        assert any(issue["type"] == "missing_data" for issue in report.issues)

    def test_outlier_flagging(self):
        """Extreme outliers should be flagged, not removed."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close[50] = 200  # Extreme outlier
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_price = low + np.random.rand(n) * (high - low)
        volume = np.random.randint(1000, 10000, n)

        data = pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=dates)

        validator = DataValidator()
        report = validator.validate(data, "TEST")

        # Outlier should be flagged
        assert any(issue["type"] == "outlier" for issue in report.issues)
        # But data should not be removed
        assert report.total_rows == n
