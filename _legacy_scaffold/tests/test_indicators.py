"""Unit tests for technical indicators — mathematical correctness and reference values."""
import numpy as np
import pandas as pd
import pytest

from quantforge.indicators import (
    SMA, EMA, WMA, RSI, MACD, BollingerBands, ATR,
    OBV, VWAP, ChaikinMoneyFlow, RelativeVolume,
    Stochastic, ROC, KeltnerChannels, RollingVolatility,
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
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


class TestSMA:
    def test_basic(self, sample_data):
        sma = SMA(20)
        result = sma.calculate(sample_data["close"])
        assert len(result) == len(sample_data)
        assert result.iloc[:19].isna().all()  # Warmup period
        assert result.iloc[19:].notna().all()

    def test_reference_value(self):
        """Test against known reference value."""
        data = pd.Series([1, 2, 3, 4, 5])
        sma = SMA(3)
        result = sma.calculate(data)
        assert result.iloc[2] == pytest.approx(2.0)  # (1+2+3)/3
        assert result.iloc[3] == pytest.approx(3.0)  # (2+3+4)/3
        assert result.iloc[4] == pytest.approx(4.0)  # (3+4+5)/3


class TestEMA:
    def test_basic(self, sample_data):
        ema = EMA(20)
        result = ema.calculate(sample_data["close"])
        assert len(result) == len(sample_data)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()

    def test_reference_value(self):
        """EMA with span=3: α = 2/(3+1) = 0.5."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = EMA(3)
        result = ema.calculate(data)
        # First valid value is SMA of first 3: (1+2+3)/3 = 2.0
        assert result.iloc[2] == pytest.approx(2.0)
        # EMA[3] = 0.5 * 4 + 0.5 * 2.0 = 3.0
        assert result.iloc[3] == pytest.approx(3.0)


class TestWMA:
    def test_basic(self, sample_data):
        wma = WMA(20)
        result = wma.calculate(sample_data["close"])
        assert len(result) == len(sample_data)
        assert result.iloc[:19].isna().all()

    def test_reference_value(self):
        """WMA with period=3: weights [1, 2, 3]/6."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        wma = WMA(3)
        result = wma.calculate(data)
        # WMA[2] = (1*1 + 2*2 + 3*3) / 6 = 14/6 ≈ 2.333
        assert result.iloc[2] == pytest.approx(14.0 / 6.0)


class TestRSI:
    def test_basic(self, sample_data):
        rsi = RSI(14)
        result = rsi.calculate(sample_data["close"])
        assert len(result) == len(sample_data)
        assert result.iloc[:14].isna().all()
        assert result.iloc[14:].notna().all()
        assert (result.dropna() >= 0).all() and (result.dropna() <= 100).all()

    def test_reference_value(self):
        """RSI with all gains should be 100."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                          11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        rsi = RSI(14)
        result = rsi.calculate(data)
        assert result.iloc[-1] == pytest.approx(100.0)


class TestMACD:
    def test_basic(self, sample_data):
        macd = MACD()
        result = macd.calculate(sample_data["close"])
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns
        assert len(result) == len(sample_data)


class TestBollingerBands:
    def test_basic(self, sample_data):
        bb = BollingerBands(20, 2.0)
        result = bb.calculate(sample_data["close"])
        assert "mid" in result.columns
        assert "upper" in result.columns
        assert "lower" in result.columns
        assert "bandwidth" in result.columns
        # Upper >= Mid >= Lower
        valid = result.dropna()
        assert (valid["upper"] >= valid["mid"]).all()
        assert (valid["mid"] >= valid["lower"]).all()


class TestATR:
    def test_basic(self, sample_data):
        atr = ATR(14)
        result = atr.calculate(sample_data["high"], sample_data["low"], sample_data["close"])
        assert len(result) == len(sample_data)
        assert result.iloc[:14].isna().all()
        assert (result.dropna() >= 0).all()


class TestOBV:
    def test_basic(self, sample_data):
        obv = OBV()
        result = obv.calculate(sample_data["close"], sample_data["volume"])
        assert len(result) == len(sample_data)
        assert result.iloc[0] == 0  # First value is 0


class TestVWAP:
    def test_basic(self, sample_data):
        vwap = VWAP(20)
        result = vwap.calculate(
            sample_data["high"], sample_data["low"],
            sample_data["close"], sample_data["volume"]
        )
        assert len(result) == len(sample_data)


class TestRelativeVolume:
    def test_basic(self, sample_data):
        rv = RelativeVolume(20)
        result = rv.calculate(sample_data["volume"])
        assert len(result) == len(sample_data)
        assert result.iloc[:20].isna().all()


class TestStochastic:
    def test_basic(self, sample_data):
        stoch = Stochastic(14, 3)
        result = stoch.calculate(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        assert "pct_k" in result.columns
        assert "pct_d" in result.columns
        valid = result.dropna()
        assert (valid["pct_k"] >= 0).all() and (valid["pct_k"] <= 100).all()


class TestROC:
    def test_basic(self, sample_data):
        roc = ROC(10)
        result = roc.calculate(sample_data["close"])
        assert len(result) == len(sample_data)
        assert result.iloc[:10].isna().all()

    def test_reference_value(self):
        data = pd.Series([100.0, 110.0])
        roc = ROC(1)
        result = roc.calculate(data)
        assert result.iloc[1] == pytest.approx(10.0)  # (110/100 - 1) * 100
