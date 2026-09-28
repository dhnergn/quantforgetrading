"""Unit tests for risk engine — position sizing, stops, and portfolio constraints."""
import pandas as pd
import pytest

from quantforge.risk.position_sizing import (
    ATRSizer, FixedFractionalSizer, KellySizer, SizingContext, VolatilitySizer,
)
from quantforge.risk.portfolio import PortfolioRiskManager, RiskLimits
from quantforge.risk.stops import Stop, StopConfig, StopManager, StopType


class TestFixedFractionalSizer:
    def test_basic_sizing(self):
        sizer = FixedFractionalSizer()
        context = SizingContext(
            equity=100_000,
            current_price=100.0,
            stop_distance=2.0,
            risk_fraction=0.01,
        )
        size = sizer.compute_size(context)
        # risk_amount = 1000, stop_distance = 2.0 → size = 500
        assert size == pytest.approx(500.0)

    def test_hard_limits(self):
        """Position size should be capped by max_position_pct."""
        sizer = FixedFractionalSizer()
        context = SizingContext(
            equity=100_000,
            current_price=1.0,
            stop_distance=0.01,
            risk_fraction=0.5,  # Would be 5,000,000 shares without limit
            max_position_pct=0.20,
        )
        size = sizer.compute_size(context)
        # Max position value = 20,000, price = 1.0 → max 20,000 shares
        assert size <= 20_000


class TestATRSizer:
    def test_basic_sizing(self):
        sizer = ATRSizer(atr_multiplier=2.0)
        context = SizingContext(
            equity=100_000,
            current_price=100.0,
            atr=2.0,
            risk_fraction=0.01,
        )
        size = sizer.compute_size(context)
        # risk_amount = 1000, atr*mult = 4.0 → size = 250
        assert size == pytest.approx(250.0)

    def test_no_atr(self):
        """Should return 0 if ATR is not available."""
        sizer = ATRSizer()
        context = SizingContext(equity=100_000, current_price=100.0, atr=None)
        assert sizer.compute_size(context) == 0


class TestKellySizer:
    def test_basic_sizing(self):
        sizer = KellySizer(kelly_fraction=0.25)
        context = SizingContext(equity=100_000, current_price=100.0)
        size = sizer.compute_size(context)
        assert size > 0


class TestPortfolioRiskManager:
    def test_concentration_limit(self):
        """Order exceeding concentration limit should be rejected."""
        limits = RiskLimits(max_single_position_pct=0.20)
        manager = PortfolioRiskManager(limits)

        result = manager.check_order(
            instrument_id="AAPL",
            side="buy",
            quantity=1000,
            price=200.0,  # $200,000 position
            current_positions=[],
            equity=100_000,  # 200% of equity — exceeds limit
        )
        assert not result.approved

    def test_gross_exposure_limit(self):
        """Order exceeding gross exposure limit should be rejected."""
        limits = RiskLimits(max_gross_exposure=1.5)
        manager = PortfolioRiskManager(limits)

        result = manager.check_order(
            instrument_id="AAPL",
            side="buy",
            quantity=1000,
            price=200.0,
            current_positions=[],
            equity=100_000,
        )
        assert not result.approved

    def test_drawdown_halt(self):
        """Trading should halt when drawdown exceeds limit."""
        limits = RiskLimits(max_drawdown_pct=0.10)
        manager = PortfolioRiskManager(limits)

        # Simulate 15% drawdown
        manager._peak_equity = 100_000
        result = manager.check_drawdown(85_000)
        assert not result.approved
        assert manager.is_trading_halted

    def test_var_computation(self):
        """VaR should be computed correctly."""
        limits = RiskLimits()
        manager = PortfolioRiskManager(limits)

        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005, -0.03, 0.02, -0.015, 0.01, -0.005])
        var = manager.compute_var(returns, 0.95)
        assert var > 0  # VaR should be positive (loss amount)

    def test_cvar_computation(self):
        """CVaR should be >= VaR."""
        limits = RiskLimits()
        manager = PortfolioRiskManager(limits)

        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005, -0.03, 0.02, -0.015, 0.01, -0.005])
        var = manager.compute_var(returns, 0.95)
        cvar = manager.compute_cvar(returns, 0.95)
        assert cvar >= var


class TestStopManager:
    def test_percentage_stop(self):
        """Stop should trigger when price falls to stop level."""
        manager = StopManager(slippage_pct=0.001)
        stop = Stop(
            instrument_id="AAPL",
            stop_type=StopType.PERCENTAGE,
            entry_price=100.0,
            stop_price=95.0,
            quantity=100,
            created_at=pd.Timestamp("2023-01-01"),
        )
        manager.add_stop(stop)

        # Price above stop — no trigger
        triggered = manager.check_stops("AAPL", 96.0, 96.5, 95.5, pd.Timestamp("2023-01-02"))
        assert len(triggered) == 0

        # Price at stop — trigger
        triggered = manager.check_stops("AAPL", 95.0, 95.5, 94.5, pd.Timestamp("2023-01-03"))
        assert len(triggered) == 1

    def test_trailing_stop(self):
        """Trailing stop should update with price."""
        manager = StopManager(slippage_pct=0.001)
        stop = Stop(
            instrument_id="AAPL",
            stop_type=StopType.TRAILING,
            entry_price=100.0,
            stop_price=95.0,
            quantity=100,
            created_at=pd.Timestamp("2023-01-01"),
            trailing=True,
            highest_price=100.0,
        )
        manager.add_stop(stop)

        # Price rises — trailing stop should update
        manager.check_stops("AAPL", 110.0, 111.0, 109.0, pd.Timestamp("2023-01-02"))
        active = manager.get_active_stops("AAPL")
        assert len(active) == 1
        assert active[0].highest_price == 110.0
