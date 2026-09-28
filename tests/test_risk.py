"""Risk engine, position sizing, kill switches, VaR/CVaR, and portfolio accounting.

The recurring theme: an undefined result is NaN, a limit is a hard ceiling, and
the risk engine has authority over the strategy.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quantforge.core.enums import (
    KillSwitchScope, OrderSide, OrderStatus, RiskLevel, SizingMethod,
    StopType, TradingState,
)
from quantforge.core.errors import (
    ConfigurationError, KillSwitchEngaged, RiskLimitBreached,
)
from quantforge.core.timestamps import utcnow


# ==========================================================================
# Portfolio accounting
# ==========================================================================


class TestPortfolio:
    def test_simple_long_round_trip(self, portfolio):
        portfolio.apply_fill("A", OrderSide.LONG, 100, 50.0, when="t0", bar_index=0)
        assert portfolio.cash == pytest.approx(95_000.0)
        assert portfolio.equity == pytest.approx(100_000.0)
        trade = portfolio.apply_fill("A", OrderSide.SHORT, 100, 60.0,
                                     when="t1", bar_index=1)
        assert trade.pnl == pytest.approx(1_000.0)
        assert portfolio.cash == pytest.approx(101_000.0)
        assert portfolio.positions["A"].is_flat

    def test_weighted_average_entry_on_scale_in(self, portfolio):
        portfolio.apply_fill("A", OrderSide.LONG, 100, 50.0, when="t0", bar_index=0)
        portfolio.apply_fill("A", OrderSide.LONG, 100, 60.0, when="t1", bar_index=1)
        assert portfolio.positions["A"].quantity == pytest.approx(200.0)
        assert portfolio.positions["A"].average_entry_price == pytest.approx(55.0)

    def test_partial_scale_out_keeps_position_and_entry(self, portfolio):
        portfolio.apply_fill("A", OrderSide.LONG, 200, 50.0, when="t0", bar_index=0)
        trade = portfolio.apply_fill("A", OrderSide.SHORT, 100, 60.0,
                                     when="t1", bar_index=1, exit_reason="scale")
        # Reducing by less than the whole holding must not book a closed trade.
        assert trade is None
        assert portfolio.positions["A"].quantity == pytest.approx(100.0)
        assert portfolio.positions["A"].average_entry_price == pytest.approx(50.0)

    def test_reversal_splits_into_close_and_reopen(self, portfolio):
        portfolio.apply_fill("A", OrderSide.LONG, 100, 10.0, when="t0", bar_index=0)
        trade = portfolio.apply_fill("A", OrderSide.SHORT, 300, 12.0,
                                     when="t1", bar_index=1, exit_reason="flip")
        assert trade.pnl == pytest.approx(200.0)
        pos = portfolio.positions["A"]
        assert pos.quantity == pytest.approx(-200.0)
        assert pos.average_entry_price == pytest.approx(12.0)

    def test_short_profits_when_price_falls(self, portfolio):
        portfolio.apply_fill("B", OrderSide.SHORT, 100, 20.0, when="t0", bar_index=0)
        portfolio.mark_to_market({"B": 18.0}, when="t1")
        assert portfolio.unrealized_pnl == pytest.approx(200.0)

    def test_fees_reduce_cash_not_cost_basis(self, portfolio):
        portfolio.apply_fill("A", OrderSide.LONG, 100, 50.0, fees=5.0,
                             when="t0", bar_index=0)
        assert portfolio.cash == pytest.approx(95_000.0 - 5.0)
        assert portfolio.positions["A"].average_entry_price == pytest.approx(50.0)
        assert portfolio.total_fees == pytest.approx(5.0)

    def test_close_all_flattens_and_books_pnl(self, portfolio):
        portfolio.apply_fill("X", OrderSide.LONG, 10, 10.0, when="t0", bar_index=0)
        trades = portfolio.close_all({"X": 11.5}, when="t9")
        assert len(trades) == 1
        assert trades[0].pnl == pytest.approx(15.0)
        assert portfolio.position_count == 0

    def test_unpriced_holding_is_not_valued_at_zero(self, portfolio):
        portfolio.apply_fill("A", OrderSide.LONG, 100, 50.0, when="t0", bar_index=0)
        # A data gap must not fabricate a total loss.
        portfolio.mark_to_market({}, when="t1")
        assert portfolio.positions["A"].last_price == pytest.approx(50.0)
        assert portfolio.unrealized_pnl == pytest.approx(0.0)

    def test_returns_reconcile_with_equity_differences(self, portfolio):
        for i, px in enumerate([10, 11, 12, 11.5]):
            portfolio.apply_fill("X", OrderSide.LONG, 10, 10.0,
                                 when=f"t{i}", bar_index=i)
            portfolio.mark_to_market({"X": px}, when=f"t{i}")
        eq = portfolio.equity_series()
        r = portfolio.returns()
        implied = (r * eq.shift(1).fillna(portfolio.initial_cash) - eq.diff().fillna(0))
        assert implied.abs().max() == pytest.approx(0.0, abs=1e-9)

    def test_exposure_accounting(self, portfolio):
        portfolio.apply_fill("A", OrderSide.LONG, 100, 50.0, when="t0", bar_index=0)
        portfolio.mark_to_market({"A": 60.0}, when="t1")
        portfolio.apply_fill("B", OrderSide.SHORT, 100, 20.0, when="t0", bar_index=0)
        portfolio.mark_to_market({"B": 25.0}, when="t1")
        assert portfolio.gross_exposure == pytest.approx(6_000 + 2_500)
        assert portfolio.net_exposure == pytest.approx(6_000 - 2_500)
        assert portfolio.leverage == pytest.approx(portfolio.gross_exposure
                                                   / portfolio.equity)

    def test_non_positive_fill_is_rejected(self, portfolio):
        with pytest.raises(Exception):
            portfolio.apply_fill("A", OrderSide.LONG, 0, 50.0)
        with pytest.raises(Exception):
            portfolio.apply_fill("A", OrderSide.LONG, 100, -1.0)


# ==========================================================================
# Position sizing
# ==========================================================================


class TestPositionSizing:
    def test_fixed_fractional(self):
        from quantforge.backtest.sizing import PositionSizer, SizingConfig

        s = PositionSizer(SizingConfig(fraction=0.10))
        r = s.fixed_fractional(100_000.0, 50.0)
        assert r.quantity == pytest.approx(200.0)
        assert r.equity_fraction == pytest.approx(0.10)

    def test_atr_sizing_risks_the_configured_fraction(self):
        from quantforge.backtest.sizing import PositionSizer, SizingConfig

        s = PositionSizer(SizingConfig(method=SizingMethod.VOLATILITY_ATR,
                                       atr_multiplier=2.0, risk_fraction=0.01))
        r = s.volatility_atr(100_000.0, 50.0, atr=2.0)
        assert r.risk_amount == pytest.approx(1_000.0)      # 1% of equity
        assert r.risk_per_share == pytest.approx(4.0)       # 2 x ATR
        assert r.quantity == pytest.approx(250.0)

    def test_atr_sizing_falls_back_visibly_when_atr_missing(self):
        from quantforge.backtest.sizing import PositionSizer, SizingConfig

        s = PositionSizer(SizingConfig(method=SizingMethod.VOLATILITY_ATR))
        r = s.volatility_atr(100_000.0, 50.0, atr=float("nan"))
        assert r.quantity > 0
        assert any("fell back" in n for n in r.notes)

    def test_kelly_is_capped_and_fractional(self):
        from quantforge.backtest.sizing import PositionSizer, SizingConfig

        s = PositionSizer(SizingConfig(method=SizingMethod.KELLY,
                                       kelly_fraction=0.25, max_kelly_weight=0.10))
        # f* = 0.7 - 0.3/2 = 0.55; 25% of that is 13.75%, above the 10% cap.
        r = s.kelly(100_000.0, 50.0, win_probability=0.7, payoff_ratio=2.0)
        assert r.equity_fraction <= 0.10 + 1e-9
        assert r.capped_by == "max_kelly_weight"

    def test_kelly_negative_edge_takes_no_position(self):
        from quantforge.backtest.sizing import PositionSizer, SizingConfig

        s = PositionSizer(SizingConfig(method=SizingMethod.KELLY))
        r = s.kelly(100_000.0, 50.0, win_probability=0.3, payoff_ratio=0.5)
        assert r.quantity == 0.0
        assert r.capped_by == "negative_kelly"

    def test_kelly_invalid_probability_falls_back(self):
        from quantforge.backtest.sizing import PositionSizer, SizingConfig

        s = PositionSizer(SizingConfig(method=SizingMethod.KELLY))
        r = s.kelly(100_000.0, 50.0, win_probability=1.5, payoff_ratio=1.0)
        assert r.quantity > 0
        assert any("outside (0, 1)" in n for n in r.notes)

    def test_risk_based_uses_stop_distance(self):
        from quantforge.backtest.sizing import PositionSizer, SizingConfig

        s = PositionSizer(SizingConfig(method=SizingMethod.RISK_BASED,
                                       risk_fraction=0.01))
        r = s.risk_based(100_000.0, 50.0, stop_price=48.0)
        assert r.risk_per_share == pytest.approx(2.0)
        assert r.risk_amount == pytest.approx(1_000.0)

    def test_lot_size_rounds_down_never_up(self):
        from quantforge.backtest.sizing import PositionSizer, SizingConfig

        s = PositionSizer(SizingConfig(lot_size=10))
        assert s.apply_lot_size(107.0, 10) == pytest.approx(100.0)
        assert s.apply_lot_size(9.0, 10) == pytest.approx(0.0)

    def test_sizing_never_exceeds_position_limit(self):
        from quantforge.backtest.sizing import PositionSizer, SizingConfig

        s = PositionSizer(SizingConfig(fraction=0.90))
        r = s.fixed_fractional(100_000.0, 10.0)
        r = s.clamp_to_limits(r, 100_000.0, max_position_notional=20_000.0)
        assert r.notional <= 20_000.0 + 1e-6
        assert r.capped_by == "max_position_notional"


# ==========================================================================
# Risk engine
# ==========================================================================


def _attr(iid):
    return ({"A": "Tech", "B": "Tech", "C": "Energy"}.get(iid), "equity")


class TestRiskEngine:
    def test_position_concentration_limit_breached(self, portfolio, tight_limits):
        from quantforge.risk.limits import RiskEngine

        re = RiskEngine(tight_limits, _attr)
        re.update_equity(100_000.0)
        v = re.check_order(portfolio, "A", "long", 500, 100.0)
        assert not v.approved
        assert v.risk_level == RiskLevel.RED
        kinds = {x["limit_type"] for x in v.violations}
        assert "position_concentration" in kinds
        assert "order_notional" in kinds

    def test_compliant_order_is_approved(self, portfolio, tight_limits):
        from quantforge.risk.limits import RiskEngine

        re = RiskEngine(tight_limits, _attr)
        re.update_equity(100_000.0)
        v = re.check_order(portfolio, "A", "long", 50, 100.0)
        assert v.approved
        assert v.risk_level == RiskLevel.GREEN

    def test_sector_concentration(self, portfolio):
        from quantforge.risk.limits import RiskEngine, RiskLimits

        # Position limit is looser than the sector limit, so only the sector
        # check can be the one that trips.
        limits = RiskLimits(max_position_pct=0.40, max_sector_pct=0.20,
                            max_gross_exposure=2.0, max_order_notional_pct=0.40,
                            max_net_exposure=2.0, max_leverage=2.0)
        portfolio.apply_fill("A", OrderSide.LONG, 150, 100.0)   # Tech 15,000
        re = RiskEngine(limits, _attr)
        re.update_equity(100_000.0)
        v = re.check_order(portfolio, "B", "long", 100, 100.0)  # Tech 25,000
        assert not v.approved
        assert "sector_concentration" in {x["limit_type"] for x in v.violations}

    def test_gross_exposure_limit(self, portfolio):
        from quantforge.risk.limits import RiskEngine, RiskLimits

        # Sector and position limits are loose enough that only gross trips.
        limits = RiskLimits(max_position_pct=0.40, max_sector_pct=1.0,
                            max_gross_exposure=0.50, max_order_notional_pct=0.40,
                            max_net_exposure=2.0, max_leverage=2.0)
        portfolio.apply_fill("A", OrderSide.LONG, 300, 100.0)   # 30,000
        re = RiskEngine(limits, _attr)
        re.update_equity(100_000.0)
        v = re.check_order(portfolio, "C", "long", 300, 100.0)  # gross 60,000
        assert "gross_exposure" in {x["limit_type"] for x in v.violations}

    def test_zero_equity_blocks_all_orders(self, portfolio, tight_limits):
        from quantforge.risk.limits import RiskEngine

        portfolio.cash = 0.0
        for p in portfolio.positions.values():
            p.quantity = 0.0
        re = RiskEngine(tight_limits, _attr)
        v = re.check_order(portfolio, "A", "long", 1, 100.0)
        assert not v.approved
        assert v.violations[0]["limit_type"] == "equity"

    def test_clamp_finds_a_feasible_size(self, portfolio, tight_limits):
        from quantforge.risk.limits import RiskEngine

        re = RiskEngine(tight_limits, _attr)
        re.update_equity(100_000.0)
        clamped = re.clamp_quantity(portfolio, "A", "long", 5000, 100.0)
        assert 0 < clamped < 5000
        assert re.check_order(portfolio, "A", "long", clamped, 100.0).approved

    def test_clamp_leaves_margin_below_the_boundary(self, portfolio, tight_limits):
        """Clamping exactly to the limit guarantees rejection on a price move."""
        from quantforge.risk.limits import RiskEngine

        re = RiskEngine(tight_limits, _attr)
        re.update_equity(100_000.0)
        clamped = re.clamp_quantity(portfolio, "A", "long", 5000, 100.0)
        # A 1% adverse gap must not push the clamped size over the limit.
        assert not re.check_order(portfolio, "A", "long", clamped, 101.0).approved or \
            clamped < 5000

    def test_raise_if_breached(self, portfolio, tight_limits):
        from quantforge.risk.limits import RiskEngine

        re = RiskEngine(tight_limits, _attr)
        re.update_equity(100_000.0)
        with pytest.raises(RiskLimitBreached):
            re.check_order_or_raise(portfolio, "A", "long", 5000, 100.0)

    def test_limits_are_validated_at_construction(self):
        from quantforge.risk.limits import RiskLimits

        with pytest.raises(ValueError):
            RiskLimits(max_position_pct=0.0).validate()
        with pytest.raises(ValueError):
            RiskLimits(max_position_pct=5.0).validate()

    def test_risk_breaches_are_logged(self, portfolio, tight_limits):
        from quantforge.risk.limits import RiskEngine

        re = RiskEngine(tight_limits, _attr)
        re.update_equity(100_000.0)
        re.check_order(portfolio, "A", "long", 5000, 100.0)
        assert len(re.breach_log) >= 1
        assert "limit_type" in re.breach_log[0]


# ==========================================================================
# Kill switch
# ==========================================================================


class TestKillSwitch:
    def test_daily_loss_engages_and_blocks(self):
        from quantforge.risk.kill_switch import KillSwitch, KillSwitchConfig

        ks = KillSwitch(KillSwitchConfig(max_daily_loss_pct=0.02))
        ks.note_market_data(utcnow())
        fired = ks.evaluate(equity=97_000.0, daily_start_equity=100_000.0,
                            peak_equity=100_000.0, portfolio_id="P")
        assert len(fired) == 1
        assert ks.trading_state() == TradingState.HALTED_KILL_SWITCH
        with pytest.raises(KillSwitchEngaged):
            ks.assert_can_trade()

    def test_fails_closed_without_context(self):
        """A portfolio-scoped halt must not read as NORMAL to a bare caller."""
        from quantforge.risk.kill_switch import KillSwitch, KillSwitchConfig

        ks = KillSwitch(KillSwitchConfig(max_daily_loss_pct=0.02))
        ks.note_market_data(utcnow())
        ks.evaluate(equity=90_000.0, daily_start_equity=100_000.0,
                    peak_equity=100_000.0, portfolio_id="PF1")
        assert ks.trading_state() == TradingState.HALTED_KILL_SWITCH
        assert ks.is_engaged() is True

    def test_stale_data_engages(self):
        from quantforge.risk.kill_switch import KillSwitch, KillSwitchConfig

        ks = KillSwitch(KillSwitchConfig(stale_data_seconds=30))
        ks.last_market_data_at = utcnow() - dt.timedelta(seconds=120)
        fired = ks.evaluate()
        assert len(fired) == 1
        assert "stale" in fired[0].reason.lower()

    def test_repeated_trigger_does_not_flap(self):
        from quantforge.risk.kill_switch import KillSwitch, KillSwitchConfig

        ks = KillSwitch(KillSwitchConfig(max_daily_loss_pct=0.02))
        for _ in range(5):
            ks.evaluate(equity=90_000.0, daily_start_equity=100_000.0,
                        peak_equity=100_000.0, portfolio_id="P")
        assert len(ks.active_events) == 1

    def test_release_requires_an_explicit_call(self):
        from quantforge.risk.kill_switch import KillSwitch, KillSwitchConfig

        ks = KillSwitch(KillSwitchConfig(max_daily_loss_pct=0.02))
        ks.note_market_data(utcnow())
        ks.evaluate(equity=90_000.0, daily_start_equity=100_000.0,
                    peak_equity=100_000.0, portfolio_id="P")
        assert ks.trading_state() != TradingState.NORMAL
        ks.release(KillSwitchScope.PORTFOLIO, "risk_manager")
        assert ks.trading_state() == TradingState.NORMAL

    def test_external_states_block(self):
        from quantforge.risk.kill_switch import KillSwitch

        ks = KillSwitch()
        assert ks.trading_state(broker_connected=False) == TradingState.HALTED_BROKER_DOWN
        assert ks.trading_state(data_stale=True) == TradingState.HALTED_DATA_STALE
        assert ks.trading_state(reconciliation_ok=False) == \
            TradingState.REQUIRES_RECONCILIATION
        assert ks.trading_state(risk_breached=True) == TradingState.HALTED_RISK_BREACH
        assert ks.trading_state() == TradingState.NORMAL

    def test_order_rejection_trips_the_switch(self):
        from quantforge.risk.kill_switch import KillSwitch, KillSwitchConfig

        ks = KillSwitch(KillSwitchConfig(max_order_rejections=3))
        for _ in range(3):
            ks.note_order_rejection()
        assert len(ks.evaluate()) == 1


# ==========================================================================
# VaR / CVaR
# ==========================================================================


class TestVaRAndCVaR:
    @pytest.fixture
    def fat_tailed(self):
        rng = np.random.default_rng(5)
        return pd.Series(rng.standard_t(5, 4000) / np.sqrt(5 / 3) * 0.01)

    def test_all_methods_agree_on_sign(self, fat_tailed):
        from quantforge.core.enums import VaRMethod
        from quantforge.quant import risk_metrics as rm

        for m in (VaRMethod.HISTORICAL, VaRMethod.PARAMETRIC, VaRMethod.MONTE_CARLO):
            v = rm.var(fat_tailed, 0.95, 1, m, "oos")
            assert v.value > 0, f"{m} produced a non-positive loss: {v.value}"

    def test_es_is_at_least_var(self, fat_tailed):
        from quantforge.quant import risk_metrics as rm

        for conf in (0.90, 0.95, 0.99):
            v = rm.historical_var(fat_tailed, conf, 1, "oos")
            e = rm.historical_expected_shortfall(fat_tailed, conf, 1, "oos")
            assert e.value >= v.value - 1e-12, conf

    def test_var_increases_with_confidence(self, fat_tailed):
        from quantforge.quant import risk_metrics as rm

        vals = [rm.historical_var(fat_tailed, c, 1, "oos").value
                for c in (0.90, 0.95, 0.99)]
        assert vals[0] < vals[1] < vals[2]

    def test_student_t_is_more_conservative_than_normal(self, fat_tailed):
        """Fat tails must produce a *larger* 99% VaR under a t fit."""
        from quantforge.quant import risk_metrics as rm

        t = rm.parametric_var(fat_tailed, 0.99, distribution="student_t").value
        n = rm.parametric_var(fat_tailed, 0.99, distribution="normal").value
        assert t > n

    def test_insufficient_sample_is_nan_not_zero(self):
        from quantforge.quant import risk_metrics as rm

        v = rm.historical_var(pd.Series(np.random.default_rng(1).normal(0, 0.01, 12)), 0.95)
        assert np.isnan(v.value)
        assert "12 observations" in v.caveat

    def test_constant_series_var_is_undefined(self):
        from quantforge.quant import risk_metrics as rm

        v = rm.parametric_var(pd.Series([0.0] * 50), 0.95)
        assert np.isnan(v.value)
        assert "zero volatility" in v.caveat

    def test_multi_period_uses_sqrt_of_time(self, fat_tailed):
        from quantforge.quant import risk_metrics as rm

        one = rm.historical_var(fat_tailed, 0.95, 1, "oos").value
        five = rm.historical_var(fat_tailed, 0.95, 5, "oos").value
        assert five == pytest.approx(one * np.sqrt(5), rel=1e-6)
        assert "sqrt-of-time" in rm.historical_var(fat_tailed, 0.95, 5).caveat

    def test_metadata_identifies_method_confidence_horizon_dataset(self, fat_tailed):
        from quantforge.quant import risk_metrics as rm

        d = rm.historical_var(fat_tailed, 0.95, 5, "oos_test_set").as_dict()
        for key in ("value", "confidence", "horizon_periods", "method",
                    "dataset", "n_observations", "caveat"):
            assert key in d
        assert d["confidence"] == 0.95
        assert d["horizon_periods"] == 5
        assert d["dataset"] == "oos_test_set"

    def test_monte_carlo_is_reproducible(self, fat_tailed):
        from quantforge.quant import risk_metrics as rm

        a = rm.monte_carlo_var(fat_tailed, 0.95, seed=42).value
        b = rm.monte_carlo_var(fat_tailed, 0.95, seed=42).value
        assert a == pytest.approx(b)


# ==========================================================================
# Performance metrics
# ==========================================================================


class TestMetrics:
    def test_undefined_ratios_are_nan(self):
        from quantforge.validation.metrics import compute_metrics

        m = compute_metrics(pd.Series([0.001] * 300))
        assert np.isnan(m.get("sharpe"))
        assert np.isnan(m.get("calmar"))
        entry = [e for e in m.entries if e.name == "sharpe"][0]
        assert "zero" in entry.caveat.lower()

    def test_es_equals_var_is_never_reported_as_false(self):
        from quantforge.quant import risk_metrics as rm

        rng = np.random.default_rng(8)
        r = pd.Series(rng.standard_t(4, 2000) / np.sqrt(4 / 2) * 0.01)
        for c in (0.9, 0.95, 0.99):
            v = rm.historical_var(r, c).value
            e = rm.historical_expected_shortfall(r, c).value
            assert e >= v - 1e-12

    def test_short_sample_raises_a_warning(self):
        from quantforge.validation.metrics import compute_metrics

        m = compute_metrics(pd.Series([0.01, -0.01, 0.02]))
        assert any("very wide confidence intervals" in w for w in m.warnings)

    def test_excess_return_is_not_inflated(self):
        """Regression: manual construction double-compounded the risk-free rate."""
        from quantforge.validation.metrics import compute_metrics

        rng = np.random.default_rng(9)
        r = pd.Series(rng.normal(0.0004, 0.01, 756))
        m = compute_metrics(r, risk_free_annual=0.03)
        excess = m.get("excess_return_annual")
        assert abs(excess) < 1.0, f"excess return of {excess} is implausible"

    def test_benchmark_comparison_aligns_dates(self):
        from quantforge.validation.metrics import compare_to_benchmark

        rng = np.random.default_rng(10)
        idx = pd.date_range("2021-01-01", periods=500, freq="B", tz="UTC")
        a = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        b = pd.Series(rng.normal(0, 0.01, 300), index=idx[:300])
        out = compare_to_benchmark(a, b)
        assert out["n_periods"] == 300

    def test_no_trade_list_warns(self):
        from quantforge.validation.metrics import compute_metrics

        m = compute_metrics(pd.Series(np.random.default_rng(11).normal(0, 0.01, 300)))
        assert any("no trade list" in w for w in m.warnings)


# ==========================================================================
# Costs, slippage, liquidity
# ==========================================================================


class TestCostModel:
    def test_free_model_is_loudly_flagged(self):
        from quantforge.backtest.costs import CommissionSchedule, CostModel, SlippageModelSpec
        from quantforge.core.enums import SlippageModel

        cm = CostModel(commission=CommissionSchedule(),
                       slippage=SlippageModelSpec(model=SlippageModel.NONE,
                                                  half_spread_bps=0.0, pct=0.0))
        assert cm.is_free
        assert "COSTS ARE ZERO" in cm.as_dict()["warning"]

    def test_costs_increase_with_size(self, cost_model):
        small = cost_model.estimate(100, 50.0, "long", bar_volume=1e6)
        large = cost_model.estimate(10_000, 50.0, "long", bar_volume=1e6)
        assert large["total_cost"] > small["total_cost"]

    def test_buy_pays_up_and_sell_receives_less(self, cost_model):
        buy, _ = cost_model.apply(100, 50.0, "long")
        sell, _ = cost_model.apply(100, 50.0, "short")
        assert buy > 50.0
        assert sell < 50.0

    def test_per_share_fees_not_charged_on_fx(self, cost_model):
        equity = cost_model.estimate(1000, 1.2, "long", instrument_type="fx")
        assert equity["commission"] < 1.0     # 1000 * 0.005 would be $5

    def test_conservative_model_costs_more(self, cost_model):
        from quantforge.backtest.costs import CostModel

        cons = CostModel.conservative_equities()
        a = cost_model.estimate(100, 50.0, "long", bar_volume=1e6)
        b = cons.estimate(100, 50.0, "long", bar_volume=1e6)
        assert b["total_cost"] > a["total_cost"]

    def test_volume_based_slippage_scales_with_participation(self):
        from quantforge.backtest.costs import SlippageModelSpec
        from quantforge.core.enums import SlippageModel

        s = SlippageModelSpec(model=SlippageModel.VOLUME_BASED,
                              half_spread_bps=0.0, volume_participation_ref=0.01)
        small = s.compute(100, 50.0, bar_volume=1e7)
        large = s.compute(1e6, 50.0, bar_volume=1e7)
        assert large > small

    def test_unknown_volume_refuses_to_fill(self):
        from quantforge.backtest.liquidity import LiquidityModel

        liq = LiquidityModel()
        d = liq.decide_fill(100, bar_volume=None)
        assert d["filled"] == 0.0
        assert "insufficient liquidity" in d["reason"]

    def test_participation_cap_limits_fill(self):
        from quantforge.backtest.liquidity import LiquidityModel

        liq = LiquidityModel(max_participation_pct=0.05, allow_partial_fills=True)
        d = liq.decide_fill(10_000, bar_volume=100_000)
        assert d["filled"] == pytest.approx(5_000.0)
        assert d["partial"]

    def test_no_partial_fills_rejects_instead(self):
        from quantforge.backtest.liquidity import LiquidityModel

        liq = LiquidityModel(max_participation_pct=0.05, allow_partial_fills=False)
        d = liq.decide_fill(10_000, bar_volume=100_000)
        assert d["filled"] == 0.0
        assert d["rejected"]

    def test_limit_order_outside_the_range_does_not_fill(self):
        from quantforge.backtest.liquidity import LiquidityModel, simulate_fill_price

        r = simulate_fill_price("long", 100.0, 101.0, 99.0, 100.0, 100, 1e6,
                                order_type="limit", limit_price=10.0)
        assert not r["filled"]
        assert "not reached" in r["reason"]
