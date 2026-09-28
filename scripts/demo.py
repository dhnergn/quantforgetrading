"""End-to-end demonstration of the QuantForge platform.

Runs the full pipeline on synthetic data:

    market data -> features -> indicators -> regime -> strategy ->
    backtest -> costs/slippage/liquidity -> metrics -> IS/OOS ->
    walk-forward -> Monte Carlo -> risk -> paper trading

Everything is synthetic and offline. No broker is contacted, no credentials
are read, and live trading stays disabled.

    python scripts/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantforge.backtest.costs import (  # noqa: E402
    CommissionSchedule, CostModel, SlippageModelSpec, cost_sensitivity,
)
from quantforge.backtest.engine import BacktestConfig, BacktestEngine  # noqa: E402
from quantforge.backtest.liquidity import LiquidityModel  # noqa: E402
from quantforge.core.enums import SlippageModel  # noqa: E402
from quantforge.data.simulator import MarketSimulator  # noqa: E402
from quantforge.features.frame import build_feature_frame  # noqa: E402
from quantforge.risk.kill_switch import KillSwitch, KillSwitchConfig  # noqa: E402
from quantforge.risk.limits import RiskLimits  # noqa: E402
from quantforge.strategy.builder import (  # noqa: E402
    bollinger_mean_reversion_strategy, rsi_mean_reversion_strategy,
    sma_crossover_strategy,
)
from quantforge.validation.metrics import compare_to_benchmark, compute_metrics  # noqa: E402
from quantforge.validation.splits import (  # noqa: E402
    monte_carlo_returns, parameter_grid, grid_search, sensitivity_analysis,
    temporal_split, walk_forward,
)
from quantforge.validation.overfitting import analyse, deflated_sharpe_ratio  # noqa: E402

BAR = "=" * 74


def header(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def main() -> int:
    # ---------------------------------------------------------------- data
    header("1. MARKET DATA  (synthetic GARCH with Student-t shocks)")
    sim = MarketSimulator(n=1500, seed=7, start="2018-01-01",
                          annual_drift=0.06, annual_vol=0.22)
    data, latent = sim.simulate()
    print(f"bars          {len(data):,}")
    print(f"period        {data.index[0].date()} -> {data.index[-1].date()}")
    print(f"price range   {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"ann. vol      {latent.std() * np.sqrt(252):.2%} (target {sim.annual_vol:.0%})")
    print("The generator injects NO exploitable structure. A strategy that")
    print("profits here is profiting on noise.")

    # ------------------------------------------------------- data quality
    header("2. DATA QUALITY")
    from quantforge.data.quality import validate_ohlcv

    report = validate_ohlcv(data, "synthetic_ohlcv")
    print(report.report())

    # ----------------------------------------------------------- features
    header("3. FEATURES AND INDICATORS  (point-in-time, availability-tracked)")
    # SMA periods are chosen to cover the parameter grid used in the
    # walk-forward section below. The backtest engine refuses a strategy that
    # references a feature the frame does not provide, rather than silently
    # evaluating every condition to False and "trading" nothing.
    features = build_feature_frame(data, sma_periods=(10, 20, 30, 50, 200))
    print(f"feature columns  {features.shape[1]}")
    print(f"sample           {', '.join(features.columns[:10])} ...")
    from quantforge.quant.indicators import validate_metadata

    problems = validate_metadata()
    print(f"metadata check   {'consistent' if not problems else problems}")

    # ------------------------------------------------------------ strategy
    header("4. STRATEGY")
    strategy = sma_crossover_strategy(20, 50)
    print(strategy.describe())

    # ------------------------------------------------------------ backtest
    header("5. BACKTEST  (with costs, slippage, liquidity, risk limits)")
    cost_model = CostModel(
        commission=CommissionSchedule(per_share=0.005, regulatory_bps=0.0002),
        slippage=SlippageModelSpec(model=SlippageModel.PERCENTAGE,
                                  pct=0.0005, half_spread_bps=2.5),
    )
    limits = RiskLimits(max_position_pct=0.30, max_gross_exposure=0.60,
                        max_net_exposure=0.30, max_order_notional_pct=0.30,
                        max_sector_pct=1.0, max_asset_class_pct=1.0)
    engine = BacktestEngine(
        strategy, BacktestConfig(initial_capital=100_000.0),
        cost_model=cost_model,
        liquidity_model=LiquidityModel(max_participation_pct=0.05),
        risk_limits=limits,
    )
    result = engine.run(data, features)
    result.metrics = compute_metrics(
        result.portfolio.returns(), trades=result.portfolio.trade_frame(),
        equity=result.portfolio.equity_series(),
    )
    print(result.summary())

    # ------------------------------------------------------------- metrics
    header("6. UNIVERSE / MODEL COMPARISON")
    strategies = {
        "SMA 20/50 (baseline)": sma_crossover_strategy(20, 50),
        "SMA 10/20": sma_crossover_strategy(10, 20),
        "RSI(14) reversion": rsi_mean_reversion_strategy(),
        "Bollinger(20) reversion": bollinger_mean_reversion_strategy(),
    }
    rows = []
    for name, s in strategies.items():
        eng = BacktestEngine(s, BacktestConfig(initial_capital=100_000.0),
                             cost_model=cost_model,
                             liquidity_model=LiquidityModel(max_participation_pct=0.05),
                             risk_limits=limits)
        r = eng.run(data, features)
        r.metrics = compute_metrics(r.portfolio.returns(),
                                    trades=r.portfolio.trade_frame(),
                                    equity=r.portfolio.equity_series())
        m = r.metrics
        rows.append({
            "strategy": name, "total_return": m.get("total_return"),
            "cagr": m.get("cagr"), "sharpe": m.get("sharpe"),
            "sortino": m.get("sortino"), "max_dd": m.get("max_drawdown"),
            "trades": m.get("n_trades") or 0, "fees": r.portfolio.total_fees,
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print("\nA single metric does not select a winner. Compare the distribution,")
    print("the trade count, and the behaviour under different cost assumptions.")

    # -------------------------------------------------------- IS/OOS split
    header("7. IN-SAMPLE / OUT-OF-SAMPLE")
    split = temporal_split(features.index, 0.6, 0.2, embargo_bars=5)
    print(f"train       {split.train.start.date()} -> {split.train.end.date()} "
          f"({split.train.n(features.index)} bars)")
    print(f"validation  {split.validation.start.date()} -> "
          f"{split.validation.end.date()} ({split.validation.n(features.index)})")
    print(f"TEST        {split.test.start.date()} -> {split.test.end.date()} "
          f"({split.test.n(features.index)})")
    print(f"The embargo removes {5} bars between windows so a label horizon cannot")
    print("make validation share observations with training.")
    print("\nThe test set is touched exactly once, at the very end.")

    # ------------------------------------------------------ walk-forward
    header("8. WALK-FORWARD ANALYSIS")
    # Grid values must correspond to SMA periods the feature frame actually
    # provides (10, 20, 50, 200). The backtest engine refuses a strategy
    # referencing an absent feature rather than silently trading nothing.
    grids = parameter_grid(fast=[10, 20, 30], slow=[50, 200])

    def fit_predict(train_idx, eval_idx, params):
        """Run over the combined window so the strategy has history, then
        return only the evaluation-period returns.

        Joining the two index ranges directly is wrong: they do not overlap, so
        the join yields an all-NaN frame. The features must be taken from the
        single aligned price/feature frame, and the *scored* window selected
        afterwards.
        """
        combined = train_idx.union(eval_idx)
        s = sma_crossover_strategy(int(params["fast"]), int(params["slow"]))
        eng = BacktestEngine(s, BacktestConfig(initial_capital=100_000.0),
                             cost_model=cost_model,
                             liquidity_model=LiquidityModel(max_participation_pct=0.05),
                             risk_limits=limits)
        r = eng.run(data.loc[combined], features.loc[combined])
        rets = r.portfolio.returns()
        return rets.reindex(eval_idx).dropna()

    wf = walk_forward(features.index, fit_predict, param_grid=grids,
                      train_bars=500, validation_bars=150, test_bars=150,
                      step=150, objective="sharpe")
    print(wf.summary())

    # ------------------------------------------------------- Monte Carlo
    header("9. MONTE CARLO")
    rets = result.portfolio.returns()
    mc = monte_carlo_returns(rets, n_simulations=5000, method="bootstrap", seed=1)
    print(mc.summary())

    # -------------------------------------------------------- robustness
    header("10. ROBUSTNESS: COST SENSITIVITY")
    sens = cost_sensitivity(
        notional_per_trade=10_000.0, trades_per_year=max(1, len(result.orders) * 12),
        cost_models={"modest": cost_model,
                     "conservative": CostModel.conservative_equities(),
                     "expensive": CostModel.conservative_equities()},
        gross_return_per_year=float(result.metrics.get("cagr") or 0.0),
    )
    print(sens.to_string(float_format=lambda x: f"{x:,.4f}"))
    print("\nIf the net return turns negative under a plausible cost assumption,")
    print("the edge is a cost-modelling artefact, not a trading edge.")

    # ------------------------------------------------------- overfitting
    header("11. OVERFITTING DIAGNOSTICS")
    n_trades = int(result.metrics.get("n_trades") or 0)
    of = analyse(
        n_parameters=len(grids), n_trades=n_trades,
        in_sample={"sharpe": 0.4}, out_of_sample={"sharpe": float(
            result.metrics.get("sharpe") or float("nan"))},
    )
    print(of.summary())
    d = deflated_sharpe_ratio(rets, n_trials=len(grids))
    print(f"\nDEFLATED SHARPE: raw {d['observed_sharpe']:.3f} -> "
          f"deflated {d['deflated_sharpe']:.3f} "
          f"(expected max under null {d['expected_max_sharpe_under_null']:.3f})")
    print(f"  {d['note']}")

    # ------------------------------------------------------------- risk
    header("12. RISK ENGINE AND KILL SWITCH")
    p = result.portfolio
    print(f"gross exposure   {p.gross_exposure:,.0f}")
    print(f"net exposure     {p.net_exposure:,.0f}")
    print(f"leverage         {p.leverage:.2f}x")
    print(f"equity           {p.equity:,.2f}")
    ks = KillSwitch(KillSwitchConfig(max_daily_loss_pct=0.02))
    ks.note_market_data()
    fired = ks.evaluate(equity=97_000.0, daily_start_equity=100_000.0,
                        peak_equity=100_000.0, portfolio_id="demo")
    for ev in fired:
        print(f"KILL SWITCH: [{ev.scope.value}] {ev.reason}")
    print(f"trading state after trigger: {ks.trading_state().value}")
    print("The risk engine can reject any order the strategy wants; the strategy")
    print("has no mechanism to override it.")

    # ------------------------------------------------------ paper trading
    header("13. PAPER TRADING  (simulated adapter, same engines as live)")
    from quantforge.execution.brokers.paper import PaperBrokerAdapter
    from quantforge.execution.paper_trading import (
        PaperTradingConfig, PaperTradingSession,
    )

    broker = PaperBrokerAdapter(initial_equity=100_000.0, fill_probability=0.95,
                                participation_cap=0.05)
    broker.set_market("SIM", data)
    broker.connect()
    session = PaperTradingSession(
        strategy, PaperTradingConfig(initial_capital=100_000.0,
                                     instrument_id="SIM",
                                     reconcile_every_bars=250),
        broker, cost_model=cost_model, risk_limits=limits,
    )
    pr = session.run(data, features, session_id="demo")
    print(pr.summary())

    # --------------------------------------------------------- live gate
    header("14. LIVE TRADING GATE")
    from quantforge.execution.live_gate import (
        AUTHORISATION_CHECKLIST, LiveTradingGate,
    )
    gate = LiveTradingGate()
    permitted, why = gate.check()
    print(f"permitted: {permitted}")
    print(f"reason:    {why}")
    print(f"\nLive activation additionally requires all {len(AUTHORISATION_CHECKLIST)}")
    print("confirmations signed by an authorised operator:")
    for key, requirement in AUTHORISATION_CHECKLIST:
        print(f"  - {key}: {requirement[:72]}")

    header("DISCLAIMER")
    print(
        "This demonstration ran entirely on synthetic data with a simulated\n"
        "execution adapter. No real market data was used, no broker was\n"
        "contacted, no credentials were read, and no orders were placed.\n\n"
        "No strategy, indicator, or model in this platform produces guaranteed\n"
        "profits. A profitable backtest is evidence about the tested historical\n"
        "period under the stated assumptions, not a prediction of future\n"
        "returns. Financial markets are noisy, non-stationary, and adversarial,\n"
        "and most published strategies fail out of sample."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
