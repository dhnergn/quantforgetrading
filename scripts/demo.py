"""Demo script — demonstrates the complete QuantForge AI signal bot workflow.

This script shows:
1. Data generation
2. Data validation
3. Signal generation
4. Backtesting
5. Results summary

Live trading is DISABLED. This is for research and demonstration only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantforge.backtest.engine import BacktestConfig, BacktestEngine
from quantforge.data.validation import DataValidator
from quantforge.features.engine import FeatureEngine
from quantforge.signals.generator import SignalConfig, SignalGenerator
from quantforge.strategies.crossover import CrossoverConfig, CrossoverStrategy
from quantforge.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from quantforge.strategies.breakout import BreakoutConfig, BreakoutStrategy


def generate_data(symbol: str = "DEMO", periods: int = 500) -> pd.DataFrame:
    """Generate realistic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")

    returns = np.random.randn(periods) * 0.02 + 0.0003
    close = 100 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.randn(periods) * 0.01))
    low = close * (1 - np.abs(np.random.randn(periods) * 0.01))
    open_price = low + np.random.rand(periods) * (high - low)
    volume = np.random.randint(100_000, 1_000_000, periods)

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def run_demo():
    """Run the complete demo."""
    print("=" * 70)
    print("QuantForge AI — Signal Bot Demo")
    print("=" * 70)
    print()
    print("DISCLAIMER: No strategy, indicator, or model guarantees profits.")
    print("This is a research and demonstration tool only.")
    print("Live trading is DISABLED.")
    print()

    # Step 1: Generate data
    print("[1/5] Generating sample data...")
    data = generate_data("DEMO", 500)
    print(f"  Generated {len(data)} rows")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print()

    # Step 2: Validate
    print("[2/5] Validating data...")
    validator = DataValidator()
    report = validator.validate(data, "DEMO")
    print(f"  Total rows: {report.total_rows}")
    print(f"  Valid rows: {report.valid_rows}")
    print(f"  Issues: {len(report.issues)}")
    print(f"  Valid: {report.is_valid}")
    print()

    # Step 3: Generate signals with multiple strategies
    print("[3/5] Generating signals...")
    strategies = [
        ("Crossover", CrossoverStrategy(CrossoverConfig(name="crossover", volume_confirm=False))),
        ("Mean Reversion", MeanReversionStrategy(MeanReversionConfig(name="mean_reversion"))),
        ("Breakout", BreakoutStrategy(BreakoutConfig(name="breakout"))),
    ]

    for name, strategy in strategies:
        config = SignalConfig(
            strategy=strategy,
            feature_engine=FeatureEngine(),
            instruments=["DEMO"],
            min_confidence=0.5,
        )
        generator = SignalGenerator(config)
        signals = generator.generate_signals(data)
        print(f"  {name}: {len(signals)} signals generated")
    print()

    # Step 4: Backtest
    print("[4/5] Running backtests...")
    print()

    for name, strategy in strategies:
        config = BacktestConfig(
            strategy=strategy,
            data=data,
            instrument_ids=["DEMO"],
            start_date=data.index[50],
            end_date=data.index[-1],
            initial_capital=100_000,
        )
        engine = BacktestEngine(config)
        result = engine.run()

        print(f"  {name} Results:")
        print(f"    Total Return:     {result.total_return:>10.2%}")
        print(f"    CAGR:             {result.cagr:>10.2%}")
        print(f"    Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
        print(f"    Sortino Ratio:    {result.sortino_ratio:>10.2f}")
        print(f"    Max Drawdown:     {result.max_drawdown:>10.2%}")
        print(f"    Volatility:       {result.volatility:>10.2%}")
        print(f"    Total Trades:     {result.total_trades:>10d}")
        print(f"    Win Rate:         {result.win_rate:>10.2%}")
        print(f"    Profit Factor:    {result.profit_factor:>10.2f}")
        print(f"    VaR (95%):        {result.var_95:>10.2%}")
        print(f"    CVaR (95%):       {result.cvar_95:>10.2%}")
        print()

    # Step 5: Summary
    print("[5/5] Summary")
    print()
    print("  This demo showed:")
    "  - Data generation and validation"
    "  - Signal generation from multiple strategies"
    "  - Backtesting with transaction costs and slippage"
    "  - Risk metrics (Sharpe, Sortino, VaR, CVaR)"
    print()
    print("  Remember:")
    print("  - A profitable historical backtest is NOT proof of future profitability")
    print("  - Past performance does not guarantee future results")
    print("  - Live trading is DISABLED and requires explicit authorization")
    print()
    print("=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
