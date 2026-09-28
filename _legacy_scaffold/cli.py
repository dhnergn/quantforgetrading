"""Command-line interface for QuantForge AI signal bot.

Live trading is DISABLED. This CLI provides signal generation,
backtesting, and research capabilities only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from quantforge.backtest.engine import BacktestConfig, BacktestEngine
from quantforge.data.ingestion import DataIngestion
from quantforge.data.validation import DataValidator
from quantforge.features.engine import FeatureEngine
from quantforge.signals.generator import SignalConfig, SignalGenerator
from quantforge.strategies.breakout import BreakoutConfig, BreakoutStrategy
from quantforge.strategies.crossover import CrossoverConfig, CrossoverStrategy
from quantforge.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy


def create_strategy(name: str):
    """Create a strategy by name."""
    if name == "crossover":
        return CrossoverStrategy(CrossoverConfig(name="crossover"))
    elif name == "mean_reversion":
        return MeanReversionStrategy(MeanReversionConfig(name="mean_reversion"))
    elif name == "breakout":
        return BreakoutStrategy(BreakoutConfig(name="breakout"))
    return None


def cmd_generate_signals(args) -> None:
    """Generate trading signals."""
    print(f"Loading data from {args.data}...")
    data = pd.read_csv(args.data, index_col=0, parse_dates=True)

    instruments = args.instruments or ["default"]
    strategy = create_strategy(args.strategy)
    if strategy is None:
        print(f"Unknown strategy: {args.strategy}")
        sys.exit(1)

    config = SignalConfig(
        strategy=strategy,
        feature_engine=FeatureEngine(),
        instruments=instruments,
        min_confidence=args.min_confidence,
    )
    generator = SignalGenerator(config)

    print(f"Generating signals with {args.strategy} strategy...")
    signals = generator.generate_signals(data)

    if not signals:
        print("No signals generated.")
        return

    print(f"\nGenerated {len(signals)} signals:\n")
    for sig in signals:
        print(f"  {sig.timestamp} | {sig.instrument_id:12s} | {sig.direction:12s} | confidence={sig.confidence:.2f}")

    if args.output:
        df = generator.to_dataframe()
        df.to_csv(args.output, index=False)
        print(f"\nSignals saved to {args.output}")


def cmd_backtest(args) -> None:
    """Run a backtest."""
    print(f"Loading data from {args.data}...")
    data = pd.read_csv(args.data, index_col=0, parse_dates=True)

    instruments = args.instruments or ["default"]
    strategy = create_strategy(args.strategy)
    if strategy is None:
        print(f"Unknown strategy: {args.strategy}")
        sys.exit(1)

    config = BacktestConfig(
        strategy=strategy,
        data=data,
        instrument_ids=instruments,
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(args.end_date, "%Y-%m-%d"),
        initial_capital=args.initial_capital,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
    )

    print(f"Running backtest: {args.strategy} from {args.start_date} to {args.end_date}...")
    engine = BacktestEngine(config)
    result = engine.run()

    print(f"\n{'='*60}")
    print(f"Backtest Results")
    print(f"{'='*60}")
    print(f"Run ID:           {result.run_id}")
    print(f"Config Hash:      {result.config_hash[:16]}...")
    print(f"Total Return:     {result.total_return:.2%}")
    print(f"CAGR:             {result.cagr:.2%}")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:    {result.sortino_ratio:.2f}")
    print(f"Calmar Ratio:     {result.calmar_ratio:.2f}")
    print(f"Max Drawdown:     {result.max_drawdown:.2%}")
    print(f"Volatility:       {result.volatility:.2%}")
    print(f"Total Trades:     {result.total_trades}")
    print(f"Win Rate:         {result.win_rate:.2%}")
    print(f"Profit Factor:    {result.profit_factor:.2f}")
    print(f"VaR (95%):        {result.var_95:.2%}")
    print(f"CVaR (95%):       {result.cvar_95:.2%}")
    print(f"{'='*60}")

    if args.output:
        output_data = {
            "run_id": result.run_id,
            "config_hash": result.config_hash,
            "total_return": result.total_return,
            "cagr": result.cagr,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "max_drawdown": result.max_drawdown,
            "volatility": result.volatility,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "var_95": result.var_95,
            "cvar_95": result.cvar_95,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


def cmd_validate(args) -> None:
    """Validate market data."""
    print(f"Loading data from {args.data}...")
    data = pd.read_csv(args.data, index_col=0, parse_dates=True)

    validator = DataValidator()
    report = validator.validate(data, args.instrument)

    print(f"\nValidation Report:")
    print(f"  Instrument: {report.instrument_id}")
    print(f"  Total Rows: {report.total_rows}")
    print(f"  Valid Rows: {report.valid_rows}")
    print(f"  Issues: {len(report.issues)}")
    print(f"  Valid: {report.is_valid}")

    if report.issues:
        print(f"\nIssues:")
        for issue in report.issues[:20]:
            severity = issue.get("severity", "error")
            print(f"  [{severity}] {issue['type']}: {issue['message']}")
        if len(report.issues) > 20:
            print(f"  ... and {len(report.issues) - 20} more issues")


def cmd_ingest(args) -> None:
    """Ingest market data from CSV."""
    print(f"Ingesting data from {args.data}...")
    ingestion = DataIngestion(source="csv")
    df = ingestion.ingest_csv(args.data, args.instrument)

    print(f"Ingested {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    if args.output:
        df.to_csv(args.output)
        print(f"Saved to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="quantforge",
        description="QuantForge AI — Research-grade quantitative trading signal platform. Live trading is DISABLED.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate signals
    sig_parser = subparsers.add_parser("signals", help="Generate trading signals")
    sig_parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    sig_parser.add_argument("--instruments", nargs="+", help="Instrument IDs")
    sig_parser.add_argument("--strategy", default="crossover", choices=["crossover", "mean_reversion", "breakout"])
    sig_parser.add_argument("--min-confidence", type=float, default=0.5)
    sig_parser.add_argument("--output", "-o", help="Output CSV file")

    # Backtest
    bt_parser = subparsers.add_parser("backtest", help="Run a backtest")
    bt_parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    bt_parser.add_argument("--instruments", nargs="+", help="Instrument IDs")
    bt_parser.add_argument("--strategy", default="crossover", choices=["crossover", "mean_reversion", "breakout"])
    bt_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    bt_parser.add_argument("--initial-capital", type=float, default=100_000.0)
    bt_parser.add_argument("--commission", type=float, default=0.001)
    bt_parser.add_argument("--slippage", type=float, default=0.001)
    bt_parser.add_argument("--output", "-o", help="Output JSON file")

    # Validate
    val_parser = subparsers.add_parser("validate", help="Validate market data")
    val_parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    val_parser.add_argument("--instrument", default="default", help="Instrument ID")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest market data")
    ingest_parser.add_argument("--data", required=True, help="Path to CSV file")
    ingest_parser.add_argument("--instrument", required=True, help="Instrument ID")
    ingest_parser.add_argument("--output", "-o", help="Output CSV file")

    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "signals":
        cmd_generate_signals(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
