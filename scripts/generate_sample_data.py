"""Generate sample OHLCV data for testing and demonstration."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_ohlcv(
    symbol: str,
    start_date: str = "2023-01-01",
    periods: int = 500,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0002,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic OHLCV data.

    Args:
        symbol: Instrument symbol.
        start_date: Start date.
        periods: Number of periods.
        initial_price: Initial price.
        volatility: Daily volatility.
        trend: Daily trend (drift).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with OHLCV data.
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, periods=periods, freq="D")

    # Generate returns with trend and volatility
    returns = np.random.randn(periods) * volatility + trend
    close = initial_price * np.exp(np.cumsum(returns))

    # Generate realistic OHLC from close
    intraday_vol = volatility * 0.5
    high = close * (1 + np.abs(np.random.randn(periods) * intraday_vol))
    low = close * (1 - np.abs(np.random.randn(periods) * intraday_vol))
    open_price = low + np.random.rand(periods) * (high - low)

    # Ensure OHLC relationships
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Volume with some realistic patterns
    base_volume = 1_000_000
    volume = np.random.randint(base_volume // 2, base_volume * 2, periods)

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sample OHLCV data")
    parser.add_argument("--symbol", default="AAPL", help="Instrument symbol")
    parser.add_argument("--output", default="data/sample.csv", help="Output file path")
    parser.add_argument("--periods", type=int, default=500, help="Number of periods")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = generate_ohlcv(
        symbol=args.symbol,
        start_date=args.start_date,
        periods=args.periods,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Generated {len(df)} rows for {args.symbol}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
