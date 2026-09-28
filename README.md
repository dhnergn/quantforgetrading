# QuantForge AI

**Research-grade quantitative trading signal platform.**

> **DISCLAIMER:** No strategy, indicator, or ML model guarantees profits or predicts markets with certainty. A profitable historical backtest is evidence about the tested historical assumptions, not proof of future profitability. Live trading is **DISABLED** by default.

## Features

- **Signal Generation** — Generate trading signals from multiple strategies
- **Backtesting** — Event-driven historical simulation with transaction costs and slippage
- **Risk Management** — Position sizing, stops, portfolio constraints, VaR/CVaR
- **Data Validation** — OHLC relationship checks, outlier detection, quality reports
- **Paper Trading** — Simulated execution with realistic fills
- **Broker Abstraction** — Pluggable broker adapters (live trading disabled)

## Quick Start

### Installation

```bash
cd quantforge
pip install -r requirements.txt
```

### Generate Signals

```bash
python -m quantforge signals --data data/AAPL.csv --strategy crossover --instruments AAPL
```

### Run Backtest

```bash
python -m quantforge backtest --data data/AAPL.csv --strategy crossover --instruments AAPL --start-date 2023-01-01 --end-date 2024-01-01
```

### Validate Data

```bash
python -m quantforge validate --data data/AAPL.csv --instrument AAPL
```

### Start API Server

```bash
python -m quantforge api
# or
uvicorn quantforge.api.main:create_app --factory --host 0.0.0.0 --port 8000
```

## Strategies

| Strategy | Type | Description |
|---|---|---|
| `crossover` | Trend Following | EMA(12)/EMA(26) crossover with volume confirmation |
| `mean_reversion` | Mean Reversion | RSI extremes with Bollinger Band filter |
| `breakout` | Breakout | SMA breakout with volume and ATR confirmation |

## Architecture

```
DATA → FEATURES → STRATEGIES → SIGNALS → BACKTESTING → RISK → PAPER TRADING
```

- **Data Layer** — Immutable raw data, normalized storage, point-in-time access
- **Feature Engine** — Versioned features with full provenance
- **Strategy Engine** — Deterministic signal generation from conditions
- **Backtest Engine** — Event-driven simulation with costs and slippage
- **Risk Engine** — Position sizing, stops, portfolio constraints (authoritative)
- **Signal Generator** — Core signal bot logic with full logging

## Project Structure

```
quantforge/
├── quantforge/           # Main package
│   ├── api/              # FastAPI REST API
│   ├── backtest/         # Event-driven backtesting engine
│   ├── broker/           # Broker abstraction (paper trading)
│   ├── data/             # Data ingestion and validation
│   ├── features/         # Feature engineering engine
│   ├── indicators/       # Technical indicator library
│   ├── models/           # Database models
│   ├── risk/             # Risk engine (position sizing, stops, limits)
│   ├── signals/          # Signal generation (core signal bot)
│   └── strategies/       # Strategy engine
├── data/                 # Data files
├── logs/                 # Log files
├── tests/                # Test suite
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Live Trading

**Live trading is DISABLED.** This platform is for research and signal generation only. The broker abstraction supports live trading, but it requires:

1. Explicit authorization
2. Broker credentials in a secure vault
3. Risk limits configured
4. Kill switch configured
5. Paper trading validation completed
6. Multiple approval checkpoints

## License

Proprietary. For research use only.
