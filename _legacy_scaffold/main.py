"""FastAPI application — REST API for QuantForge AI.

Live trading is DISABLED. This API provides signal generation,
backtesting, and research capabilities only.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from quantforge.backtest.engine import BacktestConfig, BacktestEngine
from quantforge.config import get_config
from quantforge.data.validation import DataValidator
from quantforge.features.engine import FeatureEngine
from quantforge.signals.generator import SignalConfig, SignalGenerator
from quantforge.strategies.crossover import CrossoverConfig, CrossoverStrategy
from quantforge.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from quantforge.strategies.breakout import BreakoutConfig, BreakoutStrategy


# --- Pydantic models ---

class SignalResponse(BaseModel):
    signal_id: str
    timestamp: datetime
    instrument_id: str
    signal_type: str
    direction: str
    confidence: float
    strategy_name: str
    decision: str


class BacktestRequest(BaseModel):
    instrument_ids: List[str]
    start_date: datetime
    end_date: datetime
    strategy: str = "crossover"
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.001


class BacktestResponse(BaseModel):
    run_id: str
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    volatility: float
    var_95: float
    cvar_95: float


class HealthResponse(BaseModel):
    status: str
    version: str
    live_trading: bool = False
    timestamp: datetime


# --- Application factory ---

def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="QuantForge AI",
        description="Research-grade quantitative trading signal platform. Live trading is DISABLED.",
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            live_trading=False,
            timestamp=datetime.utcnow(),
        )

    @app.get("/strategies")
    async def list_strategies() -> List[Dict[str, Any]]:
        """List available strategies."""
        return [
            {
                "name": "crossover",
                "description": "Moving average crossover strategy",
                "type": "trend_following",
            },
            {
                "name": "mean_reversion",
                "description": "RSI-based mean reversion strategy",
                "type": "mean_reversion",
            },
            {
                "name": "breakout",
                "description": "Price breakout strategy with volume confirmation",
                "type": "breakout",
            },
        ]

    @app.post("/signals/generate", response_model=List[SignalResponse])
    async def generate_signals(
        instrument_ids: List[str] = Query(...),
        strategy: str = "crossover",
        data_path: Optional[str] = None,
    ) -> List[SignalResponse]:
        """Generate trading signals for specified instruments.

        This is a signal bot — it generates signals for research and
        paper trading. Live trading is DISABLED.
        """
        import pandas as pd

        # Load data (in production, this would come from the database)
        if data_path is None:
            raise HTTPException(status_code=400, detail="data_path is required")

        try:
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load data: {e}")

        # Create strategy
        strat = _create_strategy(strategy)
        if strat is None:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

        # Create signal generator
        config = SignalConfig(
            strategy=strat,
            feature_engine=FeatureEngine(),
            instruments=instrument_ids,
        )
        generator = SignalGenerator(config)

        # Generate signals
        signals = generator.generate_signals(data)

        return [
            SignalResponse(
                signal_id=s.signal_id,
                timestamp=s.timestamp,
                instrument_id=s.instrument_id,
                signal_type=s.signal_type.value,
                direction=s.direction,
                confidence=s.confidence,
                strategy_name=s.strategy_name,
                decision=s.decision,
            )
            for s in signals
        ]

    @app.post("/backtest", response_model=BacktestResponse)
    async def run_backtest(request: BacktestRequest) -> BacktestResponse:
        """Run a backtest.

        Backtesting is for research purposes. A profitable historical
        backtest is evidence about the tested historical assumptions,
        not proof of future profitability.
        """
        import pandas as pd

        # In production, this would load data from the database
        raise HTTPException(
            status_code=501,
            detail="Backtest endpoint requires data source configuration. Use the CLI for backtesting.",
        )

    @app.get("/data/validate")
    async def validate_data(
        instrument_id: str = Query(...),
        data_path: str = Query(...),
    ) -> Dict[str, Any]:
        """Validate market data quality."""
        import pandas as pd

        try:
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load data: {e}")

        validator = DataValidator()
        report = validator.validate(data, instrument_id)
        return report.to_dict()

    return app


def _create_strategy(name: str):
    """Create a strategy by name."""
    if name == "crossover":
        return CrossoverStrategy(CrossoverConfig(name="crossover"))
    elif name == "mean_reversion":
        return MeanReversionStrategy(MeanReversionConfig(name="mean_reversion"))
    elif name == "breakout":
        return BreakoutStrategy(BreakoutConfig(name="breakout"))
    return None
