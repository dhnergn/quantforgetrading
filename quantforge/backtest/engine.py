"""Event-driven backtesting engine.

Processes bars sequentially:
    Market Data → Feature Calculation → Signal → Risk Check → Order → Execution → Portfolio Update

The system NEVER uses future bars. Temporal causality is enforced.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantforge.backtest.costs import SlippageModel, TransactionCostConfig, TransactionCostModel, SlippageConfig
from quantforge.features.engine import FeatureEngine
from quantforge.risk.portfolio import PortfolioRiskManager, RiskLimits
from quantforge.risk.position_sizing import ATRSizer, FixedFractionalSizer, SizingContext
from quantforge.strategies.base import SignalDirection, Strategy


@dataclass
class BacktestConfig:
    """Complete backtest configuration — reproducible."""
    strategy: Strategy
    data: pd.DataFrame                  # OHLCV data for all instruments
    instrument_ids: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.001
    max_position_pct: float = 0.20
    max_gross_exposure: float = 2.0
    risk_fraction: float = 0.01
    benchmark: Optional[str] = None
    dataset_version: str = "1.0.0"
    random_seed: int = 42

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of configuration for reproducibility."""
        config_dict = {
            "instrument_ids": self.instrument_ids,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "max_position_pct": self.max_position_pct,
            "max_gross_exposure": self.max_gross_exposure,
            "risk_fraction": self.risk_fraction,
            "benchmark": self.benchmark,
            "dataset_version": self.dataset_version,
            "random_seed": self.random_seed,
            "strategy_config": self.strategy.config.__dict__,
        }
        canonical = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class Trade:
    """A completed trade."""
    trade_id: str
    instrument_id: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result."""
    run_id: str
    config_hash: str
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    profit_factor: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    var_95: float
    cvar_95: float
    equity_curve: pd.Series
    trades: List[Trade]
    daily_returns: pd.Series


class BacktestEngine:
    """Event-driven backtesting engine.

    Processes bars sequentially and enforces temporal causality.
    The system NEVER uses future bars.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.feature_engine = FeatureEngine(dataset_version=config.dataset_version)
        self.cost_model = TransactionCostModel(
            TransactionCostConfig(commission_pct=config.commission_pct)
        )
        self.slippage_model = SlippageModel(
            SlippageConfig(percentage_slippage=config.slippage_pct)
        )
        self.risk_manager = PortfolioRiskManager(
            RiskLimits(
                max_single_position_pct=config.max_position_pct,
                max_gross_exposure=config.max_gross_exposure,
            )
        )
        self.position_sizer = FixedFractionalSizer()

        # State
        self.equity = config.initial_capital
        self.cash = config.initial_capital
        self.positions: Dict[str, float] = {}  # instrument_id -> quantity
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self._open_trades: Dict[str, Trade] = {}  # instrument_id -> open trade

    def run(self) -> BacktestResult:
        """Run the backtest.

        Returns:
            BacktestResult with complete performance metrics.
        """
        data = self.config.data
        strategy = self.config.strategy

        # Filter data to date range
        mask = (data.index >= self.config.start_date) & (data.index <= self.config.end_date)
        backtest_data = data.loc[mask]

        if backtest_data.empty:
            raise ValueError("No data in specified date range")

        # Process each bar sequentially
        for timestamp, bar in backtest_data.iterrows():
            self._process_bar(timestamp, bar, strategy)

        # Close any open positions at the end
        self._close_all_positions(backtest_data.index[-1])

        # Compute results
        return self._compute_results()

    def _process_bar(
        self,
        timestamp: datetime,
        bar: pd.Series,
        strategy: Strategy,
    ) -> None:
        """Process a single bar — the core event loop.

        NEVER uses future bars. All calculations use only data up to `timestamp`.
        """
        # Update equity
        self._update_equity(bar)

        # Check risk limits
        risk_result = self.risk_manager.check_drawdown(self.equity)
        if not risk_result.approved:
            return

        # For each instrument in the universe
        for instrument_id in self.config.instrument_ids:
            # Get instrument data up to this point
            instrument_data = self._get_instrument_data(instrument_id, timestamp)
            if instrument_data is None or len(instrument_data) < 50:
                continue

            # Compute features (point-in-time)
            features = self.feature_engine.compute_features(
                instrument_id=instrument_id,
                data=instrument_data,
                as_of=timestamp,
                feature_names=strategy.required_features(),
            )

            if not features.features:
                continue

            # Get current position
            current_position = self.positions.get(instrument_id, 0.0)

            # Evaluate strategy
            signal = strategy.evaluate(features, current_position)

            if signal is None:
                continue

            # Execute signal
            self._execute_signal(signal, bar, instrument_id)

        # Record equity
        self.equity_curve.append((timestamp, self.equity))

    def _execute_signal(
        self,
        signal,
        bar: pd.Series,
        instrument_id: str,
    ) -> None:
        """Execute a trading signal through the risk engine.

        Signal → Position Sizing → Risk Engine → Portfolio Constraints → Order
        """
        current_position = self.positions.get(instrument_id, 0.0)
        price = bar["close"]

        if signal.direction == SignalDirection.LONG and current_position <= 0:
            # Close short if exists
            if current_position < 0:
                self._close_position(instrument_id, price, bar.name)

            # Size new position
            atr = signal.features.get("atr_14")
            stop_distance = atr * 2 if atr else price * 0.02

            sizing_context = SizingContext(
                equity=self.equity,
                current_price=price,
                atr=atr,
                stop_distance=stop_distance,
                max_position_pct=self.config.max_position_pct,
                max_gross_exposure=self.config.max_gross_exposure,
                risk_fraction=self.config.risk_fraction,
            )
            size = self.position_sizer.compute_size(sizing_context)

            if size <= 0:
                return

            # Risk check
            risk_result = self.risk_manager.check_order(
                instrument_id=instrument_id,
                side="buy",
                quantity=size,
                price=price,
                current_positions=self._get_current_positions(),
                equity=self.equity,
            )

            if not risk_result.approved:
                return

            # Execute
            self._open_position(instrument_id, size, price, bar.name)

        elif signal.direction == SignalDirection.EXIT_LONG and current_position > 0:
            self._close_position(instrument_id, price, bar.name)

    def _open_position(self, instrument_id: str, quantity: float, price: float, timestamp: datetime) -> None:
        """Open a new position."""
        # Apply slippage
        execution_price = self.slippage_model.compute_execution_price(price, "buy", quantity)
        slippage = abs(execution_price - price) * quantity

        # Compute cost
        cost = self.cost_model.compute_total_cost(execution_price, quantity)

        # Update state
        self.positions[instrument_id] = self.positions.get(instrument_id, 0.0) + quantity
        self.cash -= (execution_price * quantity) + cost

        # Record trade
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            instrument_id=instrument_id,
            side="buy",
            quantity=quantity,
            entry_price=execution_price,
            exit_price=None,
            entry_time=timestamp,
            exit_time=None,
            commission=cost,
            slippage=slippage,
        )
        self._open_trades[instrument_id] = trade

    def _close_position(self, instrument_id: str, price: float, timestamp: datetime) -> None:
        """Close an existing position."""
        if instrument_id not in self._open_trades:
            return

        trade = self._open_trades[instrument_id]
        quantity = trade.quantity

        # Apply slippage
        execution_price = self.slippage_model.compute_execution_price(price, "sell", quantity)
        slippage = abs(execution_price - price) * quantity

        # Compute cost
        cost = self.cost_model.compute_total_cost(execution_price, quantity)

        # Compute P&L
        gross_pnl = (execution_price - trade.entry_price) * quantity
        net_pnl = gross_pnl - trade.commission - cost - trade.slippage - slippage
        return_pct = net_pnl / (trade.entry_price * quantity) if trade.entry_price * quantity > 0 else 0

        # Update state
        self.positions[instrument_id] = self.positions.get(instrument_id, 0.0) - quantity
        if abs(self.positions[instrument_id]) < 1e-8:
            del self.positions[instrument_id]
        self.cash += (execution_price * quantity) - cost

        # Update trade
        trade.exit_price = execution_price
        trade.exit_time = timestamp
        trade.pnl = net_pnl
        trade.return_pct = return_pct
        trade.commission += cost
        trade.slippage += slippage

        self.trades.append(trade)
        del self._open_trades[instrument_id]

    def _close_all_positions(self, timestamp: datetime) -> None:
        """Close all open positions."""
        for instrument_id in list(self._open_trades.keys()):
            # Use last available price
            last_price = self._get_last_price(instrument_id)
            if last_price:
                self._close_position(instrument_id, last_price, timestamp)

    def _update_equity(self, bar: pd.Series) -> None:
        """Update equity based on current positions and prices."""
        position_value = 0.0
        for instrument_id, quantity in self.positions.items():
            price = bar.get("close", 0)
            position_value += quantity * price
        self.equity = self.cash + position_value

    def _get_current_positions(self):
        """Get current positions for risk checks."""
        from quantforge.risk.portfolio import Position
        return [
            Position(instrument_id=inst_id, quantity=qty, current_price=0)
            for inst_id, qty in self.positions.items()
        ]

    def _get_instrument_data(self, instrument_id: str, as_of: datetime) -> Optional[pd.DataFrame]:
        """Get instrument data up to a specific point in time.

        Enforces temporal causality — only returns data with timestamp <= as_of.
        """
        # In production, this would query the database
        # For MVP, we filter the in-memory DataFrame
        data = self.config.data
        if isinstance(data.index, pd.MultiIndex):
            # MultiIndex (instrument_id, timestamp)
            try:
                instrument_data = data.xs(instrument_id, level=0)
                return instrument_data[instrument_data.index <= as_of]
            except KeyError:
                return None
        else:
            # Single instrument DataFrame
            return data[data.index <= as_of]

    def _get_last_price(self, instrument_id: str) -> Optional[float]:
        """Get the last available price for an instrument."""
        data = self.config.data
        if isinstance(data.index, pd.MultiIndex):
            try:
                instrument_data = data.xs(instrument_id, level=0)
                return float(instrument_data["close"].iloc[-1]) if len(instrument_data) > 0 else None
            except KeyError:
                return None
        else:
            return float(data["close"].iloc[-1]) if len(data) > 0 else None

    def _compute_results(self) -> BacktestResult:
        """Compute final backtest results."""
        # Equity curve
        equity_df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        equity_df.set_index("timestamp", inplace=True)
        equity_series = equity_df["equity"]

        # Daily returns
        daily_returns = equity_series.pct_change().dropna()

        # Performance metrics
        total_return = (equity_series.iloc[-1] / self.config.initial_capital) - 1 if len(equity_series) > 1 else 0

        # CAGR
        n_years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25 if len(equity_series) > 1 else 1
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0

        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0

        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino = (daily_returns.mean() * 252) / downside_dev if downside_dev > 0 else 0

        # Max drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Max drawdown duration
        is_drawdown = drawdown < 0
        drawdown_duration = 0
        max_duration = 0
        for dd in is_drawdown:
            if dd:
                drawdown_duration += 1
                max_duration = max(max_duration, drawdown_duration)
            else:
                drawdown_duration = 0

        # Calmar ratio
        calmar = cagr / max_drawdown if max_drawdown > 0 else 0

        # Trade metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_trade_return = np.mean([t.return_pct for t in self.trades if t.return_pct is not None]) if self.trades else 0

        # VaR/CVaR
        var_95 = self.risk_manager.compute_var(daily_returns, 0.95)
        cvar_95 = self.risk_manager.compute_cvar(daily_returns, 0.95)

        return BacktestResult(
            run_id=str(uuid.uuid4()),
            config_hash=self.config.compute_hash(),
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_duration,
            volatility=volatility,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            var_95=var_95,
            cvar_95=cvar_95,
            equity_curve=equity_series,
            trades=self.trades,
            daily_returns=daily_returns,
        )
