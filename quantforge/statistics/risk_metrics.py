"""Risk metrics — Sharpe, Sortino, Calmar, VaR, CVaR, drawdown, etc.

Undefined metrics remain undefined (NaN) rather than silently converting to zero.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class RiskMetrics:
    """Performance and risk metrics with proper edge case handling."""

    @staticmethod
    def total_return(equity_curve: pd.Series) -> float:
        """Total return over the period."""
        if len(equity_curve) < 2:
            return float("nan")
        return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    @staticmethod
    def cagr(equity_curve: pd.Series) -> float:
        """Compound Annual Growth Rate."""
        if len(equity_curve) < 2:
            return float("nan")
        total = RiskMetrics.total_return(equity_curve)
        n_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        if n_years <= 0:
            return float("nan")
        return float((1 + total) ** (1 / n_years) - 1)

    @staticmethod
    def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Annualized volatility."""
        clean = returns.dropna()
        if len(clean) < 2:
            return float("nan")
        return float(clean.std() * np.sqrt(periods_per_year))

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Sharpe ratio. Returns NaN if volatility is 0."""
        clean = returns.dropna()
        if len(clean) < 2:
            return float("nan")
        excess = clean - risk_free_rate / periods_per_year
        vol = clean.std() * np.sqrt(periods_per_year)
        if vol == 0:
            return float("nan")
        return float(excess.mean() * periods_per_year / vol)

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Sortino ratio. Returns NaN if downside deviation is 0."""
        clean = returns.dropna()
        if len(clean) < 2:
            return float("nan")
        excess = clean - risk_free_rate / periods_per_year
        downside = clean[clean < 0]
        if len(downside) < 2:
            return float("nan")
        downside_dev = downside.std() * np.sqrt(periods_per_year)
        if downside_dev == 0:
            return float("nan")
        return float(excess.mean() * periods_per_year / downside_dev)

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """Maximum drawdown (positive number)."""
        if len(equity_curve) < 2:
            return float("nan")
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return float(abs(drawdown.min()))

    @staticmethod
    def max_drawdown_duration(equity_curve: pd.Series) -> int:
        """Maximum drawdown duration in periods."""
        if len(equity_curve) < 2:
            return 0
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        is_drawdown = drawdown < 0

        max_duration = 0
        current = 0
        for dd in is_drawdown:
            if dd:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0
        return max_duration

    @staticmethod
    def calmar_ratio(equity_curve: pd.Series) -> float:
        """Calmar ratio = CAGR / Max Drawdown."""
        cagr = RiskMetrics.cagr(equity_curve)
        mdd = RiskMetrics.max_drawdown(equity_curve)
        if np.isnan(cagr) or np.isnan(mdd) or mdd == 0:
            return float("nan")
        return float(cagr / mdd)

    @staticmethod
    def profit_factor(trades: pd.Series) -> float:
        """Profit factor = gross profit / gross loss."""
        if len(trades) == 0:
            return float("nan")
        gross_profit = trades[trades > 0].sum()
        gross_loss = abs(trades[trades < 0].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else float("nan")
        return float(gross_profit / gross_loss)

    @staticmethod
    def win_rate(trades: pd.Series) -> float:
        """Win rate = winning trades / total trades."""
        if len(trades) == 0:
            return float("nan")
        return float((trades > 0).sum() / len(trades))

    @staticmethod
    def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical Value at Risk.

        Returns:
            VaR as a positive number (loss amount).
            Returns NaN if insufficient data.
        """
        clean = returns.dropna()
        if len(clean) < 30:
            return float("nan")
        return float(abs(np.percentile(clean, (1 - confidence) * 100)))

    @staticmethod
    def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical Conditional VaR (Expected Shortfall).

        Returns:
            CVaR as a positive number (average loss beyond VaR).
            Returns NaN if insufficient data.
        """
        clean = returns.dropna()
        if len(clean) < 30:
            return float("nan")
        var = RiskMetrics.historical_var(clean, confidence)
        if np.isnan(var):
            return float("nan")
        tail = clean[clean <= -var]
        if len(tail) == 0:
            return var
        return float(abs(tail.mean()))

    @staticmethod
    def parametric_var(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Parametric (normal) VaR."""
        clean = returns.dropna()
        if len(clean) < 2:
            return float("nan")
        mean = clean.mean()
        std = clean.std()
        z = abs(np.percentile(np.random.randn(10000), (1 - confidence) * 100))
        return float(abs(mean - z * std))
