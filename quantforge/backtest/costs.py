"""Transaction cost and slippage models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TransactionCostConfig:
    """Transaction cost configuration."""
    commission_pct: float = 0.001       # 0.1% commission
    min_commission: float = 1.0         # Minimum commission per trade
    max_commission_pct: float = 0.01    # Max 1% commission
    exchange_fee: float = 0.0           # Fixed exchange fee
    regulatory_fee: float = 0.0         # SEC/FINRA fees


@dataclass
class SlippageConfig:
    """Slippage configuration."""
    model: str = "percentage"           # "fixed", "percentage", "spread", "volatility"
    fixed_slippage: float = 0.01        # Fixed slippage in currency
    percentage_slippage: float = 0.001  # 0.1% slippage
    spread_fraction: float = 0.5        # Fraction of spread
    volatility_multiplier: float = 0.1  # Volatility-based multiplier


class TransactionCostModel:
    """Models transaction costs: commission, exchange fees, regulatory fees."""

    def __init__(self, config: TransactionCostConfig):
        self.config = config

    def compute_commission(self, price: float, quantity: float) -> float:
        """Compute commission for a trade."""
        trade_value = abs(price * quantity)
        commission = trade_value * self.config.commission_pct
        commission = max(commission, self.config.min_commission)
        max_commission = trade_value * self.config.max_commission_pct
        commission = min(commission, max_commission)
        return commission

    def compute_total_cost(self, price: float, quantity: float) -> float:
        """Compute total transaction cost."""
        commission = self.compute_commission(price, quantity)
        return commission + self.config.exchange_fee + self.config.regulatory_fee


class SlippageModel:
    """Models slippage — the difference between expected and actual execution price.

    Never assumes unlimited liquidity or perfect fills.
    """

    def __init__(self, config: SlippageConfig):
        self.config = config

    def compute_slippage(
        self,
        price: float,
        quantity: float,
        spread: Optional[float] = None,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> float:
        """Compute slippage for a trade.

        Returns:
            Slippage as a positive number (always against the trader).
        """
        if self.config.model == "fixed":
            return self.config.fixed_slippage

        elif self.config.model == "percentage":
            return price * self.config.percentage_slippage

        elif self.config.model == "spread":
            if spread is not None and spread > 0:
                return spread * self.config.spread_fraction
            return price * self.config.percentage_slippage

        elif self.config.model == "volatility":
            if volatility is not None and volatility > 0:
                return price * volatility * self.config.volatility_multiplier
            return price * self.config.percentage_slippage

        return price * self.config.percentage_slippage

    def compute_execution_price(
        self,
        price: float,
        side: str,  # "buy" or "sell"
        quantity: float,
        spread: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> float:
        """Compute actual execution price including slippage.

        For buys: execution price is higher (slippage against trader).
        For sells: execution price is lower (slippage against trader).
        """
        slippage = self.compute_slippage(price, quantity, spread, volatility)
        if side == "buy":
            return price + slippage
        else:
            return price - slippage
