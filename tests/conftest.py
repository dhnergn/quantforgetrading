"""Shared fixtures.

Synthetic data only. No test in this suite contacts a network, a broker, or a
data vendor, and none of them can place a real order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantforge.core.enums import Role  # noqa: E402
from quantforge.security.access import User  # noqa: E402


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    """Daily OHLCV from the GARCH simulator."""
    from quantforge.data.simulator import MarketSimulator

    df, _ = MarketSimulator(n=600, seed=42, start="2019-01-01").simulate()
    return df


@pytest.fixture(scope="session")
def ohlcv_long() -> pd.DataFrame:
    from quantforge.data.simulator import MarketSimulator

    df, _ = MarketSimulator(n=1500, seed=7, start="2018-01-01").simulate()
    return df


@pytest.fixture(scope="session")
def features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    from quantforge.features.frame import build_feature_frame

    return build_feature_frame(ohlcv)


@pytest.fixture(scope="session")
def cost_model():
    from quantforge.backtest.costs import CommissionSchedule, CostModel, SlippageModelSpec
    from quantforge.core.enums import SlippageModel

    return CostModel(
        commission=CommissionSchedule(per_share=0.005, regulatory_bps=0.0002),
        slippage=SlippageModelSpec(
            model=SlippageModel.PERCENTAGE, pct=0.0005, half_spread_bps=2.5
        ),
    )


@pytest.fixture(scope="session")
def permissive_limits():
    """Limits loose enough not to interfere with tests unrelated to risk."""
    from quantforge.risk.limits import RiskLimits

    return RiskLimits(
        max_position_pct=0.60, max_gross_exposure=1.0, max_net_exposure=0.60,
        max_order_notional_pct=0.60, max_sector_pct=1.0,
        max_asset_class_pct=1.0, max_leverage=1.0,
    )


@pytest.fixture(scope="session")
def tight_limits():
    from quantforge.risk.limits import RiskLimits

    return RiskLimits(
        max_position_pct=0.10, max_sector_pct=0.20, max_gross_exposure=0.50,
        max_net_exposure=0.30, max_order_notional_pct=0.10, max_leverage=0.50,
    )


@pytest.fixture
def portfolio():
    from quantforge.backtest.portfolio import Portfolio

    return Portfolio(100_000.0)


@pytest.fixture
def order_book():
    from quantforge.execution.orders import OrderBook

    return OrderBook()


@pytest.fixture
def paper_broker(ohlcv: pd.DataFrame):
    from quantforge.execution.brokers.paper import PaperBrokerAdapter

    b = PaperBrokerAdapter(initial_equity=100_000.0)
    b.set_market("TEST", ohlcv)
    b.connect()
    return b


@pytest.fixture
def users() -> dict[str, User]:
    return {
        "researcher": User("researcher", "researcher", frozenset({Role.RESEARCHER})),
        "quant": User("quant", "quant_dev", frozenset({Role.QUANT_DEVELOPER})),
        "pm": User("pm", "pm", frozenset({Role.PORTFOLIO_MANAGER}), mfa_enabled=True),
        "risk": User("risk", "risk", frozenset({Role.RISK_MANAGER}), mfa_enabled=True),
        "admin": User("admin", "admin", frozenset({Role.ADMINISTRATOR}), mfa_enabled=True),
    }


@pytest.fixture
def ml_frame():
    """A learnable but non-trivial feature/target pair."""
    rng = np.random.default_rng(99)
    n = 900
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    f1 = rng.normal(0, 1, n)
    X = pd.DataFrame({
        "f1": f1,
        "f2": rng.normal(0, 1, n),
        "f3": f1 * 0.5 + rng.normal(0, 0.5, n),
    }, index=idx)
    y = pd.Series(
        np.where(f1 * 0.4 + rng.normal(0, 0.5, n) > 0, "up", "down"), index=idx
    )
    y_reg = pd.Series(f1 * 0.3 + rng.normal(0, 0.4, n), index=idx)
    return X, y, y_reg, idx[:700], idx[700:]


#: Markers so a test run can assert nothing external was contacted.
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test")
    config.addinivalue_line("markers", "leakage: look-ahead / leakage guards")
