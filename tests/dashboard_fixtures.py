"""A small helper that builds a realistic portfolio for dashboard tests.

Kept in the test tree rather than the package: a fixture that ships in the
library invites someone to use it as an example of a well-formed portfolio,
and it is not one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.backtest.portfolio import Portfolio
from quantforge.core.instrument import AssetClass, Instrument


def make_instruments() -> dict[str, Instrument]:
    """Four instruments, deliberately uneven metadata coverage.

    Two equities carry a sector. An ETF and an FX pair do not, because an ETF
    is a fund and a currency pair has no sector at all. That gap is what the
    sector-coverage assertions are about.
    """
    return {
        "AAPL": Instrument("AAPL", "AAPL", AssetClass.EQUITY, "XNAS", "USD",
                           sector="Technology"),
        "MSFT": Instrument("MSFT", "MSFT", AssetClass.EQUITY, "XNAS", "USD",
                           sector="Technology"),
        "TLT": Instrument("TLT", "TLT", AssetClass.ETF, "XNAS", "USD",
                          sector=None),
        "EURUSD": Instrument("EURUSD", "EURUSD", AssetClass.CURRENCY, "FX", "USD",
                             sector=None),
    }


def make_price_series(n: int = 500, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    returns = rng.normal(0.0004, 0.012, n)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx)


def build_portfolio(
    n: int = 500,
    seed: int = 3,
    initial_cash: float = 1_000_000.0,
    with_closed_trades: bool = True,
    flat_returns: bool = False,
) -> Portfolio:
    """A portfolio with four positions, a marked equity curve, and trades.

    ``flat_returns`` produces a constant-equity curve, which is how the
    dashboard's "undefined rather than misleading" paths get exercised.
    """
    instruments = make_instruments()
    prices = make_price_series(n, seed)
    if flat_returns:
        prices = pd.Series(100.0, index=prices.index)

    portfolio = Portfolio(initial_cash=initial_cash)
    specs = [
        ("AAPL", 1_200, "long"),
        ("MSFT", 500, "short"),     # a short: gross and net must differ
        ("TLT", 3_000, "long"),
        ("EURUSD", 90_000, "long"),
    ]
    for i, (symbol, qty, side) in enumerate(specs):
        entry = float(prices.iloc[i * 20])
        portfolio.get_or_create(symbol, average_entry_price=entry,
                                currency="USD")
        portfolio.apply_fill(symbol, side, qty, entry, fees=1.0,
                             when=prices.index[i * 20], strategy_id="trix")

    # Mark to market on a grid, recording equity at each mark.
    for k in range(0, n, 10):
        ts = prices.index[k]
        portfolio.mark_to_market({s: float(prices.iloc[k]) for s in instruments},
                                 when=ts)
        portfolio.record_equity(ts)
    last = float(prices.iloc[-1])
    portfolio.mark_to_market({s: last for s in instruments}, when=prices.index[-1])
    portfolio.record_equity(prices.index[-1])

    if with_closed_trades:
        # Close the whole AAPL long so there is a round trip to report. Closing
        # means filling in the *opposite* direction to the open side, so a long
        # is closed with a short fill - and the size must be at least the open
        # quantity or the position is only partially reduced and no Trade is
        # recorded. ``get_or_create`` is deliberately not called again here:
        # it would reset the entry bookkeeping the fill is about to consume.
        exit_price = float(prices.iloc[120])
        portfolio.apply_fill("AAPL", "short", 1_200, exit_price, fees=2.0,
                             when=prices.index[120], exit_reason="target")

    return portfolio


def make_benchmark(n: int = 500, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, n))),
                     index=idx)


def make_per_instrument_returns(n: int = 500, seed: int = 5) -> dict[str, pd.Series]:
    """Correlated-ish per-instrument returns for the correlation component."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    market = rng.normal(0.0002, 0.010, n)
    out = {}
    for name, beta in (("AAPL", 1.3), ("MSFT", 1.1), ("TLT", -0.2)):
        idio = rng.normal(0.0, 0.006, n)
        out[name] = pd.Series(beta * market + idio, index=idx)
    return out
