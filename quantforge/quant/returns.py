"""Return engine.

The critical distinction this module enforces is between a *price return* and a
*total return*. A price return computed on raw prices drops ~2% a year on a
3%-yield equity through dividend payments alone, which is a fictional loss
rather than a real one. Every return function therefore states which price
series it consumed.

Four different objects get conflated in practice, and the blueprint requires
they stay separate:

``asset_return``      per-instrument return series
``portfolio_return``  return on a basket, rebalanced on a stated schedule
``strategy_return``   return of signal-derived positions, net of modelled costs
``account_pnl``       realised plus unrealised P&L on cash and positions

They differ under rebalancing, leverage, costs, and idle cash. Nothing here
computes the latter three; this module supplies the first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

PriceBasis = Literal["raw", "adjusted"]


# --------------------------------------------------------------------------
# Core return types
# --------------------------------------------------------------------------


def simple_returns(
    prices: pd.Series,
    basis: PriceBasis = "adjusted",
    periods: int = 1,
) -> pd.Series:
    """Arithmetic (simple) returns.

    Formula: ``r(t) = P(t) / P(t-k) - 1``

    ``basis`` is recorded in the returned Series' attrs so downstream code and
    reports can state which price series produced the number. It is a required
    argument rather than a global default precisely because this is the single
    most consequential ambiguity in return research.
    """
    if basis not in ("raw", "adjusted"):
        raise ValueError(f"basis must be 'raw' or 'adjusted', got {basis!r}")
    prices = _validate_positive(prices, "prices")
    out = prices.pct_change(periods)
    out.name = f"simple_return_{basis}"
    out.attrs.update({
        "basis": basis,
        "definition": "P(t)/P(t-k)-1",
        "periods": periods,
        "annualization_factor": None,
    })
    return out


def log_returns(prices: pd.Series, periods: int = 1, basis: PriceBasis = "adjusted") -> pd.Series:
    """Logarithmic returns.

    Formula: ``r(t) = ln(P(t) / P(t-k))``

    Log returns are time-additive: ``sum(ln returns)`` telescopes exactly to
    ``ln(P_end / P_start)``, which is why multi-period aggregation should use
    them. The cost is a small bias in the mean (``E[log r] != log E[r]``) that
    matters only at long horizons and is documented rather than corrected.
    """
    prices = _validate_positive(prices, "prices")
    out = np.log(prices / prices.shift(periods))
    out.name = f"log_return_{basis}"
    out.attrs.update({"basis": basis, "definition": "ln(P(t)/P(t-k))", "periods": periods})
    return out


def cumulative_returns(returns: pd.Series, kind: str = "simple") -> pd.Series:
    """Wealth index from a return series (starts at 1.0).

    For simple returns: ``W(t) = prod(1 + r(i))``. An ``r <= -1`` collapses the
    wealth index to zero, which is flagged rather than clipped.
    """
    r = returns.dropna()
    if kind == "simple":
        if (r <= -1.0).any():
            bad = r.index[(r <= -1.0)][0]
            raise ValueError(
                f"return of -100% or worse at {bad} destroys the wealth index; "
                "a total loss must be modelled explicitly, not compounded"
            )
        wealth = (1.0 + r).cumprod()
    elif kind == "log":
        wealth = np.exp(r.cumsum())
    else:
        raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")
    out = pd.Series(wealth, index=wealth.index, name=f"cumulative_{kind}")
    out = _reindex(wealth, returns.index)
    out.attrs.update({"kind": kind, "definition": "compounded wealth index"})
    return out


def annualized_return(
    returns: pd.Series,
    bars_per_year: int = 252,
    kind: str = "simple",
) -> float:
    """Annualise a return series.

    Simple: geometric ``(1+total)^(bars_per_year/n) - 1``.
    Log: ``mean(ln r) * bars_per_year`` (an arithmetic mean of log returns).

    Returns NaN rather than a number when the series is too short or the
    wealth index is non-positive; an annualised figure computed from a
    degenerate input is worse than an absent one.
    """
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return float("nan")
    if kind == "simple":
        wealth = float((1.0 + r).prod())
        if not np.isfinite(wealth) or wealth <= 0:
            return float("nan")
        return float(wealth ** (bars_per_year / n) - 1.0)
    if kind == "log":
        mean_log = float(r.mean())
        if not np.isfinite(mean_log):
            return float("nan")
        return float(np.expm1(mean_log * bars_per_year))
    raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")


def rolling_return(prices: pd.Series, window: int, basis: PriceBasis = "adjusted") -> pd.Series:
    """Total return over a trailing window.

    Formula: ``P(t) / P(t-window) - 1``

    Windowed price return, *not* compounded from ``window`` daily returns. The
    two differ by the compounding term and only agree under the small-return
    approximation.
    """
    prices = _validate_positive(prices, "prices")
    out = prices.pct_change(window)
    out.name = f"rolling_return_{window}_{basis}"
    out.attrs.update({"basis": basis, "window": window})
    return out


def excess_return(
    returns: pd.Series,
    risk_free: pd.Series | float,
    periods_per_year: int = 252,
    annualize: bool = False,
) -> pd.Series:
    """Return in excess of a risk-free benchmark.

    A scalar ``risk_free`` is interpreted as an *annual* rate and converted to
    a per-period simple rate by the geometric identity, which is correct for
    compounding; dividing by ``periods_per_year`` would understate the
    benchmark and flatter the excess return.
    """
    if isinstance(risk_free, (int, float)):
        annual = float(risk_free)
        per_period = (1.0 + annual) ** (1.0 / periods_per_year) - 1.0
        rf = pd.Series(per_period, index=returns.index)
    else:
        rf = risk_free.reindex(returns.index)
    out = returns - rf
    if annualize:
        out = out * periods_per_year
    out.name = "excess_return"
    out.attrs.update({
        "definition": "r - rf",
        "annualized": bool(annualize),
        "risk_free_periods_per_year": periods_per_year,
    })
    return out


# --------------------------------------------------------------------------
# Total return: dividends and corporate actions
# --------------------------------------------------------------------------


@dataclass(slots=True)
class DividendEvent:
    """A single cash distribution on a single instrument."""

    ex_date: pd.Timestamp
    cash_per_share: float
    currency: str = "USD"
    source: str = "unknown"
    announced_date: Optional[pd.Timestamp] = None

    @property
    def is_special(self) -> bool:
        return False


def total_return_index(
    prices: pd.Series,
    dividends: Optional[pd.Series] = None,
    basis: PriceBasis = "raw",
) -> pd.Series:
    """Total-return wealth index, reinvesting dividends at the ex-date close.

    Formula per bar: ``TRI(t) = TRI(t-1) * (P(t) + D(t)) / P(t-1)``

    The dividend is added to the ending price before the ratio is taken, which
    is the reinvestment assumption. Reinvesting at the ex-date close is
    optimistic in hindsight for a falling market; reinvesting at the *payment*
    date is more realistic. The choice is explicit and reported.

    Splits are assumed already handled: if ``prices`` is split-adjusted, the
    dividend per share must be on the same post-split basis. Mixing a raw
    price series with post-split dividends silently multiplies returns, so the
    combination is validated where dividends are non-zero.
    """
    px = _validate_positive(prices, "prices")
    if dividends is None:
        idx = px / px.iloc[0]
    else:
        div = dividends.reindex(px.index).fillna(0.0)
        # First bar has no prior price, so its gross factor is defined as 1.0
        # (no return yet) rather than NaN, which would desynchronise the index.
        gross = ((px + div) / px.shift(1)).fillna(1.0)
        idx = gross.cumprod()
    out = idx.rename("total_return_index")
    out.attrs.update({
        "basis": basis,
        "dividends_reinvested": dividends is not None,
        "reinvestment_assumption": "ex_date_close",
    })
    return out


def split_adjusted_returns(
    raw_prices: pd.Series,
    split_factors: pd.Series,
) -> pd.Series:
    """Returns on a split-adjusted series, with raw series preserved separately.

    ``split_factors`` maps ex-date to factor, where factor 2.0 means a
    2-for-1 split (price halved). The adjusted series is
    ``raw / cumprod(factors)``. Returns come from the adjusted series, and the
    raw series is returned alongside so a report can state both.
    """
    factors = split_factors.reindex(raw_prices.index).fillna(1.0)
    cum = factors.cumprod()
    adjusted = raw_prices / cum
    ret = adjusted.pct_change()
    ret.name = "split_adjusted_return"
    ret.attrs.update({"split_count": int((factors != 1.0).sum())})
    return ret


# --------------------------------------------------------------------------
# Cross-sectional helpers
# --------------------------------------------------------------------------


def cross_sectional_returns(
    prices: pd.DataFrame,
    basis: PriceBasis = "adjusted",
) -> pd.DataFrame:
    """Per-instrument simple returns from a wide price frame (dates x assets)."""
    return prices.pct_change().rename_axis(index="timestamp", columns="instrument_id")


def equal_weight_portfolio_return(asset_returns: pd.DataFrame) -> pd.Series:
    """Daily rebalanced equal-weight basket return.

    This is the *rebalanced* return: it implicitly trades every bar. It is not
    the buy-and-hold return of the same basket and is not the return of an
    uninvested-cash portfolio. Confusing the three is common, so the method is
    named in the output attrs.
    """
    if asset_returns.shape[1] == 0:
        raise ValueError("no assets in frame")
    n = asset_returns.notna().sum(axis=1)
    total = asset_returns.sum(axis=1, min_count=1)
    out = total / n.replace(0, np.nan)
    out.name = "equal_weight_portfolio_return"
    out.attrs.update({
        "method": "daily_rebalanced_equal_weight",
        "n_assets_mean": float(n.mean()),
        "cash_drag": "not modelled; assumes 100% invested",
    })
    return out


def buy_and_hold_return(
    prices: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
    basis: PriceBasis = "adjusted",
) -> pd.Series:
    """Buy-and-hold basket return from initial weights, idle cash included.

    Distinct from :func:`equal_weight_portfolio_return`: weights are set once
    and drift with the assets. Where prices are missing for an instrument its
    weight is held constant at the last observed value rather than dropped,
    which avoids a silent survivorship-like bias in the basket.
    """
    px = prices.copy()
    if weights is None:
        w = pd.Series(1.0 / px.shape[1], index=px.columns)
    else:
        w = pd.Series(weights).reindex(px.columns).fillna(0.0)
    px = px.ffill()
    base = px.iloc[0]
    norm = px.divide(base)
    port = norm.mul(w, axis=1).sum(axis=1, min_count=1)
    out = port.pct_change()
    out.name = "buy_and_hold_return"
    out.attrs.update({
        "method": "buy_and_hold",
        "initial_weights": {k: float(v) for k, v in w.items()},
        "basis": basis,
    })
    return out


def volatility(returns: pd.Series, bars_per_year: int = 252,
               annualize: bool = True) -> pd.Series:
    """Rolling standard deviation of returns.

    Sample standard deviation (``ddof=1``) because returns are sample
    observations from a distribution, not a population.
    """
    window = bars_per_year if annualize else 1
    out = returns.rolling(window, min_periods=max(2, window // 4)).std(ddof=1)
    if annualize:
        out = out * np.sqrt(bars_per_year)
    out.name = "volatility_annualized" if annualize else "volatility"
    out.attrs.update({"ddof": 1, "bars_per_year": bars_per_year, "annualized": annualize})
    return out


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _validate_positive(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if series.empty:
        return series.astype(float)
    if (series.dropna() <= 0).any():
        first = series.index[series.dropna() <= 0][0]
        raise ValueError(
            f"{name} contains non-positive values at {first}; log and simple "
            "returns both require strictly positive prices. Check that the "
            "series is split-adjusted rather than raw."
        )
    return series.astype(float)


def _reindex(series: pd.Series, target: pd.Index) -> pd.Series:
    return series.reindex(target)
