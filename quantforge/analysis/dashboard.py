"""Portfolio dashboard: the 14 components blueprint Section 56 names.

Why a separate layer
--------------------
A dashboard is the one part of a quant system where a number can be
*technically correct and still deceive*. An equity curve without a baseline, a
Sharpe ratio without a sample size, a VaR without a confidence level, a sector
pie that silently omits the instruments with no sector metadata - each of these
is easy to compute, easy to render, and each misleads in a specific direction.

So every component here carries its own basis:

===============  ==========================================================
``n_observations``  how many data points produced the number
``period``          the span the number describes
``caveat``          what the number does not tell you
``available``       false with a ``reason`` when it genuinely cannot be computed
===============  ==========================================================

A component that cannot be computed is reported as unavailable **with a
reason**, never as a zero and never as an empty chart. Those two failure modes
are what make dashboards dangerous: a zero drawdown and a blank pie both look
like "nothing to report", and both are indistinguishable from a bug.

Framework-free on purpose. This module imports nothing from ``fastapi``, so
the same code backs the API, the tests, and a CLI report.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.quant import risk_metrics as rm
from quantforge.quant import statistics as st


#: Below this many observations a ratio is reported but flagged. A Sharpe ratio
#: from 12 daily returns is a number, not an estimate.
MIN_OBS_FOR_RATIO = 30

#: Below this the correlation matrix is not worth rendering.
MIN_OBS_FOR_CORRELATION = 20


@dataclass(slots=True)
class Component:
    """One dashboard component, with the basis needed to read it honestly."""

    name: str
    available: bool
    reason: str = ""
    data: Any = None
    n_observations: int = 0
    period: str = ""
    caveat: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "available": self.available,
            "reason": self.reason,
            "data": _jsonable(self.data),
            "n_observations": self.n_observations,
            "period": self.period,
            "caveat": self.caveat,
        }


def _unavailable(name: str, reason: str) -> Component:
    return Component(name=name, available=False, reason=reason)


def _equity_series(portfolio: Any) -> Optional[pd.Series]:
    """The portfolio's equity curve, or ``None`` if there isn't one.

    ``Portfolio.equity_series`` is a *method*. Reading it with ``getattr`` and
    forgetting to call it yields the bound method, and ``pd.Series(<bound
    method>)`` builds a one-element Series holding the method's repr rather
    than raising. Every component that consumes the curve would then report
    "no equity curve" for a portfolio that has one, which is exactly the kind
    of failure a dashboard hides: it looks like an empty account.

    Accepts either a method or an already-materialised Series, because a caller
    may reasonably hand in either.
    """
    raw = getattr(portfolio, "equity_series", None)
    if raw is None:
        return None
    if callable(raw):
        try:
            raw = raw()
        except TypeError:
            return None
    if isinstance(raw, pd.Series):
        return raw
    try:
        return pd.Series(raw)
    except (TypeError, ValueError):
        return None


def _jsonable(obj: Any) -> Any:
    """Make numpy/pandas types survive JSON serialisation.

    A ``NaN`` in a payload is a silent trap: ``json.dumps`` emits a bare
    ``NaN`` literal, which is invalid JSON and which some clients parse into a
    number. Undefined becomes ``null`` here, at the boundary, once.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, dt.datetime)):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    return str(obj)


def _years_covered(series: pd.Series, periods_per_year: int) -> tuple[float, str]:
    """Years a series actually spans, and how that was determined.

    Returns ``(years, basis)``. The basis string travels into the report so a
    reader can tell a date-derived span from an assumed one.
    """
    n = len(series)
    if n == 0:
        return 0.0, "no observations"
    fallback = max(n / periods_per_year, 1e-9)
    try:
        idx = pd.DatetimeIndex(series.index)
    except (TypeError, ValueError):
        return fallback, (f"assumed {periods_per_year} periods per year, because "
                          "the index carries no dates")
    if len(idx) < 2 or idx[-1] <= idx[0]:
        return fallback, (f"assumed {periods_per_year} periods per year, because "
                          "the index has no measurable span")
    span_years = (idx[-1] - idx[0]).total_seconds() / (86400.0 * 365.25)
    if span_years <= 0:
        return fallback, "assumed, because the measured span was zero"
    return span_years, "measured from the index dates"


def _period_of(index: Any) -> str:
    """Describe the span a series covers, or say why it cannot."""
    try:
        idx = pd.DatetimeIndex(index)
    except (TypeError, ValueError):
        return ""
    if len(idx) < 2:
        return ""
    span = idx[-1] - idx[0]
    days = span.total_seconds() / 86400.0
    if days < 1:
        return f"{len(idx)} periods"
    if days < 60:
        return f"{days:.0f} days"
    if days < 730:
        return f"{days / 30.44:.0f} months"
    return f"{days / 365.25:.1f} years"


# ==========================================================================
# Performance components
# ==========================================================================


def equity_curve_component(
    portfolio: Any,
    benchmark: Optional[pd.Series] = None,
) -> Component:
    """Equity curve, normalised against a benchmark when one is supplied.

    Normalisation matters: two equity curves on different starting capital
    cannot be overlaid, and comparing a curve that starts at 1,000,000 with one
    that starts at 10,000 produces a chart whose vertical axis means nothing.
    Both are rebased to 1.0 so the comparison is of *returns*, which is what a
    reader is actually looking at.

    Without a benchmark the curve is still shown, but the component says so.
    A strategy's own curve is not evidence it beat anything.
    """
    series = _equity_series(portfolio)
    if series is None:
        return _unavailable("equity_curve",
                            "the portfolio has no recorded equity curve")
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return _unavailable(
            "equity_curve",
            f"only {len(s)} equity point(s) recorded; a curve needs at least 2")

    rebased = s / s.iloc[0]
    data: dict[str, Any] = {
        "dates": [str(d) for d in s.index],
        "equity": [float(v) for v in rebased],
        "start_equity": float(s.iloc[0]),
        "end_equity": float(s.iloc[-1]),
    }
    caveat = "rebased to 1.0 at the first point"
    n = len(s)

    if benchmark is not None:
        b = pd.Series(benchmark).dropna()
        if len(b) < 2:
            return Component("equity_curve", True, "", data, n,
                             _period_of(s.index),
                             caveat + "; benchmark supplied but too short to use")
        # Align on index, then align on dates. A benchmark indexed by position
        # would silently pair the wrong bar with the wrong bar.
        b = b.reindex(s.index).ffill()
        if b.notna().sum() < 2:
            common = pd.Series(b.index).isin(s.index).sum()
            return Component("equity_curve", True, "", data, n, _period_of(s.index),
                             caveat + f"; benchmark does not overlap ({common} "
                                       "matching points)")
        b = b.dropna()
        common_index = b.index.intersection(s.index)
        data["benchmark"] = [float(v) for v in (b.loc[common_index] /
                                               b.loc[common_index].iloc[0])]
        data["benchmark_dates"] = [str(d) for d in common_index]
        caveat += ", compared against the benchmark over their common dates"
        n = min(n, len(b))
    else:
        caveat += "; no benchmark supplied, so this shows what happened, not " \
                  "whether it beat anything"

    return Component("equity_curve", True, "", data, n, _period_of(s.index), caveat)


def daily_pnl_component(portfolio: Any) -> Component:
    """Per-period P&L with the cumulative path, as currency not fractions.

    Currency rather than percent because the question a P&L chart answers is
    "how much money moved", and a percentage chart makes a 2% loss on 10,000
    look identical to a 2% loss on 10,000,000.
    """
    series = _equity_series(portfolio)
    if series is None:
        return _unavailable("daily_pnl", "the portfolio has no equity curve")
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return _unavailable("daily_pnl",
                            f"only {len(s)} equity point(s); a change needs 2")
    delta = s.diff().dropna()
    return Component(
        "daily_pnl", True, "",
        {
            "dates": [str(d) for d in delta.index],
            "pnl": [float(v) for v in delta],
            "cumulative": [float(v) for v in s - s.iloc[0]],
        },
        len(delta), _period_of(s.index),
        "period-on-period change in equity, in the portfolio's base currency",
    )


def drawdown_component(portfolio: Any) -> Component:
    """Underwater chart plus the max drawdown and its dates.

    The current drawdown is reported next to the maximum. A portfolio 0.1% under
    its peak reads very differently when the worst drawdown was 40%.
    """
    series = _equity_series(portfolio)
    if series is None:
        return _unavailable("drawdown", "the portfolio has no equity curve")
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return _unavailable("drawdown",
                            f"only {len(s)} equity point(s); a drawdown needs 2")
    if (s <= 0).any():
        return _unavailable(
            "drawdown",
            "equity reached zero or below, so a percentage drawdown is not defined")

    dd = rm.drawdown_series(s)
    res = rm.max_drawdown(s)
    return Component(
        "drawdown", True, "",
        {
            "dates": [str(d) for d in dd.index],
            "drawdown": [float(v) for v in dd],
            "max_drawdown": float(res.max_drawdown),
            "current_drawdown": float(res.current_drawdown),
            "peak": str(res.peak_index),
            "trough": str(res.trough_index),
            "recovery": str(res.recovery_index) if res.recovery_index is not None else None,
            "time_underwater_periods": res.time_underwater,
        },
        len(s), _period_of(s.index),
        "underwater means below the previous peak; an unrecovered drawdown has "
        "no end date",
    )


# ==========================================================================
# Positions
# ==========================================================================


def positions_component(portfolio: Any) -> Component:
    """Every open position with entry, current price, and P&L."""
    positions = getattr(portfolio, "positions", None)
    if positions is None:
        return _unavailable("positions", "the portfolio has no position book")
    rows = []
    for p in positions.values():
        if p.is_flat:
            continue
        rows.append({
            "instrument_id": p.instrument_id,
            "quantity": float(p.quantity),
            "direction": "long" if p.is_long else "short",
            "average_entry_price": float(p.average_entry_price),
            "last_price": float(p.last_price),
            "market_value": float(p.market_value),
            "cost_basis": float(p.cost_basis),
            "unrealized_pnl": float(p.unrealized_pnl),
            "unrealized_pct": (float(p.unrealized_pnl / p.cost_basis)
                               if p.cost_basis else None),
            "realized_pnl": float(p.realized_pnl),
            "fees": float(p.total_fees),
            "currency": p.currency,
            "strategy_id": p.strategy_id,
        })
    rows.sort(key=lambda r: -abs(r["market_value"]))
    if not rows:
        return Component("positions", True, "", [], 0, "",
                         "no open positions; this is a flat book, not a failure "
                         "to read it")
    return Component("positions", True, "", rows, len(rows), "",
                     "market value uses the last marked price, which is stale "
                     "between marks")


def realized_pnl_component(portfolio: Any) -> Component:
    """Closed trades, and the cumulative realised P&L path."""
    trades = list(getattr(portfolio, "trades", []) or [])
    if not trades:
        return Component("realized_pnl", True, "", [], 0, "",
                         "no closed trades yet")
    rows = []
    cumulative = 0.0
    for t in trades:
        cumulative += float(t.pnl) - float(t.fees)
        rows.append({
            "instrument_id": t.instrument_id,
            "entry_time": str(t.entry_time),
            "exit_time": str(t.exit_time),
            "quantity": float(t.quantity),
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price),
            "pnl": float(t.pnl),
            "fees": float(t.fees),
            "net_pnl": float(t.pnl) - float(t.fees),
            "return_pct": float(t.return_pct),
            "bars_held": int(t.bars_held),
            "exit_reason": t.exit_reason,
            "mae": float(t.mae),
            "mfe": float(t.mfe),
            "cumulative_net_pnl": cumulative,
        })
    wins = sum(1 for r in rows if r["net_pnl"] > 0)
    return Component(
        "realized_pnl", True, "", rows, len(rows), "",
        f"net of fees; {wins} of {len(rows)} trades profitable",
    )


# ==========================================================================
# Exposure
# ==========================================================================


def exposure_component(portfolio: Any) -> Component:
    """Gross, net, and leverage, with the denominators stated.

    Gross and net are reported as fractions *of equity* and also as currency,
    because a leverage of 1.8 means nothing without knowing whether equity was
    10,000 or 10,000,000.
    """
    positions = getattr(portfolio, "positions", None)
    if positions is None:
        return _unavailable("exposure", "the portfolio has no position book")
    equity = float(portfolio.equity)
    if equity == 0:
        return _unavailable("exposure", "equity is zero, so exposure ratios are "
                                        "undefined")
    gross = float(portfolio.gross_exposure)
    net = float(portfolio.net_exposure)
    return Component(
        "exposure", True, "",
        {
            "equity": equity,
            "cash": float(portfolio.cash),
            "gross_exposure": gross,
            "net_exposure": net,
            "gross_pct_equity": gross / equity,
            "net_pct_equity": net / equity,
            "leverage": gross / equity,
            "market_value": float(portfolio.market_value),
            "position_count": int(portfolio.position_count),
        },
        int(portfolio.position_count), "",
        "gross counts long and short at full size; net nets them. A book that "
        "is 200% long and 200% short is 200% gross and 0% net, and those are "
        "very different risks",
    )


def exposure_breakdown(
    portfolio: Any,
    key_fn: Callable[[Any], Optional[str]],
    component_name: str,
    label: str,
) -> Component:
    """Group exposure by any instrument attribute, reporting coverage.

    The coverage figure is the point. Grouping by sector when only 40% of
    positions carry a sector label silently redistributes that missing 60%, and
    a pie chart does not show you the missing slice. This reports what fraction
    of exposure was actually classified.
    """
    positions = getattr(portfolio, "positions", None)
    if positions is None:
        return _unavailable(component_name, "the portfolio has no position book")
    open_positions = [p for p in positions.values() if not p.is_flat]
    if not open_positions:
        return Component(component_name, True, "", [], 0, "",
                         "no open positions, so there is no exposure to group")

    groups: dict[str, float] = {}
    unclassified = 0.0
    for p in open_positions:
        key = key_fn(p)
        if key is None:
            unclassified += abs(p.market_value)
        else:
            groups[str(key)] = groups.get(str(key), 0.0) + abs(p.market_value)

    total = sum(groups.values()) + unclassified
    if total == 0:
        return _unavailable(component_name,
                            "all positions have zero market value, so exposure "
                            "cannot be grouped")
    classified = total - unclassified
    coverage = classified / total

    rows = [
        {"label": k, "exposure": v, "pct_of_total": v / total}
        for k, v in sorted(groups.items(), key=lambda kv: -kv[1])
    ]
    if unclassified > 0:
        rows.append({
            "label": "UNCLASSIFIED",
            "exposure": unclassified,
            "pct_of_total": unclassified / total,
        })

    caveat = f"{coverage:.1%} of exposure carries a {label}"
    if coverage < 1.0:
        caveat += (f"; the {unclassified / total:.1%} without one is shown "
                   "separately rather than being folded into a group it does "
                   "not belong to")
    if coverage == 0.0:
        return _unavailable(
            component_name,
            f"no position carries a {label}. This instrument set may not have "
            f"such a classification - a currency pair or a bond has no sector, "
            f"and assigning one would be an invention.")
    return Component(component_name, True, "", rows, len(open_positions), "", caveat)


# ==========================================================================
# Risk
# ==========================================================================


def var_component(
    portfolio: Any,
    confidence: float = 0.95,
    horizon_periods: int = 1,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> Component:
    """VaR and expected shortfall with method, confidence, and sample size.

    VaR and ES are reported together on purpose. VaR states the loss that is
    *not* expected to be exceeded; it says nothing about how much worse the
    losses beyond it are. ES answers that question, and quoting VaR alone is
    the standard way to understate tail risk.
    """
    series = _equity_series(portfolio)
    if series is None:
        return _unavailable("var", "the portfolio has no equity curve")
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return _unavailable("var", f"only {len(s)} equity point(s); a tail "
                                    "measure needs a return distribution")
    returns = s.pct_change(fill_method=None).dropna()
    if len(returns) < MIN_OBS_FOR_RATIO:
        return _unavailable(
            "var",
            f"{len(returns)} return observation(s). A tail percentile is not "
            f"meaningful below about {MIN_OBS_FOR_RATIO}; the number would be "
            "read as precise and would not be"
        )
    if st.is_constant(returns):
        return _unavailable("var",
                            "returns are constant, so there is no distribution "
                            "to take a percentile of")

    v = rm.var(returns, confidence=confidence, horizon_periods=horizon_periods,
               dataset="portfolio equity curve")
    es = rm.historical_expected_shortfall(returns, confidence=confidence)
    equity = float(s.iloc[-1])
    return Component(
        "var", True, "",
        {
            "var_return": float(v.value),
            "var_currency": float(v.value) * equity,
            "es_return": float(es),
            "es_currency": float(es) * equity,
            "confidence": float(v.confidence),
            "horizon_periods": int(v.horizon_periods),
            "method": v.method.value if hasattr(v.method, "value") else str(v.method),
            "equity_at_measurement": equity,
        },
        int(v.n_observations), _period_of(s.index),
        "historical simulation. VaR is the loss not expected to be exceeded; "
        "it does not describe how much worse the tail is, which is what ES "
        "measures. Neither is a maximum loss",
    )


def correlation_component(
    returns_by_instrument: Mapping[str, pd.Series],
    min_observations: int = MIN_OBS_FOR_CORRELATION,
) -> Component:
    """Correlation heatmap over the instruments actually held.

    Unavailable when there is too little overlap. A correlation matrix
    computed from a handful of overlapping observations produces confident
    numbers that are meaningless, and a heatmap makes that especially easy to
    miss.
    """
    if len(returns_by_instrument) < 2:
        return _unavailable(
            "correlations",
            f"{len(returns_by_instrument)} instrument(s) supplied; a correlation "
            "matrix needs at least 2"
        )
    frame = pd.DataFrame(dict(returns_by_instrument)).dropna(how="any")
    n = len(frame)
    if n < min_observations:
        return _unavailable(
            "correlations",
            f"only {n} periods where all {len(returns_by_instrument)} "
            f"instruments have data; below {min_observations} the correlations "
            "would be noise presented as structure"
        )
    matrix = frame.corr()
    if np.isnan(matrix.to_numpy()).any():
        # Constant columns have no correlation with anything, including
        # themselves. Dropping them is better than rendering a NaN cell that
        # a chart will draw as zero.
        keep = [c for c in matrix.columns if matrix[c].notna().any()]
        if len(keep) < 2:
            return _unavailable(
                "correlations",
                "every instrument has a constant return series, so no "
                "correlation is defined. Constant returns are usually a data "
                "error rather than a market condition"
            )
        matrix = matrix.loc[keep, keep]
        note = f"{len(returns_by_instrument) - len(keep)} instrument(s) omitted " \
               "for having constant returns, which have no correlation with " \
               "anything"
        n = len(frame)
    else:
        note = ""
    return Component(
        "correlations", True, "",
        {
            "instruments": [str(c) for c in matrix.columns],
            "matrix": [[float(v) for v in row] for row in matrix.to_numpy()],
        },
        n, "",
        "Pearson correlation of overlapping period returns"
        + (f"; {note}" if note else ""),
    )


def summary_statistics_component(
    portfolio: Any,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> Component:
    """Sharpe, Sortino, Calmar, and the counts needed to judge them.

    A Sharpe of 3.0 from 8 observations and one from 2,000 are printed the
    same way by most tools. The observation count travels with the value here,
    and small samples are flagged rather than left for the reader to notice.
    """
    series = _equity_series(portfolio)
    if series is None:
        return _unavailable("summary_statistics", "the portfolio has no equity curve")
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return _unavailable("summary_statistics",
                            f"only {len(s)} equity point(s)")
    returns = s.pct_change(fill_method=None).dropna()
    n = len(returns)
    if n == 0:
        return _unavailable("summary_statistics",
                            "the equity curve is flat, so there are no returns "
                            "to characterise")
    if st.is_constant(returns):
        return _unavailable(
            "summary_statistics",
            "returns are constant. A constant non-zero return gives an infinite "
            "Sharpe, and a constant zero return gives an undefined one; neither "
            "is a risk-adjusted performance figure"
        )

    sharpe = rm.sharpe_ratio(returns, risk_free=risk_free,
                             periods_per_year=periods_per_year)
    # Sortino takes a minimum acceptable return, not a risk-free rate. They are
    # numerically the same at a zero MAR, and the argument name is what stops a
    # reader assuming a risk-free rate was used when it was not.
    sortino = rm.sortino_ratio(returns, mar=risk_free,
                               periods_per_year=periods_per_year)
    dd = rm.max_drawdown(s)
    calmar = rm.calmar_ratio(s, periods_per_year=periods_per_year,
                             is_equity=True)

    total_return = float(s.iloc[-1] / s.iloc[0] - 1.0)

    # Years must come from the actual span of the index, not from the
    # observation count. An equity curve sampled every 10th bar has far fewer
    # points than it has days, so `n / periods_per_year` treats two years of
    # history as two months and inflates the annualised return by an order of
    # magnitude. When the index is not a date index there is no span to read,
    # and the observation count is the only estimate available - the caveat
    # then says so rather than leaving the number unqualified.
    years, basis = _years_covered(s, periods_per_year)
    trades = list(getattr(portfolio, "trades", []) or [])
    net_pnls = [float(t.pnl) - float(t.fees) for t in trades]

    data: dict[str, Any] = {
        "total_return": total_return,
        "annualised_return": ((1.0 + total_return) ** (1.0 / years) - 1.0
                              if years > 0 and total_return > -1.0 else None),
        "volatility": float(returns.std(ddof=1) * math.sqrt(periods_per_year))
                      if n > 1 else None,
        "sharpe": float(sharpe.value),
        "sharpe_caveat": sharpe.caveat,
        "sortino": float(sortino.value),
        "calmar": float(calmar.value),
        "max_drawdown": float(dd.max_drawdown),
        "current_drawdown": float(dd.current_drawdown),
        "n_trades": len(trades),
        "years_covered": years,
        "annualisation_basis": basis,
    }
    if net_pnls:
        data["win_rate"] = rm.win_rate(net_pnls).value
        data["profit_factor"] = float(rm.profit_factor(net_pnls).value)

    caveats = [f"{n} return observations", f"annualised over {years:.2f} years "
              f"({basis})"]
    if n < MIN_OBS_FOR_RATIO:
        caveats.append(
            f"FEWER THAN {MIN_OBS_FOR_RATIO} OBSERVATIONS: these ratios are "
            "dominated by sampling error and should not be compared against "
            "ratios from longer samples")
    if returns.std(ddof=1) == 0:
        caveats.append("zero volatility makes the ratios meaningless")
    if calmar.value == 0 or (isinstance(calmar.value, float)
                             and not math.isfinite(calmar.value)):
        caveats.append("Calmar is undefined when there is no drawdown to recover from")
    if len(trades) < 30:
        caveats.append(f"only {len(trades)} closed trade(s); per-trade statistics "
                       "are not meaningful at this count")

    return Component("summary_statistics", True, "", data, n,
                     _period_of(s.index), "; ".join(caveats))


def beta_component(
    portfolio: Any,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> Component:
    """Beta against a benchmark, with the overlap stated.

    Unavailable without a benchmark, and that is the point: a strategy with no
    benchmark has no beta, and reporting a number against an implied or assumed
    market would be inventing the reference.
    """
    if benchmark_returns is None:
        return _unavailable(
            "beta",
            "no benchmark supplied. Beta is a statement about behaviour relative "
            "to a reference; without one there is nothing to be relative to")
    series = _equity_series(portfolio)
    if series is None:
        return _unavailable("beta", "the portfolio has no equity curve")
    s = pd.Series(series).dropna()
    b = pd.Series(benchmark_returns).dropna()
    if len(s) < 2 or len(b) < 2:
        return _unavailable("beta", "not enough observations on either side")

    r = s.pct_change(fill_method=None).dropna()
    b = b.reindex(r.index).ffill().dropna()
    common = r.index.intersection(b.index)
    if len(common) < MIN_OBS_FOR_RATIO:
        return _unavailable(
            "beta",
            f"only {len(common)} periods where both the portfolio and the "
            f"benchmark have returns; below {MIN_OBS_FOR_RATIO} a regression "
            "slope is not distinguishable from noise"
        )
    r = r.loc[common]
    b = b.loc[common]
    if st.is_constant(b):
        return _unavailable("beta", "benchmark returns are constant, so no "
                                    "slope can be estimated")

    beta = st.beta(r, b, risk_free=risk_free)
    alpha = st.alpha(r, b, periods_per_year=periods_per_year,
                     risk_free_annual=risk_free)
    return Component(
        "beta", True, "",
        {
            "beta": float(beta),
            "alpha": float(alpha),
            "n_overlapping_periods": len(common),
            "benchmark_volatility": float(b.std(ddof=1) * math.sqrt(periods_per_year)),
        },
        len(common), _period_of(common),
        f"estimated on {len(common)} overlapping periods. Alpha is the "
        "annualised return not explained by the benchmark, and it is the "
        "larger number to distrust: it is a residual from a regression, so it "
        "absorbs everything the model left out",
    )


# ==========================================================================
# Report
# ==========================================================================


@dataclass(slots=True)
class DashboardReport:
    """All 14 components plus the disclosure that belongs above them."""

    components: dict[str, Component] = field(default_factory=dict)
    generated_at: dt.datetime = field(default_factory=dt.datetime.now)
    base_currency: str = "USD"

    COMPONENT_NAMES = (
        "equity_curve", "daily_pnl", "unrealized_pnl", "realized_pnl",
        "exposure", "positions", "sector_exposure", "asset_class_exposure",
        "correlations", "portfolio_volatility", "var", "drawdown",
        "summary_statistics", "beta",
    )

    def __getitem__(self, key: str) -> Component:
        return self.components[key]

    def available_components(self) -> list[str]:
        return [k for k in self.COMPONENT_NAMES
                if k in self.components and self.components[k].available]

    def unavailable_components(self) -> list[Component]:
        return [self.components[k] for k in self.COMPONENT_NAMES
                if k in self.components and not self.components[k].available]

    def as_dict(self) -> dict:
        return {
            "generated_at": str(self.generated_at),
            "base_currency": self.base_currency,
            "disclosure": (
                "Historical figures describe the tested period under the stated "
                "assumptions. Past performance does not predict future returns, "
                "and no figure here is a forecast. A backtest is evidence about "
                "a period, not a claim about what comes next."
            ),
            "n_available": len(self.available_components()),
            "n_total": len(self.COMPONENT_NAMES),
            "components": {k: v.as_dict() for k, v in self.components.items()},
        }


def build_dashboard(
    portfolio: Any,
    instruments: Optional[Mapping[str, Any]] = None,
    benchmark_equity: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    returns_by_instrument: Optional[Mapping[str, pd.Series]] = None,
    confidence: float = 0.95,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> DashboardReport:
    """Assemble all 14 components.

    Every component is attempted. A component that cannot be computed is
    recorded as unavailable with the reason, never omitted - a reader who sees
    13 of 14 components can tell something is missing, whereas a reader who
    sees 13 rows has no way to know a 14th was expected.
    """
    instruments = dict(instruments or {})

    def asset_class_of(position: Any) -> Optional[str]:
        inst = instruments.get(position.instrument_id)
        ac = getattr(inst, "asset_class", None)
        return getattr(ac, "value", ac) if ac is not None else None

    def sector_of(position: Any) -> Optional[str]:
        inst = instruments.get(position.instrument_id)
        return getattr(inst, "sector", None) if inst is not None else None

    report = DashboardReport(
        base_currency=getattr(portfolio, "base_currency", "USD")
    )
    report.components["equity_curve"] = equity_curve_component(
        portfolio, benchmark_equity)
    report.components["daily_pnl"] = daily_pnl_component(portfolio)
    report.components["drawdown"] = drawdown_component(portfolio)
    report.components["positions"] = positions_component(portfolio)
    report.components["realized_pnl"] = realized_pnl_component(portfolio)
    report.components["exposure"] = exposure_component(portfolio)
    report.components["sector_exposure"] = exposure_breakdown(
        portfolio, sector_of, "sector_exposure", "sector")
    report.components["asset_class_exposure"] = exposure_breakdown(
        portfolio, asset_class_of, "asset_class_exposure", "asset class")
    report.components["var"] = var_component(
        portfolio, confidence=confidence, risk_free=risk_free,
        periods_per_year=periods_per_year)
    report.components["summary_statistics"] = summary_statistics_component(
        portfolio, risk_free=risk_free, periods_per_year=periods_per_year)
    report.components["beta"] = beta_component(
        portfolio, benchmark_returns, risk_free=risk_free,
        periods_per_year=periods_per_year)
    report.components["correlations"] = correlation_component(
        returns_by_instrument or {})

    # Unrealised P&L and rolling volatility are the two components that need
    # their own inputs rather than only the portfolio.
    series = _equity_series(portfolio)
    if series is not None and len(pd.Series(series).dropna()) >= 1:
        report.components["unrealized_pnl"] = Component(
            "unrealized_pnl", True, "",
            {
                "total": float(portfolio.unrealized_pnl),
                "realized": float(portfolio.realized_pnl),
                "fees_paid": float(portfolio.total_fees),
                "equity": float(portfolio.equity),
            },
            int(portfolio.position_count), "",
            "unrealised P&L is marked at the last price seen, so between marks "
            "it does not reflect the current market",
        )
    else:
        report.components["unrealized_pnl"] = _unavailable(
            "unrealized_pnl", "the portfolio has no equity curve")

    s = pd.Series(series).dropna() if series is not None else pd.Series(dtype=float)
    if len(s) >= 3:
        returns = s.pct_change(fill_method=None).dropna()
        window = min(60, max(5, len(returns) // 2))
        rolling = (returns.rolling(window, min_periods=window).std()
                   * math.sqrt(periods_per_year))
        report.components["portfolio_volatility"] = Component(
            "portfolio_volatility", True, "",
            {
                "dates": [str(d) for d in rolling.dropna().index],
                "volatility": [float(v) for v in rolling.dropna()],
                "window": window,
                "current": float(rolling.dropna().iloc[-1]) if rolling.notna().any() else None,
            },
            int(rolling.notna().sum()), _period_of(s.index),
            f"{window}-period rolling standard deviation, annualised. The first "
            f"{window - 1} periods have no value, which is why the series is "
            "shorter than the equity curve",
        )
    else:
        report.components["portfolio_volatility"] = _unavailable(
            "portfolio_volatility",
            f"{len(s)} equity point(s); rolling volatility needs a window of "
            "returns before it has anything to report")

    return report
