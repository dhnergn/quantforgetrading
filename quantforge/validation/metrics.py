"""Performance metrics.

The governing rule from blueprint Section 45: **an undefined metric is NaN, not
zero.** A Sortino of infinity, a Sharpe of zero on a constant series, and a
Calmar of zero with no drawdown all read as "ordinary results" in a table while
meaning something entirely different. Each such case returns NaN and carries a
caveat explaining the condition, so a report shows the absence of information
rather than a fabricated number.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.quant import risk_metrics as rm
from quantforge.quant.returns import annualized_return as _ann_return
from quantforge.quant.returns import simple_returns

__all__ = [
    "PerformanceMetrics", "compute_metrics", "metrics_table",
    "annualized_return", "sharpe", "sortino", "calmar", "profit_factor",
    "win_rate", "drawdown_metrics", "compare_to_benchmark",
]

#: Below this many return observations, annualised statistics are unreliable
#: enough that they should be treated as indicative at best.
MIN_PERIODS_FOR_ANNUALISATION = 30


def _finite(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _fmt(v: Optional[float], pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.2%}" if pct else f"{v:.4f}"


@dataclass(slots=True)
class MetricEntry:
    name: str
    value: Optional[float]
    display: str
    category: str
    caveat: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name, "value": self.value, "display": self.display,
            "category": self.category, "caveat": self.caveat,
        }


@dataclass(slots=True)
class PerformanceMetrics:
    """A full metric set. Undefined metrics are ``None``, never 0."""

    entries: list[MetricEntry] = field(default_factory=list)
    n_periods: int = 0
    years: float = 0.0
    periods_per_year: int = 252
    warnings: list[str] = field(default_factory=list)

    def add(self, name, value, category, caveat="", pct=False, dp=4) -> None:
        v = _finite(value) if value is not None else None
        self.entries.append(
            MetricEntry(name, v, _fmt(v, pct) if v is not None else "n/a",
                        category, caveat)
        )

    def get(self, name: str) -> Optional[float]:
        for e in self.entries:
            if e.name == name:
                return e.value
        return None

    def as_dict(self) -> dict:
        return {e.name: e.value for e in self.entries}

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"metric": e.name, "value": e.display, "category": e.category,
                 "caveat": e.caveat}
                for e in self.entries
            ]
        )

    def summary(self) -> str:
        lines = ["=" * 68, "PERFORMANCE METRICS", "=" * 68,
                 f"Periods: {self.n_periods}   Years: {self.years:.2f}   "
                 f"Annualisation: {self.periods_per_year}/year", ""]
        for cat in ("returns", "risk", "risk_adjusted", "trading", "drawdown"):
            rows = [e for e in self.entries if e.category == cat]
            if not rows:
                continue
            lines.append(f"{cat.upper()}")
            for e in rows:
                lines.append(f"  {e.name:<26} {e.display:>12}")
            lines.append("")
        if self.warnings:
            lines.append("CAVEATS")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("")
        lines.append(
            "These are statistics computed on a historical sample under stated "
            "assumptions. They are not a forecast and do not indicate that the "
            "strategy will be profitable in future."
        )
        lines.append("=" * 68)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------


def compute_metrics(
    returns: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    equity: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    risk_free_annual: float = 0.0,
    var_confidence: float = 0.95,
    initial_capital: Optional[float] = None,
) -> PerformanceMetrics:
    """Compute the full metric set from a return series.

    ``returns`` is the strategy's net return series, i.e. **after** costs. The
    caller is responsible for that: computing metrics on a gross return series
    and separately reporting costs is how a backtest convinces itself it is
    profitable when it is not.
    """
    m = PerformanceMetrics(periods_per_year=periods_per_year)
    r = pd.Series(returns).astype("float64").dropna()
    m.n_periods = int(len(r))
    m.years = m.n_periods / periods_per_year if periods_per_year else 0.0

    if r.empty:
        m.warnings.append("no return observations: every metric is undefined")
        for name in ("total_return", "cagr", "volatility", "sharpe", "max_drawdown"):
            m.add(name, None, "returns" if "return" in name else "risk",
                  "no observations")
        return m

    if equity is None:
        equity = (1.0 + r).cumprod()
    equity = equity.reindex(r.index).ffill()

    if m.n_periods < MIN_PERIODS_FOR_ANNUALISATION:
        m.warnings.append(
            f"only {m.n_periods} periods ({m.years:.2f} years). Annualised "
            "statistics on a short sample have very wide confidence intervals "
            "and should not be compared with longer-sample figures."
        )

    # -- returns ----------------------------------------------------------
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    m.add("total_return", total, "returns", pct=True)
    cagr = _ann_return(r, periods_per_year, "simple")
    m.add("cagr", cagr, "returns", pct=True,
          caveat="geometric annualisation; on a short sample this is highly "
                 "sensitive to the endpoints")
    cagr_log = _ann_return(r, periods_per_year, "log")
    m.add("cagr_log", cagr_log, "returns", pct=True,
          caveat="arithmetic mean of log returns annualised; differs from CAGR "
                 "by Jensen's inequality")
    m.add("mean_period_return", float(r.mean()), "returns", pct=True)
    m.add("median_period_return", float(r.median()), "returns", pct=True)
    m.add("best_period", float(r.max()), "returns", pct=True)
    m.add("worst_period", float(r.min()), "returns", pct=True)
    m.add("positive_periods", float((r > 0).sum()) / len(r), "returns", pct=True,
          caveat="fraction of periods with a positive return, not trade win rate")

    if risk_free_annual:
        from quantforge.quant.returns import excess_return

        # excess_return handles the geometric conversion of the annual risk-free
        # rate to a per-period rate. Building the subtraction by hand as
        # `r - (1+rf)**(1/ppy) + 1 - 1` compounds an extra (1 + rf) factor into
        # every period, which inflates the result by orders of magnitude.
        excess = excess_return(r, risk_free_annual, periods_per_year)
        m.add("excess_return_annual",
              float(excess.mean() * periods_per_year), "returns", pct=True,
              caveat=f"arithmetic mean excess return over a {risk_free_annual:.2%} "
                     "risk-free rate, annualised linearly")

    # -- risk -------------------------------------------------------------
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    m.add("volatility", vol, "risk", pct=True,
          caveat="annualised with the sqrt-of-time assumption, which understates "
                 "risk when returns are autocorrelated")
    m.add("variance_annualized", vol**2, "risk")
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(periods_per_year))
    m.add("downside_deviation", downside, "risk", pct=True)
    semi = float(np.mean(np.minimum(r, 0.0) ** 2) * periods_per_year)
    m.add("semivariance_annualized", semi, "risk")
    m.add("skewness", float(pd.Series(r).skew()), "risk",
          caveat="negative skew means occasional large losses relative to gains")
    m.add("excess_kurtosis", float(pd.Series(r).kurt()), "risk",
          caveat="normal is 0; positive means fat tails, so Gaussian risk models "
                 "understate the probability of extreme outcomes")

    # -- risk-adjusted ----------------------------------------------------
    sh = rm.sharpe_ratio(r, risk_free_annual, periods_per_year)
    m.add("sharpe", sh.value, "risk_adjusted", caveat=sh.caveat)
    so = rm.sortino_ratio(r, 0.0, periods_per_year)
    m.add("sortino", so.value, "risk_adjusted", caveat=so.caveat)
    m.add("calmar", rm.calmar_ratio(equity, periods_per_year, is_equity=True).value,
          "risk_adjusted", caveat="CAGR divided by maximum drawdown")
    m.add("return_to_vol", float(cagr / vol) if vol > 0 and np.isfinite(cagr) else float("nan"),
          "risk_adjusted", caveat="CAGR / annualised volatility; a trailing, "
                                  "end-point-sensitive ratio")

    # -- drawdown ---------------------------------------------------------
    dd = rm.max_drawdown(equity)
    m.add("max_drawdown", dd.max_drawdown, "drawdown", pct=True)
    m.add("current_drawdown", dd.current_drawdown, "drawdown", pct=True)
    m.add("drawdown_duration_periods", dd.duration_periods, "drawdown",
          caveat="None when the drawdown has not yet recovered; truncated "
                 "duration understates the true recovery time")
    m.add("time_underwater_periods", dd.time_underwater, "drawdown",
          caveat="periods since the last equity peak")

    # -- tail risk --------------------------------------------------------
    v = rm.historical_var(r, var_confidence, 1, "strategy net returns")
    m.add(f"var_{int(var_confidence * 100)}", v.value, "risk", pct=True,
          caveat=v.caveat)
    es = rm.historical_expected_shortfall(r, var_confidence, 1, "strategy net returns")
    m.add(f"cvar_{int(var_confidence * 100)}", es.value, "risk", pct=True,
          caveat=es.caveat)
    m.add("var_95_1d", rm.historical_var(r, 0.95, 1, "strategy net returns").value,
          "risk", pct=True)

    # -- trading statistics -----------------------------------------------
    if trades is not None and len(trades):
        pnls = pd.Series(trades["pnl"]).astype("float64") if "pnl" in trades else None
        if pnls is not None and len(pnls):
            pf = rm.profit_factor(pnls)
            m.add("profit_factor", pf.value, "trading", caveat=pf.caveat)
            wr = rm.win_rate(pnls)
            m.add("win_rate", wr.value, "trading", pct=True, caveat=wr.caveat)
            m.add("n_trades", float(len(pnls)), "trading")
            m.add("mean_trade_pnl", float(pnls.mean()), "trading")
            m.add("median_trade_pnl", float(pnls.median()), "trading")
            m.add("avg_win", float(pnls[pnls > 0].mean()) if (pnls > 0).any() else float("nan"),
                  "trading")
            m.add("avg_loss", float(pnls[pnls < 0].mean()) if (pnls < 0).any() else float("nan"),
                  "trading")
            gross_win = float(pnls[pnls > 0].sum())
            gross_loss = float(-pnls[pnls < 0].sum())
            m.add("payoff_ratio",
                  gross_win / gross_loss if gross_loss > 0 else float("nan"),
                  "trading", caveat="average win / average loss")
            if "bars_held" in trades:
                m.add("avg_bars_held", float(pd.Series(trades["bars_held"]).mean()),
                      "trading")
        if "entry_time" in trades and "exit_time" in trades:
            holding = (pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
                       - pd.to_datetime(trades["entry_time"], utc=True, errors="coerce"))
            days = holding.dt.total_seconds() / 86400.0
            if days.notna().any():
                m.add("avg_holding_days", float(days.mean()), "trading")
                m.add("median_holding_days", float(days.median()), "trading")
    else:
        m.warnings.append(
            "no trade list supplied: win rate, profit factor and trade counts are "
            "unavailable. Period-level statistics are not a substitute for "
            "trade-level ones."
        )
    return m


def annualized_return(returns, periods_per_year: int = 252) -> float:
    return _ann_return(pd.Series(returns), periods_per_year, "simple")


def sharpe(returns, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    return rm.sharpe_ratio(pd.Series(returns), risk_free, periods_per_year).value


def sortino(returns, mar: float = 0.0, periods_per_year: int = 252) -> float:
    return rm.sortino_ratio(pd.Series(returns), mar, periods_per_year).value


def calmar(equity, periods_per_year: int = 252) -> float:
    return rm.calmar_ratio(pd.Series(equity), periods_per_year, is_equity=True).value


def profit_factor(trade_pnls) -> float:
    return rm.profit_factor(pd.Series(trade_pnls)).value


def win_rate(trade_pnls) -> float:
    return rm.win_rate(pd.Series(trade_pnls)).value


def drawdown_metrics(equity) -> rm.DrawdownResult:
    return rm.max_drawdown(pd.Series(equity))


# --------------------------------------------------------------------------
# Benchmark comparison
# --------------------------------------------------------------------------


def compare_to_benchmark(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_annual: float = 0.0,
) -> dict:
    """Compare a strategy against a benchmark on aligned data.

    The join is on index, and observations where either side is missing are
    dropped. Comparing over different date ranges produces meaningless
    differences in level even when the strategies are identical.
    """
    from quantforge.quant import statistics as st

    r = pd.Series(returns).astype("float64")
    b = pd.Series(benchmark_returns).astype("float64")
    joined = pd.concat([r.rename("strategy"), b.rename("benchmark")], axis=1).dropna()
    n = len(joined)
    if n < 2:
        return {"n_periods": n, "note": "insufficient overlapping observations"}

    rs, rb = joined["strategy"], joined["benchmark"]
    eq_s = (1.0 + rs).cumprod()
    eq_b = (1.0 + rb).cumprod()
    total_s = float(eq_s.iloc[-1] - 1.0)
    total_b = float(eq_b.iloc[-1] - 1.0)

    yrs = n / periods_per_year
    cagr_s = float(eq_s.iloc[-1] ** (1 / yrs) - 1.0) if yrs > 0 else float("nan")
    cagr_b = float(eq_b.iloc[-1] ** (1 / yrs) - 1.0) if yrs > 0 else float("nan")

    dd_s = rm.max_drawdown(eq_s)
    dd_b = rm.max_drawdown(eq_b)
    beta = st.beta(rs, rb, risk_free=None if not risk_free_annual else
                   pd.Series(0.0, index=rs.index))
    return {
        "n_periods": n,
        "years": yrs,
        "strategy_total_return": total_s,
        "benchmark_total_return": total_b,
        "excess_total_return": total_s - total_b,
        "strategy_cagr": cagr_s,
        "benchmark_cagr": cagr_b,
        "excess_cagr": cagr_s - cagr_b,
        "beta": beta,
        "correlation": st.correlation(rs, rb),
        "tracking_error": st.tracking_error(rs, rb, periods_per_year),
        "information_ratio": st.information_ratio(rs, rb, periods_per_year),
        "strategy_max_drawdown": dd_s.max_drawdown,
        "benchmark_max_drawdown": dd_b.max_drawdown,
        "relative_drawdown": dd_s.max_drawdown - dd_b.max_drawdown,
        "strategy_sharpe": rm.sharpe_ratio(rs, risk_free_annual, periods_per_year).value,
        "benchmark_sharpe": rm.sharpe_ratio(rb, risk_free_annual, periods_per_year).value,
        "up_capture": _capture(rs, rb, "up"),
        "down_capture": _capture(rs, rb, "down"),
        "note": (
            "Capture ratios describe behaviour on the benchmark's own up and down "
            "periods. Low downside capture with high upside capture is a favourable "
            "profile; its persistence out of sample is the question to test."
        ),
    }


def _capture(strategy: pd.Series, benchmark: pd.Series, side: str) -> float:
    mask = benchmark > 0 if side == "up" else benchmark < 0
    if mask.sum() < 2:
        return float("nan")
    return float(strategy[mask].mean() / benchmark[mask].mean())


def metrics_table(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    **kwargs,
) -> pd.DataFrame:
    """One metric table, optionally with a benchmark column for comparison."""
    m = compute_metrics(returns, **kwargs)
    table = m.table()
    if benchmark is not None:
        b = compute_metrics(benchmark, **{
            k: v for k, v in kwargs.items() if k in ("periods_per_year", "risk_free_annual")
        })
        bmap = {e.name: e.display for e in b.entries}
        table["benchmark"] = table["metric"].map(bmap)
    return table
