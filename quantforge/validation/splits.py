"""Data splitting, walk-forward analysis, Monte Carlo, and robustness.

The unifying principle across this module: **the final out-of-sample period is
touched exactly once, at the very end.** Every other operation — parameter
search, feature selection, model selection, threshold choice — happens inside
the training or validation windows. A researcher who peeks at the test set and
then adjusts anything has converted it into a validation set, and the reported
performance is no longer out-of-sample regardless of what the report says.
"""
from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.errors import ValidationError
from quantforge.core.reproducibility import set_seed
from quantforge.core.timestamps import UTC


# ==========================================================================
# Splits
# ==========================================================================


@dataclass(slots=True)
class Split:
    """One time window. ``start``/``end`` are inclusive."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def mask(self, index: pd.DatetimeIndex) -> pd.Series:
        return (index >= self.start) & (index <= self.end)

    def slice(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame.index >= self.start][frame.index <= self.end]

    def n(self, index: pd.DatetimeIndex) -> int:
        return int(self.mask(index).sum())

    def __repr__(self) -> str:
        return f"Split({self.name}: {self.start.date()} to {self.end.date()})"


@dataclass(slots=True)
class DataSplit:
    """Training / validation / test partition."""

    train: Split
    validation: Split
    test: Split
    #: True once the test window has been consumed. Set by :meth:`seal_test`.
    test_sealed: bool = False

    def as_dict(self) -> dict:
        return {
            "train": {"start": str(self.train.start), "end": str(self.train.end)},
            "validation": {"start": str(self.validation.start),
                           "end": str(self.validation.end)},
            "test": {"start": str(self.test.start), "end": str(self.test.end)},
            "test_sealed": self.test_sealed,
        }


def _as_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)


def temporal_split(
    index: pd.DatetimeIndex,
    train_frac: float = 0.60,
    validation_frac: float = 0.20,
    embargo_bars: int = 0,
) -> DataSplit:
    """Chronological train/validation/test split.

    ``embargo_bars`` drops a gap between adjacent windows. This matters more
    than it looks: a label with a 5-day horizon at the end of training uses
    prices up to 5 days into the validation period, so training and validation
    share observations and validation performance is partly in-sample. The
    embargo must be at least the label horizon.
    """
    n = len(index)
    if n < 10:
        raise ValidationError(f"need at least 10 observations to split, got {n}")
    for name, frac in (("train_frac", train_frac), ("validation_frac", validation_frac)):
        if not 0 < frac < 1:
            raise ValidationError(f"{name} must be in (0, 1), got {frac}")
    if train_frac + validation_frac >= 1.0:
        raise ValidationError(
            f"train_frac + validation_frac = {train_frac + validation_frac} leaves no "
            "test period; an out-of-sample test is not optional"
        )
    if embargo_bars < 0:
        raise ValidationError("embargo_bars must be >= 0")

    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + validation_frac))

    train_end = i_train - embargo_bars - 1
    val_start = i_train + embargo_bars
    val_end = i_val - embargo_bars - 1
    test_start = i_val + embargo_bars

    # Each window must contain at least one bar. The previous check compared
    # i_val-1 against test_start-1, which rejected every zero-embargo split
    # (959 <= 959) even though the test window was perfectly well formed.
    if train_end < 0:
        raise ValidationError(
            f"training window is empty (n={n}, train_frac={train_frac}, "
            f"embargo={embargo_bars})"
        )
    if val_end < val_start:
        raise ValidationError(
            f"validation window is empty (n={n}, embargo={embargo_bars}); the "
            "embargo must be smaller than the validation fraction"
        )
    if test_start > n - 1:
        raise ValidationError(
            f"test window is empty (n={n}, embargo={embargo_bars}); the embargo "
            "leaves no out-of-sample data, which would leave nothing to test on"
        )

    return DataSplit(
        train=Split("train", index[0], index[train_end]),
        validation=Split("validation", index[val_start], index[val_end]),
        test=Split("test", index[test_start], index[-1]),
    )


def seal_test(split: DataSplit, reason: str = "model selection complete") -> None:
    """Mark the test window as consumed. Called before the final evaluation.

    The point is procedural discipline made explicit: once this is called, any
    subsequent change to features, parameters, or model choice invalidates the
    test result and a new split is required.
    """
    split.test_sealed = True
    split.test.start  # no-op access keeps the intent explicit
    if not reason:
        raise ValidationError("a reason is required when sealing the test set")


# ==========================================================================
# Walk-forward analysis
# ==========================================================================


@dataclass(slots=True)
class WalkForwardWindow:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    n_train: int
    n_valid: int
    n_oos: int
    best_params: dict = field(default_factory=dict)
    train_metrics: dict = field(default_factory=dict)
    valid_metrics: dict = field(default_factory=dict)
    oos_metrics: dict = field(default_factory=dict)
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "train": [str(self.train_start), str(self.train_end)],
            "validation": [str(self.valid_start), str(self.valid_end)],
            "oos": [str(self.oos_start), str(self.oos_end)],
            "n_train": self.n_train, "n_valid": self.n_valid, "n_oos": self.n_oos,
            "best_params": self.best_params,
            "train_metrics": self.train_metrics,
            "valid_metrics": self.valid_metrics,
            "oos_metrics": self.oos_metrics,
            "error": self.error,
        }


@dataclass(slots=True)
class WalkForwardResult:
    windows: list[WalkForwardWindow] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    stability: dict = field(default_factory=dict)
    failed_windows: int = 0

    def frame(self) -> pd.DataFrame:
        if not self.windows:
            return pd.DataFrame()
        return pd.DataFrame([w.as_dict() for w in self.windows])

    def summary(self) -> str:
        lines = ["=" * 72, "WALK-FORWARD ANALYSIS", "=" * 72,
                 f"windows: {len(self.windows)}   failed: {self.failed_windows}"]
        if self.aggregate:
            lines.append("")
            lines.append("AGGREGATE OUT-OF-SAMPLE")
            for k, v in self.aggregate.items():
                lines.append(f"  {k:<28} {v}")
        if self.stability:
            lines.append("")
            lines.append("PARAMETER STABILITY")
            for k, v in self.stability.items():
                lines.append(f"  {k:<28} {v}")
        lines.append("")
        lines.append(
            "Walk-forward performance dispersion across windows is itself a result. "
            "A strategy with a good mean and large dispersion was driven by a few "
            "windows and is not a dependable process."
        )
        lines.append("=" * 72)
        return "\n".join(lines)


def walk_forward(
    index: pd.DatetimeIndex,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame, dict], Any],
    param_grid: Optional[Sequence[dict]] = None,
    train_bars: int = 500,
    validation_bars: int = 100,
    test_bars: int = 100,
    step: Optional[int] = None,
    metric_fn: Optional[Callable[[Any], float]] = None,
    objective: str = "sharpe",
    higher_is_better: bool = True,
) -> WalkForwardResult:
    """Rolling train -> validate -> test, then advance.

    ``fit_predict(train_frame, valid_frame, params)`` must fit on ``train_frame``
    and return a performance score computed on ``valid_frame``. Parameter
    selection uses **only** that score.

    ``param_grid`` is a list of parameter dicts, not a nested grid, so the
    caller controls the search explicitly. An empty or missing grid means one
    pass with the default parameters.

    Concatenating the per-window out-of-sample segments produces a return
    series that was never used for any decision — the honest estimate of what
    the process would have achieved.
    """
    step = step or test_bars
    total_needed = train_bars + validation_bars + test_bars
    if len(index) < total_needed:
        raise ValidationError(
            f"need at least {total_needed} bars for one walk-forward window, "
            f"have {len(index)}"
        )
    grid = list(param_grid) if param_grid else [{}]
    result = WalkForwardResult()
    oos_segments: list[pd.Series] = []
    window_no = 0
    start = 0

    while start + total_needed <= len(index):
        tr = index[start: start + train_bars]
        va = index[start + train_bars: start + train_bars + validation_bars]
        te = index[start + train_bars + validation_bars:
                   start + total_needed]
        w = WalkForwardWindow(
            index=window_no,
            train_start=tr[0], train_end=tr[-1],
            valid_start=va[0], valid_end=va[-1],
            oos_start=te[0], oos_end=te[-1],
            n_train=len(tr), n_valid=len(va), n_oos=len(te),
        )
        try:
            best_score = -np.inf if higher_is_better else np.inf
            best_params = grid[0]
            for params in grid:
                out = fit_predict(tr, va, params)
                score = metric_fn(out) if metric_fn else _extract_metric(out, objective)
                if not np.isfinite(score):
                    continue
                if (higher_is_better and score > best_score) or \
                   (not higher_is_better and score < best_score):
                    best_score, best_params = score, params
            w.best_params = dict(best_params)
            w.valid_metrics = {"score": best_score, "objective": objective}
            oos = fit_predict(tr, te, best_params)
            w.oos_metrics = metric_fn(oos) if metric_fn else {"score": _extract_metric(oos, objective)}
            w.oos_metrics["objective"] = objective
            seg = oos if isinstance(oos, pd.Series) else (
                oos["returns"] if isinstance(oos, dict) and "returns" in oos else None
            )
            if isinstance(seg, pd.Series) and len(seg):
                oos_segments.append(seg)
        except Exception as exc:
            w.error = f"{type(exc).__name__}: {exc}"
            result.failed_windows += 1
        result.windows.append(w)
        window_no += 1
        start += step

    if oos_segments:
        combined = pd.concat(oos_segments)
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        result.aggregate = _aggregate_walkforward(combined, result.windows)
    result.stability = _parameter_stability(result.windows)
    return result


def _extract_metric(out: Any, objective: str) -> float:
    if isinstance(out, dict):
        for key in (objective, f"{objective}_oos", "score"):
            if key in out:
                try:
                    return float(out[key])
                except (TypeError, ValueError):
                    continue
        m = out.get("metrics")
        if isinstance(m, dict) and objective in m:
            v = m[objective]
            return float(v if not hasattr(v, "value") else v.value)
        return float("nan")
    if hasattr(out, objective):
        v = getattr(out, objective)
        return float(v if not hasattr(v, "value") else v.value)
    return float("nan")


def _aggregate_walkforward(oos: pd.Series, windows: list[WalkForwardWindow]) -> dict:
    from quantforge.quant import risk_metrics as rm

    r = oos.dropna()
    if r.empty:
        return {"n_periods": 0, "note": "no out-of-sample returns were produced"}
    equity = (1.0 + r).cumprod()
    dd = rm.max_drawdown(equity)
    per_window = [w.oos_metrics.get("score") for w in windows
                  if isinstance(w.oos_metrics, dict) and "score" in w.oos_metrics]
    per_window = [s for s in per_window if s is not None and np.isfinite(s)]
    return {
        "n_periods": len(r),
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": _cagr(equity, len(r)),
        "volatility": float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else float("nan"),
        "sharpe": rm.sharpe_ratio(r).value,
        "sortino": rm.sortino_ratio(r).value,
        "max_drawdown": dd.max_drawdown,
        "var_95": rm.historical_var(r, 0.95, 1, "walk-forward OOS").value,
        "worst_window_score": min(per_window) if per_window else float("nan"),
        "best_window_score": max(per_window) if per_window else float("nan"),
        "positive_window_fraction": (
            float(np.mean([s > 0 for s in per_window])) if per_window else float("nan")
        ),
    }


def _cagr(equity: pd.Series, n: int) -> float:
    if n <= 0 or equity.empty or equity.iloc[-1] <= 0:
        return float("nan")
    years = n / 252.0
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def _parameter_stability(windows: list[WalkForwardWindow]) -> dict:
    """How much the selected parameters moved between windows.

    Parameters that change every window indicate the search is fitting noise.
    """
    if not windows:
        return {}
    keys = set()
    for w in windows:
        keys.update(w.best_params.keys())
    out: dict[str, Any] = {}
    n_unique_total = 0
    for k in sorted(keys):
        values = [w.best_params[k] for w in windows if k in w.best_params]
        uniq = sorted(set(map(str, values)))
        out[f"{k}_unique_values"] = len(uniq)
        out[f"{k}_modal_share"] = (
            float(max(values.count(v) for v in values) / len(values)) if values else float("nan")
        )
        n_unique_total += len(uniq)
    out["mean_unique_params_per_key"] = n_unique_total / max(1, len(keys))
    out["interpretation"] = (
        "A key with many unique values across windows is being re-selected on "
        "noise. A modal share near 1.0 indicates a stable optimum; a share near "
        "1/len(windows) indicates the search is arbitrary."
    )
    return out


# ==========================================================================
# Monte Carlo
# ==========================================================================


@dataclass(slots=True)
class MonteCarloResult:
    n_simulations: int
    statistics: dict = field(default_factory=dict)
    percentiles: dict = field(default_factory=dict)
    probability_of_loss: float = float("nan")
    method: str = ""
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=" * 72, f"MONTE CARLO ({self.n_simulations:,} simulations, "
                              f"{self.method})", "=" * 72]
        for k, v in self.statistics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<28} {v:>12.4f}")
            else:
                lines.append(f"  {k:<28} {v:>12}")
        lines.append("")
        lines.append("PERCENTILE DISTRIBUTIONS")
        for k, vals in self.percentiles.items():
            lines.append(f"  {k}")
            for p, v in vals.items():
                lines.append(f"      {p:<8} {v:>12.4f}")
        lines.append("")
        lines.append(f"  probability of a loss        {self.probability_of_loss:.2%}")
        if self.warnings:
            lines.append("")
            lines.append("CAVEATS")
            for w in self.warnings:
                lines.append(f"  - {w}")
        lines.append("")
        lines.append(
            "Resampling a historical sample reproduces its distribution; it does "
            "not reveal outcomes the sample never contained. A fat-tailed history "
            "resampled will not produce a crisis it never saw."
        )
        lines.append("=" * 72)
        return "\n".join(lines)


def monte_carlo_returns(
    returns: pd.Series,
    n_simulations: int = 5_000,
    method: str = "bootstrap",
    horizon: Optional[int] = None,
    seed: int = 0,
    include_bootstrap: bool = True,
) -> MonteCarloResult:
    """Resample or permute a return series to build a distribution of outcomes.

    Methods:

    ``bootstrap``
        Iid resample with replacement. Preserves the empirical distribution
        including its fat tails, but **cannot produce a loss worse than the
        worst observed day** — the single most misleading property of this
        technique for risk purposes.
    ``block_bootstrap``
        Circular block resampling, preserving short-range dependence.
    ``shuffle``
        Permutation of the observed sequence, which destroys order but keeps
        the exact multiset of returns.
    ``parametric``
        Fitted normal/Student-t, which can produce arbitrarily extreme values
        but only as far as the fitted assumption permits.
    """
    r = pd.Series(returns).astype("float64").dropna()
    if len(r) < 20:
        res = MonteCarloResult(
            n_simulations=0, method=method,
            warnings=[f"only {len(r)} return observations; 20 is the minimum for a "
                      "resampling distribution. No simulation was run."],
        )
        res.statistics = {"note": "insufficient observations"}
        return res

    rng = np.random.default_rng(seed)
    values = r.to_numpy()
    h = horizon or len(values)

    if method == "bootstrap":
        paths = rng.choice(values, size=(n_simulations, h), replace=True)
    elif method == "block_bootstrap":
        block = max(2, min(20, len(values) // 10))
        n_blocks = int(np.ceil(h / block))
        starts = rng.integers(0, len(values), size=(n_simulations, n_blocks))
        offsets = np.arange(block)[None, :]
        idx = (starts[:, :, None] + offsets[None, :, :]).reshape(n_simulations, -1)[:, :h]
        idx %= len(values)
        paths = values[idx]
    elif method == "shuffle":
        paths = np.vstack([rng.permutation(values) for _ in range(n_simulations)])[:, :h]
    elif method == "parametric":
        mu, sigma = float(values.mean()), float(values.std(ddof=1))
        paths = rng.normal(mu, sigma, size=(n_simulations, h))
    else:
        raise ValidationError(
            f"unknown method {method!r}; use bootstrap, block_bootstrap, shuffle, "
            "or parametric"
        )

    if include_bootstrap:
        # Append the observed path so the distribution is anchored to what
        # actually happened. It must be reshaped to the simulation horizon:
        # appending the full series alongside horizon-length paths raises a
        # shape error whenever horizon != len(returns).
        observed = values[:h] if h <= values.size else np.resize(values, h)
        paths = np.vstack([paths, observed[None, :]])

    finals = np.prod(1.0 + paths, axis=1) - 1.0
    from quantforge.quant import risk_metrics as rm

    equity_paths = np.cumprod(1.0 + paths, axis=1)
    running_max = np.maximum.accumulate(equity_paths, axis=1)
    max_dd = np.max(1.0 - equity_paths / running_max, axis=1)

    stats = {
        "mean_final_return": float(np.mean(finals)),
        "median_final_return": float(np.median(finals)),
        "std_final_return": float(np.std(finals, ddof=1)),
        "p05_final_return": float(np.quantile(finals, 0.05)),
        "p95_final_return": float(np.quantile(finals, 0.95)),
        "worst_final_return": float(np.min(finals)),
        "best_final_return": float(np.max(finals)),
        "mean_max_drawdown": float(np.mean(max_dd)),
        "p95_max_drawdown": float(np.quantile(max_dd, 0.95)),
        "worst_max_drawdown": float(np.max(max_dd)),
    }
    pcts = {}
    for name, arr in (("final_return", finals), ("max_drawdown", max_dd)):
        pcts[name] = {
            f"p{int(q * 100):02d}": float(np.quantile(arr, q))
            for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
        }
    warns = []
    if method == "bootstrap":
        warns.append(
            "iid bootstrap cannot generate a loss beyond the worst observation in "
            f"the sample ({float(np.min(values)):.2%} per period). True tail risk is "
            "therefore understated by construction."
        )
    if len(r) < 252:
        warns.append(
            f"{len(r)} observations is under one year of daily data; the simulated "
            "distribution inherits a short sample's regime mix."
        )
    return MonteCarloResult(
        n_simulations=n_simulations, statistics=stats, percentiles=pcts,
        probability_of_loss=float(np.mean(finals < 0)), method=method, warnings=warns,
    )


def monte_carlo_trade_sequence(
    trade_pnls: Sequence[float],
    n_simulations: int = 5_000,
    seed: int = 0,
) -> MonteCarloResult:
    """Resample per-trade P&L.

    Useful for trade-level strategies, but it assumes the trade sequence is
    exchangeable, which is exactly the assumption that fails when a strategy's
    edge depends on market regime.
    """
    p = np.asarray(list(trade_pnls), dtype="float64")
    p = p[np.isfinite(p)]
    if p.size < 5:
        return MonteCarloResult(n_simulations=0, method="trade_bootstrap",
                                warnings=[f"only {p.size} trades; at least 5 are "
                                          "required for a resampling distribution"],
                                statistics={"note": "insufficient trades"})
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, p.size, size=(n_simulations, p.size))
    finals = p[idx].sum(axis=1)
    equity = np.cumsum(p[idx], axis=1)
    running = np.maximum.accumulate(equity, axis=1)
    peak = np.cumsum(np.maximum.accumulate(np.cumsum(p[idx][:, ::-1], axis=1)[:, ::-1], axis=1), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak != 0, np.abs(peak - equity) / np.abs(peak), 0.0)
    max_dd = np.nanmax(dd, axis=1)
    stats = {
        "mean_pnl": float(np.mean(finals)),
        "median_pnl": float(np.median(finals)),
        "p05_pnl": float(np.quantile(finals, 0.05)),
        "p95_pnl": float(np.quantile(finals, 0.95)),
        "worst_pnl": float(np.min(finals)),
        "mean_max_drawdown": float(np.mean(max_dd)),
        "worst_max_drawdown": float(np.max(max_dd)),
    }
    return MonteCarloResult(
        n_simulations=n_simulations, statistics=stats,
        percentiles={"total_pnl": {f"p{int(q*100):02d}": float(np.quantile(finals, q))
                                   for q in (0.05, 0.25, 0.5, 0.75, 0.95)}},
        probability_of_loss=float(np.mean(finals < 0)),
        method="trade_bootstrap",
        warnings=["resampling trades assumes they are exchangeable in time, which "
                  "fails for strategies whose edge depends on regime"],
    )


# ==========================================================================
# Parameter search, sensitivity, robustness
# ==========================================================================


def parameter_grid(**axes) -> list[dict]:
    """Cartesian product of named axes into a list of parameter dicts."""
    if not axes:
        return [{}]
    keys = list(axes)
    for k, v in axes.items():
        if not isinstance(v, (list, tuple)):
            raise ValidationError(f"axis {k!r} must be a list or tuple")
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes.values())]


def grid_search(
    evaluate: Callable[[dict], dict],
    grid: Sequence[dict],
    objective: str = "sharpe",
    higher_is_better: bool = True,
    min_trades: int = 0,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate a parameter grid, returning the best and the full table.

    ``min_trades`` filters out configurations with too few trades. A
    parameter combination with two trades and a Sharpe of 4 is not a finding;
    it is a division of two observations. The filter is applied *before* the
    optimum is chosen, not reported afterwards.

    Selecting the top of this table is the first step of a selection process,
    not the end of one. The chosen point is still in-sample and must survive
    walk-forward and out-of-sample testing before it means anything.
    """
    rows = []
    for params in grid:
        try:
            res = evaluate(params)
        except Exception as exc:
            rows.append({**params, "error": f"{type(exc).__name__}: {exc}",
                         "objective": float("nan")})
            continue
        n_trades = int(res.get("n_trades", res.get("trades", 0)) or 0)
        score = float(res.get(objective, float("nan")))
        row = {**params, "objective": score, "n_trades": n_trades}
        for k in ("total_return", "cagr", "max_drawdown", "sortino",
                  "profit_factor", "win_rate", "calmar"):
            if k in res:
                row[k] = res[k]
        if res.get("error"):
            row["error"] = res["error"]
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return {}, table
    eligible = table[table["n_trades"] >= min_trades] if min_trades else table
    if eligible.empty:
        return {}, table
    col = "objective" if objective in eligible.columns else "objective"
    valid = eligible[np.isfinite(eligible[col])]
    if valid.empty:
        return {}, table
    idx = valid[col].idxmax() if higher_is_better else valid[col].idxmin()
    best = {k: v for k, v in table.loc[idx].items()
            if k not in ("objective", "n_trades", "error")}
    return best, table


def sensitivity_analysis(
    evaluate: Callable[[dict], dict],
    base: dict,
    axes: dict,
    objective: str = "sharpe",
    n_trades_min: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Vary one parameter at a time and measure how the result moves.

    A strategy whose performance collapses when one parameter moves a small
    amount was fitted to a specific point in parameter space, not to a
    phenomenon. The stability score below is deliberately blunt.
    """
    rows = []
    for name, values in axes.items():
        for v in values:
            params = {**base, name: v}
            try:
                res = evaluate(params)
                n_trades = int(res.get("n_trades", 0) or 0)
                rows.append({
                    "parameter": name, "value": v,
                    objective: float(res.get(objective, float("nan"))),
                    "n_trades": n_trades,
                    "total_return": float(res.get("total_return", float("nan"))),
                    "error": res.get("error"),
                })
            except Exception as exc:
                rows.append({"parameter": name, "value": v,
                             objective: float("nan"), "n_trades": 0,
                             "total_return": float("nan"),
                             "error": f"{type(exc).__name__}: {exc}"})
    table = pd.DataFrame(rows)

    stability: dict[str, Any] = {}
    for name, group in table.groupby("parameter"):
        valid = group[np.isfinite(group[objective])]
        if len(valid) < 2:
            continue
        base_value = group[group["value"] == base.get(name)]
        if base_value.empty or not np.isfinite(base_value[objective].iloc[0]):
            continue
        base_score = float(base_value[objective].iloc[0])
        scores = valid[objective].to_numpy()
        # A plateau near the optimum is the desirable shape; a spike is not.
        spread = float(np.std(scores))
        best = float(np.max(scores)) if np.isfinite(scores).any() else float("nan")
        tolerance = 0.5 * best if best > 0 else 0.5 * abs(base_score)
        near_optimal = int(np.sum(scores >= base_score - abs(tolerance)))
        stability[name] = {
            "score_at_base": base_score,
            "best_score": best,
            "score_std": spread,
            "n_values_evaluated": int(len(valid)),
            "n_near_optimal": near_optimal,
            "fraction_near_optimal": near_optimal / len(valid),
            "is_plateau": bool(near_optimal / len(valid) >= 0.4),
            "verdict": (
                "stable plateau: performance persists across the neighbourhood"
                if near_optimal / len(valid) >= 0.4
                else "NARROW PEAK: performance depends on precise parameter values"
            ),
        }
    return table, stability


def robustness_tests(
    run_variant: Callable[[dict], dict],
    base_config: dict,
    objective: str = "sharpe",
) -> pd.DataFrame:
    """Stress a strategy against adverse assumption changes.

    Each variant perturbs one assumption at a time. A strategy that survives
    higher costs, delayed execution, and missing bars is more likely to be
    measuring something real than one that only works at the chosen settings.
    """
    variants = {
        "base": {},
        "costs_2x": {"cost_multiplier": 2.0},
        "costs_5x": {"cost_multiplier": 5.0},
        "slippage_2x": {"slippage_multiplier": 2.0},
        "delay_1bar": {"execution_delay_bars": 1},
        "delay_2bar": {"execution_delay_bars": 2},
        "no_costs": {"cost_multiplier": 0.0},
        "half_capital": {"capital_multiplier": 0.5},
    }
    rows = []
    base_score = float("nan")
    for name, override in variants.items():
        cfg = {**base_config, **override}
        try:
            res = run_variant(cfg)
            score = float(res.get(objective, float("nan")))
            if name == "base":
                base_score = score
            rows.append({
                "variant": name, objective: score,
                "total_return": float(res.get("total_return", float("nan"))),
                "max_drawdown": float(res.get("max_drawdown", float("nan"))),
                "n_trades": int(res.get("n_trades", 0) or 0),
                "degradation": (
                    float("nan") if not (np.isfinite(score) and np.isfinite(base_score))
                    or base_score == 0 else (base_score - score) / abs(base_score)
                ),
                "error": res.get("error"),
            })
        except Exception as exc:
            rows.append({"variant": name, objective: float("nan"),
                         "degradation": float("nan"),
                         "error": f"{type(exc).__name__}: {exc}"})
    table = pd.DataFrame(rows)
    return table


def regime_breakdown(
    returns: pd.Series,
    regime_labels: pd.Series,
    periods_per_year: int = 252,
    trade_pnls: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Evaluate performance separately within each market regime.

    A strategy that works only in one regime is a regime bet. That may be a
    legitimate choice, but it has to be a deliberate one made with knowledge of
    the regime's historical frequency — not a discovery made after seeing that
    one slice worked.
    """
    from quantforge.quant import risk_metrics as rm

    r = pd.Series(returns).astype("float64")
    labels = pd.Series(regime_labels).reindex(r.index)
    rows = []
    for regime in sorted(set(labels.dropna().unique())):
        mask = labels == regime
        sub = r[mask].dropna()
        if sub.empty:
            rows.append({"regime": regime, "n_periods": 0})
            continue
        equity = (1.0 + sub).cumprod()
        dd = rm.max_drawdown(equity)
        row = {
            "regime": regime,
            "n_periods": len(sub),
            "total_return": float(equity.iloc[-1] - 1.0),
            "mean_period": float(sub.mean()),
            "volatility": float(sub.std(ddof=1) * np.sqrt(periods_per_year))
            if len(sub) > 1 else float("nan"),
            "sharpe": rm.sharpe_ratio(sub, periods_per_year=periods_per_year).value,
            "sortino": rm.sortino_ratio(sub, periods_per_year=periods_per_year).value,
            "max_drawdown": dd.max_drawdown,
            "win_rate": float((sub > 0).mean()),
        }
        if trade_pnls is not None:
            tp = pd.Series(trade_pnls).reindex(sub.index).dropna()
            if len(tp):
                row["n_trades"] = len(tp)
                row["profit_factor"] = rm.profit_factor(tp).value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_periods", ascending=False)
