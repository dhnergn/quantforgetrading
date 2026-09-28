"""Event-driven backtesting engine.

Processes bars strictly in order:

    bar t data -> signal(t) -> risk check -> order -> fill(t+1) -> mark(t+1)

The single most important property is the **one-bar gap between deciding and
filling**. A signal computed from bar ``t``'s close cannot be filled at bar
``t``'s close, because that close is only known once the bar has ended. The
engine therefore:

* computes the signal using features available at the close of bar ``t``;
* places the order for execution on bar ``t+1``;
* fills at bar ``t+1``'s open by default, adjusted for gap and impact;
* checks stops against bar ``t+1``'s range, filling at the stop only if the
  bar actually traded through it, and at the *open* if it gapped past.

Every result carries the decision chain timestamps so a reader can verify
which bars a trade depended on.
"""
from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.backtest.costs import CostModel
from quantforge.backtest.liquidity import LiquidityModel, simulate_fill_price
from quantforge.backtest.portfolio import Portfolio
from quantforge.backtest.sizing import PositionSizer, SizingConfig
from quantforge.core.enums import (
    ExecutionMode, OrderSide, OrderStatus, OrderType, RiskLevel, SignalType,
)
from quantforge.core.errors import BacktestError
from quantforge.core.reproducibility import ExperimentManifest, content_hash
from quantforge.core.timestamps import UTC, TemporalCausalityError, utcnow
from quantforge.quant.returns import simple_returns
from quantforge.risk.kill_switch import KillSwitch
from quantforge.risk.limits import RiskEngine, RiskLimits, RiskVerdict
from quantforge.strategy.builder import Strategy


# --------------------------------------------------------------------------
# Configuration and results
# --------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    base_currency: str = "USD"
    start: Optional[dt.datetime] = None
    end: Optional[dt.datetime] = None
    periods_per_year: int = 252
    risk_free_annual: float = 0.0
    var_confidence: float = 0.95
    allow_shorting: bool = False
    close_positions_at_end: bool = True
    #: Signal is acted on at the next bar's open by default. Setting this to
    #: False fills at the signal bar's close, which assumes an execution that
    #: was not available at that moment.
    execute_at_next_open: bool = True
    record_decision_chain: bool = True
    random_seed: int = 0

    def as_dict(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "base_currency": self.base_currency,
            "start": str(self.start) if self.start else None,
            "end": str(self.end) if self.end else None,
            "periods_per_year": self.periods_per_year,
            "risk_free_annual": self.risk_free_annual,
            "allow_shorting": self.allow_shorting,
            "close_positions_at_end": self.close_positions_at_end,
            "execute_at_next_open": self.execute_at_next_open,
            "random_seed": self.random_seed,
        }


@dataclass(slots=True)
class OrderRecord:
    order_id: str
    client_order_id: str
    instrument_id: str
    side: str
    quantity: float
    order_type: str
    status: str
    requested_bar: object
    filled_bar: Optional[object] = None
    requested_price: float = 0.0
    fill_price: Optional[float] = None
    fees: float = 0.0
    slippage: float = 0.0
    reason: str = ""
    risk_level: str = "green"
    risk_violations: list = field(default_factory=list)
    signal_bar: Optional[object] = None
    fill_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "instrument_id": self.instrument_id,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status,
            "requested_bar": str(self.requested_bar),
            "filled_bar": str(self.filled_bar) if self.filled_bar is not None else None,
            "signal_bar": str(self.signal_bar) if self.signal_bar is not None else None,
            "requested_price": self.requested_price,
            "fill_price": self.fill_price,
            "fees": self.fees,
            "slippage": self.slippage,
            "total_cost": self.fees + self.slippage,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "risk_violations": self.risk_violations,
            "fill_reason": self.fill_reason,
        }


@dataclass(slots=True)
class SignalRecord:
    timestamp: object
    instrument_id: str
    strategy_id: str
    strategy_version: str
    signal: str
    feature_values: dict = field(default_factory=dict)
    decision: str = ""
    position_before: float = 0.0
    order_id: Optional[str] = None
    risk_level: str = "green"
    risk_approved: Optional[bool] = None
    regime: Optional[str] = None
    model_version: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "timestamp": str(self.timestamp),
            "instrument_id": self.instrument_id,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "signal": self.signal,
            "feature_values": self.feature_values,
            "decision": self.decision,
            "position_before": self.position_before,
            "order_id": self.order_id,
            "risk_level": self.risk_level,
            "risk_approved": self.risk_approved,
            "regime": self.regime,
            "model_version": self.model_version,
        }


@dataclass(slots=True)
class BacktestResult:
    run_id: str
    config: BacktestConfig
    portfolio: Portfolio
    orders: list[OrderRecord]
    signals: list[SignalRecord]
    strategy: Strategy
    manifest: ExperimentManifest
    metrics: Any = None
    benchmark_returns: Optional[pd.Series] = None
    warnings: list[str] = field(default_factory=list)
    rejected_orders: list[dict] = field(default_factory=list)
    #: Features the strategy references but the feature frame does not provide.
    missing_features: list[str] = field(default_factory=list)

    def order_frame(self) -> pd.DataFrame:
        if not self.orders:
            return pd.DataFrame()
        return pd.DataFrame([o.as_dict() for o in self.orders])

    def signal_frame(self) -> pd.DataFrame:
        if not self.signals:
            return pd.DataFrame()
        return pd.DataFrame([s.as_dict() for s in self.signals])

    def filled_orders(self) -> pd.DataFrame:
        return self.order_frame().query("status == 'filled'")

    def summary(self) -> str:
        lines = [
            "=" * 72,
            f"BACKTEST {self.run_id}",
            "=" * 72,
            f"strategy:        {self.strategy.strategy_id}@{self.strategy.version}",
            f"config hash:     {self.manifest.compute_hash()}",
            f"initial capital: {self.config.initial_capital:,.2f}",
            f"final equity:    {self.portfolio.equity:,.2f}",
            f"orders placed:   {len(self.orders)} "
            f"({len(self.filled_orders())} filled)",
            f"trades closed:   {len(self.portfolio.trades)}",
            f"total fees:      {self.portfolio.total_fees:,.2f}",
            "",
        ]
        if self.metrics is not None:
            lines.append(self.metrics.summary())
        if self.warnings:
            lines.append("WARNINGS")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("")
        lines.append(
            "A profitable result under these assumptions is evidence about the "
            "tested historical period only. It is not evidence of future "
            "profitability, and it does not account for market impact beyond the "
            "configured model."
        )
        lines.append("=" * 72)
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "config": self.config.as_dict(),
            "strategy": self.strategy.as_dict(),
            "manifest": self.manifest.to_dict(),
            "equity_curve": {
                str(k): float(v) for k, v in self.portfolio.equity_series().items()
            },
            "orders": [o.as_dict() for o in self.orders],
            "signals": [s.as_dict() for s in self.signals],
            "trades": [t.as_dict() for t in self.portfolio.trades],
            "metrics": self.metrics.as_dict() if self.metrics is not None else None,
            "rejected_orders": self.rejected_orders,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


class BacktestEngine:
    """Single-instrument event-driven backtester.

    Multi-instrument runs orchestrate several of these against a shared
    portfolio and risk engine; keeping the single-instrument loop self-contained
    is what makes its temporal guarantees auditable.
    """

    def __init__(
        self,
        strategy: Strategy,
        config: Optional[BacktestConfig] = None,
        cost_model: Optional[CostModel] = None,
        liquidity_model: Optional[LiquidityModel] = None,
        sizing_config: Optional[SizingConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
        kill_switch: Optional[KillSwitch] = None,
        attribute_fn: Optional[Callable] = None,
        sizing_headroom: float = 0.01,
    ) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.cost_model = cost_model or CostModel.from_config(
            _FallbackConfig(self.config)
        )
        self.liquidity = liquidity_model or LiquidityModel()
        self.sizer = PositionSizer(sizing_config or SizingConfig(
            method=_sizing_method(strategy)
        ))
        self.risk = RiskEngine(risk_limits, attribute_fn)
        self.kill_switch = kill_switch or KillSwitch()
        self.sizing_headroom = sizing_headroom
        self._order_seq = 0

    # -- main loop ---------------------------------------------------------

    def _required_features(self) -> list[str]:
        """Feature names the strategy's rules reference.

        Walks the condition tree rather than relying on a hand-maintained list,
        which would drift out of sync with the rule the moment either changed.
        """
        from quantforge.strategy.conditions import Condition, ConditionGroup

        names: list[str] = []

        def walk(rule) -> None:
            if isinstance(rule, Condition):
                names.append(rule.feature)
                if rule.ref:
                    names.append(rule.ref)
            elif isinstance(rule, ConditionGroup):
                for child in rule.children:
                    walk(child)

        walk(self.strategy.entry_rule)
        for rules in (self.strategy.long_exit_rules, self.strategy.short_exit_rules):
            for rule in rules:
                if rule.kind in ("indicator", "indicator_cross"):
                    names.append(rule.params.get("feature", ""))
                    if rule.params.get("ref"):
                        names.append(rule.params["ref"])
        # Stops read ATR directly off the feature row.
        if self.strategy.stop.stop_type.value.startswith("atr") or \
           self.strategy.stop.stop_type.value.startswith("trailing"):
            names.append("atr_14")
        return [n for n in dict.fromkeys(names) if n]

    def run(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Run the backtest.

        ``data`` needs open/high/low/close/volume; ``features`` must be indexed
        identically and contain every feature the strategy references. Both are
        sliced to the configured window.
        """
        cfg = self.config
        data, features = self._align(data, features)

        if data.empty:
            raise BacktestError("no bars to simulate")

        # Preflight: a strategy referencing a feature the frame does not
        # provide would evaluate every condition to False and "trade" nothing.
        # A zero-signal backtest looks like a valid result, so this is raised
        # rather than warned about.
        missing = [n for n in self._required_features() if n not in features.columns]
        if missing:
            raise BacktestError(
                f"strategy {self.strategy.strategy_id}@{self.strategy.version} "
                f"references features absent from the feature frame: {missing}. "
                "Rebuild the feature frame with those columns. Without them every "
                "condition evaluates False and the backtest would report a flat "
                "result rather than failing."
            )

        portfolio = Portfolio(cfg.initial_capital, cfg.base_currency,
                              cfg.allow_shorting)
        self.risk.reset_session(cfg.initial_capital)
        self.risk.update_equity(cfg.initial_capital)

        orders: list[OrderRecord] = []
        signals: list[SignalRecord] = []
        rejected: list[dict] = []
        warnings: list[str] = []

        index = data.index
        n = len(index)
        pending: Optional[dict] = None
        entry_bar: Optional[int] = None
        stop_price: Optional[float] = None
        peak_price: Optional[float] = None
        position_side: Optional[str] = None
        bars_in_position = 0
        last_equity = cfg.initial_capital
        daily_anchor = cfg.initial_capital

        for i in range(n):
            ts = index[i]
            bar = data.iloc[i]
            frow = features.iloc[i]
            history = features.iloc[: i + 1]
            close = float(bar["close"])
            self.risk.update_equity(portfolio.equity)

            # ---- 1. execute a pending order from the previous bar's signal --
            if pending is not None:
                pending = self._execute(
                    pending, bar, ts, i, portfolio, orders, entry_bar
                )
                if pending is not None and not pending.get("resolved"):
                    # Still open (partial fill); carry it forward.
                    pass
                else:
                    pending = None
                    if portfolio.position_count == 0:
                        position_side = None
                        stop_price = None
                        peak_price = None
                        entry_bar = None
                        bars_in_position = 0

            # ---- 2. stop handling against this bar's range ----------------
            if stop_price is not None and position_side is not None:
                hit = (
                    (position_side == "long" and float(bar["low"]) <= stop_price)
                    or (position_side == "short" and float(bar["high"]) >= stop_price)
                )
                if hit:
                    fill = simulate_fill_price(
                        "long" if position_side == "long" else "short",
                        float(bar["open"]), float(bar["high"]), float(bar["low"]),
                        close, abs(portfolio.positions[
                            self.strategy.strategy_id].quantity)
                        if self.strategy.strategy_id in portfolio.positions else 1,
                        float(bar["volume"]), self.liquidity,
                        order_type="stop", stop_price=stop_price,
                    )
                    if fill["filled"]:
                        self._close_position(portfolio, position_side, fill, ts, i,
                                             orders, f"stop at {stop_price:g}",
                                             fill.get("fill_price"))
                        stop_price = None
                        peak_price = None
                        position_side = None
                        entry_bar = None
                        bars_in_position = 0
                        pending = None

            # ---- 3. mark to market ----------------------------------------
            iid = self.strategy.strategy_id
            portfolio.mark_to_market({iid: close}, when=ts)

            equity = portfolio.equity
            if self.kill_switch is not None:
                self.kill_switch.evaluate(
                    equity=equity, daily_start_equity=daily_anchor,
                    peak_equity=max(cfg.initial_capital, portfolio.peak_equity
                                    if hasattr(portfolio, "peak_equity") else 0.0,
                                    self.risk.peak_equity or 0.0),
                )
            if equity <= 0:
                warnings.append(
                    f"portfolio equity reached {equity:,.2f} at {ts}; simulation "
                    "stopped because further bars cannot be traded"
                )
                break

            self.risk.record_return(equity / last_equity - 1.0) if last_equity else None
            last_equity = equity

            # ---- 4. time stop -------------------------------------------
            if position_side and entry_bar is not None:
                bars_in_position = i - entry_bar
                if (self.strategy.stop.time_stop_bars
                        and bars_in_position >= self.strategy.stop.time_stop_bars):
                    self._close_position(portfolio, position_side, None, ts, i,
                                         orders, "time stop", close)
                    position_side = None
                    stop_price = None
                    entry_bar = None
                    bars_in_position = 0

            # ---- 5. strategy signal (uses only information up to here) ---
            try:
                signal, feature_values = self.strategy.evaluate_entry(frow, history)
            except Exception as exc:
                raise BacktestError(
                    f"strategy evaluation failed at {ts}: {type(exc).__name__}: {exc}"
                ) from exc

            pos_qty = (portfolio.positions[iid].quantity
                       if iid in portfolio.positions else 0.0)

            if signal == SignalType.LONG and pos_qty <= 0 and pending is None:
                sig = SignalRecord(
                    timestamp=ts, instrument_id=iid,
                    strategy_id=self.strategy.strategy_id,
                    strategy_version=self.strategy.version,
                    signal="long", feature_values=feature_values,
                    position_before=pos_qty, risk_level="green",
                )
                signals.append(sig)
                pending = {
                    "side": "long", "signal_bar": ts, "signal_index": i,
                    "quantity": self._target_quantity(portfolio, close, frow, "long"),
                    "feature_values": feature_values, "signal_record": sig,
                }
                sig.decision = f"entry signal at {ts}; order queued for {index[i+1] if i+1 < n else 'end'}"

            elif position_side and self.strategy.should_exit(
                position_side, frow, bars_in_position, history
            )[0]:
                exit_flag, why = self.strategy.should_exit(
                    position_side, frow, bars_in_position, history)
                sig = SignalRecord(
                    timestamp=ts, instrument_id=iid,
                    strategy_id=self.strategy.strategy_id,
                    strategy_version=self.strategy.version,
                    signal="flat", feature_values=feature_values,
                    position_before=pos_qty, risk_level="green",
                )
                sig.decision = f"exit: {why}"
                signals.append(sig)
                pending = {
                    "side": "exit", "signal_bar": ts, "signal_index": i,
                    "quantity": abs(pos_qty), "feature_values": feature_values,
                    "signal_record": sig, "exit_reason": why,
                }

            # ---- 6. open-position bookkeeping -----------------------------
            if position_side is None and iid in portfolio.positions:
                p = portfolio.positions[iid]
                if not p.is_flat:
                    position_side = "long" if p.quantity > 0 else "short"
                    entry_bar = p.entry_bar if isinstance(p.entry_bar, int) else i
                    peak_price = p.average_entry_price
                    atr = frow.get("atr_14", np.nan)
                    stop_price = self.strategy.stop_price(
                        p.average_entry_price, position_side,
                        float(atr) if np.isfinite(atr) else None, peak_price)
                    bars_in_position = 0
            if position_side:
                peak_price = (max(peak_price, close) if position_side == "long"
                              else min(peak_price, close) if peak_price is not None else close)
                if self.strategy.stop.stop_type.value.startswith("trailing"):
                    atr = frow.get("atr_14", np.nan)
                    stop_price = self.strategy.stop_price(
                        peak_price, position_side,
                        float(atr) if np.isfinite(atr) else None, peak_price)

        # ---- end of run ---------------------------------------------------
        final_bar = data.iloc[-1]
        if cfg.close_positions_at_end and portfolio.position_count:
            portfolio.close_all(
                {iid: float(final_bar["close"]) for iid in portfolio.positions},
                when=index[-1], reason="end_of_backtest",
            )
            portfolio.mark_to_market(
                {iid: float(final_bar["close"]) for iid in portfolio.positions},
                when=index[-1],
            )
        if pending is not None and pending.get("side") == "long" and portfolio.cash > 0:
            warnings.append(
                "a signal on the final bar produced an order that could not be "
                "filled because no subsequent bar exists. It was discarded, which "
                "is the correct treatment: a backtest must not fill an order it "
                "cannot observe being filled."
            )

        manifest = self._manifest(data, features)
        return BacktestResult(
            run_id=uuid.uuid4().hex[:12],
            config=cfg,
            portfolio=portfolio,
            orders=orders,
            signals=signals,
            strategy=self.strategy,
            manifest=manifest,
            benchmark_returns=benchmark,
            warnings=warnings + (
                ["transaction costs are zero: this result is an upper bound no "
                 "real execution could achieve"] if self.cost_model.is_free else []
            ),
            rejected_orders=rejected,
        )

    # -- helpers -----------------------------------------------------------

    def _align(self, data: pd.DataFrame, features: pd.DataFrame):
        if not data.index.equals(features.index):
            common = data.index.intersection(features.index)
            if len(common) < 2:
                raise BacktestError(
                    "price and feature indices do not overlap; the strategy would "
                    "have no causal inputs"
                )
            data = data.loc[common]
            features = features.loc[common]
        cfg = self.config
        if cfg.start is not None:
            data = data[data.index >= pd.Timestamp(cfg.start).tz_localize(UTC)
                        if data.index.tz is None else data.index >= pd.Timestamp(cfg.start)]
            features = features.loc[data.index]
        if cfg.end is not None:
            end = pd.Timestamp(cfg.end)
            end = end.tz_localize(UTC) if data.index.tz is None else end
            data = data[data.index <= end]
            features = features.loc[data.index]
        return data, features

    def _target_quantity(self, portfolio, price, frow, side) -> float:
        """Size the entry, leaving headroom for the next bar's price move.

        Sizing exactly to a risk limit guarantees the order is rejected: the
        quantity is computed at bar ``t``'s close but filled at bar ``t+1``'s
        open, so any adverse gap pushes the notional past the ceiling. The
        headroom fraction is the largest one-day move the instrument has shown
        in its recent history, so the allowance scales with the instrument's
        own volatility rather than being an arbitrary constant.
        """
        atr = frow.get("atr_14", np.nan)
        res = self.sizer.size(
            equity=portfolio.equity, price=price, atr=float(atr) if np.isfinite(atr) else None,
        )
        qty = abs(res.quantity)
        headroom = float(self.sizing_headroom)
        if np.isfinite(atr) and np.isfinite(price) and price > 0:
            # Cap the headroom at 5%: beyond that a single ATR is a limit
            # breach, not a gap, and shrinking the position further would
            # silently disable the strategy on a volatile instrument.
            headroom = float(min(0.05, max(0.002, 2.0 * atr / price)))
        qty *= (1.0 - headroom)
        qty = self.risk.clamp_quantity(portfolio, self.strategy.strategy_id, side, qty, price)
        return qty

    def _execute(self, pending, bar, ts, i, portfolio, orders, entry_bar):
        """Fill a pending order against the current bar."""
        side = pending["side"]
        iid = self.strategy.strategy_id
        self._order_seq += 1
        order_id = uuid.uuid4().hex[:16]
        client_order_id = f"qf-{self._order_seq:08d}"

        if side == "exit":
            qty = pending["quantity"]
            fill_side = "long" if portfolio.positions[iid].quantity < 0 else "short"
        else:
            qty = pending["quantity"]
            fill_side = "long"

        if qty <= 0:
            return None

        rec = OrderRecord(
            order_id=order_id, client_order_id=client_order_id, instrument_id=iid,
            side=fill_side, quantity=qty, order_type="market", status="created",
            requested_bar=ts, requested_price=float(bar["open"]),
            signal_bar=pending["signal_bar"],
        )

        # Risk check happens at execution time, on the post-fill state.
        verdict = self.risk.check_order(portfolio, iid, fill_side, qty,
                                        float(bar["open"]))
        rec.risk_level = verdict.risk_level.value
        rec.risk_violations = verdict.violations
        if not verdict.approved:
            rec.status = "rejected"
            rec.reason = verdict.reason
            orders.append(rec)
            if pending.get("signal_record"):
                pending["signal_record"].order_id = order_id
                pending["signal_record"].risk_approved = False
                pending["signal_record"].risk_level = verdict.risk_level.value
            return None

        # Liquidity decides how much actually fills.
        decision = self.liquidity.decide_fill(qty, float(bar["volume"]), "market")
        rec.reason = decision["reason"]
        if decision["filled"] <= 0:
            rec.status = "rejected" if decision["rejected"] else "cancelled"
            rec.fill_reason = decision["reason"]
            orders.append(rec)
            if pending.get("signal_record"):
                pending["signal_record"].order_id = order_id
                pending["signal_record"].risk_approved = True
            return None

        fill_qty = decision["filled"]
        fill = simulate_fill_price(
            fill_side, float(bar["open"]), float(bar["high"]), float(bar["low"]),
            float(bar["close"]), fill_qty, float(bar["volume"]), self.liquidity,
            order_type="market",
            signal_time="bar_open" if self.config.execute_at_next_open else "bar_close",
        )
        if not fill["filled"]:
            rec.status = "cancelled"
            rec.fill_reason = fill["reason"]
            orders.append(rec)
            return None

        raw_price = fill["fill_price"]
        ref_price = float(bar["open"]) if self.config.execute_at_next_open else float(bar["close"])
        est = self.cost_model.estimate(
            fill_qty, ref_price, fill_side,
            bar_volume=float(bar["volume"]),
            bar_volatility=float(bar.get("true_range", 0.0) or 0.0) / max(ref_price, 1e-9),
        )
        # The simulator already added impact; commission is charged on cash.
        rec.fees = est["commission"]
        rec.slippage = abs(fill.get("impact", 0.0)) * fill_qty
        rec.fill_price = raw_price
        rec.filled_bar = ts
        rec.status = "filled" if fill_qty >= qty else "partially_filled"

        from quantforge.core.enums import OrderSide as _OS

        if side == "exit":
            trade = portfolio.apply_fill(
                iid, _OS.LONG if fill_side == "long" else _OS.SHORT,
                fill_qty, raw_price, fees=rec.fees, when=ts,
                strategy_id=self.strategy.strategy_id, bar_index=i,
                exit_reason=pending.get("exit_reason", "signal"),
            )
        else:
            portfolio.apply_fill(
                iid, _OS.LONG, fill_qty, raw_price, fees=rec.fees, when=ts,
                strategy_id=self.strategy.strategy_id, bar_index=i,
            )
        orders.append(rec)
        if pending.get("signal_record"):
            pending["signal_record"].order_id = order_id
            pending["signal_record"].risk_approved = True
            pending["signal_record"].risk_level = verdict.risk_level.value
        pending["resolved"] = True
        return pending

    def _close_position(self, portfolio, side, fill, ts, i, orders, reason, price):
        iid = self.strategy.strategy_id
        if iid not in portfolio.positions:
            return
        qty = abs(portfolio.positions[iid].quantity)
        if qty <= 0:
            return
        from quantforge.core.enums import OrderSide as _OS

        if fill is not None:
            px = float(fill["fill_price"])
        elif price is not None:
            est = self.cost_model.estimate(qty, float(price), "long")
            px = float(price) + est["total_cost"] / qty
        else:
            px = float(portfolio.positions[iid].last_price)
        self._order_seq += 1
        rec = OrderRecord(
            order_id=uuid.uuid4().hex[:16], client_order_id=f"qf-{self._order_seq:08d}",
            instrument_id=iid, side="short" if side == "long" else "long",
            quantity=qty, order_type="stop" if "stop" in reason else "market",
            status="filled", requested_bar=ts, filled_bar=ts,
            requested_price=px, fill_price=px, reason=reason,
        )
        portfolio.apply_fill(
            iid, _OS.SHORT if side == "long" else _OS.LONG, qty, px,
            fees=0.0, when=ts, strategy_id=self.strategy.strategy_id,
            bar_index=i, exit_reason=reason,
        )
        orders.append(rec)

    def _manifest(self, data: pd.DataFrame, features: pd.DataFrame) -> ExperimentManifest:
        cfg = self.config
        return ExperimentManifest(
            experiment_type="backtest",
            dataset_version=content_hash({
                "ohlcv": data.to_numpy().tobytes()[:4096].hex(),
                "index": [str(i) for i in data.index[:5]] + [str(i) for i in data.index[-5:]],
                "n": len(data),
            }),
            feature_set_version=content_hash(list(features.columns)),
            strategy_version=self.strategy.version,
            parameters={
                "strategy_fingerprint": self.strategy.fingerprint(),
                "entry_rule": self.strategy.entry_rule.describe(),
            },
            random_seed=cfg.random_seed,
            transaction_costs=self.cost_model.as_dict(),
            slippage=self.cost_model.slippage.as_dict(),
            universe={"instruments": [self.strategy.strategy_id]},
            time_period={
                "start": str(data.index[0]), "end": str(data.index[-1]),
                "n_bars": len(data),
            },
            risk_configuration=self.risk.limits.as_dict(),
            preprocessing={"liquidity": self.liquidity.as_dict()},
            notes=f"strategy={self.strategy.strategy_id}@{self.strategy.version}",
        )


class _FallbackConfig:
    """Adapts BacktestConfig to the fields CostModel.from_config expects."""

    def __init__(self, cfg: BacktestConfig) -> None:
        self.commission_per_share = 0.0
        self.commission_pct = 0.0
        self.min_commission = 0.0
        self.regulatory_fee_bps = 0.0
        self.slippage_model = "percentage"
        self.slippage_pct = 0.001
        self.slippage_fixed = 0.0
        self.assume_spread_bps = 5.0


def _sizing_method(strategy: Strategy):
    from quantforge.core.enums import SizingMethod

    m = strategy.sizing.get("method", "fixed_fractional")
    try:
        return SizingMethod(m)
    except ValueError:
        return SizingMethod.FIXED_FRACTIONAL
