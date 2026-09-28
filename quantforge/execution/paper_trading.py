"""Paper trading.

Blueprint Section 58: paper trading uses the *same* strategy, feature, risk,
portfolio, and order engines as live trading. Only the execution adapter
differs. This module is therefore a thin loop, not a parallel implementation —
a second paper-trading path would drift from the live one, and the drift would
surface at the worst possible moment.

The loop per bar is:

    market data -> strategy -> risk engine -> order -> adapter fill ->
    portfolio update -> signal log -> latency log -> reconciliation

Every stage the live path has is present, including the ones that are
uncomfortable in simulation: staleness detection, kill switches, risk
vetoes, and reconciliation. A paper-trading harness that omits them validates
a system that does not exist.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.backtest.costs import CostModel
from quantforge.backtest.liquidity import LiquidityModel
from quantforge.backtest.portfolio import Portfolio
from quantforge.backtest.sizing import PositionSizer, SizingConfig
from quantforge.core.enums import (
    ExecutionMode, OrderSide, OrderStatus, OrderType, RiskLevel, TradingState,
)
from quantforge.core.errors import ExecutionStateError, RiskLimitBreached
from quantforge.core.timestamps import utcnow
from quantforge.execution.brokers.base import BrokerAdapter, OrderAck
from quantforge.execution.orders import Order, OrderBook
from quantforge.execution.reconciliation import PositionReconciler, ReconciliationReport
from quantforge.monitoring.observability import LatencyTracker
from quantforge.monitoring.observability import SignalLogger
from quantforge.risk.kill_switch import KillSwitch
from quantforge.risk.limits import RiskEngine, RiskLimits
from quantforge.strategy.builder import Strategy


@dataclass(slots=True)
class PaperTradingConfig:
    """Configuration. Nothing here can enable live trading."""

    initial_capital: float = 100_000.0
    base_currency: str = "USD"
    instrument_id: str = ""
    rebalance_frequency: str = "every_bar"
    reconcile_every_bars: int = 0        # 0 = never (paper does not need it, but
                                         # the capability must exist and be tested)
    max_bars: Optional[int] = None
    close_at_end: bool = True
    use_kill_switch: bool = True
    use_risk_engine: bool = True
    allow_shorting: bool = False
    order_type: OrderType = OrderType.MARKET
    take_profit_price: Optional[float] = None
    #: Replay drives the staleness check from the *bar* clock rather than wall
    #: time. A historical bar is decades old by the wall clock, so without this
    #: every replayed session would trip the staleness kill switch on bar one
    #: and trade nothing. Live sessions must leave this False.
    simulated_clock: bool = True

    def as_dict(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "base_currency": self.base_currency,
            "instrument_id": self.instrument_id,
            "rebalance_frequency": self.rebalance_frequency,
            "reconcile_every_bars": self.reconcile_every_bars,
            "max_bars": self.max_bars,
            "use_kill_switch": self.use_kill_switch,
            "use_risk_engine": self.use_risk_engine,
            "order_type": self.order_type.value,
        }


@dataclass(slots=True)
class PaperTradingResult:
    session_id: str
    config: PaperTradingConfig
    portfolio: Portfolio
    order_book: OrderBook
    orders: list[dict] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    reconciliations: list[dict] = field(default_factory=list)
    risk_verdicts: list[dict] = field(default_factory=list)
    latency: dict = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""
    bars_processed: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "config": self.config.as_dict(),
            "orders": self.orders,
            "signals": self.signals,
            "equity_curve": {str(k): float(v) for k, v in self.equity_curve.items()},
            "reconciliations": self.reconciliations,
            "risk_verdicts": self.risk_verdicts,
            "latency": self.latency,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "bars_processed": self.bars_processed,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        eq = self.equity_curve
        total = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else 0.0
        return "\n".join([
            "=" * 68,
            f"PAPER TRADING SESSION {self.session_id}",
            "=" * 68,
            f"instrument:    {self.config.instrument_id}",
            f"bars processed: {self.bars_processed}",
            f"final equity:  {float(eq.iloc[-1]):,.2f}" if len(eq) else "final equity: n/a",
            f"total return:  {total:.2%}",
            f"orders:        {len(self.orders)}",
            f"signals:       {len(self.signals)}",
            f"halted:        {self.halted}" + (f" ({self.halt_reason})" if self.halted else ""),
            "",
            "This was a SIMULATION using a simulated execution adapter. No real",
            "orders were placed and no real money was at risk. Results here do",
            "not establish that the same strategy would trade successfully for",
            "real, where fills, latency, liquidity, and counterparty behaviour",
            "all differ from the simulation.",
            "=" * 68,
        ])


class PaperTradingSession:
    """Bar-by-bar paper trading on a historical or replayed feed."""

    mode = ExecutionMode.PAPER

    def __init__(
        self,
        strategy: Strategy,
        config: PaperTradingConfig,
        adapter: BrokerAdapter,
        risk_limits: Optional[RiskLimits] = None,
        kill_switch: Optional[KillSwitch] = None,
        cost_model: Optional[CostModel] = None,
        sizing_config: Optional[SizingConfig] = None,
        attribute_fn: Optional[Callable] = None,
    ) -> None:
        self.strategy = strategy
        self.config = config
        self.adapter = adapter
        self.portfolio = Portfolio(config.initial_capital, config.base_currency,
                                   config.allow_shorting)
        self.order_book = OrderBook()
        self.risk = RiskEngine(risk_limits, attribute_fn)
        self.kill_switch = kill_switch or KillSwitch()
        self.cost_model = cost_model or CostModel()
        self.sizer = PositionSizer(sizing_config or SizingConfig())
        self.reconciler = PositionReconciler()
        self._last_features = None
        self.latency = LatencyTracker()
        self.signal_log = SignalLogger()
        self._session_id = ""
        self._peak_price: Optional[float] = None
        self._entry_price: Optional[float] = None

    def run(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        session_id: str = "",
    ) -> PaperTradingResult:
        """Replay bars through the full pipeline."""
        import uuid

        self._session_id = session_id or uuid.uuid4().hex[:10]
        cfg = self.config
        iid = cfg.instrument_id

        if not hasattr(self.adapter, "set_market"):
            raise ExecutionStateError(
                f"{self.adapter.name} adapter has no set_market(); paper trading "
                "requires an adapter that can present simulated market data"
            )

        data, features = self._align(data, features)
        self.risk.reset_session(cfg.initial_capital)
        self.risk.update_equity(cfg.initial_capital)

        result = PaperTradingResult(
            session_id=self._session_id, config=cfg, portfolio=self.portfolio,
            order_book=self.order_book,
        )
        equity_curve: list[tuple] = []
        daily_anchor = cfg.initial_capital
        last_equity = cfg.initial_capital
        position_side: Optional[str] = None
        entry_index: Optional[int] = None
        pending: Optional[dict] = None

        for i in range(len(data)):
            if cfg.max_bars and i >= cfg.max_bars:
                break
            ts = data.index[i]
            bar = data.iloc[i]
            close = float(bar["close"])
            frow = features.iloc[i]
            history = features.iloc[: i + 1]
            self.latency.mark("market_data", when=ts)

            # -- 1. execute pending order from the previous bar's signal ----
            if pending is not None:
                self._execute_pending(pending, bar, ts, i, result)
                pending = None

            # -- 2. exit on stop-loss / take-profit -----------------------
            if position_side and self._entry_price is not None:
                stop = self.strategy.stop_price(
                    self._entry_price, position_side,
                    float(frow["atr_14"]) if "atr_14" in frow and
                    np.isfinite(frow["atr_14"]) else None,
                    self._peak_price,
                )
                take = cfg.take_profit_price
                hit = False
                if stop is not None and ((position_side == "long" and float(bar["low"]) <= stop)
                                         or (position_side == "short" and float(bar["high"]) >= stop)):
                    hit, why = True, f"stop at {stop:.2f}"
                elif take is not None and ((position_side == "long" and close >= take)
                                            or (position_side == "short" and close <= take)):
                    hit, why = True, f"target at {take:.2f}"
                if hit:
                    self._close(position_side, bar, ts, i, result, why)
                    position_side, self._entry_price, self._peak_price = None, None, None
                    entry_index = None

            # -- 3. mark to market and update risk state ------------------
            self.portfolio.mark_to_market({iid: close}, when=ts)
            equity = self.portfolio.equity
            self.risk.update_equity(equity)
            self.risk.record_return((equity / last_equity - 1.0) if last_equity else 0.0)
            last_equity = equity
            equity_curve.append((ts, equity))

            if cfg.use_kill_switch:
                self.kill_switch.note_market_data(ts)
                # The simulated clock is the bar's own timestamp, so staleness
                # measures the gap between bars in the replay rather than the
                # age of a historical bar relative to wall-clock time.
                self.kill_switch.evaluate(
                    equity=equity, daily_start_equity=daily_anchor,
                    peak_equity=self.risk.peak_equity, portfolio_id=cfg.instrument_id,
                    return_history=self.risk.return_history,
                    now=(ts.to_pydatetime() if cfg.simulated_clock else None),
                )
                if self.kill_switch.trading_state() != TradingState.NORMAL:
                    result.halted = True
                    result.halt_reason = "; ".join(
                        f"{e.scope.value}: {e.reason}"
                        for e in self.kill_switch.active_events
                    )
                    result.warnings.append(
                        f"session halted at {ts} by kill switch: {result.halt_reason}"
                    )
                    break

            # -- 4. reconciliation on schedule -----------------------------
            if cfg.reconcile_every_bars and (i % cfg.reconcile_every_bars == 0):
                rec = self.reconciler.reconcile_portfolio(
                    self.portfolio, self.adapter.get_positions(), when=ts
                )
                result.reconciliations.append(rec.as_dict())
                if not rec.matched:
                    result.halted = True
                    result.halt_reason = "reconciliation mismatch"
                    result.warnings.append(
                        f"halted at {ts}: internal and broker positions disagree"
                    )
                    break

            # -- 5. strategy decision --------------------------------------
            pos_qty = (self.portfolio.positions[iid].quantity
                       if iid in self.portfolio.positions else 0.0)
            signal, feature_values = self.strategy.evaluate_entry(frow, history)
            self.latency.mark("feature", when=ts)
            self.latency.mark("signal", when=ts)

            if signal.value == "long" and pos_qty <= 0 and pending is None:
                sig = self.signal_log.record(
                    instrument_id=iid, strategy_id=self.strategy.strategy_id,
                    strategy_version=self.strategy.version, signal="long",
                    feature_values=feature_values, position_before=pos_qty,
                    decision=f"entry signal at {ts}",
                )
                pending = {"side": "long", "signal": sig, "signal_bar": ts}
            elif position_side and self.strategy.should_exit(
                position_side, frow, (i - entry_index) if entry_index else 0, history
            )[0]:
                why = self.strategy.should_exit(
                    position_side, frow, (i - entry_index) if entry_index else 0, history
                )[1]
                sig = self.signal_log.record(
                    instrument_id=iid, strategy_id=self.strategy.strategy_id,
                    strategy_version=self.strategy.version, signal="flat",
                    feature_values=feature_values, position_before=pos_qty,
                    decision=f"exit: {why}",
                )
                pending = {"side": "exit", "signal": sig, "signal_bar": ts,
                           "reason": why}

            if position_side is None and iid in self.portfolio.positions \
                    and not self.portfolio.positions[iid].is_flat:
                p = self.portfolio.positions[iid]
                position_side = "long" if p.quantity > 0 else "short"
                self._entry_price = p.average_entry_price
                self._peak_price = p.average_entry_price
                entry_index = p.entry_bar if isinstance(p.entry_bar, int) else i
            if position_side:
                self._peak_price = (max(self._peak_price, close)
                                    if position_side == "long"
                                    else min(self._peak_price, close))

        if cfg.close_at_end and self.portfolio.position_count and not result.halted:
            last = data.iloc[-1]
            for iid_held in list(self.portfolio.positions):
                p = self.portfolio.positions[iid_held]
                if not p.is_flat:
                    side = "short" if p.quantity > 0 else "long"
                    self._close(side, last, data.index[-1], len(data) - 1, result,
                                "session end")
            self.portfolio.mark_to_market(
                {iid: float(last["close"]) for iid in self.portfolio.positions},
                when=data.index[-1],
            )
            equity_curve.append((data.index[-1], self.portfolio.equity))

        result.equity_curve = pd.Series(
            [v for _, v in equity_curve],
            index=pd.DatetimeIndex([t for t, _ in equity_curve]),
        )
        result.orders = self.order_book.frame().to_dict("records") if len(self.order_book) else []
        result.signals = self.signal_log.records()
        result.bars_processed = len(data) if not cfg.max_bars else min(len(data), cfg.max_bars)
        result.latency = self.latency.summary()
        return result

    def _align(self, data: pd.DataFrame, features: pd.DataFrame):
        if not data.index.equals(features.index):
            common = data.index.intersection(features.index)
            if len(common) < 2:
                raise ExecutionStateError(
                    "price and feature indices do not overlap; the strategy would "
                    "have no causal inputs"
                )
            data, features = data.loc[common], features.loc[common]
        return data, features

    def _execute_pending(self, pending, bar, ts, i, result) -> None:
        side = pending["side"]
        iid = self.config.instrument_id
        pos = self.portfolio.positions.get(iid)
        current_qty = pos.quantity if pos else 0.0

        if side == "long":
            if current_qty > 0:
                return
            qty = self._size(bar, ts, "long")
            if qty <= 0:
                return
            order_side = OrderSide.LONG
        else:
            qty = abs(current_qty)
            if qty <= 0:
                return
            order_side = OrderSide.SHORT if current_qty > 0 else OrderSide.LONG

        order = self.order_book.new_order(
            instrument_id=iid, side=order_side, quantity=qty,
            order_type=self.config.order_type, strategy_id=self.strategy.strategy_id,
            portfolio_id=iid,
            decision_time=pending["signal_bar"],
            information_cutoff=pending["signal_bar"],
            model_version=self.strategy.version,
        )
        pending["signal"].order_id = order.order_id

        if self.config.use_risk_engine:
            verdict = self.risk.check_order(self.portfolio, iid, order_side.value,
                                            qty, float(bar["open"]))
            result.risk_verdicts.append(verdict.as_dict())
            if not verdict.approved:
                self.order_book.transition(
                    order.order_id, OrderStatus.REJECTED,
                    reason=f"risk engine: {verdict.reason}",
                    reject_reason=verdict.reason,
                )
                pending["signal"].risk_approved = False
                pending["signal"].risk_level = verdict.risk_level.value
                return
            pending["signal"].risk_approved = True
            pending["signal"].risk_level = verdict.risk_level.value

        self.order_book.transition(order.order_id, OrderStatus.SUBMITTED)
        self.latency.mark("order_submission", when=ts)
        try:
            ack = self.adapter.submit_order(order)
        except Exception as exc:
            # Any submission failure, including a timeout, leaves the order
            # state unknown. Reconciliation — not a retry — is the way out.
            self.order_book.transition(
                order.order_id, OrderStatus.UNKNOWN,
                reason=f"submission failed: {type(exc).__name__}: {exc}",
            )
            pending["signal"].decision += f" | submission failed: {exc}"
            result.warnings.append(
                f"order {order.order_id} submission outcome unknown ({exc}); "
                "reconciliation required before any retry"
            )
            return

        self.latency.mark("broker_response", when=ts)
        if ack.broker_order_id:
            order.broker_order_id = ack.broker_order_id
        if ack.status == OrderStatus.UNKNOWN:
            self.order_book.transition(order.order_id, OrderStatus.UNKNOWN,
                                        reason="broker reported unknown")
            return
        self.order_book.transition(order.order_id, ack.status,
                                    broker_order_id=ack.broker_order_id)
        if order.filled_quantity > 0:
            self._book_fill(order, ts, i)

    def _book_fill(self, order: Order, ts, i) -> None:
        """Apply a fill reported by the adapter to the internal portfolio."""
        iid = order.instrument_id
        est = self.cost_model.estimate(order.filled_quantity, order.average_fill_price or 0.0,
                                       order.side.value)
        order.fees = est["commission"]
        from quantforge.core.enums import OrderSide as _OS

        self.portfolio.apply_fill(
            iid, _OS.LONG if order.side == OrderSide.LONG else _OS.SHORT,
            order.filled_quantity, order.average_fill_price or 0.0,
            fees=est["commission"], when=ts,
            strategy_id=order.strategy_id, bar_index=i,
        )
        self.latency.mark("execution", when=ts)

    def _size(self, bar, ts, side) -> float:
        frow = self._last_features
        atr = None
        if frow is not None and "atr_14" in frow and np.isfinite(frow["atr_14"]):
            atr = float(frow["atr_14"])
        res = self.sizer.size(equity=self.portfolio.equity,
                              price=float(bar["close"]), atr=atr)
        qty = abs(res.quantity)
        if self.config.use_risk_engine:
            qty = self.risk.clamp_quantity(self.portfolio, self.config.instrument_id,
                                           side, qty, float(bar["close"]))
        return qty

    def _close(self, position_side, bar, ts, i, result, reason) -> None:
        iid = self.config.instrument_id
        pos = self.portfolio.positions.get(iid)
        if pos is None or pos.is_flat:
            return
        qty = abs(pos.quantity)
        px = float(bar["close"])
        est = self.cost_model.estimate(qty, px, "short" if position_side == "long" else "long")
        fill_px = px + (est["total_cost"] / qty if qty else 0.0) if position_side == "long" \
            else px - (est["total_cost"] / qty if qty else 0.0)
        order = self.order_book.new_order(
            instrument_id=iid,
            side=OrderSide.SHORT if position_side == "long" else OrderSide.LONG,
            quantity=qty, strategy_id=self.strategy.strategy_id,
            portfolio_id=iid, decision_time=ts, information_cutoff=ts,
        )
        self.order_book.transition(order.order_id, OrderStatus.SUBMITTED)
        try:
            ack = self.adapter.submit_order(order)
            order.broker_order_id = ack.broker_order_id
            if ack.status not in (OrderStatus.UNKNOWN,):
                self.order_book.transition(order.order_id, ack.status,
                                            broker_order_id=ack.broker_order_id)
        except Exception as exc:
            self.order_book.transition(order.order_id, OrderStatus.UNKNOWN,
                                        reason=f"exit submission failed: {exc}")
        from quantforge.core.enums import OrderSide as _OS

        self.portfolio.apply_fill(
            iid, _OS.SHORT if position_side == "long" else _OS.LONG,
            qty, fill_px, fees=est["commission"], when=ts,
            strategy_id=self.strategy.strategy_id, bar_index=i, exit_reason=reason,
        )
        result.signals.append({
            "timestamp": str(ts), "instrument_id": iid, "signal": "flat",
            "decision": f"exit: {reason}", "order_id": order.order_id,
        })
