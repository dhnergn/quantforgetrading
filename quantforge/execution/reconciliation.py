"""Position reconciliation and internal state.

Blueprint Section 64 requires that a discrepancy between internal and broker
positions puts the system into ``REQUIRES_RECONCILIATION`` rather than
continuing to trade on a wrong picture.

The reconciliation policy is deliberately conservative: when the two disagree,
the **broker** is treated as the source of truth for quantities, because the
broker's records are what a counterparty will settle against. Internal
positions are corrected to match, and the correction is recorded. The
alternative — trusting internal state because it is the one the strategy
reasoned about — is how a platform ends up confident about a position it does
not hold.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from quantforge.core.enums import RiskLevel, TradingState
from quantforge.core.errors import ReconciliationError
from quantforge.core.timestamps import utcnow
from quantforge.execution.brokers.base import BrokerPosition


@dataclass(slots=True)
class PositionSnapshot:
    """One side's view of positions at a point in time."""

    source: str
    as_of: dt.datetime
    positions: dict[str, float] = field(default_factory=dict)
    quantities: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"source": self.source, "as_of": self.as_of.isoformat(),
                "positions": dict(self.positions)}


@dataclass(slots=True)
class Discrepancy:
    instrument_id: str
    internal_quantity: float
    broker_quantity: float
    difference: float
    internal_price: float = 0.0
    broker_price: float = 0.0
    price_difference: float = 0.0
    severity: str = "warning"
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "internal_quantity": self.internal_quantity,
            "broker_quantity": self.broker_quantity,
            "difference": self.difference,
            "internal_price": self.internal_price,
            "broker_price": self.broker_price,
            "price_difference": self.price_difference,
            "severity": self.severity,
            "explanation": self.explanation,
        }


@dataclass(slots=True)
class ReconciliationReport:
    matched: bool
    state: TradingState
    risk_level: RiskLevel
    discrepancies: list[Discrepancy] = field(default_factory=list)
    corrections_applied: list[dict] = field(default_factory=list)
    performed_at: dt.datetime = field(default_factory=utcnow)
    n_internal: int = 0
    n_broker: int = 0
    max_abs_quantity_difference: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "matched": self.matched,
            "state": self.state.value,
            "risk_level": self.risk_level.value,
            "performed_at": self.performed_at.isoformat(),
            "n_internal": self.n_internal,
            "n_broker": self.n_broker,
            "max_abs_quantity_difference": self.max_abs_quantity_difference,
            "discrepancies": [d.as_dict() for d in self.discrepancies],
            "corrections_applied": self.corrections_applied,
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            f"Reconciliation at {self.performed_at.isoformat()}: "
            f"{'MATCHED' if self.matched else 'MISMATCH'}",
            f"  internal positions: {self.n_internal}   broker positions: {self.n_broker}",
            f"  state: {self.state.value}   risk: {self.risk_level.value}",
            f"  max |quantity difference|: {self.max_abs_quantity_difference:.6f}",
        ]
        if self.discrepancies:
            lines.append("  DISCREPANCIES:")
            for d in self.discrepancies[:20]:
                lines.append(
                    f"    {d.instrument_id}: internal {d.internal_quantity:+.4f} vs "
                    f"broker {d.broker_quantity:+.4f} (diff {d.difference:+.4f}) "
                    f"[{d.severity}] {d.explanation}"
                )
        for n in self.notes:
            lines.append(f"  note: {n}")
        if not self.matched:
            lines.append("")
            lines.append(
                "  Trading is blocked. Do not trade against a position record that "
                "does not match the broker: the internal view is the one that is "
                "wrong, and acting on it will compound the discrepancy."
            )
        return "\n".join(lines)


class PositionReconciler:
    """Compares internal and broker state and reports differences."""

    def __init__(
        self,
        quantity_tolerance: float = 1e-6,
        price_tolerance_pct: float = 0.01,
        auto_correct: bool = False,
    ) -> None:
        self.quantity_tolerance = quantity_tolerance
        self.price_tolerance_pct = price_tolerance_pct
        #: Corrections are **not** applied automatically by default. A quantity
        #: difference has a cause — a missed fill, a manual trade, a corporate
        #: action — and overwriting the internal record destroys the evidence.
        self.auto_correct = auto_correct
        self.history: list[ReconciliationReport] = []

    def snapshot_internal(self, portfolio, when: Optional[dt.datetime] = None) -> PositionSnapshot:
        return PositionSnapshot(
            source="internal", as_of=when or utcnow(),
            positions={i: p.quantity for i, p in portfolio.positions.items()
                       if not p.is_flat},
            quantities={i: p.average_entry_price for i, p in portfolio.positions.items()
                        if not p.is_flat},
        )

    def snapshot_broker(
        self, broker_positions: Sequence[BrokerPosition],
        when: Optional[dt.datetime] = None,
    ) -> PositionSnapshot:
        return PositionSnapshot(
            source="broker", as_of=when or utcnow(),
            positions={p.instrument_id: p.quantity for p in broker_positions
                       if abs(p.quantity) > self.quantity_tolerance},
            quantities={p.instrument_id: p.average_entry_price
                        for p in broker_positions},
        )

    def reconcile(
        self,
        internal: PositionSnapshot,
        broker: PositionSnapshot,
        block_on_mismatch: bool = True,
    ) -> ReconciliationReport:
        """Compare and classify. Never silently agrees a difference away."""
        instruments = sorted(set(internal.positions) | set(broker.positions))
        discs: list[Discrepancy] = []
        max_diff = 0.0

        for iid in instruments:
            qi = float(internal.positions.get(iid, 0.0))
            qb = float(broker.positions.get(iid, 0.0))
            diff = qi - qb
            if abs(diff) > self.quantity_tolerance:
                max_diff = max(max_diff, abs(diff))
                internal_only = abs(qi) > self.quantity_tolerance
                broker_only = abs(qb) > self.quantity_tolerance
                if internal_only and broker_only:
                    explanation = (
                        "both sides hold a position but the quantities differ; a "
                        "partial fill is unaccounted for on one side"
                    )
                    severity = "critical"
                elif internal_only:
                    explanation = (
                        "the platform believes a position that the broker does not "
                        "have. Either a fill was recorded that never happened, or "
                        "the order was rejected after being marked filled. Do not "
                        "trade this instrument until resolved."
                    )
                    severity = "critical"
                else:
                    explanation = (
                        "the broker holds a position the platform does not know "
                        "about. A manual trade outside the platform, or a fill "
                        "whose acknowledgement was lost, would produce this."
                    )
                    severity = "critical"
                discs.append(Discrepancy(
                    instrument_id=iid, internal_quantity=qi, broker_quantity=qb,
                    difference=diff,
                    internal_price=float(internal.quantities.get(iid, 0.0)),
                    broker_price=float(broker.quantities.get(iid, 0.0)),
                    severity=severity, explanation=explanation,
                ))
            else:
                pi = float(internal.quantities.get(iid, 0.0))
                pb = float(broker.quantities.get(iid, 0.0))
                if pi > 0 and pb > 0:
                    pdiff = (pi - pb) / pb
                    if abs(pdiff) > self.price_tolerance_pct:
                        discs.append(Discrepancy(
                            instrument_id=iid, internal_quantity=qi,
                            broker_quantity=qb, difference=0.0,
                            internal_price=pi, broker_price=pb,
                            price_difference=pdiff, severity="warning",
                            explanation=(
                                f"quantities agree but average entry prices differ "
                                f"by {pdiff:.2%}. Usually a partial-fill accounting "
                                "difference rather than a position error."
                            ),
                        ))

        matched = not any(d.severity == "critical" for d in discs)
        report = ReconciliationReport(
            matched=matched,
            state=TradingState.NORMAL if matched else TradingState.REQUIRES_RECONCILIATION,
            risk_level=RiskLevel.GREEN if matched else RiskLevel.RED,
            discrepancies=discs,
            n_internal=len(internal.positions),
            n_broker=len(broker.positions),
            max_abs_quantity_difference=max_diff,
        )
        if not matched:
            report.notes.append(
                f"{len([d for d in discs if d.severity == 'critical'])} critical "
                "discrepancy(ies). New orders are blocked."
            )
        self.history.append(report)
        return report

    def reconcile_portfolio(
        self, portfolio, broker_positions: Sequence[BrokerPosition],
        block_on_mismatch: bool = True,
        when: Optional[dt.datetime] = None,
    ) -> ReconciliationReport:
        internal = self.snapshot_internal(portfolio, when)
        broker = self.snapshot_broker(broker_positions, when)
        return self.reconcile(internal, broker, block_on_mismatch)

    def apply_corrections(
        self, portfolio, report: ReconciliationReport, actor: str = "reconciler"
    ) -> list[dict]:
        """Overwrite internal quantities with broker quantities.

        Only called explicitly, and every change is recorded with the previous
        value. Automatic correction is off by default because it destroys the
        evidence of *why* the two diverged, and the cause is usually the thing
        worth understanding.
        """
        applied: list[dict] = []
        for d in report.discrepancies:
            if d.severity != "critical" or abs(d.difference) <= self.quantity_tolerance:
                continue
            pos = portfolio.get_or_create(d.instrument_id)
            before = pos.quantity
            pos.quantity = d.broker_quantity
            if abs(d.broker_quantity) < 1e-12:
                pos.average_entry_price = 0.0
            elif d.broker_price > 0:
                pos.average_entry_price = d.broker_price
            applied.append({
                "instrument_id": d.instrument_id,
                "previous_quantity": before,
                "new_quantity": d.broker_quantity,
                "actor": actor,
                "at": utcnow().isoformat(),
                "reason": "internal state corrected to broker state",
                "original_discrepancy": d.as_dict(),
            })
        if applied:
            report.corrections_applied = applied
        return applied

    def reconcile_unknown_orders(
        self, adapter, order_book, max_attempts: int = 3
    ) -> dict:
        """Resolve every order in UNKNOWN state against the broker."""
        unknown = order_book.unknown_orders()
        if not unknown:
            return {"n_unknown": 0, "resolved": [], "unresolved": []}
        outcomes = adapter.reconcile_unknown_orders(
            [o.client_order_id for o in unknown]
        )
        resolved, unresolved = [], []
        from quantforge.core.enums import OrderStatus

        for o in unknown:
            outcome = outcomes.get(o.client_order_id)
            if outcome is None or not outcome.get("resolved"):
                unresolved.append({"client_order_id": o.client_order_id,
                                   "detail": outcome})
                continue
            if outcome["state"] == "not_found":
                # Never reached the broker: the order is genuinely unplaced.
                order_book.transition(o.order_id, OrderStatus.REJECTED,
                                      reason="reconciliation: broker has no record",
                                      reject_reason="not submitted to broker")
                resolved.append({"client_order_id": o.client_order_id,
                                 "action": "marked rejected; safe to resubmit"})
            else:
                state_map = {
                    "new": OrderStatus.ACCEPTED, "accepted": OrderStatus.ACCEPTED,
                    "pending": OrderStatus.ACCEPTED, "open": OrderStatus.ACCEPTED,
                    "partially_filled": OrderStatus.PARTIALLY_FILLED,
                    "filled": OrderStatus.FILLED,
                    "canceled": OrderStatus.CANCELLED,
                    "cancelled": OrderStatus.CANCELLED,
                    "rejected": OrderStatus.REJECTED,
                    "expired": OrderStatus.EXPIRED,
                }
                new_state = state_map.get(outcome["state"])
                if new_state is None:
                    # An unmapped state is not resolved. Defaulting it to
                    # ACCEPTED would invent a live order.
                    unresolved.append({
                        "client_order_id": o.client_order_id,
                        "detail": f"broker reported unmapped state "
                                  f"{outcome['state']!r}; escalated, not assumed",
                    })
                    continue
                remote = outcome.get("remote", {})
                try:
                    order_book.transition(
                        o.order_id, new_state,
                        reason=f"reconciliation resolved state to {outcome['state']}",
                        filled_quantity=remote.get("filled_quantity"),
                        fill_price=remote.get("average_fill_price"),
                        broker_order_id=remote.get("broker_order_id"),
                    )
                    resolved.append({"client_order_id": o.client_order_id,
                                     "action": f"state set to {new_state.value}",
                                     "was_live": outcome.get("action") == "cancel"})
                except Exception as exc:
                    unresolved.append({"client_order_id": o.client_order_id,
                                       "detail": str(exc)})
        return {"n_unknown": len(unknown), "resolved": resolved,
                "unresolved": unresolved}

    def latest(self) -> Optional[ReconciliationReport]:
        return self.history[-1] if self.history else None
