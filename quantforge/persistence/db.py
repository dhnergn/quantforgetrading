"""SQLAlchemy persistence for operational state.

The design split is deliberate: **heavy time-series research data lives in
Parquet/columnar storage, and only small operational state lives here** —
orders, positions, model registry metadata, audit records, and configuration
history. Time-series tables grow by millions of rows per instrument per year
and belong in a columnar store, not a row store.

Portable types only (no PostgreSQL- or SQLite-specific constructs) so the same
models run on SQLite for development and PostgreSQL/TimescaleDB in production.
"""
from __future__ import annotations

import datetime as dt
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

from quantforge.core.timestamps import utcnow

from sqlalchemy import (
    JSON, Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Index,
    Integer, String, Text, UniqueConstraint, create_engine, func,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, Session, mapped_column, sessionmaker,
)


class Base(DeclarativeBase):
    pass


# ==========================================================================
# Timestamp columns
# ==========================================================================
#
# Every timestamp column here is written timezone-aware, by ``utcnow``. SQLite
# has no native ``timestamptz``: the offset is stored as text and is **not**
# restored on read, so a value written as ``datetime(...,tzinfo=UTC)`` comes
# back **naive** in UTC.
#
# That asymmetry is a trap, because comparing a naive column value against an
# aware in-memory value raises ``TypeError: can't compare offset-naive and
# offset-aware datetimes`` rather than producing a wrong answer. A reader that
# needs to compare must re-attach UTC with ``ensure_aware``. The alternative -
# writing naive values, as this module did via ``datetime.utcnow`` - trades a
# loud failure for a silent one, since a naive timestamp compared against a
# naive local time is wrong by the machine's offset and never complains.


# ==========================================================================
# Engine / session
# ==========================================================================

_engine = None
_SessionFactory: Optional[sessionmaker] = None


def get_engine(url: Optional[str] = None, echo: bool = False):
    global _engine
    if _engine is None or url is not None:
        target = url or "sqlite:///./quantforge.db"
        kwargs: dict[str, Any] = {"echo": echo, "future": True}
        if target.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs.update({"pool_pre_ping": True, "pool_size": 5,
                           "pool_recycle": 1800})
        _engine = create_engine(target, **kwargs)
    return _engine


def get_session_factory(url: Optional[str] = None) -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None or url is not None:
        _SessionFactory = sessionmaker(
            bind=get_engine(url), autocommit=False, autoflush=False, future=True
        )
    return _SessionFactory


@contextmanager
def session_scope(url: Optional[str] = None) -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception."""
    s = get_session_factory(url)()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db(url: Optional[str] = None) -> None:
    Base.metadata.create_all(bind=get_engine(url))


# ==========================================================================
# Models
# ==========================================================================


class InstrumentRow(Base):
    __tablename__ = "instruments"

    instrument_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    asset_class: Mapped[str] = mapped_column(String(32), index=True)
    exchange: Mapped[str] = mapped_column(String(64))
    currency: Mapped[str] = mapped_column(String(3))
    country: Mapped[str] = mapped_column(String(3), default="USA")
    sector: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    tick_size: Mapped[float] = mapped_column(Float, default=0.01)
    lot_size: Mapped[int] = mapped_column(Integer, default=1)
    trading_calendar: Mapped[str] = mapped_column(String(16), default="XNYS")
    timezone: Mapped[str] = mapped_column(String(64), default="America/New_York")
    status: Mapped[str] = mapped_column(String(16), default="active")
    listed_date: Mapped[Optional[dt.date]] = mapped_column(nullable=True)
    delisted_date: Mapped[Optional[dt.date]] = mapped_column(nullable=True)
    #: Survivorship-bias control: an instrument that left the universe stays
    #: queryable, with a reason, rather than disappearing.
    delisting_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    contract_spec: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    fingerprint: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class StrategyRow(Base):
    __tablename__ = "strategies"
    __table_args__ = (
        UniqueConstraint("strategy_id", "version", name="uq_strategy_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(256))
    kind: Mapped[str] = mapped_column(String(32))
    definition: Mapped[dict] = mapped_column(JSON)
    fingerprint: Mapped[str] = mapped_column(String(32), index=True)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    locked_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class BacktestRow(Base):
    __tablename__ = "backtests"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(String(64), index=True)
    strategy_version: Mapped[str] = mapped_column(String(32))
    config_hash: Mapped[str] = mapped_column(String(32), index=True)
    dataset_version: Mapped[str] = mapped_column(String(32))
    feature_set_version: Mapped[str] = mapped_column(String(32))
    manifest: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="completed")
    initial_capital: Mapped[float] = mapped_column(Float)
    final_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_orders: Mapped[int] = mapped_column(Integer, default=0)
    n_trades: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    warnings: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class OrderRow(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_book", "portfolio_id", "status"),
        Index("ix_orders_client", "client_order_id", unique=True),
    )

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(48), unique=True)
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True,
                                                           index=True)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True,
                                                           index=True)
    instrument_id: Mapped[str] = mapped_column(String(64), index=True)
    side: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[float] = mapped_column(Float)
    order_type: Mapped[str] = mapped_column(String(16))
    limit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    strategy_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    portfolio_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    signal_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    filled_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    average_fill_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, default=0.0)
    reject_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow,
                                                    index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime,
                                                    default=utcnow,
                                                    onupdate=utcnow)


class PositionRow(Base):
    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("portfolio_id", "instrument_id", name="uq_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[str] = mapped_column(String(64), index=True)
    instrument_id: Mapped[str] = mapped_column(String(64), index=True)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    average_entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    last_price: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    total_fees: Mapped[float] = mapped_column(Float, default=0.0)
    strategy_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    #: Set when internal and broker state disagree. Trading is blocked while set.
    reconciliation_required: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime,
                                                    default=utcnow,
                                                    onupdate=utcnow)


class ModelRow(Base):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("model_id", "model_version", name="uq_model_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[str] = mapped_column(String(32))
    task: Mapped[str] = mapped_column(String(32))
    stage: Mapped[str] = mapped_column(String(24), index=True)
    fingerprint: Mapped[str] = mapped_column(String(32), index=True)
    dataset_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    feature_set_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    strategy_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    random_seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    preprocessing: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    live_authorisation_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    #: Serialised estimator bytes. Kept out of the row's hot path.
    artifact_uri: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    created_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class AuditLogRow(Base):
    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_actor_action", "actor", "action"),
        Index("ix_audit_at", "at"),
    )

    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    actor: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(48), index=True)
    resource: Mapped[str] = mapped_column(String(256), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    previous_hash: Mapped[str] = mapped_column(String(64))
    entry_hash: Mapped[str] = mapped_column(String(64), index=True)
    signature: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class KillSwitchRow(Base):
    __tablename__ = "kill_switch_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope: Mapped[str] = mapped_column(String(16), index=True)
    reason: Mapped[str] = mapped_column(Text)
    automatic: Mapped[bool] = mapped_column(Boolean, default=False)
    detail: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    engaged_at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    engaged_by: Mapped[str] = mapped_column(String(64))
    released_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    released_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class ExperimentRow(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_type: Mapped[str] = mapped_column(String(32), index=True)
    config_hash: Mapped[str] = mapped_column(String(32), index=True)
    manifest: Mapped[dict] = mapped_column(JSON)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow,
                                                    index=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


# ==========================================================================
# Repositories
# ==========================================================================


class BacktestRepository:
    """Stores backtest runs so a result can be located by its config hash."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def save(self, result: Any, created_by: Optional[str] = None) -> BacktestRow:
        row = BacktestRow(
            run_id=result.run_id,
            strategy_id=result.strategy.strategy_id,
            strategy_version=result.strategy.version,
            config_hash=result.manifest.compute_hash(),
            dataset_version=result.manifest.dataset_version,
            feature_set_version=result.manifest.feature_set_version,
            manifest=result.manifest.to_dict(),
            status="completed",
            initial_capital=result.config.initial_capital,
            final_equity=result.portfolio.equity,
            n_orders=len(result.orders),
            n_trades=len(result.portfolio.trades),
            metrics=result.metrics.as_dict() if result.metrics else None,
            warnings=result.warnings,
            created_by=created_by,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def get(self, run_id: str) -> Optional[BacktestRow]:
        return self.session.get(BacktestRow, run_id)

    def find_by_config(self, config_hash: str) -> list[BacktestRow]:
        return (self.session.query(BacktestRow)
                .filter(BacktestRow.config_hash == config_hash).all())

    def recent(self, limit: int = 50) -> list[BacktestRow]:
        return (self.session.query(BacktestRow)
                .order_by(BacktestRow.created_at.desc()).limit(limit).all())


class OrderRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, order: Any) -> OrderRow:
        row = (self.session.query(OrderRow)
               .filter(OrderRow.order_id == order.order_id).one_or_none())
        if row is None:
            row = OrderRow(order_id=order.order_id,
                           client_order_id=order.client_order_id)
            self.session.add(row)
        row.broker_order_id = order.broker_order_id
        row.idempotency_key = order.idempotency_key
        row.instrument_id = order.instrument_id
        row.side = order.side.value
        row.quantity = order.quantity
        row.order_type = order.order_type.value
        row.limit_price = order.limit_price
        row.stop_price = order.stop_price
        row.status = order.status.value
        row.strategy_id = order.strategy_id
        row.portfolio_id = order.portfolio_id
        row.signal_id = order.signal_id
        row.model_version = order.model_version
        row.filled_quantity = order.filled_quantity
        row.average_fill_price = order.average_fill_price
        row.fees = order.fees
        row.slippage = order.slippage
        row.reject_reason = order.reject_reason
        row.attempts = order.attempts
        self.session.flush()
        return row

    def by_idempotency_key(self, key: str) -> Optional[OrderRow]:
        """Duplicate detection that survives a process restart.

        In-memory dedupe is lost on restart; this query is what makes the
        protection durable.
        """
        return (self.session.query(OrderRow)
                .filter(OrderRow.idempotency_key == key)
                .filter(OrderRow.status.notin_(["filled", "cancelled", "rejected",
                                                "expired"]))
                .one_or_none())

    def unknown_status(self) -> list[OrderRow]:
        return (self.session.query(OrderRow)
                .filter(OrderRow.status == "unknown").all())

    def open_orders(self, portfolio_id: Optional[str] = None) -> list[OrderRow]:
        q = self.session.query(OrderRow).filter(
            OrderRow.status.in_(["created", "submitted", "accepted",
                                 "partially_filled", "unknown"]))
        if portfolio_id:
            q = q.filter(OrderRow.portfolio_id == portfolio_id)
        return q.all()


class ModelRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, entry: Any) -> ModelRow:
        row = (self.session.query(ModelRow)
               .filter(ModelRow.model_id == entry.model_id)
               .filter(ModelRow.model_version == entry.model_version)
               .one_or_none())
        if row is None:
            row = ModelRow(model_id=entry.model_id, model_version=entry.model_version)
            self.session.add(row)
        row.task = entry.task.value
        row.stage = entry.stage.value
        row.fingerprint = entry.fingerprint
        row.locked = entry.locked
        row.approved_by = entry.approved_by
        row.live_authorisation_id = entry.live_authorisation_id
        if entry.manifest is not None:
            row.dataset_version = entry.manifest.dataset_version
            row.feature_set_version = entry.manifest.feature_set_version
            row.strategy_version = entry.manifest.strategy_version
            row.hyperparameters = entry.manifest.parameters
            row.random_seed = entry.manifest.random_seed
            row.preprocessing = entry.manifest.preprocessing
        self.session.flush()
        return row

    def by_stage(self, stage: str) -> list[ModelRow]:
        return self.session.query(ModelRow).filter(ModelRow.stage == stage).all()

    def live_eligible(self) -> list[ModelRow]:
        return self.by_stage("live_eligible")


def table_counts(session: Session) -> dict:
    """Row counts, for the system monitoring view."""
    out: dict[str, int] = {}
    for model in (InstrumentRow, StrategyRow, BacktestRow, OrderRow,
                  PositionRow, ModelRow, AuditLogRow, KillSwitchRow,
                  ExperimentRow):
        try:
            out[model.__tablename__] = session.query(model).count()
        except Exception:
            out[model.__tablename__] = -1
    return out
