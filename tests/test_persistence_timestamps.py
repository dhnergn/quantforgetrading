"""Timestamp handling in the persistence layer.

SQLite has no native ``timestamptz``. It stores the offset as text and does not
restore it on read, so a timezone-aware value written to a ``DateTime`` column
comes back **naive in UTC**. This is the shape of the problem these tests pin
down, because the two available responses both have a failure mode:

* write **naive** values (what ``datetime.utcnow`` did) - a naive timestamp
  compared against a naive local time is wrong by the machine's offset and
  never complains;
* write **aware** values and re-attach UTC on read - a reader who forgets gets
  ``TypeError: can't compare offset-naive and offset-aware datetimes``, which is
  loud.

The module now writes aware values. These tests exist so the loud failure stays
loud rather than becoming a silent wrong answer, and so nobody "fixes" the
warning by going back to ``datetime.utcnow``.
"""
from __future__ import annotations

import datetime as dt
import tempfile
import warnings

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantforge.core.timestamps import ensure_aware, utcnow
from quantforge.persistence.db import Base, InstrumentRow, KillSwitchRow, OrderRow


@pytest.fixture
def session():
    """A throwaway SQLite database. Created in a temp dir so a failure cannot
    leave a database behind for the next run to trip over."""
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    engine = create_engine(f"sqlite:///{handle.name}")
    with warnings.catch_warnings():
        # Creating the schema runs every column default once.
        warnings.simplefilter("error", DeprecationWarning)
        Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    try:
        yield factory
    finally:
        engine.dispose()
        import os

        os.unlink(handle.name)


class TestWritesAreAware:
    def test_module_no_longer_uses_the_deprecated_utcnow(self):
        """``datetime.utcnow`` is deprecated from 3.12 and returns a naive
        value, which is precisely what this module now avoids."""
        import pathlib

        source = pathlib.Path("quantforge/persistence/db.py").read_text()
        assert "dt.datetime.utcnow" not in source

    def test_writing_does_not_raise_a_deprecation_warning(self, session):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with session() as s:
                s.add(InstrumentRow(instrument_id="X", symbol="X",
                                    asset_class="EQUITY", exchange="XNAS",
                                    currency="USD"))
                s.commit()

    def test_every_timestamp_column_has_a_utc_aware_default(self):
        """A timestamp column with no default would be NULL, and a NULL
        ordering key sorts unpredictably."""
        import inspect

        from quantforge.persistence import db as dbmod

        checked = 0
        for name, obj in vars(dbmod).items():
            if not inspect.isclass(obj) or not hasattr(obj, "__table__"):
                continue
            for column in obj.__table__.columns:
                if not column.name.endswith(("_at", "_time")):
                    continue
                default = column.default
                if default is None and column.onupdate is None:
                    continue
                callable_ = default.arg if default is not None else column.onupdate.arg
                assert callable(callable_), (
                    f"{name}.{column.name} has a non-callable default"
                )
                checked += 1
        assert checked >= 8, f"only {checked} timestamp columns inspected"


class TestReadsComeBackNaive:
    def test_sqlite_drops_the_offset_on_read(self, session):
        """The behaviour these tests exist to document. It is a property of
        SQLite, not a bug introduced here."""
        with session() as s:
            s.add(InstrumentRow(instrument_id="X", symbol="X",
                                asset_class="EQUITY", exchange="XNAS",
                                currency="USD"))
            s.commit()
        with session() as s:
            row = s.query(InstrumentRow).first()
            assert row.created_at is not None
            assert row.created_at.tzinfo is None, (
                "if this now passes, the target database is not SQLite and the "
                "re-attach-on-read guidance in persistence/db.py needs revisiting"
            )

    def test_comparing_stored_value_against_aware_now_raises(self, session):
        """The loud failure. Documented so nobody treats it as a nuisance to
        silence."""
        with session() as s:
            s.add(InstrumentRow(instrument_id="X", symbol="X",
                                asset_class="EQUITY", exchange="XNAS",
                                currency="USD"))
            s.commit()
        with session() as s:
            row = s.query(InstrumentRow).first()
            with pytest.raises(TypeError, match="naive and offset-aware"):
                _ = row.created_at < utcnow()

    def test_ensure_aware_makes_it_comparable_again(self, session):
        """The documented repair, and the one readers must use."""
        with session() as s:
            s.add(InstrumentRow(instrument_id="X", symbol="X",
                                asset_class="EQUITY", exchange="XNAS",
                                currency="USD"))
            s.commit()
        with session() as s:
            row = s.query(InstrumentRow).first()
            restored = ensure_aware(row.created_at)
            assert restored.tzinfo is not None
            # Comparable now, and the value is the same instant.
            assert restored <= utcnow()
            assert abs((utcnow() - restored).total_seconds()) < 60.0

    def test_ensure_aware_is_idempotent_on_already_aware_values(self):
        moment = utcnow()
        assert ensure_aware(moment) is moment


class TestOrderTimestampsSurviveARoundTrip:
    def test_order_created_at_reads_back_comparable(self, session):
        """Order dedupe compares timestamps across restarts, so this is the
        path where a naive/aware mismatch would actually bite."""
        with session() as s:
            s.add(OrderRow(order_id="O1", client_order_id="C1",
                           instrument_id="X", side="long", order_type="market",
                           status="new", quantity=100.0))
            s.commit()
        with session() as s:
            row = s.query(OrderRow).first()
            assert ensure_aware(row.created_at) <= utcnow()
            assert ensure_aware(row.updated_at) <= utcnow()

    def test_kill_switch_rows_round_trip(self, session):
        """A kill switch is the one row whose timestamps are read back and
        compared: a released_at earlier than engaged_at is the question an
        operator actually asks."""
        with session() as s:
            s.add(KillSwitchRow(
                scope="global", reason="test", automatic=False,
                engaged_at=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
                engaged_by="test-operator",
                released_at=dt.datetime(2024, 1, 2, tzinfo=dt.UTC),
                released_by="test-operator",
            ))
            s.commit()
        with session() as s:
            row = s.query(KillSwitchRow).first()
            engaged = ensure_aware(row.engaged_at)
            released = ensure_aware(row.released_at)
            assert released > engaged, (
                "a switch released before it was engaged is a data error, and "
                "this comparison is exactly the one that raises TypeError if "
                "the offsets are not re-attached"
            )
