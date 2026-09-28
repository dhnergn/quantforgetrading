"""Instrument abstraction for multi-asset support.

The design constraint is that nothing downstream may assume equity semantics.
A bond has no shares outstanding, a currency pair has no sector, and an index
has no order book. The contract layer carries whatever specification the asset
class actually needs, and asset-class-specific code paths key off
:attr:`Instrument.asset_class` rather than guessing.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Optional

from quantforge.core.enums import AssetClass, InstrumentStatus
from quantforge.core.errors import CalibrationError
from quantforge.core.timestamps import UTC, ensure_aware


# --------------------------------------------------------------------------
# Contract specifications per asset class
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityContractSpec:
    shares_outstanding: Optional[float] = None
    float_shares: Optional[float] = None
    shortable: bool = True
    borrow_fee_bps: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ETFContractSpec:
    aum: Optional[float] = None
    expense_ratio: Optional[float] = None
    holdings_count: Optional[int] = None
    leverage_ratio: Optional[float] = None  # 2.0 for a 2x leveraged ETF


@dataclass(frozen=True, slots=True)
class IndexContractSpec:
    """An index is not directly tradable; the tradable proxy carries this."""

    proxy_instrument_id: Optional[str] = None
    weighting: str = "market_cap"  # market_cap | price | equal | fundamental
    reconstitution_frequency: Optional[str] = None
    total_return: bool = False


@dataclass(frozen=True, slots=True)
class BondContractSpec:
    face_value: float = 100.0
    coupon_rate: Optional[float] = None  # annualised, decimal
    coupon_frequency: Optional[float] = None  # payments per year
    maturity_date: Optional[dt.date] = None
    duration: Optional[float] = None
    convexity: Optional[float] = None
    credit_rating: Optional[str] = None
    yield_to_maturity: Optional[float] = None
    accrued_interest: Optional[float] = None


@dataclass(frozen=True, slots=True)
class CommodityContractSpec:
    contract_multiplier: float = 1.0
    underlying: Optional[str] = None
    delivery_month: Optional[str] = None
    expiry_date: Optional[dt.date] = None
    contract_size: Optional[str] = None
    roll_reference: Optional[str] = None
    is_physical: bool = True


@dataclass(frozen=True, slots=True)
class CurrencyContractSpec:
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    is_cross: bool = False
    pip_size: float = 0.0001
    is_inverse: bool = False  # quote/base inversion used by some providers


@dataclass(frozen=True, slots=True)
class FutureContractSpec(CommodityContractSpec):
    """Futures and commodity futures share most fields."""


@dataclass(frozen=True, slots=True)
class SectorContractSpec:
    proxy_instrument_id: Optional[str] = None
    member_count: Optional[int] = None


@dataclass(frozen=True, slots=True)
class GenericContractSpec:
    """Fallback for anything without a dedicated spec."""

    extra: dict[str, Any] = field(default_factory=dict)


_CONTRACT_SPEC_TYPES: dict[AssetClass, type] = {
    AssetClass.EQUITY: EquityContractSpec,
    AssetClass.ETF: ETFContractSpec,
    AssetClass.INDEX: IndexContractSpec,
    AssetClass.BOND: BondContractSpec,
    AssetClass.COMMODITY: CommodityContractSpec,
    AssetClass.FUTURE: FutureContractSpec,
    AssetClass.CURRENCY: CurrencyContractSpec,
    AssetClass.SECTOR: SectorContractSpec,
}


def default_contract_spec(asset_class: AssetClass) -> Any:
    factory = _CONTRACT_SPEC_TYPES.get(asset_class, GenericContractSpec)
    return factory()


# --------------------------------------------------------------------------
# Instrument
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradable instrument.

    ``instrument_id`` is the platform's stable primary key. ``symbol`` is not
    unique over time — a ticker can be reassigned after a corporate action — so
    every price record is keyed on ``instrument_id``, never on symbol.
    """

    instrument_id: str
    symbol: str
    asset_class: AssetClass
    exchange: str
    currency: str
    country: str = "USA"
    sector: Optional[str] = None
    industry: Optional[str] = None
    tick_size: float = 0.01
    lot_size: int = 1
    trading_calendar: str = "XNYS"
    timezone: str = "America/New_York"
    status: InstrumentStatus = InstrumentStatus.ACTIVE
    contract_spec: Any = None
    listed_date: Optional[dt.date] = None
    delisted_date: Optional[dt.date] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tick_size <= 0:
            raise CalibrationError(f"{self.instrument_id}: tick_size must be > 0")
        if self.lot_size < 1:
            raise CalibrationError(f"{self.instrument_id}: lot_size must be >= 1")
        if len(self.currency) != 3:
            raise CalibrationError(
                f"{self.instrument_id}: currency must be ISO-4217 alpha-3, "
                f"got {self.currency!r}"
            )
        if self.contract_spec is None:
            object.__setattr__(
                self, "contract_spec", default_contract_spec(self.asset_class)
            )

    # -- asset-class capabilities -----------------------------------------

    @property
    def is_equity_like(self) -> bool:
        """True when share-count semantics apply (price x shares = notional)."""
        return self.asset_class in (AssetClass.EQUITY, AssetClass.ETF)

    @property
    def is_directly_tradable(self) -> bool:
        """Indices and sector baskets are analytical; they need a proxy to trade."""
        return self.asset_class not in (AssetClass.INDEX, AssetClass.SECTOR)

    @property
    def is_fixed_income(self) -> bool:
        return self.asset_class == AssetClass.BOND

    @property
    def needs_duration_risk(self) -> bool:
        spec = self.contract_spec
        return isinstance(spec, BondContractSpec)

    # -- helpers -----------------------------------------------------------

    def round_price(self, price: float) -> float:
        """Snap a price to the instrument's tick grid, away from zero."""
        if not math.isfinite(price):
            raise CalibrationError(f"non-finite price for {self.instrument_id}")
        if self.tick_size <= 0:
            return price
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def round_quantity(self, quantity: float) -> int:
        """Truncate to whole lots. Rounding down avoids over-sizing."""
        return int(math.floor(abs(quantity) / self.lot_size)) * self.lot_size

    def is_tradable_at(self, when: dt.datetime) -> bool:
        """Whether the instrument existed and had not been delisted at ``when``."""
        ts = ensure_aware(when)
        assert ts is not None
        d = ts.astimezone(UTC).date()
        if self.listed_date and d < self.listed_date:
            return False
        if self.delisted_date and d > self.delisted_date:
            return False
        return self.status != InstrumentStatus.SUSPENDED

    def to_dict(self) -> dict[str, Any]:
        spec = self.contract_spec
        if spec is None:
            spec_dict: dict[str, Any] = {}
        elif is_dataclass(spec):
            spec_dict = {
                k: (v.isoformat() if isinstance(v, (dt.date, dt.datetime)) else v)
                for k, v in asdict(spec).items()
            }
        else:  # pragma: no cover - defensive
            spec_dict = {"repr": repr(spec)}
        return {
            "instrument_id": self.instrument_id,
            "symbol": self.symbol,
            "asset_class": self.asset_class.value,
            "exchange": self.exchange,
            "currency": self.currency,
            "country": self.country,
            "sector": self.sector,
            "industry": self.industry,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "trading_calendar": self.trading_calendar,
            "timezone": self.timezone,
            "status": self.status.value,
            "listed_date": self.listed_date.isoformat() if self.listed_date else None,
            "delisted_date": (
                self.delisted_date.isoformat() if self.delisted_date else None
            ),
            "contract_spec": spec_dict,
            "metadata": self.metadata,
        }

    def fingerprint(self) -> str:
        """Stable hash of the full identity. Feeds the reproducibility hash."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Instrument registry
# --------------------------------------------------------------------------


class InstrumentRegistry:
    """Point-in-time-aware instrument master.

    Two responsibilities that are easy to conflate and expensive to get wrong:

    1. Resolve a symbol as it was known at a historical date (so a research
       universe built for 2012 does not pick up today's ticker).
    2. Answer "which instruments existed at date D", including those later
       delisted or bankrupt — the basis of survivorship-bias control.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, Instrument] = {}
        self._symbol_intervals: dict[str, list[tuple[dt.date, dt.date, str]]] = {}

    def register(self, instrument: Instrument) -> Instrument:
        if instrument.instrument_id in self._by_id:
            raise CalibrationError(
                f"instrument_id already registered: {instrument.instrument_id}"
            )
        self._by_id[instrument.instrument_id] = instrument
        start = instrument.listed_date or dt.date(1900, 1, 1)
        end = instrument.delisted_date or dt.date(2999, 12, 31)
        self._symbol_intervals.setdefault(instrument.symbol, []).append(
            (start, end, instrument.instrument_id)
        )
        for key in self._symbol_intervals:
            self._symbol_intervals[key].sort()
        return instrument

    def get(self, instrument_id: str) -> Instrument:
        try:
            return self._by_id[instrument_id]
        except KeyError:
            raise CalibrationError(f"unknown instrument_id: {instrument_id}") from None

    def has(self, instrument_id: str) -> bool:
        return instrument_id in self._by_id

    def all(self) -> list[Instrument]:
        return list(self._by_id.values())

    def resolve(self, symbol: str, as_of: Optional[dt.datetime] = None) -> Instrument:
        """Resolve a symbol to the instrument that owned it at ``as_of``.

        Ambiguity (a ticker reused across two non-overlapping entities) is
        resolved by interval; simultaneous overlap is a data error.
        """
        entries = self._symbol_intervals.get(symbol)
        if not entries:
            raise CalibrationError(f"unknown symbol: {symbol}")
        if as_of is None:
            matches = [e for e in entries]
        else:
            d = ensure_aware(as_of).astimezone(UTC).date()  # type: ignore[union-attr]
            matches = [e for e in entries if e[0] <= d <= e[1]]
        if not matches:
            raise CalibrationError(f"symbol {symbol} not resolvable as of {as_of}")
        if len(matches) > 1:
            ids = sorted({m[2] for m in matches})
            if len(ids) > 1:
                raise CalibrationError(
                    f"symbol {symbol} is ambiguous as of {as_of}: {ids}"
                )
        return self._by_id[matches[-1][2]]

    def universe_at(
        self,
        date: dt.date | dt.datetime,
        asset_classes: Optional[set[AssetClass]] = None,
        include_delisted: bool = True,
    ) -> list[Instrument]:
        """All instruments alive on ``date``.

        ``include_delisted=True`` is the survivorship-bias control: a universe
        built for 2008 must contain the firms that failed in 2009.
        """
        d = (
            ensure_aware(date).astimezone(UTC).date()  # type: ignore[union-attr]
            if isinstance(date, dt.datetime)
            else date
        )
        out: list[Instrument] = []
        for inst in self._by_id.values():
            if asset_classes and inst.asset_class not in asset_classes:
                continue
            if inst.listed_date and d < inst.listed_date:
                continue
            if inst.delisted_date and d > inst.delisted_date:
                if not include_delisted:
                    continue
                # Alive at `d`; delisting is a later event.
            if inst.status == InstrumentStatus.DELISTED and not include_delisted:
                if not inst.delisted_date or inst.delisted_date < d:
                    continue
            out.append(inst)
        return sorted(out, key=lambda i: i.instrument_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, instrument_id: object) -> bool:
        return instrument_id in self._by_id


# --------------------------------------------------------------------------
# Universe (point-in-time membership)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UniverseMembership:
    """One instrument's membership in a universe over an interval."""

    universe_id: str
    instrument_id: str
    start_date: dt.date
    end_date: Optional[dt.date] = None  # None = still a member
    entry_reason: str = "initial"
    exit_reason: Optional[str] = None

    def covers(self, on: dt.date) -> bool:
        if on < self.start_date:
            return False
        return self.end_date is None or on <= self.end_date


@dataclass(slots=True)
class Universe:
    """A historical index membership with explicit join/leave dates.

    Point-in-time universe membership is what makes it possible to study an
    index's actual historical constituents instead of today's survivors.
    """

    universe_id: str
    name: str
    benchmark_instrument_id: Optional[str] = None
    membership: list[UniverseMembership] = field(default_factory=list)

    def add(self, membership: UniverseMembership) -> None:
        self.membership.append(membership)

    def members_at(self, on: dt.date) -> list[str]:
        return sorted(
            m.instrument_id for m in self.membership if m.covers(on)
        )

    def leave(self, instrument_id: str, on: dt.date, reason: str) -> None:
        for m in self.membership:
            if m.instrument_id == instrument_id and m.end_date is None and on >= m.start_date:
                m.end_date = on
                m.exit_reason = reason
                return
        raise CalibrationError(
            f"{instrument_id} is not an active member of {self.universe_id} on {on}"
        )
