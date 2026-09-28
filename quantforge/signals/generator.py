"""Signal generator — the core of the signal bot.

Generates trading signals from strategies, logs them with full context,
and provides them to the execution layer (paper trading only).

Every signal is logged with:
    timestamp, instrument, strategy, signal, feature values,
    indicator values, model prediction, regime, risk state, position, decision
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from quantforge.features.engine import FeatureEngine, FeatureVector
from quantforge.models.signal import Signal as SignalModel, SignalType, SignalStatus
from quantforge.strategies.base import SignalDirection, Strategy


@dataclass
class SignalConfig:
    """Configuration for the signal generator."""
    strategy: Strategy
    feature_engine: FeatureEngine
    instruments: List[str]
    dataset_version: str = "1.0.0"
    min_confidence: float = 0.5  # Minimum confidence to emit a signal


@dataclass
class GeneratedSignal:
    """A generated signal with full context."""
    signal_id: str
    timestamp: datetime
    instrument_id: str
    signal_type: SignalType
    direction: str
    confidence: float
    features: Dict[str, float]
    indicators: Dict[str, float]
    strategy_name: str
    strategy_version: str
    decision: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalGenerator:
    """Generates trading signals from strategies.

    This is the core of the signal bot. It:
    1. Computes features for each instrument
    2. Evaluates the strategy
    3. Generates signals with full context
    4. Logs all signals for traceability

    Live trading is DISABLED. Signals are for research and paper trading only.
    """

    def __init__(self, config: SignalConfig):
        self.config = config
        self._signal_history: List[GeneratedSignal] = []

    def generate_signals(
        self,
        data: pd.DataFrame,
        as_of: Optional[datetime] = None,
    ) -> List[GeneratedSignal]:
        """Generate signals for all instruments at a specific point in time.

        Args:
            data: OHLCV DataFrame (can be MultiIndex for multiple instruments).
            as_of: Decision time. Only data up to this point is used.

        Returns:
            List of generated signals.
        """
        if as_of is None:
            as_of = datetime.utcnow()

        signals = []
        strategy = self.config.strategy

        for instrument_id in self.config.instruments:
            # Get instrument data
            instrument_data = self._get_instrument_data(data, instrument_id, as_of)
            if instrument_data is None or len(instrument_data) < 50:
                continue

            # Compute features (point-in-time)
            features = self.config.feature_engine.compute_features(
                instrument_id=instrument_id,
                data=instrument_data,
                as_of=as_of,
                feature_names=strategy.required_features(),
            )

            if not features.features:
                continue

            # Evaluate strategy
            signal = strategy.evaluate(features, position=0.0)

            if signal is None:
                continue

            # Filter by confidence
            if signal.confidence < self.config.min_confidence:
                continue

            # Create generated signal
            generated = GeneratedSignal(
                signal_id=str(uuid.uuid4()),
                timestamp=as_of,
                instrument_id=instrument_id,
                signal_type=self._map_signal_type(signal.direction),
                direction=signal.direction.value,
                confidence=signal.confidence,
                features=signal.features,
                indicators=signal.indicators,
                strategy_name=signal.strategy_name,
                strategy_version=signal.strategy_version,
                decision="enter" if signal.direction in (SignalDirection.LONG, SignalDirection.SHORT) else "exit",
                metadata={
                    "information_cutoff_time": as_of.isoformat(),
                    "prediction_time": as_of.isoformat(),
                    "decision_time": as_of.isoformat(),
                },
            )

            signals.append(generated)
            self._signal_history.append(generated)

        return signals

    def generate_signals_range(
        self,
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "1D",
    ) -> List[GeneratedSignal]:
        """Generate signals over a date range.

        Args:
            data: OHLCV DataFrame.
            start_date: Start date.
            end_date: End date.
            frequency: Signal frequency ("1D", "1H", etc.).

        Returns:
            List of all generated signals in the date range.
        """
        all_signals = []
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)

        for date in date_range:
            signals = self.generate_signals(data, as_of=date)
            all_signals.extend(signals)

        return all_signals

    def get_signal_history(self) -> List[GeneratedSignal]:
        """Get all generated signals."""
        return self._signal_history.copy()

    def clear_history(self) -> None:
        """Clear signal history."""
        self._signal_history.clear()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert signal history to DataFrame."""
        if not self._signal_history:
            return pd.DataFrame()

        records = []
        for sig in self._signal_history:
            records.append({
                "signal_id": sig.signal_id,
                "timestamp": sig.timestamp,
                "instrument_id": sig.instrument_id,
                "signal_type": sig.signal_type.value,
                "direction": sig.direction,
                "confidence": sig.confidence,
                "strategy_name": sig.strategy_name,
                "strategy_version": sig.strategy_version,
                "decision": sig.decision,
            })
        return pd.DataFrame(records)

    def _get_instrument_data(
        self,
        data: pd.DataFrame,
        instrument_id: str,
        as_of: datetime,
    ) -> Optional[pd.DataFrame]:
        """Get instrument data up to a specific point in time.

        Enforces temporal causality.
        """
        if isinstance(data.index, pd.MultiIndex):
            try:
                instrument_data = data.xs(instrument_id, level=0)
                return instrument_data[instrument_data.index <= as_of]
            except KeyError:
                return None
        else:
            return data[data.index <= as_of]

    def _map_signal_type(self, direction: SignalDirection) -> SignalType:
        """Map strategy signal direction to signal type."""
        mapping = {
            SignalDirection.LONG: SignalType.LONG,
            SignalDirection.SHORT: SignalType.SHORT,
            SignalDirection.EXIT_LONG: SignalType.EXIT_LONG,
            SignalDirection.EXIT_SHORT: SignalType.EXIT_SHORT,
            SignalDirection.HOLD: SignalType.HOLD,
        }
        return mapping.get(direction, SignalType.HOLD)
