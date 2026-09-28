"""Look-ahead bias and data-leakage prevention.

These are the most important tests in the suite. A failure here means a
backtest would report numbers that look plausible and describe a system that
could not have traded.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quantforge.core.enums import StopType
from quantforge.core.errors import LeakageError, LeakageGuardViolation
from quantforge.core.timestamps import (
    DecisionChain, LookAheadBiasError, TemporalCausalityError, TemporalEnvelope,
    UTC, utcnow,
)

pytestmark = pytest.mark.leakage


# ==========================================================================
# Timestamp semantics
# ==========================================================================


class TestTemporalEnvelope:
    def test_publication_before_event_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be published before"):
            TemporalEnvelope(
                event_time=dt.datetime(2024, 1, 10, tzinfo=UTC),
                publication_time=dt.datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_ingestion_before_publication_is_rejected(self):
        with pytest.raises(ValueError, match="cannot precede publication"):
            TemporalEnvelope(
                event_time=dt.datetime(2024, 1, 1, tzinfo=UTC),
                publication_time=dt.datetime(2024, 1, 5, tzinfo=UTC),
                ingestion_time=dt.datetime(2024, 1, 2, tzinfo=UTC),
            )

    def test_available_at_honours_ingested_as_of(self):
        env = TemporalEnvelope(
            event_time=dt.datetime(2024, 1, 1, tzinfo=UTC),
            publication_time=dt.datetime(2024, 1, 5, tzinfo=UTC),
            ingestion_time=dt.datetime(2024, 4, 1, tzinfo=UTC),
            ingested_as_of=dt.datetime(2024, 4, 1, tzinfo=UTC),
        )
        # Backfilled data is not treated as contemporaneously available.
        assert env.available_at == dt.datetime(2024, 4, 1, tzinfo=UTC)

    def test_available_at_falls_back_to_publication(self):
        env = TemporalEnvelope(
            event_time=dt.datetime(2024, 1, 1, tzinfo=UTC),
            publication_time=dt.datetime(2024, 1, 5, tzinfo=UTC),
        )
        assert env.available_at == dt.datetime(2024, 1, 5, tzinfo=UTC)

    def test_feed_latency_and_publication_lag(self):
        env = TemporalEnvelope(
            event_time=dt.datetime(2024, 1, 1, 12, tzinfo=UTC),
            publication_time=dt.datetime(2024, 1, 1, 12, 0, 5, tzinfo=UTC),
            ingestion_time=dt.datetime(2024, 1, 1, 12, 0, 8, tzinfo=UTC),
            exchange_time=dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        )
        assert env.publication_lag == dt.timedelta(seconds=5)
        assert env.feed_latency == dt.timedelta(seconds=8)

    def test_naive_datetimes_are_coerced_to_utc(self):
        env = TemporalEnvelope(event_time=dt.datetime(2024, 1, 1))
        assert env.event_time.tzinfo is not None


class TestDecisionChain:
    def test_monotonic_chain_validates(self):
        t0 = dt.datetime(2024, 1, 1, tzinfo=UTC)
        DecisionChain(
            information_cutoff=t0,
            prediction=t0 + dt.timedelta(seconds=1),
            decision=t0 + dt.timedelta(seconds=2),
            order=t0 + dt.timedelta(seconds=3),
            execution=t0 + dt.timedelta(seconds=4),
        ).validate()

    @pytest.mark.parametrize("bad", [
        ("prediction_before_cutoff", dict(prediction=dt.timedelta(seconds=-1))),
        ("decision_before_prediction", dict(prediction=dt.timedelta(seconds=2),
                                           decision=dt.timedelta(seconds=1))),
        ("execution_before_order", dict(order=dt.timedelta(seconds=5),
                                        execution=dt.timedelta(seconds=4))),
    ])
    def test_non_monotonic_chain_raises(self, bad):
        t0 = dt.datetime(2024, 1, 1, tzinfo=UTC)
        kwargs = {k: t0 + v for k, v in bad[1].items() if isinstance(v, dt.timedelta)}
        with pytest.raises(TemporalCausalityError):
            DecisionChain(information_cutoff=t0, **kwargs).validate()

    def test_latency_breakdown(self):
        t0 = dt.datetime(2024, 1, 1, tzinfo=UTC)
        chain = DecisionChain(
            information_cutoff=t0,
            prediction=t0 + dt.timedelta(seconds=1),
            decision=t0 + dt.timedelta(seconds=3),
            order=t0 + dt.timedelta(seconds=4),
            execution=t0 + dt.timedelta(seconds=6),
        )
        lat = chain.latencies()
        assert lat["model_latency_s"] == pytest.approx(1.0)
        assert lat["signal_latency_s"] == pytest.approx(2.0)
        assert lat["execution_latency_s"] == pytest.approx(2.0)


# ==========================================================================
# Feature store point-in-time retrieval
# ==========================================================================


class TestPointInTimeFeatureStore:
    def test_close_feature_unavailable_at_its_own_bar(self, ohlcv):
        """The core guarantee: bar T's close is not usable at bar T's close."""
        from quantforge.features.engineering import build_price_features, mvp_feature_set

        fs = mvp_feature_set()
        store = build_price_features(ohlcv, fs, "EQ001")
        bar = ohlcv.index[100]
        # At the bar's own timestamp the previous bar's close is returned.
        at_bar = store.get_point_in_time("EQ001", bar, ["close"])["close"]
        assert at_bar != pytest.approx(ohlcv.loc[bar, "close"])
        # One hour after the daily close boundary, this bar's close appears.
        after = store.get_point_in_time(
            "EQ001", bar + pd.Timedelta(hours=21), ["close"]
        )["close"]
        assert after == pytest.approx(ohlcv.loc[bar, "close"])

    def test_query_never_returns_a_future_value(self, ohlcv):
        from quantforge.features.engineering import build_price_features, mvp_feature_set

        fs = mvp_feature_set()
        store = build_price_features(ohlcv, fs, "EQ001")
        names = ["close", "rsi_14", "sma_20", "atr_14"]
        for t in ohlcv.index[50:150]:
            for n in names:
                audit = store.audit_leakage("EQ001", n, t + pd.Timedelta(hours=21))
                assert audit["no_leakage"], audit

    def test_audit_leakage_detects_an_injected_future_value(self, ohlcv):
        from quantforge.features.engineering import build_price_features, mvp_feature_set

        fs = mvp_feature_set()
        store = build_price_features(ohlcv, fs, "EQ001")
        bar = ohlcv.index[100]
        # Inject a value stamped far in the future.
        store.write("EQ001", "close", [bar], [999.0],
                    available_at=[bar + pd.Timedelta(days=365)])
        audit = store.audit_leakage("EQ001", "close", bar + pd.Timedelta(hours=21))
        assert audit["value"] != 999.0
        assert audit["no_leakage"] is True

    def test_strict_mode_raises_when_nothing_is_available(self, ohlcv):
        from quantforge.core.errors import FeatureUnavailableError
        from quantforge.features.engineering import build_price_features, mvp_feature_set

        fs = mvp_feature_set()
        store = build_price_features(ohlcv, fs, "EQ001")
        with pytest.raises(FeatureUnavailableError):
            store.get_point_in_time("EQ001", ohlcv.index[0] - pd.Timedelta(days=1),
                                     ["rsi_14"], strict=True)

    def test_nan_is_never_stored_as_an_available_value(self, ohlcv):
        from quantforge.features.engineering import build_price_features, mvp_feature_set

        fs = mvp_feature_set()
        store = build_price_features(ohlcv, fs, "EQ001")
        frame = store.as_frame()
        # NaN rows are not persisted at all, so nothing can read "missing" as zero.
        assert not frame["value"].isna().any()

    def test_assert_no_leakage_passes_on_a_correct_store(self, ohlcv):
        from quantforge.features.engineering import build_price_features, mvp_feature_set
        from quantforge.features.store import assert_no_leakage

        fs = mvp_feature_set()
        store = build_price_features(ohlcv, fs, "EQ001")
        report = assert_no_leakage(
            store, ["EQ001"], ["close", "rsi_14", "sma_20"],
            [t + pd.Timedelta(hours=21) for t in ohlcv.index[100:120]],
        )
        assert report.clean, report.summary()


# ==========================================================================
# Preprocessing leakage guards
# ==========================================================================


class TestPreprocessingLeakage:
    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(5)
        n = 400
        idx = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
        return pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
        }, index=idx)

    def test_transform_before_fit_raises(self, frame):
        from quantforge.ml.preprocess import TrainScaler

        with pytest.raises(LeakageError, match="before fit"):
            TrainScaler().transform(frame)

    def test_scaler_reuses_training_statistics(self, frame):
        from quantforge.ml.preprocess import TrainScaler

        train, val = frame.iloc[:300], frame.iloc[300:]
        s = TrainScaler().fit(train)
        out = s.transform(val)
        # Validation inherits the *training* mean, not its own.
        assert out["a"].mean() == pytest.approx(0.0, abs=0.2)
        assert out["a"].std() == pytest.approx(1.0, abs=0.2)

    def test_double_fit_raises(self, frame):
        from quantforge.ml.preprocess import default_pipeline

        p = default_pipeline().fit(frame.iloc[:300])
        with pytest.raises(LeakageError, match="already fitted"):
            p.fit(frame.iloc[300:])

    def test_audit_detects_a_full_dataset_fit(self, frame):
        """The exact mistake blueprint Section 7 is about."""
        from quantforge.ml.preprocess import PreprocessingPipeline, TrainScaler
        from quantforge.ml.preprocess import assert_no_future_fit

        train_idx = frame.index[:300]
        leaky = PreprocessingPipeline().add(TrainScaler()).fit(frame)  # full data
        with pytest.raises(LeakageGuardViolation, match="outside the training window"):
            assert_no_future_fit(leaky, train_idx)

        clean = PreprocessingPipeline().add(TrainScaler()).fit(frame.iloc[:300])
        assert_no_future_fit(clean, train_idx)   # must not raise

    def test_pca_leakage_changes_the_component_basis(self, frame):
        from quantforge.ml.preprocess import TrainPCA

        train = frame.iloc[:300]
        pca_full = TrainPCA(n_components=2).fit(frame)
        pca_train = TrainPCA(n_components=2).fit(train)
        assert not np.allclose(pca_full.components_, pca_train.components_)
        # The full-data fit's window extends past training.
        assert pca_full.state()["fit_end"] > pca_train.state()["fit_end"]

    def test_pca_rejects_nan(self, frame):
        from quantforge.ml.preprocess import TrainPCA

        with pytest.raises(LeakageError, match="NaN"):
            TrainPCA(n_components=1).fit(frame.assign(z=np.nan))

    def test_pca_rejects_too_few_observations(self, frame):
        from quantforge.ml.preprocess import TrainPCA

        with pytest.raises(LeakageError, match="more observations"):
            TrainPCA(n_components=1).fit(frame.iloc[:2])

    def test_variance_filter_fitted_on_train_only(self, frame):
        from quantforge.ml.preprocess import TrainVarianceFilter

        f = TrainVarianceFilter().fit(frame.assign(constant=1.0))
        assert "constant" in f.dropped_

    def test_imputer_uses_train_medians(self, frame):
        from quantforge.ml.preprocess import TrainMedianImputer

        holed = frame.copy()
        holed.iloc[0:5, 0] = np.nan
        train_median = holed.iloc[5:, 0].median()
        imp = TrainMedianImputer().fit(holed.iloc[5:])
        out = imp.transform(holed)
        assert out["a"].iloc[0] == pytest.approx(train_median)

    def test_drop_incomplete_rows_refuses_to_discard_too_much(self, frame):
        from quantforge.ml.preprocess import DropIncompleteRows

        mostly_nan = frame.copy()
        mostly_nan["a"] = np.nan
        mostly_nan.iloc[0, 1] = 1.0
        step = DropIncompleteRows(max_drop_fraction=0.5).fit(mostly_nan)
        # The guard fires on transform, which is where rows are actually lost.
        with pytest.raises(LeakageError, match="above the"):
            step.transform(mostly_nan)


# ==========================================================================
# Backtest causality
# ==========================================================================


class TestBacktestCausality:
    def _run(self, data, features, cost_model, limits):
        from quantforge.backtest.engine import BacktestEngine, BacktestConfig
        from quantforge.strategy.builder import sma_crossover_strategy

        eng = BacktestEngine(
            sma_crossover_strategy(10, 20), BacktestConfig(initial_capital=100_000.0),
            cost_model=cost_model, risk_limits=limits,
        )
        return eng.run(data, features)

    def test_truncated_run_matches_the_full_run_prefix(self, ohlcv, features,
                                                       cost_model, permissive_limits):
        """The definitive anti-look-ahead test.

        Running on bars 0..k must produce exactly the same equity curve and
        the same filled orders as the first k bars of a full run. If any future
        bar influenced a past decision, these diverge.
        """
        full = self._run(ohlcv, features, cost_model, permissive_limits)
        full_equity = full.portfolio.equity_series()
        for k in (300, 450, 550):
            part = self._run(ohlcv.iloc[:k], features.iloc[:k],
                             cost_model, permissive_limits)
            pe = part.portfolio.equity_series()
            common = pe.index.intersection(full_equity.index)
            assert len(common) > 0
            diff = (pe.reindex(common) - full_equity.reindex(common)).abs().max()
            assert diff == 0.0, f"prefix diverged at {k} bars by {diff}"

    def test_signal_is_acted_on_at_the_next_bar_not_the_signal_bar(
        self, ohlcv, features, cost_model, permissive_limits
    ):
        res = self._run(ohlcv, features, cost_model, permissive_limits)
        filled = [o for o in res.orders if o.filled_bar is not None]
        assert filled, "no orders were filled; the test would be vacuous"
        # Stop-triggered exits are not signals and correctly carry no
        # signal_bar; only signal-driven orders are subject to the one-bar gap.
        signal_orders = [o for o in filled if o.signal_bar is not None]
        assert signal_orders, "no signal-driven orders were filled"
        for o in signal_orders:
            # The fill bar must be strictly after the bar that produced the signal.
            assert o.filled_bar > o.signal_bar, (
                f"order {o.order_id} filled on {o.filled_bar} from a signal on "
                f"{o.signal_bar}: this is look-ahead bias"
            )
        for o in filled:
            if o.signal_bar is None:
                assert "stop" in o.order_type or "exit" in (o.reason or "").lower() \
                    or o.reason, "an order with no signal must be a risk exit"

    def test_fill_price_is_not_the_signal_bar_close(self, ohlcv, features,
                                                    cost_model, permissive_limits):
        res = self._run(ohlcv, features, cost_model, permissive_limits)
        for o in res.orders:
            if o.filled_bar is None or o.signal_bar is None:
                continue
            signal_close = float(ohlcv.loc[o.signal_bar, "close"])
            fill_open = float(ohlcv.loc[o.filled_bar, "open"])
            # The fill must sit at the *next* bar's open plus modelled cost.
            assert o.fill_price >= fill_open
            assert o.fill_price == pytest.approx(fill_open, rel=0.05)
            # The fill must be much closer to the next open than to the
            # signal close. A tolerance-based "not equal" check is unreliable
            # because overnight gaps are small; distance comparison is not.
            d_open = abs(o.fill_price - fill_open)
            d_close = abs(o.fill_price - signal_close)
            if abs(signal_close - fill_open) / signal_close > 0.002:
                assert d_open < d_close, (
                    f"fill {o.fill_price} is closer to the signal close "
                    f"({signal_close}) than to the next open ({fill_open})"
                )

    def test_missing_feature_fails_loudly(self, ohlcv, features, cost_model,
                                          permissive_limits):
        from quantforge.core.errors import BacktestError
        from quantforge.backtest.engine import BacktestEngine, BacktestConfig
        from quantforge.strategy.builder import sma_crossover_strategy

        eng = BacktestEngine(
            sma_crossover_strategy(5, 10), BacktestConfig(), cost_model=cost_model,
            risk_limits=permissive_limits,
        )
        with pytest.raises(BacktestError, match="absent from the feature frame"):
            eng.run(ohlcv, features)

    def test_stop_gap_fills_at_the_open_not_the_stop(self):
        from quantforge.backtest.liquidity import LiquidityModel, simulate_fill_price

        # A bar that gapped straight through a buy stop at 95: open 88.
        res = simulate_fill_price(
            "long", 88.0, 89.0, 87.0, 88.5, 100, 1e6, LiquidityModel(),
            order_type="stop", stop_price=95.0,
        )
        assert res["filled"]
        # Filling at 95 would be the classic stop-loss backtest error.
        assert res["fill_price"] < 95.0
        assert res["gap_adjustment"] > 0
        assert "gapped" in res["reason"]

    def test_stop_within_range_fills_at_the_stop(self):
        from quantforge.backtest.liquidity import LiquidityModel, simulate_fill_price

        res = simulate_fill_price(
            "long", 100.0, 101.0, 94.0, 95.0, 100, 1e6, LiquidityModel(),
            order_type="stop", stop_price=95.0,
        )
        assert res["filled"]
        assert res["fill_price"] == pytest.approx(95.0, rel=0.02)


# ==========================================================================
# Corporate actions and point-in-time fundamentals
# ==========================================================================


class TestCorporateActionCausality:
    def test_announced_date_bounds_availability(self):
        from quantforge.core.enums import CorporateActionType
        from quantforge.data.corporate_actions import CorporateAction

        a = CorporateAction(
            instrument_id="X", action_type=CorporateActionType.SPLIT,
            ex_date=dt.date(2024, 6, 1), factor=2.0,
            announced_date=dt.date(2024, 5, 15),
        )
        # Known from the announcement, not the ex-date.
        assert a.available_at == dt.datetime(2024, 5, 15, tzinfo=UTC)

    def test_announcement_after_ex_date_is_rejected(self):
        from quantforge.core.enums import CorporateActionType
        from quantforge.data.corporate_actions import CorporateAction
        from quantforge.core.errors import CorporateActionError

        with pytest.raises(CorporateActionError, match="announced after"):
            CorporateAction(
                instrument_id="X", action_type=CorporateActionType.SPLIT,
                ex_date=dt.date(2024, 5, 1), factor=2.0,
                announced_date=dt.date(2024, 6, 1),
            )

    def test_register_known_at_excludes_unannounced_actions(self):
        from quantforge.core.enums import CorporateActionType
        from quantforge.data.corporate_actions import CorporateAction, CorporateActionRegister

        reg = CorporateActionRegister()
        reg.add(CorporateAction(
            instrument_id="X", action_type=CorporateActionType.SPLIT,
            ex_date=dt.date(2024, 6, 1), factor=2.0,
            announced_date=dt.date(2024, 5, 15),
        ))
        before = reg.known_at("X", dt.datetime(2024, 5, 1, tzinfo=UTC))
        after = reg.known_at("X", dt.datetime(2024, 5, 20, tzinfo=UTC))
        assert before == []
        assert len(after) == 1

    def test_raw_prices_are_never_modified(self):
        from quantforge.core.enums import CorporateActionType
        from quantforge.data.corporate_actions import CorporateAction, apply_adjustments

        idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        raw = pd.Series(np.linspace(100, 60, 60), index=idx)
        original = raw.copy()
        actions = [CorporateAction(
            instrument_id="X", action_type=CorporateActionType.SPLIT,
            ex_date=dt.date(2024, 1, 31), factor=2.0,
            announced_date=dt.date(2024, 1, 15),
        )]
        res = apply_adjustments(raw, actions)
        pd.testing.assert_series_equal(res.raw, original)
        # A bar *before* the ex-date is restated; one after is left alone,
        # because it already trades in the current share basis.
        pre = raw.index[0]
        assert res.split_factor.loc[pre] == pytest.approx(2.0)
        assert res.adjusted_price.loc[pre] == pytest.approx(raw.loc[pre] / 2.0)
        post = raw.index[-1]
        assert res.split_factor.loc[post] == pytest.approx(1.0)
        assert res.adjusted_price.loc[post] == pytest.approx(raw.loc[post])

    def test_split_adjusted_returns_contain_no_jump(self):
        from quantforge.core.enums import CorporateActionType
        from quantforge.data.corporate_actions import CorporateAction, apply_adjustments

        idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        raw = pd.Series(100.0, index=idx)
        # The ex-date is index 30 (2024-01-31); the raw price halves from there.
        raw.iloc[30:] = 50.0
        actions = [CorporateAction(
            instrument_id="X", action_type=CorporateActionType.SPLIT,
            ex_date=dt.date(2024, 1, 31), factor=2.0,
            announced_date=dt.date(2024, 1, 15),
        )]
        res = apply_adjustments(raw, actions)
        adjusted = res.adjusted_price.pct_change().dropna()
        # A 2-for-1 split in raw prices is not a 50% loss.
        assert adjusted.abs().max() < 1e-9

    def test_dividend_is_scaled_by_the_split_factor(self):
        from quantforge.core.enums import CorporateActionType
        from quantforge.data.corporate_actions import CorporateAction, apply_adjustments

        idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        raw = pd.Series(100.0, index=idx)
        # Split ex-date at index 30 (2024-01-31); dividend ex-date 2024-02-09.
        raw.iloc[30:] = 50.0
        actions = [
            CorporateAction(instrument_id="X", action_type=CorporateActionType.SPLIT,
                            ex_date=dt.date(2024, 1, 31), factor=2.0,
                            announced_date=dt.date(2024, 1, 5)),
            # The split (2024-01-31) precedes the dividend (2024-02-09), so
            # the quoted $1.00 is already per post-split share and matches the
            # price basis on that date: it must NOT be rescaled.
            CorporateAction(instrument_id="X",
                            action_type=CorporateActionType.CASH_DIVIDEND,
                            ex_date=dt.date(2024, 2, 9), cash_per_share=1.00),
        ]
        res = apply_adjustments(raw, actions)
        assert res.dividend.loc[pd.Timestamp("2024-02-09", tz="UTC")] == \
            pytest.approx(1.00)


    def test_dividend_quoted_pre_split_is_rescaled(self):
        """A dividend announced before a later split must be restated.

        The share count changes, so a per-share amount quoted on the earlier
        basis has to be divided by the subsequent factor to be comparable with
        the post-split price series it is applied to.
        """
        from quantforge.core.enums import CorporateActionType
        from quantforge.data.corporate_actions import CorporateAction, apply_adjustments

        idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        raw = pd.Series(100.0, index=idx)
        raw.iloc[40:] = 50.0
        actions = [
            CorporateAction(instrument_id="X",
                            action_type=CorporateActionType.CASH_DIVIDEND,
                            ex_date=dt.date(2024, 1, 10), cash_per_share=1.00),
            CorporateAction(instrument_id="X",
                            action_type=CorporateActionType.SPLIT,
                            ex_date=dt.date(2024, 2, 10), factor=2.0),
        ]
        res = apply_adjustments(raw, actions)
        # $1.00 quoted pre-split is $0.50 in the post-split basis.
        assert res.dividend.loc[pd.Timestamp("2024-01-10", tz="UTC")] == \
            pytest.approx(0.5)
