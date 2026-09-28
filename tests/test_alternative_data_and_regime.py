"""Regime detection, fundamentals, macro, and alternative data.

The recurring theme: every one of these domains has a way to look correct while
leaking the future. A fundamentals store that returns the latest revision, a
macro store that returns the final number, a news item timestamped by
ingestion, and a regime label readable in the bar that produced it are all
plausible, and all are look-ahead bias.

The tests assert the temporal invariants directly, because the outputs look
fine either way.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quantforge.core.errors import (
    AlternativeDataError, FundamentalDataError, MacroDataError,
)
from quantforge.core.timestamps import UTC

D = dt.datetime


def T(y, m, d, h=0):
    return D(y, m, d, h, tzinfo=UTC)


@pytest.fixture
def regime_prices():
    """A series with a deliberate regime structure: calm, violent, then dull."""
    rng = np.random.default_rng(7)
    returns = np.concatenate([
        rng.normal(0.0006, 0.010, 400),
        rng.normal(0.0003, 0.022, 400),
        rng.normal(-0.0005, 0.008, 400),
    ])
    idx = pd.date_range("2018-01-01", periods=1200, freq="B", tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=idx)


# ==========================================================================
# Regime detection
# ==========================================================================


class TestRegimeLabel:
    def test_unknown_regime_rejected(self):
        from quantforge.analysis.regime import Regime, RegimeLabel

        with pytest.raises(ValueError, match="unknown regime"):
            RegimeLabel("CRASH", "rule_based", pd.Timestamp("2024-01-01"))

    def test_ml_label_requires_a_model_version(self):
        """Otherwise a model swap silently changes every label's attribution."""
        from quantforge.analysis.regime import Regime, RegimeLabel

        with pytest.raises(ValueError, match="model_version"):
            RegimeLabel(Regime.BULL, "ml_based", pd.Timestamp("2024-01-01"))

    def test_label_cannot_be_knowable_before_its_bar(self):
        from quantforge.analysis.regime import Regime, RegimeLabel

        with pytest.raises(ValueError, match="precedes the label timestamp"):
            RegimeLabel(Regime.BULL, "rule_based", pd.Timestamp("2024-01-02"),
                        available_at=pd.Timestamp("2024-01-01"))

    def test_available_at_defaults_to_the_timestamp(self):
        from quantforge.analysis.regime import Regime, RegimeLabel

        lab = RegimeLabel(Regime.BEAR, "rule_based", pd.Timestamp("2024-01-02"))
        assert lab.available_at == pd.Timestamp("2024-01-02")


class TestRuleBasedRegime:
    def test_labels_the_injected_volatility_regime(self, regime_prices):
        from quantforge.analysis.regime import RuleBasedRegimeDetector

        frame = RuleBasedRegimeDetector().detect_series(regime_prices)
        # The violent middle segment should be mostly HIGH_VOLATILITY.
        middle = frame["regime"].iloc[450:750].value_counts()
        assert middle.get("HIGH_VOLATILITY", 0) > 0.5 * middle.sum(), \
            f"the injected volatility regime was not detected: {middle.to_dict()}"

    def test_warmup_is_unlabelled_not_guessed(self, regime_prices):
        from quantforge.analysis.regime import RuleBasedRegimeDetector

        frame = RuleBasedRegimeDetector().detect_series(regime_prices)
        assert frame["regime"].iloc[:50].isna().all(), (
            "the warmup must be unlabelled; a volatility percentile needs a "
            "distribution to compare against, and inventing one is a guess"
        )
        assert frame["regime"].notna().sum() > 1000

    def test_available_at_is_strictly_after_the_label_bar(self, regime_prices):
        """The causality property. A zero lag would make labels look tradable
        in the bar that produced them."""
        from quantforge.analysis.regime import RuleBasedRegimeDetector

        frame = RuleBasedRegimeDetector().detect_series(regime_prices)
        ts = pd.Series(frame.index, index=frame.index)
        labelled = frame["regime"].notna()
        assert (frame.loc[labelled, "available_at"] > ts[labelled]).all()

    def test_zero_lag_is_still_expressible_but_marks_the_bar(self, regime_prices):
        from quantforge.analysis.regime import RuleBasedRegimeDetector

        frame = RuleBasedRegimeDetector().detect_series(regime_prices, labelling_lag=0)
        ts = pd.Series(frame.index, index=frame.index)
        assert (frame["available_at"] == ts).all()

    def test_volatility_reference_is_expanding(self, regime_prices):
        """Using the full-sample distribution would let a later crash relabel an
        earlier calm period."""
        from quantforge.analysis.regime import RuleBasedRegimeDetector

        frame = RuleBasedRegimeDetector().detect_series(regime_prices)
        hi = frame["vol_high_threshold"]
        assert hi.notna().sum() > 500
        # The threshold rises as high-vol periods enter the history.
        assert hi.iloc[-1] > hi.dropna().iloc[100]

    def test_too_short_a_series_raises(self):
        from quantforge.analysis.regime import RuleBasedRegimeDetector

        with pytest.raises(ValueError, match="too few"):
            RuleBasedRegimeDetector().detect_series(pd.Series([100.0] * 20))

    def test_single_state_label_carries_its_parameters(self, regime_prices):
        from quantforge.analysis.regime import RuleBasedRegimeDetector

        lab = RuleBasedRegimeDetector().detect(regime_prices)
        assert lab.method == "rule_based"
        assert lab.parameters            # provenance is recorded
        assert "realized_vol_20" in lab.features


class TestRegimeBreakdown:
    def test_shares_sum_to_one(self, regime_prices):
        from quantforge.analysis.regime import (
            RuleBasedRegimeDetector, regime_breakdown_report,
        )

        frame = RuleBasedRegimeDetector().detect_series(regime_prices)
        rets = regime_prices.pct_change(fill_method=None)
        report = regime_breakdown_report(rets, frame["regime"])
        assert report["share_of_history"].sum() == pytest.approx(1.0, abs=1e-9)
        assert report["n_periods"].sum() <= len(rets)

    def test_dominance_reports_a_regime_bet(self, regime_prices):
        """A strategy working only in a rare regime is a bet, and the share of
        history is the number that says so."""
        from quantforge.analysis.regime import (
            RuleBasedRegimeDetector, regime_dominance,
        )

        frame = RuleBasedRegimeDetector().detect_series(regime_prices)
        rets = regime_prices.pct_change(fill_method=None)
        d = regime_dominance(rets, frame["regime"])
        assert d["dominant_regime"] is not None
        assert 0 < d["dominant_regime_share_of_periods"] <= 1
        assert "not a forecast" in d["note"].lower()

    def test_unlabelled_input_is_handled(self, regime_prices):
        from quantforge.analysis.regime import regime_dominance

        empty = pd.Series([None] * len(regime_prices), index=regime_prices.index)
        d = regime_dominance(regime_prices.pct_change(fill_method=None), empty)
        assert d["dominant"] is None or d["dominant_regime"] is None


# ==========================================================================
# Fundamentals
# ==========================================================================


class TestFundamentalObservation:
    def test_available_at_is_publication_not_period_end(self):
        from quantforge.data.fundamentals import FundamentalObservation

        o = FundamentalObservation("EQ001", "net_income", 1e9, "2024-Q3",
                                   T(2024, 9, 30), T(2024, 11, 1))
        assert o.available_at == T(2024, 11, 1)
        assert o.available_at != T(2024, 9, 30)

    def test_ingested_as_of_caps_the_bound(self):
        """A figure backfilled in 2024 was not available to a 2023 decision."""
        from quantforge.data.fundamentals import FundamentalObservation

        o = FundamentalObservation("EQ001", "net_income", 1e9, "2024-Q3",
                                   T(2024, 9, 30), T(2024, 11, 1),
                                   ingested_as_of=T(2025, 5, 1))
        assert o.available_at == T(2025, 5, 1)

    def test_published_before_the_period_ends_is_rejected(self):
        from quantforge.data.fundamentals import FundamentalObservation

        with pytest.raises(FundamentalDataError, match="precedes period_end"):
            FundamentalObservation("X", "revenue", 1, "2024-Q3",
                                   T(2024, 9, 30), T(2024, 8, 1))

    def test_ingested_before_published_is_rejected(self):
        from quantforge.data.fundamentals import FundamentalObservation

        with pytest.raises(FundamentalDataError, match="precedes publication"):
            FundamentalObservation("X", "revenue", 1, "2024-Q3",
                                   T(2024, 9, 30), T(2024, 11, 1),
                                   ingested_as_of=T(2024, 10, 1))

    def test_negative_revision_rejected(self):
        from quantforge.data.fundamentals import FundamentalObservation

        with pytest.raises(FundamentalDataError, match="revision must be"):
            FundamentalObservation("X", "revenue", 1, "2024-Q3",
                                   T(2024, 9, 30), T(2024, 11, 1), revision=-1)

    def test_unit_is_inherited_from_the_metric(self):
        from quantforge.data.fundamentals import FundamentalObservation

        o = FundamentalObservation("X", "roe", 0.15, "2024-Q3",
                                   T(2024, 9, 30), T(2024, 11, 1))
        assert o.unit == "fraction"
        unknown = FundamentalObservation("X", "not_a_metric", 1, "2024-Q3",
                                         T(2024, 9, 30), T(2024, 11, 1))
        assert unknown.unit == "unknown"


class TestFundamentalStore:
    def _store(self):
        from quantforge.data.fundamentals import (
            FundamentalObservation, FundamentalStore,
        )

        s = FundamentalStore()
        s.add(FundamentalObservation("EQ001", "net_income", 1.0e9, "2024-Q3",
                                     T(2024, 9, 30), T(2024, 11, 1)))
        s.add(FundamentalObservation("EQ001", "net_income", 1.1e9, "2024-Q3",
                                     T(2024, 9, 30), T(2025, 3, 15), revision=1))
        return s

    def test_query_before_the_revision_returns_the_original(self):
        """The property that makes the store safe. Using the restated value on
        an earlier date is look-ahead."""
        s = self._store()
        assert s.get_value("EQ001", "net_income", "2024-12-01") == 1.0e9
        assert s.get_value("EQ001", "net_income", "2025-04-01") == 1.1e9

    def test_query_before_publication_returns_nothing(self):
        s = self._store()
        assert s.get_value("EQ001", "net_income", "2024-10-01") is None

    def test_both_revisions_are_retained(self):
        s = self._store()
        assert len(s.history("EQ001", "net_income")) == 2

    def test_revisions_are_immutable(self):
        from quantforge.data.fundamentals import FundamentalObservation

        s = self._store()
        with pytest.raises(FundamentalDataError, match="immutable"):
            s.add(FundamentalObservation("EQ001", "net_income", 9.9e9, "2024-Q3",
                                         T(2024, 9, 30), T(2024, 11, 2)))

    def test_readding_an_identical_observation_is_idempotent(self):
        from quantforge.data.fundamentals import FundamentalObservation

        s = self._store()
        duplicate = FundamentalObservation("EQ001", "net_income", 1.0e9, "2024-Q3",
                                           T(2024, 9, 30), T(2024, 11, 1))
        returned = s.add(duplicate)
        # The store hands back the canonical instance it already holds, not the
        # caller's copy, so downstream identity checks see one object.
        assert returned.value == duplicate.value
        assert returned.publication_date == duplicate.publication_date
        assert len(s) == 2, "a duplicate add must not grow the store"

    def test_max_lookback_expires_stale_values(self):
        s = self._store()
        assert s.get_value("EQ001", "net_income", "2025-01-01",
                           max_lookback_days=30) is None

    def test_available_fields_respect_timing(self):
        s = self._store()
        assert s.get_value("EQ001", "net_income", "2024-10-01") is None
        assert s.available_fields("EQ001", "2024-12-01") == ["net_income"]

    def test_audit_reports_revisions_and_long_lags(self):
        s = self._store()
        findings = s.audit_timing()
        assert any("revisions" in f["detail"] for f in findings)


class TestFundamentalRatios:
    def test_z_score_flags_distress(self):
        from quantforge.data.fundamentals import altman_z_score

        healthy = altman_z_score(1e6, 5e5, 2e6, 3e6, 1e7, 4e6, 8e6)
        distressed = altman_z_score(1e4, -5e5, -1e5, 2e6, 1e7, 9e6, 8e6)
        assert healthy > 1.8 > distressed

    def test_z_score_requires_positive_assets(self):
        from quantforge.data.fundamentals import altman_z_score

        with pytest.raises(FundamentalDataError, match="total_assets"):
            altman_z_score(0, 0, 0, 0, 0, 0, 0)

    def test_fcf_handles_either_capex_sign(self):
        """Feeds disagree on capex sign; a naive subtraction doubles it."""
        from quantforge.data.fundamentals import free_cash_flow

        assert free_cash_flow(1e8, 5e7) == 5e7
        assert free_cash_flow(1e8, -5e7) == 5e7

    def test_f_score_spans_its_range(self):
        from quantforge.data.fundamentals import piotroski_f_score

        best = piotroski_f_score(0.1, 5e8, 0.02, 1e10, 0.3, 0.5, 0.1,
                                 False, 0.02, 0.05)
        worst = piotroski_f_score(-0.1, -5e8, -0.02, 1e10, 0.8, 0.5, -0.1,
                                  True, -0.02, -0.05)
        assert 0 <= worst < best <= 9

    def test_f_score_requires_assets_for_the_accruals_test(self):
        from quantforge.data.fundamentals import piotroski_f_score

        with pytest.raises(FundamentalDataError, match="total_assets"):
            piotroski_f_score(0.1, 5e8, 0.02, 0, 0.3, 0.5, 0.1,
                              False, 0.02, 0.05)


# ==========================================================================
# Macro
# ==========================================================================


class TestMacroStore:
    def _store(self):
        from quantforge.data.alternative import MacroObservation, MacroStore

        s = MacroStore()
        s.add(MacroObservation("GDP", pd.Timestamp("2024-06-30"), 3.1,
                               publication_date=T(2024, 7, 25),
                               scheduled_release=T(2024, 7, 25), frequency="Q"))
        s.add(MacroObservation("GDP", pd.Timestamp("2024-06-30"), 3.4,
                               publication_date=T(2025, 3, 28), revision=1,
                               is_initial=False, frequency="Q"))
        return s

    def test_vintage_is_respected(self):
        """The most flattering form of macro look-ahead: using the final revised
        number for an earlier date."""
        s = self._store()
        assert s.value_at("GDP", "2024-09-01") == 3.1
        assert s.value_at("GDP", "2025-04-01") == 3.4

    def test_query_before_publication_returns_nothing(self):
        s = self._store()
        assert s.value_at("GDP", "2024-07-01") is None

    def test_published_before_the_period_is_rejected(self):
        from quantforge.data.alternative import MacroObservation

        with pytest.raises(MacroDataError, match="precedes the period"):
            MacroObservation("X", pd.Timestamp("2024-06-30"), 1.0,
                             publication_date=T(2024, 1, 1))

    def test_release_calendar_absence_is_flagged(self):
        s = self._store()
        findings = s.audit_calendar_coverage()
        assert any("scheduled release" in f["detail"] for f in findings)
        assert s.observation_at("GDP", "2024-09-01").release_calendar_verified


# ==========================================================================
# Alternative data
# ==========================================================================


class TestNews:
    def _store(self):
        from quantforge.data.alternative import NewsItem, NewsStore

        s = NewsStore()
        for i, (day, sent) in enumerate([(1, 0.6), (1, 0.2), (2, -0.4)]):
            s.add(NewsItem(f"n{i}", ("EQ001",), published_at=T(2024, 3, day),
                           sentiment=sent, sentiment_source="dictionary",
                           relevance=0.8, source_name="wire"))
        return s

    def test_available_at_is_publication(self):
        from quantforge.data.alternative import NewsItem

        item = NewsItem("n", ("EQ001",), published_at=T(2024, 3, 1),
                        ingested_at=T(2024, 3, 1, 3))
        assert item.available_at == T(2024, 3, 1)

    def test_ingested_before_published_is_rejected(self):
        from quantforge.data.alternative import NewsItem

        with pytest.raises(AlternativeDataError, match="precedes published_at"):
            NewsItem("n", ("EQ001",), published_at=T(2024, 3, 2),
                     ingested_at=T(2024, 3, 1))

    def test_sentiment_must_be_bounded(self):
        from quantforge.data.alternative import NewsItem

        with pytest.raises(AlternativeDataError, match="outside"):
            NewsItem("n", ("EQ001",), published_at=T(2024, 3, 1), sentiment=5.0,
                     sentiment_source="ml")

    def test_sentiment_requires_a_method(self):
        """A score with no provenance cannot be compared or reproduced."""
        from quantforge.data.alternative import NewsItem

        with pytest.raises(AlternativeDataError, match="no source"):
            NewsItem("n", ("EQ001",), published_at=T(2024, 3, 1), sentiment=0.5)

    def test_daily_sentiment_reports_the_article_count(self):
        """A mean of 0.9 from one article is not a signal; a mean of 0.0 from
        forty is conflicting news. The mean alone hides the difference."""
        s = self._store()
        daily = s.daily_sentiment("EQ001", "2024-03-01", "2024-03-03")
        assert "n_articles" in daily.columns
        assert daily.loc["2024-03-01", "n_articles"] == 2
        assert daily.loc["2024-03-01", "sentiment"] == pytest.approx(0.4)

    def test_relevance_filter_applies(self):
        s = self._store()
        assert len(s.items_between("2024-03-01", "2024-03-03",
                                   min_relevance=0.9)) == 0
        assert len(s.items_between("2024-03-01", "2024-03-03",
                                   min_relevance=0.5)) == 3


class TestSocial:
    def test_ingested_before_posted_is_rejected(self):
        from quantforge.data.alternative import SocialPost

        with pytest.raises(AlternativeDataError, match="ingested before posted"):
            SocialPost("p", ("EQ001",), posted_at=T(2024, 3, 2),
                       ingested_at=T(2024, 3, 1))

    def test_sentiment_bounded(self):
        from quantforge.data.alternative import SocialPost

        with pytest.raises(AlternativeDataError, match=r"\[-1, 1\]"):
            SocialPost("p", ("EQ001",), posted_at=T(2024, 3, 1), sentiment=2.0)


class TestAnalystRevisions:
    def _revs(self):
        from quantforge.data.alternative import AnalystRevision, RevisionAction

        return [
            AnalystRevision("EQ001", "a1", "B1", T(2024, 2, 1),
                            RevisionAction.UPGRADE, prior_rating="Hold",
                            new_rating="Buy", prior_target=100., new_target=120.),
            AnalystRevision("EQ001", "a2", "B2", T(2024, 2, 5),
                            RevisionAction.UPGRADE, prior_rating="Hold",
                            new_rating="Buy", prior_target=90., new_target=115.),
            AnalystRevision("EQ001", "a3", "B3", T(2024, 1, 1),
                            RevisionAction.DOWNGRADE, prior_rating="Buy",
                            new_rating="Hold"),
        ]

    def test_consensus_requires_a_broker_minimum(self):
        """One analyst's upgrade is an opinion, not consensus."""
        from quantforge.data.alternative import aggregate_revisions

        out = aggregate_revisions(self._revs(), "2024-03-01", 90, min_brokers=5)
        assert out["consensus"] is None
        assert "not consensus" in out["note"]

    def test_lookback_window_is_respected(self):
        from quantforge.data.alternative import aggregate_revisions

        assert aggregate_revisions(self._revs(), "2024-01-15", 90)["n_revisions"] == 1

    def test_target_change_is_relative(self):
        revs = self._revs()
        assert revs[0].target_change == pytest.approx(0.20)

    def test_unknown_action_rejected(self):
        from quantforge.data.alternative import AnalystRevision

        with pytest.raises(AlternativeDataError, match="unknown revision action"):
            AnalystRevision("E", "a", "b", T(2024, 1, 1), "sideways")

    def test_initiation_cannot_have_a_prior(self):
        from quantforge.data.alternative import AnalystRevision, RevisionAction

        with pytest.raises(AlternativeDataError, match="no prior rating"):
            AnalystRevision("E", "a", "b", T(2024, 1, 1),
                            RevisionAction.INITIATE, prior_rating="Hold")

    def test_id_is_deterministic(self):
        assert self._revs()[0].revision_id == self._revs()[0].revision_id
