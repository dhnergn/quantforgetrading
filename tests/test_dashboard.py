"""The portfolio dashboard, and the annualisation fix underneath it.

Two themes, both about a number that looks fine and is not.

**A dashboard is where a correct number misleads most easily.** An equity
curve with no benchmark shows what happened, not whether it beat anything. A
Sharpe ratio with no sample size is indistinguishable from one computed over
two thousand periods. A sector pie that silently drops the instruments with no
sector label looks complete. A zero drawdown and a blank chart are both
indistinguishable from a bug. So every component here carries its basis, and
every component that cannot be computed is marked unavailable *with a reason*
rather than rendered as zero.

**Annualisation must come from elapsed time, not from the observation count.**
``len(series) / periods_per_year`` is only right when the series has exactly
``periods_per_year`` points per year, and the codebase used it everywhere. On a
subsampled curve that understated 1.9 years as 0.2 and reported a 798,847%
annualised return on a 494% total.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.dashboard_fixtures import (
    build_portfolio, make_benchmark, make_instruments, make_per_instrument_returns,
    make_price_series,
)


@pytest.fixture
def portfolio():
    return build_portfolio()


@pytest.fixture
def report(portfolio):
    from quantforge.analysis.dashboard import build_dashboard

    return build_dashboard(portfolio, make_instruments(),
                           returns_by_instrument=make_per_instrument_returns())


# ==========================================================================
# The annualisation fix
# ==========================================================================


class TestElapsedYears:
    def test_measured_from_dates_not_point_count(self):
        """A subsampled series spans real time; its point count does not."""
        from quantforge.quant.risk_metrics import elapsed_years

        idx = pd.date_range("2020-01-01", periods=730, freq="D", tz="UTC")
        full = pd.Series(1.0, index=idx)
        sparse = full.iloc[::10]
        # 730 calendar days is two years. The sparse view has 73 points, which
        # at 252 points/year would read as 0.29 years.
        assert elapsed_years(full, 252) == pytest.approx(2.0, abs=0.01)
        assert elapsed_years(sparse, 252) == pytest.approx(2.0, abs=0.05)

    def test_falls_back_when_the_index_has_no_dates(self):
        from quantforge.quant.risk_metrics import elapsed_years

        plain = pd.Series([1.0, 2.0, 3.0])
        assert elapsed_years(plain, 252) == pytest.approx(3 / 252)

    def test_empty_series_is_zero_not_an_error(self):
        from quantforge.quant.risk_metrics import elapsed_years

        assert elapsed_years(pd.Series(dtype="float64"), 252) == 0.0

    def test_single_point_falls_back(self):
        from quantforge.quant.risk_metrics import elapsed_years

        one = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-01"]))
        assert elapsed_years(one, 252) == pytest.approx(1 / 252)


class TestCalmarIsSamplingInvariant:
    def _path(self):
        rng = np.random.default_rng(1)
        n = 1000
        return pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))),
            index=pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC"),
        )

    def test_same_path_same_calmar_regardless_of_sampling(self):
        """The property that was broken. A fixed return path must not produce a
        Calmar that depends on how densely it was sampled."""
        from quantforge.quant.risk_metrics import calmar_ratio

        daily = self._path()
        values = [calmar_ratio(s).value
                  for s in (daily, daily.iloc[::10], daily.iloc[::50])]
        assert max(values) / min(values) < 3.0, (
            f"calmar varies {max(values) / min(values):.1f}x across sampling "
            f"rates of the same price path: {values}"
        )

    def test_max_years_still_overrides(self):
        """For a synthetic index with no real dates, the caller supplies the span."""
        from quantforge.quant.risk_metrics import calmar_ratio

        daily = self._path()
        assert calmar_ratio(daily, max_years=1.0).value != calmar_ratio(daily).value

    def test_non_positive_start_equity_is_refused(self):
        """Compounding from a zero base produces a complex number, not a rate."""
        from quantforge.quant.risk_metrics import calmar_ratio

        path = self._path().copy()
        path.iloc[0] = 0.0
        result = calmar_ratio(path)
        assert np.isnan(result.value)
        assert "not positive" in result.caveat


class TestAnnualisedReturnUsesRealSpan:
    def test_subsampled_curve_is_not_annualised_as_if_dense(self, report):
        """The bug in the form the dashboard exposed it: 51 points over 23
        months is 1.9 years, not 0.2."""
        comp = report["summary_statistics"]
        assert comp.available
        assert comp.data["years_covered"] == pytest.approx(1.9, abs=0.2)
        assert comp.data["annualisation_basis"] == "measured from the index dates"
        # A 495% total return over 1.9 years is ~154% annualised, not ~800,000%.
        assert comp.data["annualised_return"] < 5.0

    def test_annualisation_basis_is_reported(self, report):
        """An annualised number whose denominator is invisible is not checkable."""
        comp = report["summary_statistics"]
        assert "annualised over" in comp.caveat
        assert comp.data["years_covered"] > 0


# ==========================================================================
# Honest component behaviour
# ==========================================================================


class TestComponentContract:
    def test_all_fourteen_are_always_present(self, report):
        """A reader who sees 13 of 14 can tell something is missing. A reader
        who sees 13 rows has no way to know a 14th was expected."""
        assert set(report.components) == set(report.COMPONENT_NAMES)
        assert len(report.COMPONENT_NAMES) == 14

    def test_unavailable_components_state_a_reason(self, report):
        for comp in report.unavailable_components():
            assert not comp.available
            assert comp.reason, f"{comp.name} is unavailable with no reason given"

    def test_available_components_carry_a_caveat(self, report):
        for name in report.available_components():
            assert report[name].caveat, f"{name} has no caveat"

    def test_small_samples_are_flagged_not_left_to_notice(self, portfolio):
        from quantforge.analysis.dashboard import summary_statistics_component

        comp = summary_statistics_component(portfolio)
        if comp.n_observations < 30:
            assert "FEWER THAN" in comp.caveat.upper()

    def test_report_serialises_to_strict_json(self, report):
        """A bare NaN in a payload is invalid JSON and some clients parse it
        as a number. Undefined must become null at the boundary."""
        payload = json.dumps(report.as_dict(), allow_nan=False)
        assert "NaN" not in payload
        json.loads(payload)


class TestEquityCurve:
    def test_rebased_so_two_capitals_can_be_compared(self, report):
        comp = report["equity_curve"]
        assert comp.available
        assert comp.data["equity"][0] == pytest.approx(1.0)

    def test_without_a_benchmark_it_says_so(self, report):
        """A strategy's own curve is not evidence it beat anything."""
        assert "no benchmark" in report["equity_curve"].caveat

    def test_benchmark_is_compared_over_common_dates(self, portfolio):
        from quantforge.analysis.dashboard import build_dashboard

        rep = build_dashboard(portfolio, make_instruments(),
                              benchmark_equity=make_benchmark())
        comp = rep["equity_curve"]
        assert comp.available
        assert comp.data["benchmark"][0] == pytest.approx(1.0)
        assert "common dates" in comp.caveat

    def test_benchmark_on_a_different_index_is_reported_not_mispaired(self, portfolio):
        """A benchmark indexed by position rather than date would silently pair
        the wrong bar with the wrong bar."""
        from quantforge.analysis.dashboard import build_dashboard

        misaligned = pd.Series(np.linspace(100, 200, 12))
        rep = build_dashboard(portfolio, make_instruments(),
                              benchmark_equity=misaligned)
        assert "benchmark" in rep["equity_curve"].caveat


class TestSectorCoverage:
    def test_unclassified_exposure_is_shown_separately(self, report):
        """Two of the four fixtures carry a sector. The other two must not be
        quietly folded into Technology."""
        rows = {r["label"]: r for r in report["sector_exposure"].data}
        assert "UNCLASSIFIED" in rows
        assert rows["UNCLASSIFIED"]["pct_of_total"] > 0.5
        # The classified share is stated as a fraction in the sentence, so the
        # size of the gap is legible without reading the chart.
        assert "of exposure carries a sector" in report["sector_exposure"].caveat
        assert "shown separately" in report["sector_exposure"].caveat

    def test_no_coverage_is_unavailable_with_a_reason(self, portfolio):
        """A currency pair has no sector. Assigning one would be an invention."""
        from quantforge.analysis.dashboard import build_dashboard

        no_sectors = {k: type(v)(v.instrument_id, v.symbol, v.asset_class,
                                 v.exchange, v.currency, sector=None)
                      for k, v in make_instruments().items()}
        rep = build_dashboard(portfolio, no_sectors)
        comp = rep["sector_exposure"]
        assert not comp.available
        assert "no position carries a sector" in comp.reason

    def test_asset_class_coverage_is_complete(self, report):
        """Every instrument has an asset class, so coverage must be 100%."""
        assert "100.0%" in report["asset_class_exposure"].caveat


class TestTailAndRatioGuards:
    def test_var_needs_enough_observations(self):
        from quantforge.analysis.dashboard import var_component
        from quantforge.backtest.portfolio import Portfolio

        comp = var_component(Portfolio(1000.0))
        assert not comp.available
        assert "tail" in comp.reason or "return" in comp.reason

    def test_flat_returns_refuse_rather_than_report_infinite(self):
        """A constant non-zero return gives an infinite Sharpe; a constant zero
        return gives an undefined one. Neither is a performance figure."""
        from quantforge.analysis.dashboard import summary_statistics_component

        comp = summary_statistics_component(build_portfolio(flat_returns=True))
        assert not comp.available
        assert "constant" in comp.reason

    def test_var_and_es_are_reported_together(self, report):
        """VaR says what is not expected to be exceeded and nothing about the
        tail beyond it. Quoting VaR alone understates tail risk."""
        comp = report["var"]
        assert comp.available
        assert comp.data["es_return"] >= comp.data["var_return"]
        assert comp.data["confidence"] == pytest.approx(0.95)
        assert comp.n_observations > 0


class TestBetaAndCorrelations:
    def test_beta_unavailable_without_a_benchmark(self, report):
        assert not report["beta"].available
        assert "no benchmark" in report["beta"].reason

    def test_beta_computed_when_a_benchmark_is_given(self, portfolio):
        from quantforge.analysis.dashboard import build_dashboard

        bench = make_benchmark()
        rep = build_dashboard(portfolio, make_instruments(),
                              benchmark_returns=bench.pct_change(fill_method=None))
        comp = rep["beta"]
        assert comp.available
        assert isinstance(comp.data["beta"], float)
        assert comp.data["n_overlapping_periods"] > 0

    def test_one_instrument_cannot_be_correlated(self):
        from quantforge.analysis.dashboard import correlation_component

        comp = correlation_component({"A": pd.Series([0.1, 0.2, 0.3])})
        assert not comp.available
        assert "at least 2" in comp.reason

    def test_constant_returns_have_no_correlation(self):
        """Constant returns are usually a data error, and a NaN cell renders
        as zero in most charts."""
        from quantforge.analysis.dashboard import correlation_component

        idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
        comp = correlation_component({
            "A": pd.Series(0.0, index=idx),
            "B": pd.Series(np.arange(100) * 0.01, index=idx),
        })
        assert not comp.available
        assert "constant" in comp.reason

    def test_correlation_matrix_is_symmetric_with_unit_diagonal(self, report):
        comp = report["correlations"]
        assert comp.available
        m = np.array(comp.data["matrix"])
        assert np.allclose(m, m.T)
        assert np.allclose(np.diag(m), 1.0)


class TestPositionsAndExposure:
    def test_gross_and_net_differ_when_a_short_is_held(self, report):
        """A book that is long and short at the same size is 200% gross and 0%
        net, and those are very different risks."""
        comp = report["exposure"]
        assert comp.data["gross_exposure"] != comp.data["net_exposure"]
        assert "gross counts long and short at full size" in comp.caveat

    def test_realised_pnl_is_reported_net_of_fees(self, report):
        comp = report["realized_pnl"]
        assert comp.available
        for row in comp.data:
            assert row["net_pnl"] == pytest.approx(row["pnl"] - row["fees"])
        assert "net of fees" in comp.caveat

    def test_drawdown_reports_current_alongside_maximum(self, report):
        """0.1% under peak reads very differently when the worst was 40%."""
        comp = report["drawdown"]
        assert "current_drawdown" in comp.data
        assert "max_drawdown" in comp.data

    def test_zero_equity_refuses_exposure_ratios(self):
        from quantforge.analysis.dashboard import exposure_component
        from quantforge.backtest.portfolio import Portfolio

        p = Portfolio(0.0)
        comp = exposure_component(p)
        assert not comp.available
        assert "equity is zero" in comp.reason


class TestDisclosure:
    def test_report_carries_the_risk_disclosure(self, report):
        payload = report.as_dict()
        assert "does not predict future" in payload["disclosure"]

    def test_availability_is_counted_honestly(self, report):
        payload = report.as_dict()
        assert payload["n_available"] + len(report.unavailable_components()) == 14
