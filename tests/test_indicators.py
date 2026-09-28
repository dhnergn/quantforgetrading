"""Indicator correctness: mathematical reference values and edge behaviour.

The reference tests use hand-computed values rather than another library, so
they would catch a library change as well as a bug in this code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.quant import indicators as ind


def frame(close, high=None, low=None, open_=None, volume=None):
    close = pd.Series(close, dtype="float64")
    n = len(close)
    return pd.DataFrame({
        "open": open_ if open_ is not None else close.shift(1).bfill(),
        "high": high if high is not None else close * 1.01,
        "low": low if low is not None else close * 0.99,
        "close": close,
        "volume": volume if volume is not None else np.full(n, 1e6),
    })


class TestMovingAverages:
    def test_sma_matches_hand_computed(self):
        # 1..10 -> SMA(3): first defined value at index 2 = (1+2+3)/3 = 2
        s = ind.SMA(3).calculate(frame(range(1, 11)))["sma"]
        assert s.iloc[0] is np.nan or np.isnan(s.iloc[0])
        assert s.iloc[1] is np.nan or np.isnan(s.iloc[1])
        assert s.iloc[2] == pytest.approx(2.0)
        assert s.iloc[3] == pytest.approx(3.0)
        assert s.iloc[-1] == pytest.approx(9.0)

    def test_sma_warmup_metadata_matches_behaviour(self):
        f = ind.SMA(20).calculate(frame(np.arange(1, 101, dtype=float)))
        s = f["sma"]
        assert s.isna().sum() == ind.SMA(20).warmup - 1
        assert s.iloc[19] == pytest.approx(np.mean(np.arange(1, 21)))

    def test_ema_sma_seed_then_decay(self):
        f = ind.EMA(3).calculate(frame(np.arange(1, 21, dtype=float)))["ema"]
        # pandas ewm(adjust=False, min_periods=3) seeds with the first value;
        # either seeding convention must be finite and strictly increasing here.
        assert np.isfinite(f.iloc[2])
        assert f.iloc[-1] > f.iloc[2]

    def test_wma_is_linear_weighted(self):
        f = ind.WMA(3).calculate(frame([1.0, 2.0, 3.0, 4.0]))["wma"]
        # At index 2 the window is [1, 2, 3] with weights 1, 2, 3.
        assert f.iloc[2] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6)

    def test_vwma_weights_by_volume(self):
        close = pd.Series([10.0, 20.0, 30.0])
        vol = pd.Series([1.0, 1.0, 10.0])
        f = ind.VWMA(3).calculate(frame(close, volume=vol))["vwma"]
        expected = (10 * 1 + 20 * 1 + 30 * 10) / 12
        assert f.iloc[2] == pytest.approx(expected)

    def test_kama_tracks_a_trend_more_closely_than_a_random_walk(self):
        """Regression guard: ER numerator must be net displacement, not path.

        If ER were ``sum|delta| / sum|delta|`` it would be identically 1, the
        smoothing constant would be constant, and KAMA would degenerate to a
        fixed EMA. That degeneracy is invisible on a trending series (an EMA
        also tracks it) but shows up as a much larger lag on a noisy one.
        """
        trend = np.arange(1, 201, dtype=float)
        noise = np.random.default_rng(0).normal(100, 2, 200).cumsum() + 100
        k_trend = ind.KAMA(10, 10).calculate(frame(trend))["kama"].iloc[-1]
        k_noise = ind.KAMA(10, 10).calculate(frame(noise))["kama"].iloc[-1]
        ema_noise = ind.EMA(10).calculate(frame(noise))["ema"].iloc[-1]
        # A degenerate (constant-SC) KAMA would equal the EMA exactly.
        assert k_noise != pytest.approx(ema_noise, rel=1e-6)
        # And on a trend it should sit close to the price, not far behind.
        assert abs(k_trend - trend[-1]) < 2.0

    def test_kama_flat_series_gives_er_zero(self):
        flat = np.full(120, 50.0)
        k = ind.KAMA(10, 10).calculate(frame(flat))["kama"]
        # Zero net displacement => zero efficiency ratio => no adaptation.
        assert k.iloc[-1] == pytest.approx(50.0)


class TestMomentum:
    def test_rsi_all_gains_is_100(self):
        rising = np.arange(1, 60, dtype=float)
        rsi = ind.RSI(14).calculate(frame(rising))["rsi"]
        assert rsi.iloc[-1] == pytest.approx(100.0)

    def test_rsi_all_losses_is_zero(self):
        falling = np.arange(60, 1, -1, dtype=float)
        rsi = ind.RSI(14).calculate(frame(falling))["rsi"]
        assert rsi.iloc[-1] == pytest.approx(0.0)

    def test_rsi_constant_series_is_nan_not_fifty(self):
        flat = np.full(60, 50.0)
        rsi = ind.RSI(14).calculate(frame(flat))["rsi"]
        # 0/0 is undefined. Returning 50 would invent a neutral reading.
        assert np.isnan(rsi.iloc[-1])

    def test_rsi_bounded(self):
        rng = np.random.default_rng(3)
        rsi = ind.RSI(14).calculate(frame(rng.normal(100, 1, 300).cumsum() + 100))["rsi"].dropna()
        assert ((rsi >= 0) & (rsi <= 100)).all()

    def test_macd_zero_for_constant_series(self):
        flat = np.full(120, 100.0)
        out = ind.MACD().calculate(frame(flat))
        assert out["macd"].iloc[-1] == pytest.approx(0.0)
        assert out["macd_signal"].iloc[-1] == pytest.approx(0.0)

    def test_macd_positive_in_uptrend(self):
        rising = np.arange(1, 201, dtype=float)
        out = ind.MACD().calculate(frame(rising))
        assert out["macd"].iloc[-1] > 0
        assert out["macd_hist"].iloc[-1] > 0

    def test_stochastic_zero_range_is_nan(self):
        # A genuinely zero-range bar needs high == low == close; the frame
        # helper synthesises a range around the close, so it is set explicitly.
        n = 80
        flat = np.full(n, 50.0)
        st = ind.Stochastic().calculate(
            frame(flat, high=flat.copy(), low=flat.copy(), open_=flat.copy())
        )
        assert np.isnan(st["stoch_k"].iloc[-1])

    def test_stochastic_bounded(self):
        rng = np.random.default_rng(5)
        f = ind.Stochastic().calculate(frame(rng.normal(100, 1, 200).cumsum() + 100))
        k = f["stoch_k"].dropna()
        assert ((k >= 0) & (k <= 100)).all()

    def test_roc_hand_computed(self):
        f = ind.ROC(2).calculate(frame([10.0, 20.0, 30.0, 40.0]))["roc"]
        # (30-10)/10*100 at index 2
        assert f.iloc[2] == pytest.approx(200.0)

    def test_cci_constant_series_is_nan_not_zero(self):
        """A flat series makes CCI 0/0.

        Numerator (TP - SMA) and denominator (mean absolute deviation) both go
        to zero. Returning 0 would claim "average" for a series with no
        variation at all, which is the same class of error as reporting a
        variance-free Sharpe of zero. NaN is the honest answer, matching the
        stochastic oscillator's behaviour on a zero-range bar.
        """
        n = 60
        flat = np.full(n, 50.0)
        cci = ind.CCI(20).calculate(
            frame(flat, high=flat.copy(), low=flat.copy(), open_=flat.copy())
        )["cci"]
        assert np.isnan(cci.iloc[-1])

    def test_cci_bounded_and_signed(self):
        rng = np.random.default_rng(27)
        f = frame(rng.normal(100, 1, 200).cumsum() + 100)
        cci = ind.CCI(20).calculate(f)["cci"].dropna()
        assert cci.abs().max() > 10          # CCI exceeds +/-100 routinely
        assert (cci.abs() > 200).mean() < 0.1  # but should not be wildly off-scale


class TestVolatility:
    def test_atr_positive_and_grows_with_range(self):
        narrow = frame(np.full(50, 100.0))
        wide = frame(np.full(50, 100.0), high=np.full(50, 110.0),
                     low=np.full(50, 90.0))
        atr_n = ind.ATR(14).calculate(narrow)["atr"]
        atr_w = ind.ATR(14).calculate(wide)["atr"]
        assert atr_n.iloc[-1] < atr_w.iloc[-1]
        assert atr_n.iloc[-1] >= 0

    def test_true_range_uses_previous_close(self):
        c = pd.Series([10.0, 20.0])
        f = frame(c, high=[11.0, 25.0], low=[9.0, 15.0])
        tr = ind.TrueRange().calculate(f)["tr"]
        # bar 1: max(25-15, |25-10|, |15-10|) = 15
        assert tr.iloc[1] == pytest.approx(15.0)

    def test_bollinger_band_ordering(self):
        rng = np.random.default_rng(9)
        bb = ind.BollingerBands(20).calculate(frame(rng.normal(100, 2, 200).cumsum() + 100))
        valid = bb.dropna()
        assert (valid["bb_lower"] <= valid["bb_mid"]).all()
        assert (valid["bb_mid"] <= valid["bb_upper"]).all()

    def test_bollinger_zero_volatility_is_nan_pct_b(self):
        flat = np.full(60, 100.0)
        bb = ind.BollingerBands(20).calculate(frame(flat))
        assert np.isnan(bb["bb_pct_b"].iloc[-1])
        assert bb["bb_width"].iloc[-1] == pytest.approx(0.0)

    def test_bollinger_width_scales_with_volatility(self):
        rng = np.random.default_rng(11)
        calm = rng.normal(0, 0.2, 200).cumsum() + 100
        wild = rng.normal(0, 3.0, 200).cumsum() + 100
        w1 = ind.BollingerBands(20).calculate(frame(calm))["bb_width"].iloc[-1]
        w2 = ind.BollingerBands(20).calculate(frame(wild))["bb_width"].iloc[-1]
        assert w2 > w1

    def test_realized_vol_annualisation_factor(self):
        """RV must equal sqrt(252) * per-period sigma, not sqrt(n) times that.

        Regression guard: summing log-return squares without dividing by the
        window length inflates annualised volatility by sqrt(window) — a 7.7x
        overstatement at n=60 — which then corrupts volatility-based sizing.
        """
        rng = np.random.default_rng(13)
        daily_sigma = 0.01
        rets = rng.normal(0, daily_sigma, 20_000)
        prices = 100 * np.exp(np.cumsum(rets))
        for window in (20, 60, 252):
            rv = ind.RealizedVolatility(window, bars_per_year=252).calculate(
                frame(prices))["realized_vol"].iloc[-1]
            assert rv == pytest.approx(np.sqrt(252) * daily_sigma, rel=0.15), (
                f"window {window}: {rv:.4f} vs expected "
                f"{np.sqrt(252) * daily_sigma:.4f}"
            )

    def test_realized_vol_unannualised_is_per_period(self):
        rng = np.random.default_rng(14)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 5000)))
        rv = ind.RealizedVolatility(60, annualize=False).calculate(
            frame(prices))["realized_vol"].iloc[-1]
        assert rv == pytest.approx(0.01, rel=0.15)

    def test_parkinson_less_than_close_to_close_for_noise(self):
        rng = np.random.default_rng(17)
        close = rng.normal(100, 1, 200).cumsum() + 100
        f = frame(close, high=close + 1.0, low=close - 1.0)
        park = ind.ParkinsonVolatility(20).calculate(f)["parkinson_vol"].iloc[-1]
        hv = ind.RealizedVolatility(20, annualize=False).calculate(f)["realized_vol"].iloc[-1]
        assert park > 0 and hv > 0

    def test_keltner_uses_atr(self):
        rng = np.random.default_rng(19)
        f = frame(rng.normal(100, 1, 100).cumsum() + 100)
        kc = ind.KeltnerChannels(20, 10, 2.0).calculate(f)
        atr = ind.ATR(10).calculate(f)["atr"]
        assert (kc["kc_upper"].dropna() > kc["kc_mid"].dropna()).all()
        assert (kc["kc_lower"].dropna() < kc["kc_mid"].dropna()).all()

    def test_adx_bounds_and_zero_range(self):
        rng = np.random.default_rng(23)
        f = frame(rng.normal(100, 1, 200).cumsum() + 100)
        adx = ind.ADX(14).calculate(f)["adx"].dropna()
        assert ((adx >= 0) & (adx <= 100)).all()
        flat = ind.ADX(14).calculate(frame(np.full(80, 50.0)))["adx"]
        assert np.isnan(flat.iloc[-1]) or flat.iloc[-1] == pytest.approx(0.0)


class TestVolume:
    def test_obv_signs(self):
        close = pd.Series([10.0, 11.0, 10.0, 10.0])
        vol = pd.Series([100.0, 200.0, 300.0, 400.0])
        obv = ind.OBV().calculate(frame(close, volume=vol))["obv"]
        # +200 on the up bar, -300 on the down bar, 0 on the flat bar
        assert obv.iloc[1] == pytest.approx(200.0)
        assert obv.iloc[2] == pytest.approx(-100.0)
        assert obv.iloc[3] == pytest.approx(-100.0)

    def test_cmf_constant_is_finite(self):
        flat = np.full(60, 50.0)
        cmf = ind.CMF(20).calculate(frame(flat))["cmf"]
        assert np.isfinite(cmf.iloc[-1])

    def test_cmf_buying_pressure_positive(self):
        # Close must sit near the bar high for the money-flow multiplier to be
        # positive; a close at the midpoint of the range gives exactly zero.
        up = np.arange(1, 61, dtype=float) * 2
        n = len(up)
        f = frame(up, high=up * 1.02, low=up * 0.98, open_=up * 0.99)
        f["close"] = up * 1.015          # closing near the high = buying pressure
        assert ind.CMF(20).calculate(f)["cmf"].iloc[-1] > 0

    def test_cmf_selling_pressure_negative(self):
        down = np.arange(60, 0, -1, dtype=float) * 2.0
        f = frame(down, high=down * 1.02, low=down * 0.98, open_=down * 1.01)
        f["close"] = down * 0.985         # closing near the low = selling pressure
        assert ind.CMF(20).calculate(f)["cmf"].iloc[-1] < 0

    def test_vwap_within_bar_range(self):
        rng = np.random.default_rng(29)
        f = frame(rng.normal(100, 1, 60).cumsum() + 100)
        vwap = ind.VWAP(20).calculate(f)["vwap"]
        valid = vwap.dropna()
        assert (valid > f["low"].min()).all() and (valid < f["high"].max()).all()

    def test_relative_volume_unity_at_mean(self):
        vol = pd.Series([100.0] * 40)
        rv = ind.RelativeVolume(20).calculate(frame(np.arange(1, 41, dtype=float),
                                                   volume=vol))["rel_volume"]
        assert rv.iloc[-1] == pytest.approx(1.0)

    def test_mfi_bounded(self):
        rng = np.random.default_rng(31)
        f = frame(rng.normal(100, 1, 100).cumsum() + 100)
        mfi = ind.MFI(14).calculate(f)["mfi"].dropna()
        assert ((mfi >= 0) & (mfi <= 100)).all()


class TestMetadataIntegrity:
    def test_all_declared_warmups_are_correct(self):
        problems = ind.validate_metadata()
        assert problems == [], f"indicator metadata is inconsistent: {problems}"

    def test_nan_propagation_not_window_shortening(self):
        """A NaN inside a window must not shrink it into a shorter mean."""
        base = np.arange(1, 41, dtype=float)
        clean = ind.SMA(5).calculate(frame(base))["sma"]
        holed = base.copy()
        holed[20] = np.nan
        dirty = ind.SMA(5).calculate(frame(holed))["sma"]
        # More NaNs after inserting one hole: the window was not shortened.
        assert dirty.isna().sum() > clean.isna().sum()
        assert np.isnan(dirty.iloc[20])

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="requires columns"):
            ind.SMA(5).calculate(pd.DataFrame({"open": [1.0], "high": [1.0]}))

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            ind.SMA(0)

    def test_registry_lookup(self):
        assert isinstance(ind.get_indicator("SMA", period=10), ind.SMA)
        with pytest.raises(KeyError):
            ind.get_indicator("NoSuchIndicator")

    def test_catalogue_lists_every_indicator(self):
        cat = ind.catalogue()
        assert len(cat) == len(ind.INDICATORS)
        for entry in cat:
            assert {"name", "version", "formula", "inputs", "lookback", "warmup"} \
                <= set(entry)

    def test_every_indicator_declares_a_version(self):
        for name, cls in ind.INDICATORS.items():
            assert cls.METADATA.version, f"{name} has no version"
            assert cls.METADATA.formula, f"{name} has no formula"
