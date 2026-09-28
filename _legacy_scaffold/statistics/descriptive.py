"""Descriptive statistics — mean, median, variance, std dev, skewness, kurtosis, etc.

Handles small samples, missing observations, constant series, and numerical instability.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class DescriptiveStats:
    """Descriptive statistics with proper edge case handling."""

    @staticmethod
    def mean(data: pd.Series) -> float:
        """Arithmetic mean. Returns NaN for empty series."""
        clean = data.dropna()
        if len(clean) == 0:
            return float("nan")
        return float(clean.mean())

    @staticmethod
    def median(data: pd.Series) -> float:
        """Median. Returns NaN for empty series."""
        clean = data.dropna()
        if len(clean) == 0:
            return float("nan")
        return float(clean.median())

    @staticmethod
    def variance(data: pd.Series, ddof: int = 1) -> float:
        """Variance. Returns NaN for insufficient data."""
        clean = data.dropna()
        if len(clean) <= ddof:
            return float("nan")
        return float(clean.var(ddof=ddof))

    @staticmethod
    def std_dev(data: pd.Series, ddof: int = 1) -> float:
        """Standard deviation. Returns NaN for insufficient data."""
        clean = data.dropna()
        if len(clean) <= ddof:
            return float("nan")
        return float(clean.std(ddof=ddof))

    @staticmethod
    def downside_deviation(data: pd.Series, threshold: float = 0.0) -> float:
        """Downside deviation below threshold."""
        clean = data.dropna()
        if len(clean) == 0:
            return float("nan")
        downside = clean[clean < threshold]
        if len(downside) == 0:
            return 0.0
        return float(np.sqrt(np.mean((downside - threshold) ** 2)))

    @staticmethod
    def skewness(data: pd.Series) -> float:
        """Skewness. Returns NaN for n < 3."""
        clean = data.dropna()
        if len(clean) < 3:
            return float("nan")
        return float(clean.skew())

    @staticmethod
    def kurtosis(data: pd.Series) -> float:
        """Excess kurtosis. Returns NaN for n < 4."""
        clean = data.dropna()
        if len(clean) < 4:
            return float("nan")
        return float(clean.kurtosis())

    @staticmethod
    def percentile(data: pd.Series, q: float) -> float:
        """Percentile (0-100). Returns NaN for empty series."""
        clean = data.dropna()
        if len(clean) == 0:
            return float("nan")
        return float(np.percentile(clean, q))

    @staticmethod
    def zscore(data: pd.Series) -> pd.Series:
        """Z-scores. Returns NaN where std dev is 0."""
        clean = data.dropna()
        if len(clean) < 2:
            return pd.Series(np.nan, index=data.index)
        mean = clean.mean()
        std = clean.std(ddof=1)
        if std == 0:
            return pd.Series(np.nan, index=data.index)
        return (data - mean) / std

    @staticmethod
    def covariance(x: pd.Series, y: pd.Series, ddof: int = 1) -> float:
        """Covariance between two series."""
        combined = pd.concat([x, y], axis=1).dropna()
        if len(combined) <= ddof:
            return float("nan")
        return float(combined.iloc[:, 0].cov(combined.iloc[:, 1], ddof=ddof))

    @staticmethod
    def correlation(x: pd.Series, y: pd.Series) -> float:
        """Pearson correlation. Returns NaN for insufficient data."""
        combined = pd.concat([x, y], axis=1).dropna()
        if len(combined) < 2:
            return float("nan")
        std_x = combined.iloc[:, 0].std()
        std_y = combined.iloc[:, 1].std()
        if std_x == 0 or std_y == 0:
            return float("nan")
        return float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))

    @staticmethod
    def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Beta = cov(asset, market) / var(market)."""
        combined = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if len(combined) < 2:
            return float("nan")
        cov = combined.iloc[:, 0].cov(combined.iloc[:, 1])
        var = combined.iloc[:, 1].var()
        if var == 0:
            return float("nan")
        return float(cov / var)

    @staticmethod
    def rolling_correlation(
        x: pd.Series, y: pd.Series, window: int
    ) -> pd.Series:
        """Rolling correlation between two series."""
        return x.rolling(window).corr(y)

    @staticmethod
    def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method. Returns boolean series."""
        z = DescriptiveStats.zscore(data)
        return z.abs() > threshold

    @staticmethod
    def detect_outliers_iqr(data: pd.Series, factor: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method. Returns boolean series."""
        clean = data.dropna()
        if len(clean) == 0:
            return pd.Series(False, index=data.index)
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return (data < lower) | (data > upper)
