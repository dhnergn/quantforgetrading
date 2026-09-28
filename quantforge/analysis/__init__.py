"""Analysis: regime detection, portfolio dashboard."""
from quantforge.analysis.dashboard import (
    Component, DashboardReport, build_dashboard, correlation_component,
    exposure_component, equity_curve_component, summary_statistics_component,
    var_component,
)
from quantforge.analysis.regime import (
    Regime, RegimeLabel, RegimeThresholds, RuleBasedRegimeDetector,
    regime_breakdown_report, regime_dominance, regime_features,
)

__all__ = [
    "Component", "DashboardReport", "build_dashboard", "correlation_component",
    "exposure_component", "equity_curve_component",
    "summary_statistics_component", "var_component",
    "Regime", "RegimeLabel", "RegimeThresholds", "RuleBasedRegimeDetector",
    "regime_breakdown_report", "regime_dominance", "regime_features",
]
