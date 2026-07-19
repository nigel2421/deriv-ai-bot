"""
Analytics suite for Deriv digit/trend trading:
  digit heatmaps, patterns, edge scores, quality filters, scanners.
"""
from src.analytics.digit_analysis import digit_heatmap, digit_snapshot
from src.analytics.edge_score import (
    compute_full_edge_report,
    edge_label,
    historical_edge_score,
    live_edge_score,
    pattern_strength,
    recency_weighted_performance,
)
from src.analytics.pattern_clarity import (
    pattern_clarity,
    clarity_class,
    entropy_strength,
    composite_entropy_score,
    entropy_clarity_from_digits,
)
from src.analytics.rolling_entropy import (
    RollingEntropyEngine,
    feed_ticks,
    get_engine,
)
from src.analytics.hierarchical_clarity import (
    build_hierarchical_clarity,
    weights_for_contract,
)
from src.analytics.contract_profiles import (
    evaluate_contract_setup,
    contract_clarity,
    get_base_profile,
    register_profile,
    get_weight_engine,
)
from src.analytics.historical_predictive_power import (
    get_hpp_tracker,
    composite_hpp,
    lift_score,
)
from src.analytics.hpp_timeseries import get_hpp_timeseries
from src.analytics.hpp_velocity import (
    velocity_state,
    compute_metric_velocity,
    weighted_engine_velocity,
)
from src.analytics.trade_filter import evaluate_setup
from src.analytics.edge_scanner import scan_markets
from src.analytics.no_trade_engine import (
    evaluate_no_trade,
    expected_value,
    risk_pct_from_quality,
)
from src.analytics.calibration import (
    get_calibration_tracker,
    wilson_ci,
    calibration_error,
)
from src.analytics.rise_fall_engine import analyze_rise_fall, composite_rf_score
from src.analytics.meta_validator import meta_validate
from src.analytics.momentum_persistence_engine import (
    analyze_momentum_persistence,
    final_trade_quality,
    momentum_engine,
    persistence_engine,
)

__all__ = [
    "digit_heatmap",
    "digit_snapshot",
    "edge_label",
    "historical_edge_score",
    "live_edge_score",
    "pattern_strength",
    "pattern_clarity",
    "clarity_class",
    "entropy_strength",
    "composite_entropy_score",
    "entropy_clarity_from_digits",
    "RollingEntropyEngine",
    "feed_ticks",
    "get_engine",
    "build_hierarchical_clarity",
    "weights_for_contract",
    "evaluate_contract_setup",
    "contract_clarity",
    "get_base_profile",
    "register_profile",
    "get_weight_engine",
    "get_hpp_tracker",
    "get_hpp_timeseries",
    "velocity_state",
    "compute_metric_velocity",
    "weighted_engine_velocity",
    "composite_hpp",
    "lift_score",
    "recency_weighted_performance",
    "compute_full_edge_report",
    "evaluate_setup",
    "scan_markets",
    "evaluate_no_trade",
    "expected_value",
    "risk_pct_from_quality",
    "get_calibration_tracker",
    "wilson_ci",
    "calibration_error",
    "analyze_rise_fall",
    "composite_rf_score",
    "meta_validate",
    "analyze_momentum_persistence",
    "final_trade_quality",
    "momentum_engine",
    "persistence_engine",
]
