"""No-trade / EV / calibration decision-engine tests."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.analytics.no_trade_engine import (
    edge_decay_pct,
    ensemble_votes,
    evaluate_no_trade,
    expected_value,
    map_engine_regime,
    risk_pct_from_quality,
    trade_quality_score,
)
from src.analytics.calibration import (
    CalibrationTracker,
    calibration_error,
    score_band,
    wilson_ci,
)


def test_expected_value_positive_and_negative():
    # 55% win @ 0.95 net → EV = 0.55*0.95 - 0.45*1 = 0.0725
    assert expected_value(0.55, reward=0.95, risk=1.0) > 0
    # 48% win @ 0.95 → negative EV
    assert expected_value(0.48, reward=0.95, risk=1.0) < 0
    # Fair coin at 1:1 net reward 1.0 → EV 0
    assert abs(expected_value(0.5, reward=1.0, risk=1.0)) < 1e-9


def test_trade_quality_weights():
    tq = trade_quality_score(
        pattern_strength=85,
        pattern_clarity=82,
        hpp=90,
        hpp_velocity=7.5,  # ~75 mapped
        confidence=95,
    )
    # Spec example ≈ 85
    assert 80 <= tq["trade_quality"] <= 92
    assert tq["auto_ok"] is True


def test_risk_pct_from_quality_bands():
    assert risk_pct_from_quality(95) == 1.0
    assert risk_pct_from_quality(85) == 0.5
    assert risk_pct_from_quality(79) == 0.0
    assert risk_pct_from_quality(50) == 0.0


def test_edge_decay_and_retire():
    # (92-70)/92 * 100 ≈ 23.91%
    d = edge_decay_pct(92, 70)
    assert abs(d - 23.913) < 0.1

    r = evaluate_no_trade(
        contract_type="DIGITDIFF",
        pattern_clarity=90,
        pattern_strength=90,
        hpp=70,
        hpp_velocity=5,
        entropy_stability=80,
        confidence=95,
        signal_confidence=0.9,
        p_win=0.62,
        reward=0.95,
        regime_raw="STRONG PATTERN",
        realtime_pattern_strength=80,
        peak_hpp=92,
        cold_start=False,
        entropy_buy=True,
        pattern_buy=True,
        hpp_buy=True,
        probability_buy=True,
    )
    assert r["retired"] is True or r["edge_decay_pct"] >= 20
    assert r["status"] == "REJECTED"


def test_block_low_clarity_and_negative_velocity():
    r = evaluate_no_trade(
        contract_type="DIGITDIFF",
        pattern_clarity=60,  # < 75
        pattern_strength=85,
        hpp=80,
        hpp_velocity=-8,  # < -5
        entropy_stability=80,
        confidence=90,
        signal_confidence=0.9,
        p_win=0.7,
        reward=0.95,
        regime_raw="STRONG PATTERN",
        realtime_pattern_strength=80,
        cold_start=False,
        entropy_buy=True,
        pattern_buy=True,
        hpp_buy=True,
        probability_buy=True,
    )
    assert r["status"] == "REJECTED"
    assert r["allow"] is False
    assert any("Clarity" in b or "Velocity" in b for b in r["blocks"])
    assert r["display"]["status"] == "REJECTED"


def test_block_random_regime():
    r = evaluate_no_trade(
        contract_type="DIGITDIFF",
        pattern_clarity=90,
        pattern_strength=90,
        hpp=90,
        hpp_velocity=5,
        entropy_stability=90,
        confidence=95,
        signal_confidence=0.95,
        p_win=0.7,
        reward=0.95,
        regime_raw="RANDOM",
        realtime_pattern_strength=20,
        cold_start=False,
        entropy_buy=True,
        pattern_buy=True,
        hpp_buy=True,
        probability_buy=True,
    )
    assert r["status"] == "REJECTED"
    assert r["regime"] == "RANDOM"
    assert r["regime_allowed_contracts"] == []


def test_allow_strong_setup():
    r = evaluate_no_trade(
        contract_type="DIGITDIFF",
        pattern_clarity=88,
        pattern_strength=85,
        hpp=90,
        hpp_velocity=6,
        entropy_stability=75,
        confidence=95,
        signal_confidence=0.9,
        p_win=0.62,
        reward=0.95,
        regime_raw="STRONG PATTERN",
        realtime_pattern_strength=80,
        cold_start=False,
        entropy_buy=True,
        pattern_buy=True,
        hpp_buy=True,
        probability_buy=True,
    )
    assert r["status"] == "ALLOWED"
    assert r["allow"] is True
    assert r["ev"] > 0
    assert r["risk_pct"] >= 0.5
    assert r["ensemble"]["agree"] is True


def test_block_ev_non_positive():
    r = evaluate_no_trade(
        contract_type="DIGITEVEN",
        pattern_clarity=90,
        pattern_strength=90,
        hpp=90,
        hpp_velocity=5,
        entropy_stability=80,
        confidence=90,
        signal_confidence=0.9,
        p_win=0.48,
        reward=0.95,
        regime_raw="BALANCED",
        realtime_pattern_strength=70,
        cold_start=False,
        entropy_buy=True,
        pattern_buy=True,
        hpp_buy=True,
        probability_buy=True,
    )
    assert r["status"] == "REJECTED"
    assert any("EV" in b for b in r["blocks"])


def test_ensemble_requires_all_four():
    e = ensemble_votes(
        entropy_buy=True, pattern_buy=True, hpp_buy=True, probability_buy=False
    )
    assert e["agree"] is False
    assert e["n_yes"] == 3


def test_map_regime_labels():
    assert map_engine_regime("RANDOM") == "RANDOM"
    assert map_engine_regime("STRONG PATTERN") == "STRONG PATTERN"
    assert map_engine_regime("BIASED", 80) == "BIASED"
    assert map_engine_regime("BIASED", 50) == "EMERGING PATTERN"


def test_wilson_ci_and_calibration_error():
    lo, hi = wilson_ci(78, 100)
    assert 0.68 < lo < 0.78
    assert 0.78 < hi < 0.88
    assert calibration_error(0.80, 0.78) == pytest.approx(0.02, abs=1e-6)
    assert score_band(85) == "80-90"


def test_calibration_tracker_bands(tmp_path: Path):
    cal = CalibrationTracker(
        path=tmp_path / "cal.json",
        peak_path=tmp_path / "peaks.json",
    )
    # Well-calibrated 80-90 band
    for i in range(100):
        cal.record(
            contract="DIGITDIFF",
            is_win=(i % 100) < 82,
            predicted_p=0.82,
            quality=85,
            clarity=88 if i % 2 == 0 else 50,
            hpp=90 if i < 50 else 40,
            velocity=5,
            regime="STRONG PATTERN" if i < 60 else "RANDOM",
            profit=1.0 if (i % 100) < 82 else -1.0,
        )
    rep = cal.band_report()
    b = rep["bands"]["80-90"]
    assert b["generated"] == 100
    assert b["win_rate"] is not None
    assert b["ci_low"] is not None
    assert b["calibration_error"] is not None

    drift = cal.prediction_drift(last_n=100)
    assert drift["n"] == 100
    assert drift["drift_pct"] is not None

    hpp_v = cal.metric_validation("hpp", high=80, low=60)
    assert hpp_v["n_high"] > 0

    peak = cal.peak_hpp("DIGITDIFF")
    assert peak >= 90

    checklist = cal.validation_checklist()
    assert "checks" in checklist
    assert "precision_by_score_band" in checklist["checks"]


def test_ci_wide_is_unreliable(tmp_path: Path):
    cal = CalibrationTracker(
        path=tmp_path / "cal2.json",
        peak_path=tmp_path / "peaks2.json",
    )
    for i in range(5):
        cal.record(contract="CALL", is_win=True, predicted_p=0.68, quality=90)
    ci = cal.confidence_interval_report(contract="CALL")
    assert ci["n"] == 5
    assert ci["reliable"] is False
    assert "UNRELIABLE" in ci["display"]


def test_filter_exposes_no_trade_fields():
    from src.analytics.trade_filter import evaluate_setup

    ticks = []
    for i in range(120):
        ticks.append(
            {
                "epoch": 1_700_000_000 + i,
                "quote": 100.0 + (i % 10) * 0.01,
            }
        )
    # Bias even digits for a cleaner structure
    for i in range(50):
        ticks[-(i + 1)]["quote"] = float(f"100.{(i % 5) * 2}")

    ev = evaluate_setup(
        ticks,
        symbol="R_75",
        contract_type="DIGITEVEN",
        family="digits",
        history_rows=[],
        signal_confidence=0.88,
        min_sample=100,
    )
    assert "no_trade" in ev
    assert "ev" in ev
    assert "risk_pct" in ev
    assert "regime" in ev
    assert "gates" in ev and "no_trade_ok" in ev["gates"]
    assert "ev_ok" in ev["gates"]
    assert ev["no_trade"]["status"] in {"ALLOWED", "REJECTED"}
    # Display shape
    disp = ev["no_trade"].get("display") or {}
    assert disp.get("status") in {"ALLOWED", "REJECTED"}
