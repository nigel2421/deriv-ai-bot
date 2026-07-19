"""Momentum + Persistence + Transition engine tests."""
from __future__ import annotations

from pathlib import Path

from src.analytics.momentum_persistence_engine import (
    PersistenceVelocityTracker,
    analyze_momentum_persistence,
    dual_system_blend,
    final_trade_quality,
    momentum_engine,
    momentum_persistence_score,
    persistence_confidence,
    persistence_engine,
    rf_mp_gates,
    tick_directions,
    transition_engine,
)
from src.analytics.no_trade_engine import trade_quality_score
from src.analytics.calibration import CalibrationTracker
from src.analytics.trade_filter import evaluate_setup


def _quotes_up(n: int = 25) -> list:
    q = [100.0]
    for _ in range(n):
        q.append(q[-1] + 0.01)
    return q


def _ticks_from_quotes(quotes: list) -> list:
    return [
        {"epoch": 1_700_000_000 + i, "quote": q}
        for i, q in enumerate(quotes)
    ]


def test_momentum_formula_spec():
    # 14 up, 6 down → raw = (14-6)/20 = 0.4 → score = 50 + 20 = 70
    dirs = [1] * 14 + [-1] * 6
    m = momentum_engine(dirs, window=20)
    assert m["up"] == 14
    assert m["down"] == 6
    assert abs(m["raw_momentum"] - 0.4) < 1e-6
    assert abs(m["momentum_score"] - 70.0) < 0.2
    assert m["direction"] == "BULLISH"


def test_momentum_neutral_and_strong():
    dirs = [1, -1] * 10
    m = momentum_engine(dirs, window=20)
    assert abs(m["momentum_score"] - 50.0) < 1.0
    dirs2 = [1] * 20
    m2 = momentum_engine(dirs2, window=20)
    assert m2["momentum_score"] >= 90
    assert "Strong" in m2["label"] or m2["momentum_score"] >= 90


def test_transition_matrix_shape():
    dirs = [1, 1, 1, -1, -1, 1, 1]
    tm = transition_engine(dirs)
    assert "matrix" in tm
    assert "UP" in tm["matrix"] and "DOWN" in tm["matrix"]
    assert abs(tm["p_uu"] + tm["p_ud"] - 1.0) < 1e-6 or tm["n_from_up"] == 0


def test_persistence_up_chain():
    dirs = [1] * 40
    p = persistence_engine(dirs, contract_type="CALL")
    assert p["persistence"] >= 90
    assert p["side"] == "UP→UP"
    assert p["p_uu"] >= 0.9


def test_mp_combine_formula():
    # 75*0.6 + 80*0.4 = 45 + 32 = 77
    mp = momentum_persistence_score(80, 75, sample_confidence=1.0)
    assert abs(mp["momentum_persistence"] - 77.0) < 0.2


def test_persistence_confidence_small_sample_worthless():
    c = persistence_confidence(3, required=200)
    assert c["confidence"] < 0.05
    assert c["label"] == "LOW"
    assert c["trustworthy"] is False
    c2 = persistence_confidence(800, required=200)
    assert c2["confidence"] >= 1.0
    assert c2["label"] == "HIGH"


def test_persistence_velocity_strengthening(tmp_path: Path):
    tr = PersistenceVelocityTracker(path=tmp_path / "pv.json")
    for v in (55, 57, 60, 65, 68):
        out = tr.note("R_75|CALL", v, n_transitions=400)
    # Spec example: current 68 vs prev avg ~59.25 → smooth ~+8.75
    assert out["velocity"] > 0
    assert out["smooth_velocity"] > 5
    assert out["status"] in {"STRENGTHENING", "IMPROVING"}
    assert out["velocity_score"] >= 50
    assert "fast_velocity" in out and "medium_velocity" in out
    assert "acceleration" in out
    assert out.get("persistence_engine_score") is not None

    for v in (78, 72, 65, 58):
        out2 = tr.note("R_100|PUT", v, n_transitions=400)
    assert out2["velocity"] < 0
    assert out2["status"] in {"WEAKENING", "DECLINING"}
    assert out2.get("block_late_entry") is True


def test_velocity_momentum_score_formula():
    from src.analytics.momentum_persistence_engine import velocity_momentum_score

    assert velocity_momentum_score(0) == 50
    assert abs(velocity_momentum_score(10) - 80) < 0.1
    assert velocity_momentum_score(20) == 100
    assert velocity_momentum_score(-20) == 0


def test_persistence_engine_composite():
    from src.analytics.momentum_persistence_engine import persistence_engine_score

    # 70*0.5 + 85*0.3 + 90*0.2 = 35 + 25.5 + 18 = 78.5
    e = persistence_engine_score(
        persistence=70, velocity_score=85, acceleration_score_val=90
    )
    assert abs(e["persistence_engine_score"] - 78.5) < 0.2


def test_validate_pvel_edge_and_auto_reduce(tmp_path: Path):
    from src.analytics.momentum_persistence_engine import (
        PersistenceVelocityTracker,
        validate_persistence_velocity_edge,
        get_persistence_velocity_tracker,
    )
    import src.analytics.momentum_persistence_engine as mpe

    # Isolate tracker
    mpe._pvel = PersistenceVelocityTracker(path=tmp_path / "pv2.json")
    # Build outcomes where negative velocity somehow wins more → fail → reduce weight
    outcomes = []
    for i in range(600):
        outcomes.append(
            {
                "persistence_velocity": 5.0,
                "is_win": i % 3 == 0,  # ~33% WR
                "profit": 1.0 if i % 3 == 0 else -1.0,
            }
        )
    for i in range(600):
        outcomes.append(
            {
                "persistence_velocity": -5.0,
                "is_win": i % 2 == 0,  # 50% WR — better than A → fail
                "profit": 1.0 if i % 2 == 0 else -1.0,
            }
        )
    old_w = get_persistence_velocity_tracker().velocity_weight
    rep = validate_persistence_velocity_edge(
        outcomes, min_samples=1000, auto_reduce=True
    )
    assert rep["enough_sample"] is True
    assert rep["pass"] is False
    assert get_persistence_velocity_tracker().velocity_weight < old_w


def test_final_quality_formula_with_mp():
    tq = final_trade_quality(
        pattern_strength=88,
        pattern_clarity=82,
        hpp=84,
        hpp_velocity=6,
        momentum_persistence=77,
        confidence=91,
    )
    assert tq["auto_ok"] is True
    assert tq["trade_quality"] >= 80
    assert "momentum_persistence" in tq["weights"]
    assert abs(tq["weights"]["momentum_persistence"] - 0.15) < 1e-9

    # no_trade trade_quality_score path
    tq2 = trade_quality_score(
        pattern_strength=88,
        pattern_clarity=82,
        hpp=84,
        hpp_velocity=6,
        confidence=91,
        momentum_persistence=77,
    )
    assert abs(tq2["trade_quality"] - tq["trade_quality"]) < 0.5


def test_dual_system_blend():
    d = dual_system_blend(existing_edge=80, momentum_score=70, persistence_score=75)
    # 0.5*80 + 0.3*70 + 0.2*75 = 40 + 21 + 15 = 76
    assert abs(d["dual_score"] - 76.0) < 0.2


def test_rf_gates_rise():
    analysis = {
        "contract_type": "CALL",
        "momentum": {"momentum_score": 72},
        "persistence": {
            "persistence": 65,
            "sample_confidence": {"trustworthy": True, "n": 100, "label": "MEDIUM"},
        },
        "persistence_velocity": {
            "block_late_entry": False,
            "status": "IMPROVING",
            "velocity": 3.0,
            "acceleration": 1.0,
            "n": 10,
        },
    }
    g = rf_mp_gates(
        analysis, trade_quality=85, hpp_velocity=3.0, contract_type="CALL"
    )
    assert g["allow"] is True

    g2 = rf_mp_gates(
        analysis, trade_quality=85, hpp_velocity=-2.0, contract_type="CALL"
    )
    assert g2["allow"] is False

    # Falling velocity blocks even if persistence is 65
    analysis["persistence_velocity"]["velocity"] = -3.0
    analysis["persistence_velocity"]["status"] = "DECLINING"
    analysis["persistence_velocity"]["block_late_entry"] = True
    g3 = rf_mp_gates(
        analysis, trade_quality=85, hpp_velocity=3.0, contract_type="CALL"
    )
    assert g3["allow"] is False


def test_rf_gates_fall_needs_bearish_momentum():
    analysis = {
        "contract_type": "PUT",
        "momentum": {"momentum_score": 30},  # bearish
        "persistence": {
            "persistence": 65,
            "sample_confidence": {"trustworthy": True, "n": 80, "label": "MEDIUM"},
        },
        "persistence_velocity": {
            "block_late_entry": False,
            "status": "IMPROVING",
            "velocity": 2.0,
            "acceleration": 0.5,
            "n": 8,
        },
    }
    g = rf_mp_gates(
        analysis, trade_quality=85, hpp_velocity=2.0, contract_type="PUT"
    )
    assert g["allow"] is True
    # Bullish momentum should fail PUT
    analysis["momentum"]["momentum_score"] = 70
    g2 = rf_mp_gates(
        analysis, trade_quality=85, hpp_velocity=2.0, contract_type="PUT"
    )
    assert g2["allow"] is False


def test_analyze_full_pipeline():
    ticks = _ticks_from_quotes(_quotes_up(50))
    a = analyze_momentum_persistence(
        ticks, symbol="R_75", contract_type="CALL", note_velocity=False
    )
    assert a["mp_score"] is not None
    assert a["momentum"]["momentum_score"] >= 70
    assert a["persistence"]["persistence"] >= 50
    assert "transition" in a


def test_filter_exposes_mp():
    ticks = _ticks_from_quotes(_quotes_up(60))
    # slight noise
    for i in range(10):
        ticks[i]["quote"] = 100.0 + i * 0.005
    ev = evaluate_setup(
        ticks,
        symbol="R_75",
        contract_type="CALL",
        family="rise_fall",
        history_rows=[],
        signal_confidence=0.88,
        min_sample=100,
    )
    assert "momentum_persistence" in ev
    assert "mp_score" in ev
    assert ev.get("mp_score") is not None or ev.get("momentum_persistence")


def test_feature_contribution_report(tmp_path: Path):
    cal = CalibrationTracker(
        path=tmp_path / "c.json", peak_path=tmp_path / "p.json"
    )
    for i in range(120):
        high = i % 3 != 0
        cal.record(
            contract="CALL",
            is_win=high if i % 5 != 0 else (not high),
            quality=85 if high else 55,
            momentum=80 if high else 40,
            persistence=70 if high else 45,
            momentum_persistence=77 if high else 48,
            clarity=80 if high else 50,
            hpp=80 if high else 50,
            velocity=5 if high else -3,
            profit=1.0 if (high if i % 5 != 0 else not high) else -1.0,
        )
    rep = cal.feature_contribution_report(last_n=120)
    assert rep["n"] == 120
    assert "momentum" in rep["features"]
    assert "persistence" in rep["features"]
    checklist = cal.validation_checklist()
    assert "momentum_validation" in checklist["checks"]
    assert "feature_contribution" in checklist["checks"]
