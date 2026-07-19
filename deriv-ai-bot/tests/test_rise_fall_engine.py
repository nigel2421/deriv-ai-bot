"""Rise/Fall directional engine + meta-validator tests."""
from __future__ import annotations

from src.analytics.rise_fall_engine import (
    analyze_rise_fall,
    composite_rf_score,
    directional_entropy,
    persistence_score,
    tick_directions,
    tick_momentum_score,
    transition_matrix,
    volatility_regime,
)
from src.analytics.meta_validator import meta_validate
from src.analytics.contract_profiles import get_base_profile, evaluate_contract_setup
from src.analytics.trade_filter import evaluate_setup


def _ticks_trend(n: int = 80, up_bias: float = 0.75) -> list:
    """Synthetic ticks with controllable up bias."""
    import random

    random.seed(42)
    q = 100.0
    ticks = []
    for i in range(n):
        if random.random() < up_bias:
            q += 0.02
        else:
            q -= 0.015
        ticks.append({"epoch": 1_700_000_000 + i, "quote": round(q, 4)})
    return ticks


def test_tick_momentum_strong_bullish():
    quotes = [100.0]
    for i in range(20):
        quotes.append(quotes[-1] + 0.01)  # all up
    dirs = tick_directions(quotes)
    mom = tick_momentum_score(dirs, window=20)
    assert mom["up"] >= 15
    assert mom["momentum_pct"] >= 90
    assert mom["direction"] == "BULLISH"
    assert "Bullish" in mom["label"] or "Strong" in mom["label"]


def test_transition_and_persistence():
    # Alternating → mean reversion
    quotes = [100.0]
    for i in range(40):
        quotes.append(quotes[-1] + (0.01 if i % 2 == 0 else -0.01))
    dirs = tick_directions(quotes)
    tm = transition_matrix(dirs)
    assert "p_uu" in tm and "p_ud" in tm
    # Streak of ups → high UU
    quotes2 = [100.0]
    for _ in range(30):
        quotes2.append(quotes2[-1] + 0.01)
    p = persistence_score(tick_directions(quotes2))
    assert p["p_continue_after_up"] >= 0.9
    assert p["score"] >= 70


def test_directional_entropy_edge():
    # Balanced
    dirs = [1, -1] * 40
    e = directional_entropy(dirs)
    assert e["h_ratio"] > 0.95
    assert e["score"] < 55
    # Biased up
    dirs2 = [1] * 30 + [-1] * 10
    e2 = directional_entropy(dirs2)
    assert e2["up_pct"] >= 70
    assert e2["score"] > e["score"]
    assert e2["edge"] is True


def test_volatility_regimes():
    calm = [100.0 + i * 0.0001 for i in range(80)]
    v = volatility_regime(calm)
    assert v["regime"] in {"CALM", "NORMAL"}
    assert v["tradeable"] is True

    # Expanding: quiet then large moves
    q = [100.0] * 40
    for i in range(40):
        q.append(q[-1] + (0.5 if i % 2 == 0 else -0.45))
    v2 = volatility_regime(q)
    assert v2["regime"] in {"EXPANDING", "CHAOTIC", "NORMAL"}


def test_composite_rf_weights():
    c = composite_rf_score(
        momentum=80,
        trend_strength=75,
        volatility_score=85,
        hpp=70,
        directional_entropy_score=72,
    )
    # 0.35*80 + 0.25*75 + 0.20*85 + 0.10*70 + 0.10*72
    expected = 0.35 * 80 + 0.25 * 75 + 0.20 * 85 + 0.10 * 70 + 0.10 * 72
    assert abs(c["rf_score"] - expected) < 0.5
    assert c["weights"]["momentum"] == 0.35


def test_analyze_rise_fall_call_bias():
    ticks = _ticks_trend(100, up_bias=0.80)
    a = analyze_rise_fall(ticks, contract_type="CALL")
    assert a["family"] == "rise_fall"
    assert a["rf_score"] is not None
    assert "momentum" in a
    assert "transition_matrix" in a
    assert "volatility" in a
    assert "directional_entropy" in a
    assert a["metrics"]["momentum"] >= 50  # oriented for CALL
    # PUT on same bullish stream should score lower oriented momentum
    b = analyze_rise_fall(ticks, contract_type="PUT")
    assert b["oriented_momentum"] <= a["oriented_momentum"] + 1


def test_call_put_profile_weights_are_directional():
    w = get_base_profile("CALL")
    assert w["momentum"] >= 0.30
    assert w.get("digit_entropy", 0) < 0.15  # not digit-heavy
    assert "trend_strength" in w
    assert "volatility_score" in w
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_evaluate_contract_setup_includes_rf():
    ticks = _ticks_trend(90, up_bias=0.78)
    r = evaluate_contract_setup(
        ticks,
        symbol="R_75",
        contract_type="CALL",
        sample_n=50,
        pattern_wr=0.58,
    )
    assert r.get("rise_fall") or r.get("rf_score") is not None
    assert "momentum" in (r.get("metrics") or {})
    assert "trend_strength" in (r.get("metrics") or {})


def test_meta_validator_blocks_negative_velocity():
    m = meta_validate(
        contract_type="CALL",
        pattern_strength=88,
        pattern_clarity=82,
        hpp=84,
        hpp_velocity=-15,  # decaying
        confidence=91,
        p_win=0.62,
        reward=0.95,
        regime_raw="STRONG PATTERN",
        realtime_pattern_strength=80,
        family="rise_fall",
        rf_score=82,
        vol_tradeable=True,
        cold_start=False,
    )
    assert m["status"] == "BLOCKED"
    assert m["allow"] is False
    assert "decaying" in m["reason"].lower() or any(
        c["name"] == "velocity" and not c["ok"] for c in m["checks"]
    )


def test_meta_validator_approves_aligned_setup():
    m = meta_validate(
        contract_type="DIGITDIFF",
        pattern_strength=88,
        pattern_clarity=82,
        hpp=84,
        hpp_velocity=9,
        confidence=91,
        p_win=0.62,
        reward=0.95,
        regime_raw="STRONG PATTERN",
        realtime_pattern_strength=80,
        family="digits",
        cold_start=False,
    )
    assert m["status"] == "APPROVED"
    assert m["allow"] is True
    assert m["n_ok"] == m["n_total"]


def test_filter_rise_fall_exposes_rf_fields():
    ticks = _ticks_trend(100, up_bias=0.82)
    ev = evaluate_setup(
        ticks,
        symbol="R_75",
        contract_type="CALL",
        family="rise_fall",
        history_rows=[],
        signal_confidence=0.88,
        min_sample=100,
    )
    assert "rise_fall" in ev
    assert "meta_validator" in ev
    assert "gates" in ev and "meta_ok" in ev["gates"]
    # Family model applied
    if ev.get("rise_fall"):
        assert ev["rise_fall"].get("rf_score") is not None
