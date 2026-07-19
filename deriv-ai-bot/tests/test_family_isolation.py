"""Digit vs Rise/Fall paths must not crash each other."""
from __future__ import annotations

from src.strategy.contract_types import is_digit_contract, is_rise_fall, normalize_contract_type
from src.analytics.trade_filter import evaluate_setup
from src.analytics.rise_fall_engine import analyze_rise_fall
from src.analytics.momentum_persistence_engine import analyze_momentum_persistence


def _ticks(n=100, bias="even"):
    ticks = []
    q = 100.0
    for i in range(n):
        if bias == "up":
            q += 0.02
        elif bias == "down":
            q -= 0.02
        else:
            q += 0.01 if i % 2 == 0 else -0.005
        # even last digit for digit path
        quote = float(f"{int(q)}.{0 if bias == 'even' else (i % 10)}")
        if bias in {"up", "down"}:
            quote = q
        ticks.append({"epoch": 1_700_000_000 + i, "quote": quote})
    return ticks


def test_contract_type_partition():
    digits = ["DIGITOVER", "DIGITUNDER", "DIGITEVEN", "DIGITODD", "DIGITDIFF", "DIGITMATCH"]
    rf = ["CALL", "PUT", "RISE", "FALL"]
    for d in digits:
        assert is_digit_contract(d)
        assert not is_rise_fall(d)
    for r in rf:
        assert is_rise_fall(normalize_contract_type(r) or r)
        assert not is_digit_contract(normalize_contract_type(r) or r)


def test_digit_filter_does_not_raise():
    ticks = _ticks(120, bias="even")
    ev = evaluate_setup(
        ticks,
        symbol="R_25",
        contract_type="DIGITEVEN",
        family="digits",
        history_rows=[],
        signal_confidence=0.80,
        global_samples=0,
    )
    assert "allow" in ev
    assert "no_trade" in ev
    assert ev.get("family") == "digits"


def test_rf_filter_does_not_raise():
    ticks = _ticks(120, bias="up")
    ev = evaluate_setup(
        ticks,
        symbol="R_50",
        contract_type="CALL",
        family="rise_fall",
        history_rows=[],
        signal_confidence=0.80,
        global_samples=0,
    )
    assert "allow" in ev
    assert ev.get("family") == "rise_fall"
    # RF analysis attached when directional path runs
    assert "rise_fall" in ev or "momentum_persistence" in ev


def test_rf_and_digit_analysis_independent():
    ticks_up = _ticks(100, bias="up")
    ticks_even = _ticks(100, bias="even")
    rf = analyze_rise_fall(ticks_up, contract_type="CALL")
    mp = analyze_momentum_persistence(
        ticks_up, symbol="R_50", contract_type="CALL", note_velocity=False
    )
    dig = evaluate_setup(
        ticks_even,
        symbol="R_25",
        contract_type="DIGITEVEN",
        family="digits",
        signal_confidence=0.80,
        global_samples=0,
    )
    assert rf.get("rf_score") is not None
    assert mp.get("mp_score") is not None
    assert dig.get("allow") is not None
    # Running digit filter must not corrupt RF structures
    rf2 = analyze_rise_fall(ticks_up, contract_type="PUT")
    assert "metrics" in rf2


def test_unified_min_confidence_constant():
    from src.strategy.adaptive_learner import AdaptiveLearner
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        al = AdaptiveLearner(path=Path(d) / "l.json")
        for fam, ct in (
            ("digits", "DIGITEVEN"),
            ("rise_fall", "CALL"),
            ("minute_rise_fall", "PUT"),
        ):
            assert al.effective_min_confidence(0.80, family=fam, contract_type=ct) == 0.80
