from src.strategy.digit_contracts import (
    barrier_for_predicted_digit,
    extract_last_digit,
    normalize_barrier,
    validate_digit_contract,
    would_win,
)
from src.strategy.signal_generator import SignalGenerator


def test_extract_last_digit():
    assert extract_last_digit(503.77) == 7
    assert extract_last_digit("54640.6196") == 6
    assert extract_last_digit(100) == 0
    assert extract_last_digit(None) is None


def test_would_win_over_under():
    assert would_win("DIGITOVER", 4, 7) is True
    assert would_win("DIGITOVER", 4, 4) is False
    assert would_win("DIGITUNDER", 5, 3) is True
    assert would_win("DIGITUNDER", 5, 5) is False


def test_would_win_even_odd_match():
    assert would_win("DIGITEVEN", None, 8) is True
    assert would_win("DIGITEVEN", None, 7) is False
    assert would_win("DIGITODD", None, 7) is True
    assert would_win("DIGITMATCH", 3, 3) is True
    assert would_win("DIGITDIFF", 3, 4) is True


def test_normalize_barrier_edges():
    assert normalize_barrier("DIGITOVER", 9) == 8
    assert normalize_barrier("DIGITUNDER", 0) == 1
    assert normalize_barrier("DIGITEVEN", 5) is None


def test_barrier_for_predicted_digit():
    # pred 7 → OVER barrier 6 (win if digit > 6)
    assert barrier_for_predicted_digit("DIGITOVER", 7) == 6
    # pred 2 → UNDER barrier 3
    assert barrier_for_predicted_digit("DIGITUNDER", 2) == 3
    assert barrier_for_predicted_digit("DIGITMATCH", 9) == 9
    assert barrier_for_predicted_digit("DIGITEVEN", 4) is None


def test_validate_digit_contract():
    ok, reason, b = validate_digit_contract("DIGITOVER", 4)
    assert ok and b == 4
    ok2, _, b2 = validate_digit_contract("DIGITEVEN", 9)
    assert ok2 and b2 is None


def test_signal_generator_over():
    sg = SignalGenerator()
    ct, barrier, conf = sg.generate_signal(
        {"digit": 8, "parity": True},
        confidence=0.9,
        min_confidence=0.75,
        allowed_types=["DIGITOVER", "DIGITUNDER"],
    )
    assert ct == "DIGITOVER"
    assert barrier is not None and barrier < 8
    assert conf == 0.9


def test_signal_generator_even():
    sg = SignalGenerator(prefer_parity=True)
    ct, barrier, conf = sg.generate_signal(
        {"digit": 6, "parity": True, "preferred_type": "DIGITEVEN"},
        confidence=0.85,
        min_confidence=0.75,
        allowed_types=["DIGITEVEN", "DIGITODD", "DIGITOVER"],
    )
    assert ct == "DIGITEVEN"
    assert barrier is None


def test_signal_generator_respects_min_conf():
    sg = SignalGenerator()
    ct, barrier, conf = sg.generate_signal(
        {"digit": 5}, confidence=0.5, min_confidence=0.75
    )
    assert ct is None and conf == 0.0


def test_signal_from_ticks():
    sg = SignalGenerator()
    ticks = [{"quote": 100.0 + i + (0.2 if i % 2 == 0 else 0.1)} for i in range(40)]
    ct, barrier, conf, stats = sg.generate_from_ticks(
        ticks, min_confidence=0.0, lookback=30
    )
    assert stats["n"] > 0
    # With min_confidence 0 we should usually get something
    assert ct is not None or stats["n"] == 0
