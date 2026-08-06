"""Tests for FX session hours, duration fallbacks, and offer gate."""
from __future__ import annotations

from datetime import datetime, timezone

from src.strategy.market_offer_gate import (
    REASON_DURATION,
    MarketOfferGate,
    classify_offer_error,
    duration_fallbacks,
)
from src.strategy.session_hours import (
    forex_session_open,
    is_fx_symbol,
    is_likely_session_open,
    is_spike_synthetic,
    preferred_minute_duration,
)


def test_fx_symbol_and_duration():
    assert is_fx_symbol("frxEURUSD")
    assert is_fx_symbol("frxGBPUSD")
    assert not is_fx_symbol("R_100")
    assert preferred_minute_duration("frxEURUSD") >= 15
    assert preferred_minute_duration("R_100", 2) == 2


def test_spike_synthetic_detection():
    assert is_spike_synthetic("BOOM500")
    assert is_spike_synthetic("CRASH1000")
    assert not is_spike_synthetic("frxEURUSD")
    assert not is_spike_synthetic("R_75")


def test_forex_weekend_closed():
    # Saturday UTC
    sat = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
    assert forex_session_open(sat) is False
    open_ok, reason = is_likely_session_open("frxEURUSD", sat)
    assert open_ok is False
    assert "weekend" in reason
    # Wednesday mid-day open
    wed = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
    assert forex_session_open(wed) is True


def test_duration_fallbacks_fx_and_boom():
    fx_alts = duration_fallbacks(30, "m", symbol="frxEURUSD")
    assert (15, "m") in fx_alts or (45, "m") in fx_alts
    boom_alts = duration_fallbacks(5, "m", symbol="BOOM500")
    assert any(u == "t" for _, u in boom_alts)


def test_classify_duration_error():
    assert (
        classify_offer_error("Trading is not offered for this duration.")
        == REASON_DURATION
    )


def test_offer_gate_blocks_then_expires_logic():
    gate = MarketOfferGate(duration_cooldown_sec=1)
    gate.block(
        "BOOM500",
        reason=REASON_DURATION,
        error="Trading is not offered for this duration.",
        contract_type="CALL",
        duration=5,
        duration_unit="m",
    )
    blocked, reason = gate.is_blocked(
        "BOOM500", contract_type="CALL", duration=5, duration_unit="m"
    )
    assert blocked is True
    assert reason == REASON_DURATION
    gate.note_success("BOOM500")
    blocked2, _ = gate.is_blocked(
        "BOOM500", contract_type="CALL", duration=5, duration_unit="m"
    )
    assert blocked2 is False


def test_boom_crash_contract_sanitization():
    from src.strategy.session_hours import is_boom_symbol, is_crash_symbol, sanitize_contracts_for_symbol

    assert is_boom_symbol("BOOM500") is True
    assert is_boom_symbol("BOOM1000") is True
    assert is_crash_symbol("CRASH500") is True
    assert is_crash_symbol("CRASH1000") is True

    # BOOM should only allow CALL
    assert sanitize_contracts_for_symbol("BOOM500", ["CALL", "PUT"]) == ["CALL"]
    # CRASH should only allow PUT
    assert sanitize_contracts_for_symbol("CRASH500", ["CALL", "PUT"]) == ["PUT"]
    # Normal symbols should allow both
    assert sanitize_contracts_for_symbol("R_100", ["CALL", "PUT"]) == ["CALL", "PUT"]

