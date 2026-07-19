"""Session hours + re-open policy for FX / closed markets."""
from datetime import datetime, timezone

from src.strategy.market_offer_gate import MarketOfferGate, REASON_MARKET_CLOSED
from src.strategy.session_hours import forex_session_open, is_likely_session_open


def test_synthetics_always_open():
    ok, why = is_likely_session_open("R_25")
    assert ok is True
    assert "synthetic" in why or "always" in why


def test_forex_weekend_closed():
    # Saturday 12:00 UTC
    sat = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
    assert forex_session_open(sat) is False
    ok, why = is_likely_session_open("frxEURUSD", sat)
    assert ok is False
    assert "weekend" in why or "closed" in why


def test_forex_weekday_open():
    # Wednesday 14:00 UTC
    wed = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
    assert forex_session_open(wed) is True
    ok, _ = is_likely_session_open("frxGBPUSD", wed)
    assert ok is True


def test_closed_block_expires_and_clears_on_success():
    g = MarketOfferGate(market_closed_cooldown_sec=1, max_market_closed_cooldown_sec=2)
    g.note_error("frxEURUSD", "Market is closed")
    assert g.is_symbol_blocked("frxEURUSD") is True
    # Success path clears immediately
    g.note_success("frxEURUSD")
    assert g.is_symbol_blocked("frxEURUSD") is False


def test_block_not_permanent_policy():
    g = MarketOfferGate()
    snap = g.snapshot()
    assert snap.get("policy") == "temporary_cooldown_then_reprobe"
