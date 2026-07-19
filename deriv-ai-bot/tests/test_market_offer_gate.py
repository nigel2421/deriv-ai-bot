"""Market offer gate: skip duration/closed markets."""
from src.strategy.market_offer_gate import (
    REASON_DURATION,
    REASON_MARKET_CLOSED,
    MarketOfferGate,
    classify_offer_error,
    duration_fallbacks,
)


def test_classify_duration_error():
    assert (
        classify_offer_error("Trading is not offered for this duration.")
        == REASON_DURATION
    )
    assert classify_offer_error("Market is closed") == REASON_MARKET_CLOSED


def test_block_duration_and_skip():
    g = MarketOfferGate(duration_cooldown_sec=600)
    g.note_error(
        "RDBEAR",
        "Trading is not offered for this duration.",
        contract_type="CALL",
        duration=5,
        duration_unit="t",
    )
    blocked, reason = g.is_blocked(
        "RDBEAR", contract_type="CALL", duration=5, duration_unit="t"
    )
    assert blocked is True
    assert reason == REASON_DURATION
    # Other market still open
    b2, _ = g.is_blocked("R_25", contract_type="CALL", duration=5, duration_unit="t")
    assert b2 is False


def test_market_closed_blocks_symbol():
    g = MarketOfferGate(market_closed_cooldown_sec=600)
    g.note_error("frxEURUSD", "Market is currently closed")
    assert g.is_symbol_blocked("frxEURUSD") is True
    b, r = g.is_blocked("frxEURUSD", contract_type="CALL", duration=5, duration_unit="t")
    assert b is True
    assert r == REASON_MARKET_CLOSED


def test_duration_fallbacks_exclude_primary():
    fb = duration_fallbacks(5, "t")
    assert (5, "t") not in fb
    assert any(u == "t" for _, u in fb)
