"""Market category classification and scoring engines."""
from src.strategy.market_categories import (
    BOOM,
    CRASH,
    FOREX,
    CRYPTO,
    SYNTHETIC_VOL,
    classify_market,
    filter_allowed_for_symbol,
    market_profile,
    scoring_engine,
    scoring_path,
    PATH_DIGITS_RF,
    PATH_DIRECTIONAL,
    PATH_SPIKE,
)


def test_classify_synthetics():
    assert classify_market("R_25") == SYNTHETIC_VOL
    assert classify_market("R_100") == SYNTHETIC_VOL
    assert classify_market("1HZ50V") == SYNTHETIC_VOL
    assert classify_market("1HZ10V") == SYNTHETIC_VOL


def test_classify_extended_synthetics():
    from src.strategy.market_categories import (
        STEP,
        DSI,
        JUMP,
        DEX,
        DERIVED_FX,
        TREK,
        DAILY_RESET,
    )

    assert classify_market("stpRNG") == STEP
    assert classify_market("DSI20") == DSI
    assert classify_market("JD50") == JUMP
    assert classify_market("DEX600UP") == DEX
    assert classify_market("DEX900DN") == DEX
    assert classify_market("frxEURUSDDFx10") == DERIVED_FX
    assert classify_market("GBPUSDDFx20") == DERIVED_FX
    assert classify_market("RDBULL") == DAILY_RESET
    assert classify_market("TREKUSD") == TREK


def test_classify_boom_crash():
    assert classify_market("BOOM500") == BOOM
    assert classify_market("BOOM1000") == BOOM
    assert classify_market("CRASH500") == CRASH
    assert classify_market("CRASH1000") == CRASH


def test_classify_forex_crypto():
    assert classify_market("frxEURUSD") == FOREX
    assert classify_market("frxGBPUSD") == FOREX
    assert classify_market("cryBTCUSD") == CRYPTO
    assert classify_market("cryETHUSD") == CRYPTO


def test_scoring_paths():
    assert scoring_path(SYNTHETIC_VOL) == PATH_DIGITS_RF
    assert scoring_path(BOOM) == PATH_SPIKE
    assert scoring_path(CRASH) == PATH_SPIKE
    assert scoring_path(FOREX) == PATH_DIRECTIONAL
    assert scoring_path(CRYPTO) == PATH_DIRECTIONAL


def test_synthetic_allows_digits_forex_does_not():
    syn = market_profile("R_50")
    assert syn["digits_enabled"] is True
    assert "DIGITEVEN" in syn["allowed_contracts"]
    assert "CALL" in syn["allowed_contracts"]

    fx = market_profile("frxEURUSD")
    assert fx["digits_enabled"] is False
    assert "CALL" in fx["allowed_contracts"]
    assert "DIGITOVER" not in fx["allowed_contracts"]


def test_filter_allowed_for_symbol():
    mixed = ["DIGITOVER", "CALL", "PUT", "DIGITEVEN"]
    out = filter_allowed_for_symbol("frxUSDJPY", mixed)
    assert "CALL" in out and "PUT" in out
    assert "DIGITOVER" not in out

    out2 = filter_allowed_for_symbol("R_25", mixed)
    assert "DIGITOVER" in out2 and "CALL" in out2


def test_engine_weights_normalized():
    for cat in (SYNTHETIC_VOL, FOREX, BOOM, CRYPTO):
        eng = scoring_engine(cat)
        assert abs(sum(eng.values()) - 1.0) < 1e-6
        assert all(v > 0 for v in eng.values())


def test_synthetic_engine_includes_entropy():
    eng = scoring_engine(SYNTHETIC_VOL)
    assert "entropy" in eng or "pattern_clarity" in eng
    assert "momentum" in eng


def test_forex_engine_no_digit_entropy_primary():
    eng = scoring_engine(FOREX)
    assert "momentum" in eng
    assert "persistence" in eng
    assert "entropy" not in eng  # digit entropy not primary for FX
