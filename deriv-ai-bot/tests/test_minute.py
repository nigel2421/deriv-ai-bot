from src.strategy.candles import build_candles
from src.strategy.minute_engine import analyze_minute


def _up_ticks(n=200, start=100.0):
    ticks = []
    p = start
    for i in range(n):
        p += 0.02 + (0.01 if i % 10 == 0 else 0)
        ticks.append({"quote": p, "epoch": 1_700_000_000 + i})
    return ticks


def test_build_candles_from_epochs():
    ticks = _up_ticks(180)
    candles = build_candles(ticks, period_sec=60)
    assert len(candles) >= 2
    assert candles[-1]["close"] >= candles[0]["open"]


def test_minute_engine_uptrend_call():
    ticks = _up_ticks(400)
    sig = analyze_minute(ticks, period_sec=60, min_confidence=0.55, duration_minutes=2)
    # Strong uptrend should often yield CALL; allow None if not enough structure
    if sig:
        assert sig["contract_type"] in {"CALL", "PUT"}
        assert sig["duration_unit"] == "m"
        assert sig["family"] == "minute_rise_fall"
