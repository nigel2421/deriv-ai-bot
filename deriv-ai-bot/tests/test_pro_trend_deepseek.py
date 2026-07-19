"""Pro-trend strategy and DeepSeek advisor unit tests."""
from pathlib import Path

from src.ai.deepseek_advisor import DeepSeekAdvisor
from src.strategy.pro_trend import analyze_pro_trend, market_structure


def _ticks_trend(up: bool = True, n: int = 220, start: float = 1000.0):
    ticks = []
    p = start
    for i in range(n):
        # Smooth trend with mild pullbacks
        step = 0.08 if up else -0.08
        if i % 12 == 0:
            step *= -0.35  # pullback
        p = p + step
        ticks.append({"quote": p, "epoch": 1_700_000_000 + i, "symbol": "R_75"})
    return ticks


def test_market_structure_uptrend():
    closes = [100 + i * 0.5 + (0.1 if i % 3 else 0) for i in range(40)]
    st = market_structure(closes, look=8)
    assert st["hh"] or st["uptrend"] or st["hl"]


def test_pro_trend_up_prefers_call_or_flat():
    t = analyze_pro_trend(_ticks_trend(up=True), symbol="R_75", min_confidence=0.70)
    assert t["ready"]
    if t.get("contract_type"):
        assert t["contract_type"] == "CALL"
        assert t["confidence"] > 0.5


def test_pro_trend_down_prefers_put_or_flat():
    t = analyze_pro_trend(_ticks_trend(up=False), symbol="R_100", min_confidence=0.70)
    assert t["ready"]
    if t.get("contract_type"):
        assert t["contract_type"] == "PUT"


def test_deepseek_disabled_without_key():
    adv = DeepSeekAdvisor(api_key=None, enabled=True)
    assert not adv.is_ready()
    assert adv.analyze(trades=[]) is None


def test_deepseek_apply_recommendation_sets_multipliers(tmp_path: Path):
    cache = tmp_path / "ds.json"
    adv = DeepSeekAdvisor(api_key="sk-test", enabled=True, cache_path=cache)
    adv.apply_recommendation(
        {
            "summary": "digits weak",
            "risk_score": 70,
            "trade_type_analysis": [
                {
                    "contract_type": "DIGITOVER",
                    "symbol": "R_75",
                    "verdict": "ban",
                    "suggested_confidence_mult": 0.55,
                },
                {
                    "contract_type": "CALL",
                    "symbol": "R_75",
                    "verdict": "keep",
                    "suggested_confidence_mult": 1.1,
                },
            ],
        }
    )
    assert adv.confidence_multiplier("R_75", "DIGITOVER") == 0.55
    assert adv.confidence_multiplier("R_75", "CALL") == 1.1
    assert cache.is_file()


def test_deepseek_parse_json_fenced():
    raw = '```json\n{"summary": "ok", "risk_score": 10}\n```'
    obj = DeepSeekAdvisor._parse_json_content(raw)
    assert obj is not None
    assert obj["summary"] == "ok"
    assert obj["risk_score"] == 10


def test_deepseek_note_closed_trade_every_n():
    adv = DeepSeekAdvisor(api_key="sk-test", enabled=True, analyze_every=3)
    assert adv.note_closed_trade() is False
    assert adv.note_closed_trade() is False
    assert adv.note_closed_trade() is True
