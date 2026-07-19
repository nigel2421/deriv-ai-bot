"""DeepSeek recommendation hard-execution helpers."""
from src.ai.deepseek_advisor import DeepSeekAdvisor


def test_ban_and_preferred_from_recommendation(tmp_path):
    adv = DeepSeekAdvisor(
        api_key="sk-test",
        enabled=True,
        cache_path=tmp_path / "ds.json",
        analyze_every=5,
    )
    # Don't need network for apply_recommendation
    rec = {
        "summary": "test",
        "risk_score": 20,
        "trade_type_analysis": [
            {
                "contract_type": "DIGITOVER",
                "symbol": "R_25",
                "verdict": "ban",
                "suggested_confidence_mult": 0.5,
            },
            {
                "contract_type": "CALL",
                "symbol": "R_75",
                "verdict": "keep",
                "suggested_confidence_mult": 1.15,
            },
            {
                "contract_type": "PUT",
                "symbol": "BOOM500",
                "verdict": "reduce",
                "suggested_confidence_mult": 0.8,
            },
        ],
    }
    adv.apply_recommendation(rec)

    assert adv.is_banned("R_25", "DIGITOVER") is True
    assert adv.is_banned("R_50", "DIGITOVER") is False  # only R_25|DIGITOVER banned
    assert adv.is_banned("R_75", "CALL") is False
    assert adv.confidence_multiplier("R_75", "CALL") >= 1.1
    assert adv.selection_boost("R_75", "CALL") > 0
    assert adv.selection_boost("R_25", "DIGITOVER") < 0
    assert "R_75" in adv.preferred_symbols()


def test_selector_prefers_deepseek_boost():
    from src.strategy.trade_selector import TradeSelector

    sel = TradeSelector()
    signals = [
        {
            "symbol": "R_10",
            "contract_type": "CALL",
            "confidence": 0.82,
            "learn_bonus": 0.0,
            "family": "rise_fall",
            "deepseek_mult": 0.85,
            "deepseek_boost": -0.02,
            "live_edge": 70,
            "quality_score": 70,
            "pattern_strength": 70,
            "decision_quality": 70,
            "ev": 0.05,
            "mp_score": 70,
        },
        {
            "symbol": "R_75",
            "contract_type": "CALL",
            "confidence": 0.81,
            "learn_bonus": 0.0,
            "family": "rise_fall",
            "deepseek_mult": 1.15,
            "deepseek_boost": 0.05,
            "live_edge": 70,
            "quality_score": 70,
            "pattern_strength": 70,
            "decision_quality": 70,
            "ev": 0.05,
            "mp_score": 70,
        },
    ]
    best = sel.select_best_trade(signals)
    assert best is not None
    assert best["symbol"] == "R_75"
