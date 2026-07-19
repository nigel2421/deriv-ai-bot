"""DeepSeek sample gates + per-market/strategy bucketing."""
from src.ai.deepseek_advisor import DeepSeekAdvisor


def _trade(sym, fam, ct, status, profit=1.0):
    return {
        "symbol": sym,
        "family": fam,
        "contract_type": ct,
        "status": status,
        "profit": profit,
        "stake": 1.0,
        "confidence": 0.85,
    }


def test_buckets_group_by_market_and_strategy():
    trades = [
        _trade("R_25", "rise_fall", "CALL", "win"),
        _trade("R_25", "rise_fall", "CALL", "loss", -1),
        _trade("R_25", "digits", "DIGITOVER", "win"),
        _trade("R_50", "rise_fall", "PUT", "win"),
    ]
    buckets = DeepSeekAdvisor.build_market_strategy_buckets(trades)
    keys = {b["key"] for b in buckets}
    assert "R_25|rise_fall" in keys
    assert "R_25|digits" in keys
    assert "R_50|rise_fall" in keys
    rf = next(b for b in buckets if b["key"] == "R_25|rise_fall")
    assert rf["n"] == 2
    assert any(t["contract_type"] == "CALL" for t in rf["by_contract_type"])


def test_auto_skips_small_sample(tmp_path):
    adv = DeepSeekAdvisor(
        api_key="sk-test",
        enabled=True,
        cache_path=tmp_path / "ds.json",
        analyze_every=20,
        min_sample=20,
        min_per_setup=12,
    )
    trades = [_trade("R_10", "rise_fall", "CALL", "win") for _ in range(5)]
    # note 5 closes — should not trigger
    for _ in range(5):
        assert adv.note_closed_trade("R_10", "rise_fall", "CALL") is False
    buckets, keys, reason = adv.select_buckets_for_analysis(trades, force=False)
    assert buckets == []
    assert reason == "insufficient_sample"


def test_per_setup_trigger_at_12(tmp_path):
    adv = DeepSeekAdvisor(
        api_key="sk-test",
        enabled=True,
        cache_path=tmp_path / "ds.json",
        analyze_every=20,
        min_sample=20,
        min_per_setup=12,
    )
    for i in range(11):
        assert adv.note_closed_trade("BOOM500", "rise_fall", "CALL") is False
    assert adv.note_closed_trade("BOOM500", "rise_fall", "CALL") is True
    assert "BOOM500|rise_fall" in adv.due_setup_keys()


def test_global_trigger_at_20(tmp_path):
    adv = DeepSeekAdvisor(
        api_key="sk-test",
        enabled=True,
        cache_path=tmp_path / "ds.json",
        analyze_every=20,
        min_sample=20,
        min_per_setup=12,
    )
    # Spread across many setups so no single hits 12 first
    for i in range(19):
        sym = f"R_{(i % 5) * 25 or 10}"
        assert adv.note_closed_trade(sym, "rise_fall", "CALL") is False
    assert adv.note_closed_trade("R_100", "rise_fall", "PUT") is True


def test_merge_recommendations_keep_other_markets(tmp_path):
    adv = DeepSeekAdvisor(
        api_key="sk-test",
        enabled=True,
        cache_path=tmp_path / "ds.json",
    )
    adv.apply_recommendation(
        {
            "trade_type_analysis": [
                {
                    "symbol": "R_25",
                    "contract_type": "CALL",
                    "verdict": "keep",
                    "suggested_confidence_mult": 1.15,
                }
            ]
        },
        merge=True,
    )
    adv.apply_recommendation(
        {
            "trade_type_analysis": [
                {
                    "symbol": "CRASH500",
                    "contract_type": "PUT",
                    "verdict": "ban",
                    "suggested_confidence_mult": 0.5,
                }
            ]
        },
        merge=True,
    )
    assert adv.confidence_multiplier("R_25", "CALL") >= 1.1
    assert adv.is_banned("CRASH500", "PUT") is True
    assert adv.is_banned("R_25", "CALL") is False


def test_force_allows_smaller_buckets(tmp_path):
    adv = DeepSeekAdvisor(
        api_key="sk-test",
        enabled=True,
        cache_path=tmp_path / "ds.json",
        min_sample=20,
        min_per_setup=12,
    )
    trades = [
        _trade("R_25", "rise_fall", "CALL", "win"),
        _trade("R_25", "rise_fall", "PUT", "loss", -1),
        _trade("R_25", "rise_fall", "CALL", "win"),
    ]
    buckets, keys, reason = adv.select_buckets_for_analysis(trades, force=True)
    assert reason == "force"
    assert any(b["symbol"] == "R_25" for b in buckets)
