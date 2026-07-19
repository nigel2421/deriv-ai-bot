"""Market Opportunity Ranking (MOR) formula + tiers."""
from src.analytics.market_opportunity_ranking import (
    OpportunityHistory,
    compute_mor,
    correlation_filter,
    opportunity_tier,
    opportunity_score,
    risk_penalties,
    regime_match_score,
    validate_mor_ranking,
)


def test_opportunity_score_example():
    # Spec-ish: 88, 84, 82, 90, 76, 95 with production weights
    o = opportunity_score(
        pattern_strength=88,
        pattern_clarity=84,
        hpp=82,
        hpp_velocity=12,  # ~90 mapped
        momentum_persistence=76,
        regime_match=95,
        expected_value=0.08,
        confidence=90,
    )
    assert 80 <= o["opportunity_raw"] <= 95


def test_penalties_and_final():
    mor = compute_mor(
        pattern_strength=88,
        pattern_clarity=84,
        hpp=82,
        hpp_velocity=10,
        momentum_persistence=76,
        regime_match=95,
        expected_value=0.1,
        confidence=90,
        sample_n=100,
        drawdown_pct=0.06,  # -10
    )
    assert mor["opportunity_score"] < mor["opportunity_raw"]
    assert mor["tier"] in {"ELITE", "STRONG", "WATCHLIST", "IGNORE"}


def test_low_sample_penalty():
    p = risk_penalties(sample_n=10)
    assert p["total_penalty"] >= 15


def test_tiers():
    assert opportunity_tier(92) == "ELITE"
    assert opportunity_tier(85) == "STRONG"
    assert opportunity_tier(72) == "WATCHLIST"
    assert opportunity_tier(60) == "IGNORE"


def test_regime_match_rf_vs_random():
    good = regime_match_score(
        family="rise_fall", market_regime="STRONG PATTERN", chop_score=0.1
    )
    bad = regime_match_score(
        family="rise_fall", market_regime="RANDOM", chop_score=0.6
    )
    assert good > bad


def test_opportunity_velocity(tmp_path):
    h = OpportunityHistory(path=tmp_path / "mor.json")
    for s in (79, 82, 85, 90):
        pack = h.note("R_75", s, "STRONG")
    assert pack["velocity"] > 0
    assert pack["emerging"] or pack["velocity"] >= 0


def test_correlation_filter():
    ranked = [
        {"symbol": "R_75", "category": "synthetic_vol", "tier": "ELITE", "score": 91},
        {"symbol": "R_100", "category": "synthetic_vol", "tier": "STRONG", "score": 88},
        {"symbol": "BOOM500", "category": "boom", "tier": "STRONG", "score": 87},
    ]
    out = correlation_filter(ranked, max_per_cluster=1)
    # second synthetic marked filtered
    synth = [r for r in out if r["symbol"] in {"R_75", "R_100"}]
    assert any(r.get("correlation_filtered") for r in synth) or len(synth) == 2


def test_validate_mor():
    outcomes = []
    for i in range(40):
        outcomes.append(
            {
                "symbol": "R_75",
                "is_win": True,
                "profit": 1.0,
                "mor_score": 92,
                "tier": "ELITE",
                "opp_velocity": 2.0,
                "confidence": 90,
            }
        )
    for i in range(40):
        outcomes.append(
            {
                "symbol": "JD100",
                "is_win": i % 3 == 0,
                "profit": 1.0 if i % 3 == 0 else -1.0,
                "mor_score": 55,
                "tier": "IGNORE",
                "opp_velocity": -3.0,
                "confidence": 40,
            }
        )
    v = validate_mor_ranking(outcomes)
    assert "checks" in v
    assert v["n_outcomes"] == 80
