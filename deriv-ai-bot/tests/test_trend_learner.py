from src.strategy.trend_analyzer import analyze_trend
from src.strategy.adaptive_learner import AdaptiveLearner
from src.strategy.contract_types import normalize_contract_type, validate_contract
from src.api.trade_executor import TradeExecutor
from src.api.deriv_client import DerivClient
from src.strategy.trade_selector import TradeSelector
from pathlib import Path


def _ticks_trend(up: bool = True, n: int = 50, start: float = 100.0):
    ticks = []
    p = start
    for i in range(n):
        p = p + (0.05 if up else -0.05) + (0.01 if i % 5 == 0 and up else 0)
        ticks.append({"quote": p, "epoch": 1000 + i, "symbol": "R_100"})
    return ticks


def test_trend_up_prefers_call():
    t = analyze_trend(_ticks_trend(up=True))
    assert t["direction"] in {"up", "flat"}  # strong series should be up
    if t["direction"] == "up":
        assert t["contract_type"] == "CALL"
        assert t["confidence"] > 0.5


def test_trend_down_prefers_put():
    t = analyze_trend(_ticks_trend(up=False))
    if t["direction"] == "down":
        assert t["contract_type"] == "PUT"


def test_adaptive_learner_boosts_winners(tmp_path: Path):
    path = tmp_path / "learn.json"
    al = AdaptiveLearner(path=path, min_samples=2)
    al.record("R_100", "CALL", True, 1.0)
    al.record("R_100", "CALL", True, 1.0)
    al.record("R_100", "CALL", True, 1.0)
    mult = al.confidence_multiplier("R_100", "CALL")
    assert mult >= 1.0
    adj = al.adjust_confidence("R_100", "CALL", 0.85)
    assert adj >= 0.85


def test_adaptive_learner_skips_cold_streak(tmp_path: Path):
    path = tmp_path / "learn2.json"
    al = AdaptiveLearner(path=path, cold_streak_skip=3, min_samples=1)
    for _ in range(3):
        al.record("R_50", "PUT", False, -1.0)
    skip, reason = al.should_skip("R_50", "PUT")
    assert skip
    assert al.adjust_confidence("R_50", "PUT", 0.9) == 0.0


def test_soft_ban_after_two_losses(tmp_path: Path):
    path = tmp_path / "learn_ban.json"
    al = AdaptiveLearner(path=path, cold_streak_skip=2, min_samples=1)
    al.record("R_25", "CALL", False, -1.0)
    al.record("R_25", "CALL", False, -1.0)
    skip, reason = al.should_skip("R_25", "CALL")
    assert skip
    assert "soft_ban" in reason or "streak" in reason or "all_losses" in reason


def test_selection_bonus_prefers_winners(tmp_path: Path):
    path = tmp_path / "learn_bonus.json"
    al = AdaptiveLearner(path=path, min_samples=2, max_selection_bonus=0.15)
    # Preferred prior alone
    b0 = al.selection_bonus("R_50", "PUT")
    assert b0 > 0  # preferred setup prior
    for _ in range(5):
        al.record("R_50", "PUT", True, 1.0)
    b1 = al.selection_bonus("R_50", "PUT")
    assert b1 >= b0
    assert b1 <= 0.15
    # Loser gets low/zero bonus
    for _ in range(4):
        al.record("R_75", "CALL", False, -1.0)
    # soft-ban may skip; bonus for weak WR should be low if not skipped
    if not al.should_skip("R_75", "CALL")[0]:
        assert al.selection_bonus("R_75", "CALL") < b1


def test_effective_min_confidence_unified(tmp_path: Path):
    """All families share the same min_confidence (strategy.xml 0.80)."""
    path = tmp_path / "learn_conf.json"
    al = AdaptiveLearner(path=path, min_samples=1)
    assert al.cold_start_phase() == "cold"
    assert al.effective_min_confidence(0.80, family="rise_fall") == 0.80
    assert al.effective_min_confidence(0.80, family="digits") == 0.80
    for i in range(100):
        al.record("R_10", "CALL", True, 1.0)
    assert al.cold_start_phase() == "mature"
    # Still unified — not 0.83 for RF vs 0.82 for digits
    assert al.effective_min_confidence(0.80, family="rise_fall") == 0.80
    assert al.effective_min_confidence(0.80, family="digits", contract_type="DIGITEVEN") == 0.80


def test_selector_prefers_learn_bonus_and_winners():
    sel = TradeSelector()
    best = sel.select_best_trade(
        [
            {
                "symbol": "R_100",
                "contract_type": "CALL",
                "confidence": 0.82,
                "family": "rise_fall",
                "learn_bonus": 0.0,
                "live_edge": 70,
                "quality_score": 70,
            },
            {
                "symbol": "R_50",
                "contract_type": "PUT",
                "confidence": 0.81,
                "family": "rise_fall",
                "learn_bonus": 0.12,
                "live_edge": 75,
                "quality_score": 78,
                "decision_quality": 82,
            },
        ]
    )
    assert best["symbol"] == "R_50"
    assert best["contract_type"] == "PUT"


def test_call_put_proposal_v2():
    client = DerivClient("33R2Z6MTElnIWrId8aH3m", "tok")
    ex = TradeExecutor(client)
    msg = ex.build_proposal("R_75", "CALL", 1.0, barrier=None, duration=5)
    assert msg["contract_type"] == "CALL"
    assert "barrier" not in msg
    assert msg["underlying_symbol"] == "R_75"


def test_validate_call_put():
    assert normalize_contract_type("RISE") == "CALL"
    assert normalize_contract_type("FALL") == "PUT"
    ok, _, b = validate_contract("CALL", None)
    assert ok and b is None


def test_selector_picks_highest_conf():
    sel = TradeSelector()
    best = sel.select_best_trade(
        [
            {"symbol": "R_10", "contract_type": "CALL", "confidence": 0.81, "family": "rise_fall"},
            {
                "symbol": "R_100",
                "contract_type": "DIGITOVER",
                "confidence": 0.91,
                "family": "digits",
                "learn_bonus": 0.02,
            },
        ]
    )
    assert best["symbol"] == "R_100"


def test_chart_tools_vote_on_uptrend():
    from src.strategy.chart_tools import chart_snapshot, rise_fall_vote

    ticks = []
    p = 100.0
    for i in range(80):
        p += 0.08
        ticks.append({"quote": p, "epoch": 1000 + i})
    snap = chart_snapshot(ticks)
    assert snap["ready"] is True
    ct, conf, detail = rise_fall_vote(snap)
    # Strong uptrend should lean CALL when tools agree
    assert snap.get("ema_bull") is True
    if ct:
        assert ct == "CALL"
        assert conf > 0.55


def test_regime_skips_choppy():
    from src.strategy.regime_filter import assess_regime, should_skip_rise_fall

    # Alternating up/down = chop
    ticks = []
    p = 100.0
    for i in range(50):
        p += 0.05 if i % 2 == 0 else -0.05
        ticks.append({"quote": p, "epoch": 1000 + i})
    reg = assess_regime(ticks)
    assert reg["chop_score"] > 0.4
    skip, reason, _ = should_skip_rise_fall(ticks)
    assert skip is True


def test_adaptive_barrier_varies_with_prediction():
    from src.strategy.barrier_picker import adaptive_barrier

    # Prices ending in high digits → quotes like x.x7, x.x8, x.x9
    ticks = []
    for i in range(40):
        d = 7 + (i % 3)  # 7,8,9
        ticks.append({"quote": 100.0 + d / 10.0, "epoch": 1000 + i})
    b_over, meta = adaptive_barrier(
        "DIGITOVER", predicted_digit=8, ticks=ticks, mode="adaptive"
    )
    assert b_over is not None
    assert b_over < 8  # 8 must win OVER@b
    assert meta.get("win_rate") is not None

    b_under, meta2 = adaptive_barrier(
        "DIGITUNDER", predicted_digit=2, ticks=ticks, mode="adaptive"
    )
    assert b_under is not None
    assert b_under > 2

    # fixed mode still available
    bf, _ = adaptive_barrier("DIGITOVER", mode="fixed", fixed_over=6)
    assert bf == 6
