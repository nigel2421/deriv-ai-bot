from src.strategy.martingale import MartingaleStrategy
from src.strategy.risk_manager import RiskManager
from src.strategy.zuno_strategy import ZunoStrategy
from src.strategy.xml_parser import XMLStrategyParser
from src.strategy.strategy_engine import StrategyEngine


def test_martingale_peek_and_result():
    mg = MartingaleStrategy(base_stake=2.0, max_steps=3)
    assert mg.peek_stake() == 2.0
    # Loss → double
    assert mg.on_result(False) == 4.0
    assert mg.peek_stake() == 4.0
    assert mg.on_result(False) == 8.0
    # Win → reset
    assert mg.on_result(True) == 2.0
    assert mg.current_loss_streak == 0


def test_martingale_max_steps_deactivates():
    mg = MartingaleStrategy(base_stake=1.0, max_steps=2)
    mg.on_result(False)  # streak 1, stake 2
    mg.on_result(False)  # streak 2, stake 4
    stake = mg.on_result(False)  # streak 3 > 2 → off
    assert stake == 0.0
    assert not mg.active
    assert mg.peek_stake() == 0.0
    mg.reset()
    assert mg.active
    assert mg.peek_stake() == 1.0


def test_martingale_legacy_get_next_stake():
    mg = MartingaleStrategy(base_stake=2.0, max_steps=6)
    assert mg.get_next_stake(False) > 0
    assert mg.get_next_stake(True) == 2.0


def test_zuno_switch():
    z = ZunoStrategy(
        switch_on_win="DIGITOVER",
        switch_on_loss="DIGITUNDER",
        initial_type="DIGITOVER",
    )
    assert z.peek_type() == "DIGITOVER"
    assert z.on_result(False) == "DIGITUNDER"
    assert z.peek_type() == "DIGITUNDER"
    assert z.on_result(True) == "DIGITOVER"


def test_xml_parser_full_fields():
    p = XMLStrategyParser("config/strategy.xml")
    g = p.config["global"]
    assert g["min_confidence"] == 0.80

    r100 = p.get_strategy("R_100")
    assert r100["type"] == "martingale"
    assert r100["base_stake"] == 1.0
    assert r100["max_steps"] == 3
    assert r100["over_barrier"] == 6
    assert r100["under_barrier"] == 4
    assert "DIGITOVER" in r100["contract_types"]
    assert "DIGITUNDER" in r100["contract_types"]
    assert "DIGITEVEN" in r100["contract_types"]
    assert "DIGITODD" in r100["contract_types"]
    assert "CALL" in r100["contract_types"]
    assert "PUT" in r100["contract_types"]

    r75 = p.get_strategy("R_75")
    assert r75["type"] == "martingale"
    assert r75["base_stake"] == 1.0
    assert r75["over_barrier"] == 6
    assert r75["under_barrier"] == 4

    # Portfolio symbols present (classic + 1Hz)
    for sym in (
        "R_10",
        "R_25",
        "R_50",
        "R_75",
        "R_100",
        "1HZ10V",
        "1HZ25V",
        "1HZ50V",
        "1HZ75V",
        "1HZ100V",
    ):
        assert sym in p.config["markets"]


def test_strategy_engine_martingale_stake():
    engine = StrategyEngine(XMLStrategyParser("config/strategy.xml"))
    # Adaptive barriers: pass predicted digit + ticks so OVER/UNDER vary
    ticks = [{"quote": 100.0 + (i % 10) * 0.1, "epoch": 1000 + i} for i in range(40)]
    intent = engine.apply_signal(
        "R_100", "DIGITOVER", 5, 0.9, predicted_digit=8, ticks=ticks
    )
    assert intent is not None
    assert intent["stake"] == 1.0
    assert intent["contract_type"] == "DIGITOVER"
    assert intent["barrier"] is not None
    assert 0 <= int(intent["barrier"]) <= 8
    # Predicted 8 should be in win set (barrier < 8)
    assert int(intent["barrier"]) < 8
    assert intent["strategy"] == "martingale"

    engine.on_trade_result("R_100", is_win=False)
    intent2 = engine.apply_signal(
        "R_100", "DIGITUNDER", 3, 0.9, predicted_digit=2, ticks=ticks
    )
    assert intent2 is not None
    assert intent2["stake"] == 2.0  # doubled after loss
    assert intent2["barrier"] is not None
    assert 1 <= int(intent2["barrier"]) <= 9
    assert int(intent2["barrier"]) > 2  # pred 2 in UNDER win set


def test_strategy_engine_digits_and_even():
    engine = StrategyEngine(XMLStrategyParser("config/strategy.xml"))
    # EVEN is allowed in portfolio
    intent = engine.apply_signal("R_75", "DIGITEVEN", None, 0.9)
    assert intent is not None
    assert intent["contract_type"] == "DIGITEVEN"
    assert intent["stake"] == 1.0
    assert intent["barrier"] is None

    ticks = [{"quote": 503.78, "epoch": 1000 + i} for i in range(30)]  # last digit 8
    intent2 = engine.apply_signal(
        "R_75", "DIGITOVER", None, 0.9, predicted_digit=8, ticks=ticks
    )
    assert intent2 is not None
    assert intent2["contract_type"] == "DIGITOVER"
    assert intent2["barrier"] is not None
    assert 0 <= int(intent2["barrier"]) <= 8
    assert int(intent2["barrier"]) < 8


def test_risk_blocks_unknown_balance():
    rm = RiskManager(min_balance=1.0)
    d = rm.can_trade(None)
    assert not d
    assert d.reason == "balance_unknown"


def test_risk_blocks_low_balance():
    rm = RiskManager(min_balance=10.0)
    d = rm.can_trade(5.0)
    assert not d
    assert "balance_below_min" in d.reason


def test_risk_blocks_max_open():
    rm = RiskManager(min_balance=1.0, max_open_trades=2)
    d = rm.can_trade(100.0, open_trades=2)
    assert not d
    assert "max_open_trades" in d.reason


def test_risk_stake_cap_and_clamp():
    rm = RiskManager(min_balance=1.0, max_stake_pct=5.0, min_stake=0.35)
    d = rm.can_trade(100.0, proposed_stake=10.0)
    assert not d
    assert "stake_above_cap" in d.reason
    assert rm.clamp_stake(10.0, 100.0) == 5.0


def test_daily_loss_limit():
    rm = RiskManager(min_balance=1.0, max_daily_loss_pct=5.0)
    rm.set_session_balance(100.0)
    rm.record_trade_result(-3.0)
    rm.record_trade_result(-2.5)
    d = rm.can_trade(94.5)
    assert not d
    assert "daily_loss_limit" in d.reason


def test_win_resets_streak():
    rm = RiskManager(min_balance=1.0, max_consecutive_losses=5)
    rm.record_trade_result(-1)
    rm.record_trade_result(-1)
    rm.record_trade_result(2)
    assert rm.consecutive_losses == 0
    assert rm.daily_pnl == 0.0


def test_cooldown_auto_resumes_and_resets_streak():
    """
    Regression: after max consecutive losses the bot paused 30m, but on
    expiry consecutive_losses stayed at max → infinite re-pause loop.
    """
    from datetime import datetime, timedelta, timezone

    rm = RiskManager(
        min_balance=1.0,
        max_consecutive_losses=3,
        trade_pause_minutes=30,
    )
    rm.record_trade_result(-1)
    rm.record_trade_result(-1)
    rm.record_trade_result(-1)
    assert rm.consecutive_losses == 3
    assert rm.is_paused()
    assert rm.paused_until is not None

    # Still blocked during cooldown
    d = rm.can_trade(100.0)
    assert not d
    assert "paused" in d.reason

    # Simulate timer expiry
    rm.paused_until = datetime.now(timezone.utc) - timedelta(seconds=5)
    # can_trade must auto-resume and allow trading again
    d2 = rm.can_trade(100.0)
    assert d2, d2.reason
    assert rm.consecutive_losses == 0
    assert not rm.is_paused()
    evt = rm.consume_auto_resume()
    assert evt is not None
    assert evt.get("previous_streak") == 3
    # Second can_trade must NOT re-pause forever
    d3 = rm.can_trade(100.0)
    assert d3
    assert rm.consecutive_losses == 0


def test_manual_resume_clears_pause():
    rm = RiskManager(min_balance=1.0, max_consecutive_losses=2, trade_pause_minutes=30)
    rm.record_trade_result(-1)
    rm.record_trade_result(-1)
    assert rm.is_paused()
    rm.resume(reset_streak=True)
    assert not rm.is_paused()
    assert rm.consecutive_losses == 0
    assert rm.can_trade(50.0)
