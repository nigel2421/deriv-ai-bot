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
    assert g["min_confidence"] == 0.72

    r100 = p.get_strategy("R_100")
    assert r100["type"] == "martingale"
    assert r100["base_stake"] == 2.0
    assert r100["max_steps"] == 5
    assert r100["over_barrier"] == 6
    assert r100["under_barrier"] == 4
    assert "DIGITOVER" in r100["contract_types"]
    assert "DIGITUNDER" in r100["contract_types"]
    assert "DIGITODD" not in r100["contract_types"]

    r75 = p.get_strategy("R_75")
    assert r75["type"] == "martingale"
    assert r75["base_stake"] == 1.5
    assert r75["over_barrier"] == 6
    assert r75["under_barrier"] == 4

    # Portfolio symbols present
    for sym in ("R_10", "R_25", "R_50", "R_75", "R_100"):
        assert sym in p.config["markets"]


def test_strategy_engine_martingale_stake():
    engine = StrategyEngine(XMLStrategyParser("config/strategy.xml"))
    intent = engine.apply_signal("R_100", "DIGITOVER", 5, 0.9)
    assert intent is not None
    assert intent["stake"] == 2.0
    assert intent["contract_type"] == "DIGITOVER"
    # Fixed strategy barrier OVER@6 (ignores AI barrier 5)
    assert intent["barrier"] == 6
    assert intent["strategy"] == "martingale"

    engine.on_trade_result("R_100", is_win=False)
    intent2 = engine.apply_signal("R_100", "DIGITUNDER", 3, 0.9)
    assert intent2 is not None
    assert intent2["stake"] == 4.0  # doubled after loss
    assert intent2["barrier"] == 4  # fixed UNDER@4


def test_strategy_engine_over_under_only():
    engine = StrategyEngine(XMLStrategyParser("config/strategy.xml"))
    # EVEN not allowed — falls through to allow-list OVER/UNDER
    intent = engine.apply_signal("R_75", "DIGITEVEN", None, 0.9)
    assert intent is not None
    assert intent["contract_type"] in {"DIGITOVER", "DIGITUNDER"}
    assert intent["stake"] == 1.5
    if intent["contract_type"] == "DIGITOVER":
        assert intent["barrier"] == 6
    else:
        assert intent["barrier"] == 4


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
