"""Session stop-loss, 1:3 profit target, and stake configuration tests."""
from src.strategy.risk_manager import RiskManager


def test_session_stop_loss_blocks_trading():
    rm = RiskManager(
        min_balance=1.0,
        session_stop_loss_pct=5.0,
        session_target_rr=3.0,
        max_consecutive_losses=99,
    )
    rm.set_session_balance(100.0)
    # 5% of 100 = 5
    rm.record_trade_result(-3.0)
    assert rm.can_trade(97.0)
    rm.record_trade_result(-2.5)
    assert rm.session_stop_hit
    d = rm.can_trade(94.5)
    assert not d
    assert "session_stop_loss" in d.reason


def test_session_target_1_to_3_stops_trading():
    rm = RiskManager(
        min_balance=1.0,
        session_stop_loss_pct=5.0,  # risk 5 → target 15
        session_target_rr=3.0,
        session_stop_on_target=True,
        max_consecutive_losses=99,
    )
    rm.set_session_balance(100.0)
    assert rm.session_stop_loss_amount() == 5.0
    assert rm.session_target_amount() == 15.0
    rm.record_trade_result(8.0)
    assert rm.can_trade(108.0)
    rm.record_trade_result(8.0)  # daily 16 >= 15
    assert rm.session_target_hit
    d = rm.can_trade(116.0)
    assert not d
    assert "session_target_hit" in d.reason


def test_stop_loss_pct_clamped_to_band():
    rm = RiskManager(
        min_balance=1.0,
        session_stop_loss_pct=1.0,  # below min → 5
        session_stop_loss_pct_min=5.0,
        session_stop_loss_pct_max=10.0,
    )
    assert rm.session_stop_loss_pct == 5.0
    rm.configure_session_risk(stop_loss_pct=12.0)
    assert rm.session_stop_loss_pct == 10.0
    rm.configure_session_risk(stop_loss_pct=7.5)
    assert rm.session_stop_loss_pct == 7.5


def test_resume_clears_session_flags():
    rm = RiskManager(min_balance=1.0, session_stop_loss_pct=5.0, session_target_rr=3.0)
    rm.set_session_balance(100.0)
    rm.session_stop_hit = True
    rm.session_target_hit = True
    rm.resume(reset_streak=True)
    assert not rm.session_stop_hit
    assert not rm.session_target_hit
    assert rm.can_trade(100.0)


def test_configure_base_stake():
    rm = RiskManager(min_balance=1.0, base_stake=1.0)
    snap = rm.configure_session_risk(base_stake=0.75, max_stake_pct=1.5)
    assert snap["base_stake"] == 0.75
    assert snap["max_stake_pct"] == 1.5


def test_reset_session_run():
    rm = RiskManager(min_balance=1.0, session_stop_loss_pct=5.0)
    rm.set_session_balance(100.0)
    rm.record_trade_result(-1.0)
    rm.reset_session_run(95.0)
    assert rm.daily_pnl == 0.0
    assert rm.session_start_balance == 95.0
    assert rm.session_stop_loss_amount() == 4.75
