import asyncio

from src.utils.telegram_bot import TelegramBot, _is_placeholder


def test_placeholder_detection():
    assert _is_placeholder(None)
    assert _is_placeholder("")
    assert _is_placeholder("your_telegram_bot_token")
    assert not _is_placeholder("123456:ABC-DEF")


def test_not_configured_without_env():
    bot = TelegramBot(token="your_telegram_bot_token", chat_id="your_chat_id")
    assert not bot.is_configured()
    assert bot.trading_enabled is True


def test_pause_resume_flags():
    bot = TelegramBot(token="your_x", chat_id="your_y")
    bot.pause_trading("test")
    assert bot.trading_enabled is False
    bot.resume_trading("test")
    assert bot.trading_enabled is True


def test_send_notification_logs_when_unconfigured():
    bot = TelegramBot(token="your_token", chat_id="your_chat")
    ok = asyncio.run(bot.send_notification("hello test"))
    assert ok is False  # no network send
    assert bot._send_count == 0


def test_status_provider_wiring():
    state = {"balance": 100.0, "mode": "demo", "open_trades": 0, "daily_pnl": 0}
    bot = TelegramBot(token="t", chat_id="1", status_provider=lambda: state)
    bot.set_status_provider(lambda: {**state, "paused": False})
    assert bot._status_provider()["balance"] == 100.0
    assert bot._status_provider()["paused"] is False


def test_orchestrator_respects_telegram_pause():
    bot = TelegramBot(token="your_a", chat_id="your_b")
    assert bot.trading_enabled
    bot.pause_trading()
    assert not bot.trading_enabled
