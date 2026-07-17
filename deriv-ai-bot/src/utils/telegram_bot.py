"""
Telegram notifications + remote trading controls.

Commands (when bot is started):
  /help    — list commands
  /status  — balance, open trades, pause state
  /stats   — daily PnL / win rate snapshot
  /pause   — stop opening new trades
  /resume  — allow trading again
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

StatusProvider = Callable[[], Dict[str, Any]]


def _is_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True
    v = str(value).strip().lower()
    return (
        not v
        or v.startswith("your_")
        or v in {"changeme", "todo", "xxx", "none", "null"}
    )


class TelegramBot:
    """Telegram integration for notifications and /pause /resume commands."""

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        *,
        status_provider: Optional[StatusProvider] = None,
    ):
        self.token = token if token is not None else TELEGRAM_BOT_TOKEN
        self.chat_id = str(chat_id) if chat_id is not None else (
            str(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID else None
        )
        self.app = None
        self._bot = None
        self._started = False
        # Master switch: when False, orchestrator must not open new trades
        self.is_active = True
        self._status_provider: StatusProvider = status_provider or (lambda: {})
        self._stats_provider: StatusProvider = lambda: {}
        self._send_count = 0
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------ config
    def is_configured(self) -> bool:
        return not _is_placeholder(self.token) and not _is_placeholder(self.chat_id)

    def set_status_provider(self, provider: StatusProvider) -> None:
        self._status_provider = provider

    def set_stats_provider(self, provider: StatusProvider) -> None:
        self._stats_provider = provider

    @property
    def trading_enabled(self) -> bool:
        return bool(self.is_active)

    def pause_trading(self, reason: str = "telegram") -> None:
        self.is_active = False
        logger.warning("Trading PAUSED via %s", reason)

    def resume_trading(self, reason: str = "telegram") -> None:
        self.is_active = True
        logger.info("Trading RESUMED via %s", reason)

    # ------------------------------------------------------------------ lifecycle
    async def start_bot(self) -> bool:
        """Start command polling (non-blocking alongside the trading loop)."""
        if not self.is_configured():
            logger.info(
                "Telegram not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID). "
                "Commands disabled; notifications will log only."
            )
            return False
        if self._started:
            return True

        try:
            from telegram.ext import Application, CommandHandler

            self.app = (
                Application.builder()
                .token(str(self.token))
                .build()
            )
            self._bot = self.app.bot

            self.app.add_handler(CommandHandler("help", self.cmd_help))
            self.app.add_handler(CommandHandler("start", self.cmd_help))
            self.app.add_handler(CommandHandler("status", self.cmd_status))
            self.app.add_handler(CommandHandler("stats", self.cmd_stats))
            self.app.add_handler(CommandHandler("pause", self.cmd_pause))
            self.app.add_handler(CommandHandler("resume", self.cmd_resume))

            await self.app.initialize()
            await self.app.start()
            # PTB v20+: updater lives on application
            if self.app.updater:
                await self.app.updater.start_polling(
                    drop_pending_updates=True,
                    allowed_updates=["message"],
                )
            self._started = True
            logger.info(
                "Telegram bot started (chat_id=%s). Commands: "
                "/status /pause /resume /stats /help",
                self.chat_id,
            )
            await self.send_notification(
                "🤖 Deriv AI Bot online.\n"
                f"Trading: {'ON' if self.is_active else 'PAUSED'}\n"
                "Commands: /status /pause /resume /stats /help",
                force=True,
            )
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error("Failed to start Telegram bot: %s", e)
            self._started = False
            # Fall back to bare Bot for send-only
            try:
                from telegram import Bot

                self._bot = Bot(token=str(self.token))
                logger.info("Telegram send-only mode (no command polling).")
            except Exception as e2:
                logger.error("Telegram Bot init failed: %s", e2)
                self._bot = None
            return False

    async def stop_bot(self) -> None:
        if not self._started or not self.app:
            return
        try:
            if self.app.updater and self.app.updater.running:
                await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped.")
        except Exception as e:
            logger.warning("Telegram stop error: %s", e)
        finally:
            self._started = False

    # ------------------------------------------------------------------ send
    async def send_notification(
        self,
        message: str,
        *,
        force: bool = False,
        silent: bool = False,
    ) -> bool:
        """
        Send a message to the configured chat.

        Always logs locally. Network send only when configured.
        """
        text = (message or "").strip()
        if not text:
            return False

        logger.info("TELEGRAM: %s", text)

        if not self.is_configured():
            return False

        # Ensure bot client exists even without polling
        if self._bot is None:
            try:
                from telegram import Bot

                self._bot = Bot(token=str(self.token))
            except Exception as e:
                self._last_error = str(e)
                logger.error("Telegram Bot create failed: %s", e)
                return False

        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text[:4000],  # Telegram hard limit ~4096
                disable_notification=silent,
            )
            self._send_count += 1
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error("Telegram send failed: %s", e)
            return False

    # ------------------------------------------------------------------ commands
    async def cmd_help(self, update, context) -> None:
        if not update.message:
            return
        await update.message.reply_text(
            "Deriv AI Bot commands:\n"
            "/status — runtime status\n"
            "/stats — daily trade stats\n"
            "/pause — stop new trades\n"
            "/resume — allow new trades\n"
            "/help — this message"
        )

    async def cmd_status(self, update, context) -> None:
        if not update.message:
            return
        try:
            st = self._status_provider() or {}
        except Exception as e:
            await update.message.reply_text(f"Status error: {e}")
            return

        trading = "ON ✅" if self.is_active else "PAUSED ⏸"
        risk_paused = st.get("paused")
        lines = [
            f"Trading switch: {trading}",
            f"Risk paused: {risk_paused}",
            f"Mode: {st.get('mode', '?')}",
            f"Execute trades: {st.get('execute_trades', '?')}",
            f"Balance: {st.get('balance')} {st.get('currency', '')}",
            f"Open trades: {st.get('open_trades')}/{st.get('max_open_trades', st.get('max_open_trades', '?'))}",
            f"Daily PnL: {st.get('daily_pnl')}",
            f"Consecutive losses: {st.get('consecutive_losses')}",
            f"Trades today: {st.get('trades_today')}",
        ]
        await update.message.reply_text("\n".join(str(x) for x in lines))

    async def cmd_stats(self, update, context) -> None:
        if not update.message:
            return
        try:
            st = self._stats_provider() or self._status_provider() or {}
        except Exception as e:
            await update.message.reply_text(f"Stats error: {e}")
            return

        wins = st.get("wins_today", 0) or 0
        losses = st.get("losses_today", 0) or 0
        total = st.get("trades_today", wins + losses) or 0
        wr = (wins / total * 100.0) if total else 0.0
        text = (
            f"📊 Daily stats\n"
            f"Trades: {total} (W {wins} / L {losses})\n"
            f"Win rate: {wr:.1f}%\n"
            f"Daily PnL: {st.get('daily_pnl', 0)}\n"
            f"Closed tracked: {st.get('closed_count', 0)}\n"
            f"Telegram sends: {self._send_count}"
        )
        await update.message.reply_text(text)

    async def cmd_pause(self, update, context) -> None:
        self.pause_trading("telegram:/pause")
        # Also engage risk manager pause if provider exposes it
        try:
            st = self._status_provider() or {}
            # soft signal only — orchestrator checks is_active
        except Exception:
            pass
        if update.message:
            await update.message.reply_text(
                "⏸ Trading paused. Bot stays online; no new trades.\n"
                "Use /resume to continue."
            )
        await self.send_notification("⏸ Trading PAUSED via Telegram", force=True)

    async def cmd_resume(self, update, context) -> None:
        self.resume_trading("telegram:/resume")
        if update.message:
            await update.message.reply_text("▶️ Trading resumed.")
        await self.send_notification("▶️ Trading RESUMED via Telegram", force=True)
