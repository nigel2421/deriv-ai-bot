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
from typing import Any, Callable, Dict, List, Optional

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
        raw_token = token if token is not None else TELEGRAM_BOT_TOKEN
        raw_chat = (
            chat_id
            if chat_id is not None
            else (TELEGRAM_CHAT_ID if TELEGRAM_CHAT_ID else None)
        )
        # Strip CR/LF/spaces — Secret Manager / Windows .env often leave \r
        self.token = str(raw_token).strip().replace("\r", "").replace("\n", "") if raw_token else None
        self.chat_id = (
            str(raw_chat).strip().replace("\r", "").replace("\n", "")
            if raw_chat is not None and str(raw_chat).strip()
            else None
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
        # Optional hooks set by orchestrator (clear risk cooldown, etc.)
        self.on_resume_hook = None
        self.on_pause_hook = None

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

            # Validate chat_id is a *user/group*, not the bot's own id
            try:
                me = await self._bot.get_me()
                if self.chat_id and str(me.id) == str(self.chat_id):
                    logger.error(
                        "TELEGRAM_CHAT_ID (%s) is the bot's own id. "
                        "Use your personal chat id (message the bot, then check "
                        "getUpdates / @userinfobot). Notifications cannot work.",
                        self.chat_id,
                    )
            except Exception as e:
                logger.warning("Telegram get_me failed: %s", e)

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
            bal = None
            cur = "USD"
            try:
                st = self._status_provider() or {}
                bal = st.get("balance")
                cur = st.get("currency") or "USD"
            except Exception:
                pass
            await self.send_notification(
                self.format_system(
                    "🤖 Deriv AI Bot online",
                    [
                        f"Trading: <b>{'ON ✅' if self.is_active else 'PAUSED ⏸'}</b>",
                        "Commands: /status /pause /resume /stats /help",
                    ],
                    balance=bal,
                    currency=cur,
                ),
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

    # ------------------------------------------------------------------ format helpers
    @staticmethod
    def _esc(s: Any) -> str:
        """Escape HTML for Telegram parse_mode=HTML."""
        t = str(s if s is not None else "")
        return (
            t.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    @staticmethod
    def _money(v: Any, currency: str = "USD") -> str:
        try:
            return f"{float(v):,.2f} {currency}".strip()
        except (TypeError, ValueError):
            return f"{v} {currency}".strip()

    def format_balance_line(self, balance: Any = None, currency: str = "USD") -> str:
        if balance is None and callable(self._status_provider):
            try:
                st = self._status_provider() or {}
                balance = st.get("balance")
                currency = st.get("currency") or currency
            except Exception:
                pass
        return f"💰 <b>Balance:</b> <code>{self._esc(self._money(balance, currency))}</code>"

    def format_trade_opened(
        self,
        *,
        symbol: str,
        contract_type: str,
        stake: Any,
        balance: Any,
        currency: str = "USD",
        confidence: Any = None,
        barrier: Any = None,
        duration: Any = None,
        duration_unit: str = "t",
        family: str = "",
        contract_id: Any = None,
        ask_price: Any = None,
        executed: bool = True,
    ) -> str:
        title = "🚀 TRADE OPENED" if executed else "📄 PROPOSAL ONLY"
        conf_s = "—"
        try:
            if confidence is not None:
                conf_s = f"{float(confidence) * 100:.1f}%"
        except (TypeError, ValueError):
            conf_s = str(confidence)
        horizon = "minute" if duration_unit == "m" else "tick"
        barrier_s = "—" if barrier is None else str(barrier)
        lines = [
            f"<b>{title}</b>",
            "──────────────",
            f"📊 <b>Market:</b> <code>{self._esc(symbol)}</code>",
            f"📐 <b>Type:</b> <code>{self._esc(contract_type)}</code>",
            f"🎯 <b>Barrier:</b> <code>{self._esc(barrier_s)}</code>",
            f"⏱ <b>Duration:</b> <code>{self._esc(duration)}{self._esc(duration_unit)}</code> ({horizon})",
            f"💵 <b>Stake:</b> <code>{self._esc(self._money(stake, currency))}</code>",
        ]
        if ask_price is not None:
            lines.append(
                f"🏷 <b>Ask:</b> <code>{self._esc(self._money(ask_price, currency))}</code>"
            )
        lines.append(f"📈 <b>Confidence:</b> <code>{self._esc(conf_s)}</code>")
        if family:
            lines.append(f"🏷 <b>Family:</b> <code>{self._esc(family)}</code>")
        if contract_id is not None:
            lines.append(f"🆔 <b>Contract:</b> <code>{self._esc(contract_id)}</code>")
        lines.append("──────────────")
        lines.append(self.format_balance_line(balance, currency))
        return "\n".join(lines)

    def format_trade_closed(
        self,
        *,
        status: str,
        symbol: str,
        contract_type: str,
        profit: Any,
        balance: Any,
        currency: str = "USD",
        stake: Any = None,
        barrier: Any = None,
        contract_id: Any = None,
        daily_pnl: Any = None,
        consecutive_losses: Any = None,
        family: str = "",
        duration: Any = None,
        duration_unit: str = "t",
    ) -> str:
        st = (status or "").upper()
        if st == "WIN":
            emoji = "✅"
        elif st == "LOSS":
            emoji = "❌"
        else:
            emoji = "➖"
        try:
            p = float(profit)
            profit_s = f"{p:+,.2f} {currency}"
        except (TypeError, ValueError):
            profit_s = f"{profit} {currency}"
        lines = [
            f"<b>{emoji} TRADE {self._esc(st)}</b>",
            "──────────────",
            f"📊 <b>Market:</b> <code>{self._esc(symbol)}</code>",
            f"📐 <b>Type:</b> <code>{self._esc(contract_type)}</code>",
        ]
        if barrier is not None:
            lines.append(f"🎯 <b>Barrier:</b> <code>{self._esc(barrier)}</code>")
        if duration is not None:
            lines.append(
                f"⏱ <b>Duration:</b> <code>{self._esc(duration)}{self._esc(duration_unit or '')}</code>"
            )
        if stake is not None:
            lines.append(
                f"💵 <b>Stake:</b> <code>{self._esc(self._money(stake, currency))}</code>"
            )
        lines.append(f"📉 <b>P&amp;L:</b> <code>{self._esc(profit_s)}</code>")
        if daily_pnl is not None:
            try:
                d = float(daily_pnl)
                lines.append(
                    f"📅 <b>Daily PnL:</b> <code>{self._esc(f'{d:+,.2f} {currency}')}</code>"
                )
            except (TypeError, ValueError):
                lines.append(f"📅 <b>Daily PnL:</b> <code>{self._esc(daily_pnl)}</code>")
        if consecutive_losses is not None:
            lines.append(
                f"🔥 <b>Loss streak:</b> <code>{self._esc(consecutive_losses)}</code>"
            )
        if family:
            lines.append(f"🏷 <b>Family:</b> <code>{self._esc(family)}</code>")
        if contract_id is not None:
            lines.append(f"🆔 <b>Contract:</b> <code>{self._esc(contract_id)}</code>")
        lines.append("──────────────")
        lines.append(self.format_balance_line(balance, currency))
        return "\n".join(lines)

    def format_trade_error(
        self,
        *,
        title: str,
        error: str,
        balance: Any = None,
        currency: str = "USD",
        symbol: str = "",
        contract_type: str = "",
        stake: Any = None,
    ) -> str:
        lines = [
            f"<b>⚠️ {self._esc(title)}</b>",
            "──────────────",
        ]
        if symbol:
            lines.append(f"📊 <b>Market:</b> <code>{self._esc(symbol)}</code>")
        if contract_type:
            lines.append(f"📐 <b>Type:</b> <code>{self._esc(contract_type)}</code>")
        if stake is not None:
            lines.append(
                f"💵 <b>Stake:</b> <code>{self._esc(self._money(stake, currency))}</code>"
            )
        lines.append(f"❗ <b>Error:</b> <code>{self._esc(error)}</code>")
        lines.append("──────────────")
        lines.append(self.format_balance_line(balance, currency))
        return "\n".join(lines)

    def format_system(
        self,
        title: str,
        body_lines: Optional[list] = None,
        *,
        balance: Any = None,
        currency: str = "USD",
    ) -> str:
        lines = [f"<b>{self._esc(title)}</b>", "──────────────"]
        for line in body_lines or []:
            lines.append(str(line))
        lines.append("──────────────")
        lines.append(self.format_balance_line(balance, currency))
        return "\n".join(lines)

    # ------------------------------------------------------------------ send
    async def send_notification(
        self,
        message: str,
        *,
        force: bool = False,
        silent: bool = False,
        parse_mode: str = "HTML",
    ) -> bool:
        """
        Send a message to the configured chat.

        Always logs locally. Network send only when configured.
        Default parse_mode=HTML for clean formatting.
        """
        text = (message or "").strip()
        if not text:
            return False

        logger.info("TELEGRAM: %s", text.replace("\n", " | "))

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
            kwargs: Dict[str, Any] = {
                "chat_id": self.chat_id,
                "text": text[:4000],
                "disable_notification": silent,
            }
            if parse_mode:
                kwargs["parse_mode"] = parse_mode
            await self._bot.send_message(**kwargs)
            self._send_count += 1
            return True
        except Exception as e:
            # Retry plain text if HTML fails
            if parse_mode:
                try:
                    plain = (
                        text.replace("<b>", "")
                        .replace("</b>", "")
                        .replace("<code>", "")
                        .replace("</code>", "")
                        .replace("&amp;", "&")
                        .replace("&lt;", "<")
                        .replace("&gt;", ">")
                    )
                    await self._bot.send_message(
                        chat_id=self.chat_id,
                        text=plain[:4000],
                        disable_notification=silent,
                    )
                    self._send_count += 1
                    return True
                except Exception as e2:
                    self._last_error = str(e2)
                    logger.error("Telegram send failed: %s", e2)
                    return False
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
        cur = st.get("currency") or "USD"
        text = self.format_system(
            "📋 Bot status",
            [
                f"Trading: <b>{trading}</b>",
                f"Risk paused: <code>{st.get('paused')}</code>",
                f"Mode: <code>{self._esc(st.get('mode', '?'))}</code>",
                f"Execute: <code>{st.get('execute_trades')}</code>",
                f"Open: <code>{st.get('open_trades')}/{st.get('max_open_trades', '?')}</code>",
                f"Daily PnL: <code>{self._esc(st.get('daily_pnl'))}</code>",
                f"Loss streak: <code>{st.get('consecutive_losses')}</code>",
                f"Trades today: <code>{st.get('trades_today')}</code>",
            ],
            balance=st.get("balance"),
            currency=cur,
        )
        try:
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception:
            await update.message.reply_text(text)

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
        cur = st.get("currency") or "USD"
        text = self.format_system(
            "📊 Daily stats",
            [
                f"Trades: <code>{total}</code> (W {wins} / L {losses})",
                f"Win rate: <code>{wr:.1f}%</code>",
                f"Daily PnL: <code>{self._esc(st.get('daily_pnl', 0))}</code>",
                f"Closed tracked: <code>{st.get('closed_count', 0)}</code>",
                f"Telegram sends: <code>{self._send_count}</code>",
            ],
            balance=st.get("balance"),
            currency=cur,
        )
        try:
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception:
            await update.message.reply_text(text)

    async def cmd_pause(self, update, context) -> None:
        if callable(self.on_pause_hook):
            try:
                self.on_pause_hook()
            except Exception as e:
                logger.warning("on_pause_hook failed: %s", e)
                self.pause_trading("telegram:/pause")
        else:
            self.pause_trading("telegram:/pause")
        bal = None
        cur = "USD"
        try:
            st = self._status_provider() or {}
            bal = st.get("balance")
            cur = st.get("currency") or "USD"
        except Exception:
            pass
        msg = self.format_system(
            "⏸ Trading PAUSED",
            [
                "No new trades until you /resume.",
                "Dashboard Resume also works.",
            ],
            balance=bal,
            currency=cur,
        )
        if update.message:
            try:
                await update.message.reply_text(msg, parse_mode="HTML")
            except Exception:
                await update.message.reply_text("Trading paused.")
        await self.send_notification(msg, force=True)

    async def cmd_resume(self, update, context) -> None:
        if callable(self.on_resume_hook):
            try:
                self.on_resume_hook()
            except Exception as e:
                logger.warning("on_resume_hook failed: %s", e)
                self.resume_trading("telegram:/resume")
        else:
            self.resume_trading("telegram:/resume")
        bal = None
        cur = "USD"
        try:
            st = self._status_provider() or {}
            bal = st.get("balance")
            cur = st.get("currency") or "USD"
        except Exception:
            pass
        msg = self.format_system(
            "▶️ Trading RESUMED",
            ["Risk cooldown cleared (if it was active)."],
            balance=bal,
            currency=cur,
        )
        if update.message:
            try:
                await update.message.reply_text(msg, parse_mode="HTML")
            except Exception:
                await update.message.reply_text("Trading resumed.")
        await self.send_notification(msg, force=True)
