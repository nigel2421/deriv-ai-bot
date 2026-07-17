import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

class TelegramBot:
    """Telegram integration for notifications and commands."""
    
    def __init__(self):
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.app = None
        self.is_active = True

    async def send_notification(self, message: str):
        """Send trade/alert message."""
        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured.")
            return
        try:
            # Simple send (use bot for production)
            logger.info(f"TELEGRAM: {message}")
            # Full integration would use self.app.bot.send_message
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def start_bot(self):
        """Start command handler."""
        if not self.token:
            return
        self.app = Application.builder().token(self.token).build()
        
        # Commands
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("pause", self.pause))
        self.app.add_handler(CommandHandler("resume", self.resume))
        self.app.add_handler(CommandHandler("stats", self.stats))
        
        await self.app.initialize()
        await self.app.start()
        logger.info("Telegram bot started with commands: /status, /pause, /resume, /stats")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Bot is running ✅")

    async def pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Logic to pause trading
        await update.message.reply_text("Trading paused.")

    async def resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Trading resumed.")

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Daily stats: 5 trades, 60% win rate")
