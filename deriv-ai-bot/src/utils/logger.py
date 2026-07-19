import logging
import logging.config
import os

def setup_logger():
    log_dir = 'data/logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/bot.log'),
            logging.StreamHandler()
        ]
    )
    # Avoid logging full Telegram bot tokens in request URLs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return logging.getLogger(__name__)
