import os
from dotenv import load_dotenv

load_dotenv()

DERIV_APP_ID = os.getenv('DERIV_APP_ID')
DERIV_API_TOKEN = os.getenv('DERIV_API_TOKEN')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
MODE = os.getenv('MODE', 'demo')
SYMBOLS = os.getenv('SYMBOLS', 'R_100,R_75').split(',')
