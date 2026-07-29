import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Deriv public sample app id for *legacy* WebSocket only (numeric).
# New developers.deriv.com OAuth/web apps use alphanumeric App IDs → auto v2 mode.
_DEFAULT_APP_ID = "1089"

_raw_app_id = os.getenv("DERIV_APP_ID") or ""
if not _raw_app_id or _raw_app_id.startswith("your_"):
    DERIV_APP_ID = _DEFAULT_APP_ID
else:
    DERIV_APP_ID = _raw_app_id

# Bearer token: legacy API token OR OAuth access_token OR PAT from developers.deriv.com
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")
# Optional: pin a demo/real options account id (e.g. VRTC12345)
DERIV_ACCOUNT_ID = os.getenv("DERIV_ACCOUNT_ID") or None
# auto | legacy | v2  (auto: numeric app id → legacy, else v2 OAuth/PAT)
DERIV_API_MODE = (os.getenv("DERIV_API_MODE") or "auto").strip().lower()
DERIV_API_BASE = os.getenv("DERIV_API_BASE", "https://api.derivws.com").rstrip("/")
# OAuth (web login) — client_id is usually the same as DERIV_APP_ID for new apps
DERIV_OAUTH_CLIENT_ID = os.getenv("DERIV_OAUTH_CLIENT_ID") or DERIV_APP_ID
DERIV_OAUTH_CLIENT_SECRET = os.getenv("DERIV_OAUTH_CLIENT_SECRET") or None
DERIV_OAUTH_REDIRECT_URI = os.getenv(
    "DERIV_OAUTH_REDIRECT_URI",
    "https://deriv-ai-bot-842806243906.us-central1.run.app/oauth/callback",
)
DERIV_OAUTH_AUTH_URL = os.getenv(
    "DERIV_OAUTH_AUTH_URL", "https://auth.deriv.com/oauth2/auth"
)
DERIV_OAUTH_TOKEN_URL = os.getenv(
    "DERIV_OAUTH_TOKEN_URL", "https://auth.deriv.com/oauth2/token"
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODE = os.getenv("MODE", "demo")
# Diversified portfolio: classic + 1Hz + major FX (override via SYMBOLS env)
_DEFAULT_SYMBOLS = (
    "R_10,R_25,R_50,R_75,R_100,"
    "1HZ10V,1HZ25V,1HZ50V,1HZ75V,1HZ100V,"
    "frxEURUSD,frxGBPUSD"
)
SYMBOLS = [
    s.strip()
    for s in os.getenv("SYMBOLS", _DEFAULT_SYMBOLS).split(",")
    if s.strip()
]
# Always-on adaptive learning (persists under data/ while instance is warm)
LEARNING_ALWAYS = _env_bool("LEARNING_ALWAYS", True)
LEARNING_PATH = os.getenv("LEARNING_PATH", "data/learning_state.json")

# When False: request proposals only (no buy). Demo defaults to True; real defaults False.
_execute_default = MODE != "real"
EXECUTE_TRADES = _env_bool("EXECUTE_TRADES", _execute_default)

# Contract duration in ticks for digit trades
TRADE_DURATION_TICKS = _env_int("TRADE_DURATION_TICKS", 5)

# Minimum net return on a win: profit / stake (0.5 = +50% / $1 → total payout ≥ 1.5× stake).
# Skip proposals that pay less (e.g. DIGITOVER@0 often ~+0.05–0.15).
MIN_NET_RETURN = _env_float("MIN_NET_RETURN", 0.5)

# --- Risk / money management (override strategy.xml where applicable) ---
# Minimum account balance required to open a new trade
MIN_BALANCE = _env_float("MIN_BALANCE", 1.0)
# Max concurrent open contracts
MAX_OPEN_TRADES = _env_int("MAX_OPEN_TRADES", 3)
# Max stake as % of live balance (tight default — stops $60 martingale blowups)
MAX_STAKE_PCT = _env_float("MAX_STAKE_PCT", 1.0)
# Floor/ceiling stake (Deriv demo min is often 0.35)
MIN_STAKE = _env_float("MIN_STAKE", 0.35)
# Hard max stake in account currency (blocks 2^n martingale runaway)
_raw_max_stake = os.getenv("MAX_STAKE")
MAX_STAKE = float(_raw_max_stake) if _raw_max_stake not in (None, "") else 8.0

# Tick history bootstrap (warmup AI buffers on connect)
TICK_HISTORY_COUNT = _env_int("TICK_HISTORY_COUNT", 500)
TICK_HISTORY_MIN = _env_int("TICK_HISTORY_MIN", 50)
# Save bootstrapped ticks under data/historical/ (for training)
SAVE_TICK_HISTORY = _env_bool("SAVE_TICK_HISTORY", True)

# Model training gates (also used by scripts/train_model.py)
# Random digit baseline ~0.10; require slight edge by default
MIN_MODEL_ACCURACY = _env_float("MIN_MODEL_ACCURACY", 0.12)
MIN_MODEL_LIFT = _env_float("MIN_MODEL_LIFT", 0.0)
FORCE_SAVE_MODEL = _env_bool("FORCE_SAVE_MODEL", False)
TRAIN_LSTM = _env_bool("TRAIN_LSTM", False)

# ── DeepSeek AI Advisor ────────────────────────────────────────────────────
# Per-market analysis: triggers every DEEPSEEK_ANALYZE_EVERY trades on a symbol
# Reads full GCS-backed trade_history.jsonl for maximum accuracy
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_ENABLED = _env_bool("DEEPSEEK_ENABLED", False)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# How many closed trades per symbol before triggering an analysis
DEEPSEEK_ANALYZE_EVERY = _env_int("DEEPSEEK_ANALYZE_EVERY", 100)
