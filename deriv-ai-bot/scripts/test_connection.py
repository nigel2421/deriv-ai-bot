"""Quick connectivity smoke test for Deriv WebSocket + authorize."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DERIV_APP_ID, DERIV_API_TOKEN, SYMBOLS, MODE
from src.utils.logger import setup_logger
from src.api.auth_manager import AuthManager
from src.api.deriv_client import DerivClient
from src.api.price_fetcher import PriceFetcher

logger = setup_logger()


async def test_connection():
    auth = AuthManager()
    app_id, token = auth.get_credentials()

    if not app_id or not token:
        logger.error("Set DERIV_APP_ID and DERIV_API_TOKEN in .env first.")
        return False

    if str(token).startswith("your_") or str(app_id).startswith("your_"):
        logger.error("Placeholder credentials in .env — use real Deriv demo token.")
        return False

    client = DerivClient(str(app_id), str(token), MODE or "demo")
    await client.connect()

    if not client.authorized:
        logger.error("Failed to authorize with Deriv API.")
        await client.close()
        return False

    fetcher = PriceFetcher(client)
    fetcher.subscribe_symbols(SYMBOLS)

    logger.info("Waiting for ticks on %s ...", SYMBOLS)
    for i in range(15):
        await asyncio.sleep(1)
        for symbol in SYMBOLS:
            ticks = fetcher.get_recent_data(symbol, 5)
            if ticks:
                logger.info(
                    "[%ss] %s last quote=%s (buffer=%d)",
                    i + 1,
                    symbol,
                    ticks[-1].get("quote"),
                    len(client.get_latest_ticks(symbol, 1000)),
                )

    ok = any(fetcher.get_recent_data(s, 1) for s in SYMBOLS)
    await client.close()
    if ok:
        logger.info("Connection test PASSED.")
    else:
        logger.warning("Authorized but no ticks received yet — check symbols/network.")
    return ok


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
