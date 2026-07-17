"""
Bootstrap tick history smoke test.

  python scripts/test_tick_history.py
  python scripts/test_tick_history.py --count 100
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DERIV_APP_ID, MODE, SYMBOLS, TICK_HISTORY_COUNT
from src.utils.logger import setup_logger
from src.api.auth_manager import AuthManager
from src.api.deriv_client import DerivClient
from src.api.price_fetcher import PriceFetcher
from src.ai.predictor import Predictor

logger = setup_logger()


async def run(count: int) -> int:
    auth = AuthManager()
    app_id, token = auth.get_credentials()
    if not token or str(token).startswith("your_"):
        logger.error("Set DERIV_API_TOKEN in .env")
        return 1

    client = DerivClient(str(app_id or DERIV_APP_ID), str(token), MODE or "demo")
    ok = await client.connect()
    if not ok:
        logger.error("Authorize failed")
        await client.close()
        return 1

    fetcher = PriceFetcher(client)
    fetcher.subscribe_symbols(SYMBOLS)

    sizes = await fetcher.bootstrap_history(
        SYMBOLS,
        count=count,
        min_required=50,
        save_dir=str(ROOT / "data" / "historical"),
    )
    logger.info("Buffer sizes: %s", sizes)

    pred = Predictor()
    for symbol in SYMBOLS:
        ticks = fetcher.get_recent_data(symbol, count)
        logger.info(
            "%s ticks=%d first_epoch=%s last_quote=%s",
            symbol,
            len(ticks),
            ticks[0].get("epoch") if ticks else None,
            ticks[-1].get("quote") if ticks else None,
        )
        if len(ticks) >= 50:
            out = pred.predict(ticks)
            logger.info("%s prediction: %s", symbol, out)

    failed = [s for s, n in sizes.items() if n < 50]
    await client.close()
    if failed:
        logger.error("Bootstrap incomplete for: %s", failed)
        return 1
    logger.info("Tick history bootstrap PASSED")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=TICK_HISTORY_COUNT)
    args = p.parse_args()
    sys.exit(asyncio.run(run(args.count)))


if __name__ == "__main__":
    main()
