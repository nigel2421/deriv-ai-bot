"""
Trading Engine Microservice daemon.
Runs continuous multi-market scanning, signal generation, risk management,
and trade execution. Records all trades to PostgreSQL and Redis.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DERIV_API_TOKEN, DERIV_APP_ID
from src.api.deriv_client import DerivClient
from src.bot_runtime import runtime, start_bot, stop_bot
from src.database.db import db_manager
from src.utils.logger import setup_logger

logger = setup_logger()


async def run_trading_service():
    mode = os.getenv("MODE", "demo")
    cycle_seconds = int(os.getenv("TRADE_CYCLE_SECONDS", "45"))

    logger.info(
        "Starting Trading Engine Service (mode=%s, cycle=%ds, app_id=%s)",
        mode,
        cycle_seconds,
        DERIV_APP_ID,
    )

    # Initialize bot runtime
    for attempt in range(1, 4):
        try:
            await start_bot(mode=mode, cycle_seconds=cycle_seconds)
            if runtime.status in {"running", "starting"}:
                break
        except Exception as e:
            logger.exception("Trading Engine start attempt %d failed: %s", attempt, e)
        await asyncio.sleep(5 * attempt)

    logger.info("Trading Engine Service operational.")

    # Main supervision loop
    try:
        while True:
            await asyncio.sleep(15)
            # Sync status to Redis cache for dashboard/monitoring services
            status_data = runtime.public_status()
            db_manager.cache_set("trading_engine:status", status_data, ttl_seconds=60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Stopping Trading Engine Service...")
        await stop_bot()


if __name__ == "__main__":
    try:
        asyncio.run(run_trading_service())
    except KeyboardInterrupt:
        logger.info("Trading Engine daemon shutdown.")
