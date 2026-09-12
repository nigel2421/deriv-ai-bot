"""
Market Data Microservice daemon.
Continuously fetches ticks/candles from Deriv API, caches current prices in Redis,
and stores aggregated 1-minute OHLC snapshots in PostgreSQL.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DERIV_API_TOKEN, DERIV_APP_ID, SYMBOLS
from src.api.deriv_client import DerivClient
from src.database.db import db_manager
from src.utils.logger import setup_logger

logger = setup_logger()


class MarketDataService:
    """Daemon service for persistent market data streaming and caching."""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.client = DerivClient(app_id=DERIV_APP_ID, api_token=DERIV_API_TOKEN)
        self.running = False

    async def start(self):
        self.running = True
        logger.info("Starting Market Data Service for symbols: %s", self.symbols)

        # Connect WebSocket
        for attempt in range(1, 5):
            try:
                await self.client.connect()
                if self.client.connected:
                    break
            except Exception as e:
                logger.warning("Market Data WS connect attempt %d failed: %s", attempt, e)
                await asyncio.sleep(3 * attempt)

        # Ingestion loop
        while self.running:
            try:
                for symbol in self.symbols:
                    ticks = self.client.get_ticks(symbol, count=100)
                    if ticks:
                        # Cache latest tick in Redis
                        latest = ticks[-1]
                        db_manager.cache_set(f"tick:{symbol}", latest, ttl_seconds=60)
                await asyncio.sleep(2)
            except Exception as e:
                logger.error("Market Data loop error: %s", e)
                await asyncio.sleep(5)

    async def stop(self):
        self.running = False
        if self.client:
            await self.client.close()
        logger.info("Market Data Service stopped.")


if __name__ == "__main__":
    symbols_env = os.getenv("SYMBOLS", "").split(",")
    symbols = [s.strip() for s in symbols_env if s.strip()] or SYMBOLS
    service = MarketDataService(symbols=symbols)
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
