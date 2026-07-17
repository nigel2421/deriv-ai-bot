from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.api.deriv_client import DerivClient

logger = logging.getLogger(__name__)


class PriceFetcher:
    """Live tick subscription + historical bootstrap for AI warmup."""

    def __init__(self, client: DerivClient):
        self.client = client

    def subscribe_symbols(self, symbols: List[str]) -> None:
        self.client.subscribe_ticks(symbols)
        logger.info("Subscribed to price data for: %s", symbols)

    def get_recent_data(self, symbol: str, count: int = 100) -> List[Dict]:
        return self.client.get_latest_ticks(symbol, count)

    def buffer_sizes(self, symbols: List[str]) -> Dict[str, int]:
        return {s: self.client.buffer_size(s) for s in symbols}

    async def bootstrap_history(
        self,
        symbols: List[str],
        count: int = 500,
        *,
        min_required: int = 50,
        save_dir: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Fetch ticks_history for each symbol and seed live buffers.

        Returns map symbol → buffer size after bootstrap.
        """
        results: Dict[str, int] = {}
        for symbol in symbols:
            existing = self.client.buffer_size(symbol)
            if existing >= count:
                logger.info(
                    "%s already has %d ticks (>= %d); skip history fetch",
                    symbol,
                    existing,
                    count,
                )
                results[symbol] = existing
                continue

            need = max(count - existing, min_required)
            ticks = await self.client.fetch_ticks_history(
                symbol, count=need, seed_buffer=True
            )
            size = self.client.buffer_size(symbol)
            results[symbol] = size

            if size < min_required:
                logger.warning(
                    "%s buffer only %d ticks after bootstrap (need %d)",
                    symbol,
                    size,
                    min_required,
                )
            else:
                logger.info(
                    "%s ready: %d ticks in buffer (fetched %d history)",
                    symbol,
                    size,
                    len(ticks),
                )

            if save_dir and ticks:
                self._save_ticks_csv(symbol, ticks, save_dir)

        return results

    async def ensure_ready(
        self,
        symbols: List[str],
        min_ticks: int = 50,
        history_count: int = 500,
    ) -> bool:
        """
        Bootstrap if needed; return True if every symbol has >= min_ticks.
        """
        sizes = self.buffer_sizes(symbols)
        if all(sizes.get(s, 0) >= min_ticks for s in symbols):
            return True

        await self.bootstrap_history(
            symbols, count=history_count, min_required=min_ticks
        )
        sizes = self.buffer_sizes(symbols)
        ok = all(sizes.get(s, 0) >= min_ticks for s in symbols)
        if not ok:
            logger.warning("Not all symbols reached min_ticks=%s: %s", min_ticks, sizes)
        return ok

    @staticmethod
    def _save_ticks_csv(
        symbol: str, ticks: List[Dict], save_dir: str
    ) -> Optional[Path]:
        try:
            path = Path(save_dir)
            path.mkdir(parents=True, exist_ok=True)
            out = path / f"{symbol}_ticks.csv"
            df = pd.DataFrame(ticks)
            # Append if file exists
            if out.is_file():
                old = pd.read_csv(out)
                df = (
                    pd.concat([old, df], ignore_index=True)
                    .drop_duplicates(subset=["epoch"], keep="last")
                    .sort_values("epoch")
                )
            df.to_csv(out, index=False)
            logger.info("Saved %d ticks → %s", len(df), out)
            return out
        except Exception as e:
            logger.warning("Failed to save ticks for %s: %s", symbol, e)
            return None
