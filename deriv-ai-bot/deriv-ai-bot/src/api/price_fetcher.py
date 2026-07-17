from src.api.deriv_client import DerivClient
from typing import List
import logging

logger = logging.getLogger(__name__)

class PriceFetcher:
    def __init__(self, client: DerivClient):
        self.client = client

    def subscribe_symbols(self, symbols: List[str]):
        self.client.subscribe_ticks(symbols)
        logger.info(f"Subscribed to price data for: {symbols}")

    def get_recent_data(self, symbol: str, count: int = 100):
        return self.client.get_latest_ticks(symbol, count)
