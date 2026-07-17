import json
import logging
from typing import List, Dict
import pandas as pd
from src.api.deriv_client import DerivClient

logger = logging.getLogger(__name__)

class StatementFetcher:
    """Fetches trade history for AI retraining."""
    
    def __init__(self, client: DerivClient):
        self.client = client

    def fetch_recent_statements(self, limit: int = 100) -> List[Dict]:
        """Request statement data."""
        msg = {"statement": 1, "limit": limit}
        self.client.send_message(msg)
        logger.info("Fetching trade statements...")
        # In real impl, handle response in client on_message
        return []  # Placeholder - parse from WS responses

    def save_to_csv(self, data: List[Dict], path: str = "data/training/features.csv"):
        if data:
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
            logger.info(f"Saved {len(data)} trades to {path}")
