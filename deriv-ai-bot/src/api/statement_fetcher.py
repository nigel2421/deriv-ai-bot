"""Fetch Deriv account statement (trade history) via correlated WebSocket requests."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.api.deriv_client import DerivClient

logger = logging.getLogger(__name__)


class StatementFetcher:
    """Fetches trade / statement history for review and optional retraining labels."""

    def __init__(self, client: DerivClient):
        self.client = client
        self.last_raw: Optional[Dict[str, Any]] = None

    async def fetch_recent_statements(
        self,
        limit: int = 100,
        *,
        offset: int = 0,
        description: str = "",
        timeout: float = 25.0,
    ) -> List[Dict[str, Any]]:
        """
        Request statement transactions.

        Deriv `statement` returns recent account transactions (buys, sells, etc.).
        """
        msg: Dict[str, Any] = {
            "statement": 1,
            "limit": int(limit),
            "offset": int(offset),
        }
        if description:
            msg["action_type"] = description

        logger.info("Fetching statement limit=%s offset=%s", limit, offset)
        try:
            data = await self.client.request(msg, timeout=timeout)
        except Exception as e:
            logger.error("Statement request failed: %s", e)
            return []

        self.last_raw = data
        if data.get("error"):
            logger.error("Statement error: %s", data["error"])
            return []

        statement = data.get("statement") or {}
        transactions = statement.get("transactions") or []
        if not isinstance(transactions, list):
            logger.warning("Unexpected statement payload: %s", type(transactions))
            return []

        logger.info("Received %d statement transactions", len(transactions))
        return list(transactions)

    def save_to_csv(
        self,
        data: List[Dict[str, Any]],
        path: str = "data/training/statements.csv",
    ) -> Optional[Path]:
        if not data:
            logger.warning("No statement rows to save.")
            return None
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.json_normalize(data)
        if out.is_file():
            try:
                old = pd.read_csv(out)
                df = pd.concat([old, df], ignore_index=True)
                # Prefer transaction_id dedupe when present
                for key in ("transaction_id", "id", "reference_id"):
                    if key in df.columns:
                        df = df.drop_duplicates(subset=[key], keep="last")
                        break
            except Exception:
                pass
        df.to_csv(out, index=False)
        logger.info("Saved %d statement rows → %s", len(df), out)
        return out
