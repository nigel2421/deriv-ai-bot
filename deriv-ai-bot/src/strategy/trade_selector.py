from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TradeSelector:
    """Selects best trade across markets."""

    def select_best_trade(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not signals:
            return None
        best = max(signals, key=lambda x: x.get("confidence", 0))
        logger.info("Selected best trade: %s", best)
        return best

