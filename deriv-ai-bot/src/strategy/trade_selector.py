from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TradeSelector:
    """Selects best trade across markets."""
    
    def select_best_trade(self, signals: List[Dict]) -> Dict:
        if not signals:
            return None
        # Select highest confidence
        best = max(signals, key=lambda x: x.get('confidence', 0))
        logger.info(f"Selected best trade: {best}")
        return best
