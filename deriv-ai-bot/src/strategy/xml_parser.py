import xml.etree.ElementTree as ET
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class XMLStrategyParser:
    """Parses strategy.xml for flexible configuration."""
    
    def __init__(self, config_path: str = "config/strategy.xml"):
        self.config_path = config_path
        self.config = self._parse()

    def _parse(self) -> Dict:
        try:
            tree = ET.parse(self.config_path)
            root = tree.getroot()
            config = {}
            
            # Global settings
            global_elem = root.find('global')
            if global_elem is not None:
                config['global'] = {
                    'min_confidence': float(global_elem.find('min_confidence').text),
                    'max_daily_loss_pct': float(global_elem.find('max_daily_loss_pct').text),
                    'max_consecutive_losses': int(global_elem.find('max_consecutive_losses').text),
                    'trade_pause_minutes': int(global_elem.find('trade_pause_minutes').text),
                }
            
            # Market-specific strategies
            config['markets'] = {}
            for market in root.findall('market'):
                symbol = market.get('symbol')
                strategy_elem = market.find('strategy')
                if strategy_elem is not None:
                    config['markets'][symbol] = {
                        'type': strategy_elem.get('type'),
                        'base_stake': float(strategy_elem.find('base_stake').text),
                        'max_steps': int(strategy_elem.find('max_steps').text) if strategy_elem.find('max_steps') is not None else 6,
                        # Add more as needed
                    }
            return config
        except Exception as e:
            logger.error(f"Failed to parse strategy.xml: {e}")
            return {'global': {}, 'markets': {}}

    def get_strategy(self, symbol: str) -> Dict:
        return self.config['markets'].get(symbol, self.config['markets'].get('R_100', {}))
