import asyncio
import logging
from typing import List, Dict
from config.settings import SYMBOLS
from src.api.deriv_client import DerivClient
from src.api.price_fetcher import PriceFetcher
from src.api.trade_executor import TradeExecutor
from src.api.trade_monitor import TradeMonitor
from src.strategy.xml_parser import XMLStrategyParser
from src.strategy.risk_manager import RiskManager
from src.strategy.trade_selector import TradeSelector
from src.strategy.signal_generator import SignalGenerator
from src.ai.predictor import Predictor

logger = logging.getLogger(__name__)

class TradingOrchestrator:
    """Main coordinator for multi-market scanning and trading logic."""
    
    def __init__(self, client: DerivClient, mode: str = "demo"):
        self.client = client
        self.fetcher = PriceFetcher(client)
        self.executor = TradeExecutor(client)
        self.monitor = TradeMonitor(client)
        self.parser = XMLStrategyParser()
        self.risk_manager = RiskManager()
        self.selector = TradeSelector()
        self.signal_gen = SignalGenerator()
        self.predictor = Predictor()
        self.telegram = TelegramBot()
        self.mode = mode
        self.active_symbols = SYMBOLS
        self.max_open_trades = 3

    async def scan_markets(self):
        """Multi-market scan: Predict + select best trades."""
        signals = []
        for symbol in self.active_symbols:
            ticks = self.fetcher.get_recent_data(symbol, 100)
            if not ticks:
                continue
            
            # AI Prediction
            pred = self.predictor.predict(ticks)
            confidence = pred.get('confidence', 0.5)
            
            strategy = self.parser.get_strategy(symbol)
            min_conf = self.parser.config.get('global', {}).get('min_confidence', 0.75)
            
            # Generate signal
            contract_type, barrier, conf = self.signal_gen.generate_signal(
                pred, confidence, min_conf
            )
            if contract_type:
                signals.append({
                    'symbol': symbol,
                    'contract_type': contract_type,
                    'barrier': barrier,
                    'confidence': conf,
                    'strategy': strategy.get('type')
                })
        
        # Select best
        best_trade = self.selector.select_best_trade(signals)
        return best_trade

    async def execute_trade_cycle(self):
        """Full cycle: Scan -> Risk check -> Execute."""
        if not self.risk_manager.can_trade(1000):  # Placeholder balance
            logger.warning("Risk checks failed. Skipping cycle.")
            return
        
        best = await self.scan_markets()
        if not best:
            return
        
        # Execute
        logger.info(f"Executing trade: {best}")
        await self.telegram.send_notification(f"🚀 New Trade: {best['symbol']} {best['contract_type']} Confidence: {best['confidence']:.1%}")
        
        self.executor.send_proposal(
            best['symbol'], 
            best['contract_type'], 
            2.0,  # Base stake from strategy
            best.get('barrier')
        )
        # Monitor would handle follow-up in real flow
