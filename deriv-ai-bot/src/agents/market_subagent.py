import logging
from typing import Any, Dict, List, Optional
from src.agents.base_agent import BaseAgent, AgentSignal
from src.strategy.session_hours import is_likely_session_open, is_fx_symbol, is_boom_symbol, is_crash_symbol

logger = logging.getLogger(__name__)

# Full list of 20 monitored markets and their categories
MONITORED_MARKETS = [
    {"symbol": "R_10", "name": "Volatility 10 Index", "category": "Continuous Volatilities"},
    {"symbol": "R_25", "name": "Volatility 25 Index", "category": "Continuous Volatilities"},
    {"symbol": "R_50", "name": "Volatility 50 Index", "category": "Continuous Volatilities"},
    {"symbol": "R_75", "name": "Volatility 75 Index", "category": "Continuous Volatilities"},
    {"symbol": "R_100", "name": "Volatility 100 Index", "category": "Continuous Volatilities"},
    {"symbol": "1HZ10V", "name": "Volatility 10 (1s) Index", "category": "1Hz Volatilities"},
    {"symbol": "1HZ25V", "name": "Volatility 25 (1s) Index", "category": "1Hz Volatilities"},
    {"symbol": "1HZ50V", "name": "Volatility 50 (1s) Index", "category": "1Hz Volatilities"},
    {"symbol": "1HZ75V", "name": "Volatility 75 (1s) Index", "category": "1Hz Volatilities"},
    {"symbol": "1HZ100V", "name": "Volatility 100 (1s) Index", "category": "1Hz Volatilities"},
    {"symbol": "frxEURUSD", "name": "EUR/USD", "category": "Forex Majors"},
    {"symbol": "frxGBPUSD", "name": "GBP/USD", "category": "Forex Majors"},
    {"symbol": "BOOM1000", "name": "Boom 1000 Index", "category": "Boom & Crash"},
    {"symbol": "BOOM500", "name": "Boom 500 Index", "category": "Boom & Crash"},
    {"symbol": "CRASH1000", "name": "Crash 1000 Index", "category": "Boom & Crash"},
    {"symbol": "CRASH500", "name": "Crash 500 Index", "category": "Boom & Crash"},
    {"symbol": "JD10", "name": "Jump 10 Index", "category": "Jump Indices"},
    {"symbol": "JD25", "name": "Jump 25 Index", "category": "Jump Indices"},
    {"symbol": "JD50", "name": "Jump 50 Index", "category": "Jump Indices"},
    {"symbol": "STPIDX", "name": "Step Index", "category": "Step Indices"},
]


class MarketWatcherSubAgent(BaseAgent):
    """
    Dedicated Market Sub-Agent watching a single asset 24/7.
    Tracks ticks, regime, tick speed, and session readiness for its symbol.
    """

    def __init__(self, symbol: str, category: str = "Synthetic Index", enabled: bool = True):
        super().__init__(name=f"Watcher_{symbol}", enabled=enabled)
        self.symbol = symbol
        self.category = category
        self.last_price: float = 0.0
        self.tick_count: int = 0
        self.session_open: bool = True

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        if not self.enabled:
            return []

        ticks = context.get("ticks", [])
        if not ticks:
            return []

        self.tick_count = len(ticks)
        self.last_price = ticks[-1]
        open_ok, reason = is_likely_session_open(self.symbol)
        self.session_open = open_ok

        if not open_ok:
            return []

        # Return market readiness signal
        return [
            AgentSignal(
                agent_name=self.name,
                symbol=self.symbol,
                contract_type="WATCHER_OK",
                confidence=1.0,
                weight=1.0,
                rationale=f"Active monitoring {self.symbol}: {self.tick_count} ticks, price={self.last_price:.4f}",
                metadata={
                    "category": self.category,
                    "last_price": self.last_price,
                    "tick_count": self.tick_count,
                    "session": reason,
                },
            )
        ]
