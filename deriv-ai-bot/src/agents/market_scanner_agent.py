import logging
from typing import Any, Dict, List, Optional
from src.agents.base_agent import BaseAgent, AgentSignal
from src.strategy.session_hours import is_likely_session_open, is_fx_symbol, is_boom_symbol, is_crash_symbol, sanitize_contracts_for_symbol

logger = logging.getLogger(__name__)


class MarketScannerAgent(BaseAgent):
    """
    Market Scanner Agent: Monitors market availability, session hours,
    liquidity depth, and filters candidate symbols.
    """

    def __init__(self, name: str = "MarketScannerAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        """
        Scans symbols provided in context.
        Returns a meta-signal for each tradeable symbol with market condition metadata.
        """
        if not self.enabled:
            return []

        symbols = context.get("symbols", [])
        fetcher = context.get("fetcher")
        offer_gate = context.get("offer_gate")

        scanned_signals: List[AgentSignal] = []

        for symbol in symbols:
            # 1. Check session open status
            open_ok, session_reason = is_likely_session_open(symbol)
            if not open_ok:
                logger.debug("MarketScannerAgent: %s skipped (%s)", symbol, session_reason)
                continue

            # 2. Check offer gate block
            if offer_gate and offer_gate.is_symbol_blocked(symbol):
                logger.debug("MarketScannerAgent: %s blocked by offer_gate", symbol)
                continue

            # 3. Check tick data availability
            ticks = fetcher.get_recent_data(symbol, 120) if fetcher else context.get("ticks_map", {}).get(symbol, [])
            if not ticks or len(ticks) < 10:
                logger.debug("MarketScannerAgent: insufficient ticks for %s", symbol)
                continue

            # Signal that symbol is scan-ready
            scanned_signals.append(
                AgentSignal(
                    agent_name=self.name,
                    symbol=symbol,
                    contract_type="SCAN_OK",
                    confidence=1.0,
                    weight=1.0,
                    rationale=f"Market open ({session_reason}), {len(ticks)} ticks ready",
                    metadata={
                        "ticks_count": len(ticks),
                        "is_fx": is_fx_symbol(symbol),
                        "is_boom": is_boom_symbol(symbol),
                        "is_crash": is_crash_symbol(symbol),
                        "last_price": ticks[-1] if ticks else 0.0,
                    }
                )
            )

        return scanned_signals
