import logging
from typing import Any, Dict, Optional
from src.agents.base_agent import BaseAgent
from src.api.trade_executor import TradeExecutor

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    """
    Execution Agent: Solely responsible for order placement, Deriv API trade proposals,
    buy execution, and contract settlement callbacks.
    Separates execution latency from strategy analysis.
    """

    def __init__(self, executor: TradeExecutor, name: str = "ExecutionAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        self.executor = executor

    async def execute_trade(self, trade_intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute trade intent via Deriv API executor pipeline.
        Returns trade execution outcome dict or None if execution failed.
        """
        if not self.enabled or not trade_intent:
            return None

        symbol = trade_intent["symbol"]
        contract_type = trade_intent["contract_type"]
        stake = trade_intent["stake"]
        duration = trade_intent.get("duration", 5)
        duration_unit = trade_intent.get("duration_unit", "t")
        barrier = trade_intent.get("barrier")

        logger.info(
            "ExecutionAgent sending order: %s %s stake=%.2f dur=%s%s barrier=%s",
            symbol,
            contract_type,
            stake,
            duration,
            duration_unit,
            barrier,
        )

        try:
            res = await self.executor.execute_trade(
                symbol=symbol,
                contract_type=contract_type,
                stake=stake,
                duration=duration,
                duration_unit=duration_unit,
                barrier=barrier,
            )
            if res and res.get("status") == "bought":
                logger.info(
                    "ExecutionAgent order executed successfully: contract_id=%s symbol=%s",
                    res.get("contract_id"),
                    symbol,
                )
                return {
                    **trade_intent,
                    "contract_id": res.get("contract_id"),
                    "buy_price": res.get("buy_price"),
                    "status": "executed",
                    "execution_details": res,
                }
            else:
                logger.warning("ExecutionAgent trade execution returned status: %s", res)
                return None
        except Exception as e:
            logger.error("ExecutionAgent failed to execute trade: %s", e, exc_info=True)
            return None

    async def evaluate(self, context: Dict[str, Any]) -> list:
        return []
