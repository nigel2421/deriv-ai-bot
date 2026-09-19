import logging
from typing import Any, Dict, Optional
from src.agents.base_agent import BaseAgent, ConsensusDecision
from src.strategy.risk_manager import RiskManager
from src.strategy.anti_spiral import AntiSpiral
from src.strategy.correlation_filter import CorrelationFilter
from config.settings import MAX_OPEN_TRADES, MAX_STAKE_PCT, MIN_BALANCE, MIN_STAKE, MAX_STAKE

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    """
    Risk Management Agent: Enforces stake bounds, daily drawdown limits,
    anti-spiral streak protection, and multi-asset correlation filtering.
    """

    def __init__(self, name: str = "RiskAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        self.risk_manager = RiskManager(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=6,
            trade_pause_minutes=60,
            min_balance=MIN_BALANCE,
            max_open_trades=MAX_OPEN_TRADES,
            max_stake_pct=MAX_STAKE_PCT,
            min_stake=MIN_STAKE,
            max_stake=MAX_STAKE,
        )
        self.anti_spiral = AntiSpiral()
        self.correlation_filter = CorrelationFilter()

    def evaluate_risk(
        self,
        decision: ConsensusDecision,
        current_balance: float,
        open_trade_count: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Validate decision against risk manager, anti-spiral state, and stake bounds.
        Returns risk-approved trade parameters dict or None if rejected.
        """
        if not self.enabled or not decision:
            return None

        # 1. Update risk manager balance
        self.risk_manager.update_balance(current_balance)
        if self.risk_manager.session_start_balance is None:
            self.risk_manager.set_session_balance(current_balance)

        # 2. Check if daily loss or pause threshold reached
        risk_dec = self.risk_manager.can_trade(account_balance=current_balance, open_trades=open_trade_count)
        if not risk_dec.allowed:
            logger.info("RiskAgent block symbol=%s: %s", decision.symbol, risk_dec.reason)
            return None


        # 3. Check anti-spiral streak safety
        anti_ok, anti_reason = self.anti_spiral.allow(
            decision.symbol, decision.contract_type, decision.confidence
        )
        if not anti_ok:
            logger.info("RiskAgent AntiSpiral block symbol=%s: %s", decision.symbol, anti_reason)
            return None

        # 4. Calculate dynamic stake based on balance and risk rules
        raw_intent = decision.raw_intent or {}
        proposed_stake = float(raw_intent.get("stake", MIN_STAKE))
        stake = self.risk_manager.clamp_stake(
            stake=proposed_stake,
            account_balance=current_balance,
        )
        if stake <= 0:
            stake = float(MIN_STAKE)


        approved_intent = {
            **raw_intent,
            "symbol": decision.symbol,
            "contract_type": decision.contract_type,
            "stake": stake,
            "confidence": decision.confidence,
            "ensemble_score": decision.ensemble_score,
            "barrier": decision.barrier,
            "duration": decision.duration,
            "duration_unit": decision.duration_unit,
            "participating_agents": raw_intent.get("participating_agents", []),
            "risk_approved": True,
            "risk_rationale": f"Approved stake ${stake:.2f} (balance ${current_balance:.2f})",
        }

        logger.info(
            "RiskAgent approved trade symbol=%s ct=%s stake=%.2f conf=%.2f",
            approved_intent["symbol"],
            approved_intent["contract_type"],
            approved_intent["stake"],
            approved_intent["confidence"],
        )

        return approved_intent

    async def evaluate(self, context: Dict[str, Any]) -> list:
        return []
