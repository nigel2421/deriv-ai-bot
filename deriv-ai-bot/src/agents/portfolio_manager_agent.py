import logging
from typing import Any, Dict, List, Optional
from src.agents.base_agent import BaseAgent, AgentSignal, ConsensusDecision


logger = logging.getLogger(__name__)


class PortfolioManagerAgent(BaseAgent):
    """
    Stage 5: Portfolio Manager Agent
    
    Acts as the final risk and capital allocation authority.
    Sits between Consensus voting and Trade Execution.
    No specialist agent can override the Portfolio Manager.
    
    Responsibilities:
    - Enforces 1% risk per trade capital allocation rule ($10 max risk on $1,000 balance).
    - Hard daily drawdown limit checks (5% max daily loss).
    - Max concurrent open trades check.
    - Correlated asset class exposure controls.
    """

    def __init__(
        self,
        risk_per_trade_pct: float = 1.0,
        max_daily_loss_pct: float = 5.0,
        max_open_trades: int = 20,
        stake_floor: float = 1.00,
        stake_ceiling: float = 10.00,
        name: str = "PortfolioManagerAgent",
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_trades = max_open_trades
        self.stake_floor = stake_floor
        self.stake_ceiling = stake_ceiling

    def evaluate_portfolio_decision(
        self,
        decision: ConsensusDecision,
        account_balance: float,
        open_trades: List[Dict[str, Any]],
        daily_pnl: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate consensus decision against portfolio parameters.
        Returns portfolio-approved trade intent or None if hard veto applied.
        """
        if not self.enabled or not decision:
            return None

        symbol = decision.symbol
        contract_type = decision.contract_type
        balance = max(10.0, account_balance)

        # 1. Hard Veto: Daily Drawdown Limit Check
        max_daily_loss = balance * (self.max_daily_loss_pct / 100.0)
        if daily_pnl <= -max_daily_loss:
            reason = f"Hard Veto: Daily PnL (${daily_pnl:.2f}) reached max daily drawdown (-${max_daily_loss:.2f})"
            logger.warning("PortfolioManagerAgent VETO symbol=%s: %s", symbol, reason)
            return None

        # 2. Hard Veto: Max Concurrent Open Trades Check
        if len(open_trades) >= self.max_open_trades:
            reason = f"Hard Veto: Open trades ({len(open_trades)}) >= max limit ({self.max_open_trades})"
            logger.warning("PortfolioManagerAgent VETO symbol=%s: %s", symbol, reason)
            return None

        # 3. Hard Veto: Correlated Exposure Control (Max 2 trades in same asset class)
        symbol_prefix = symbol.split("_")[0] if "_" in symbol else symbol[:3]
        matching_group = sum(1 for t in open_trades if str(t.get("symbol", "")).startswith(symbol_prefix))
        if matching_group >= 2:
            reason = f"Hard Veto: Group exposure limit reached for asset class '{symbol_prefix}' ({matching_group} active)"
            logger.warning("PortfolioManagerAgent VETO symbol=%s: %s", symbol, reason)
            return None

        # 4. Position Sizing Engine: 1% Account Risk Sizing
        calculated_stake = balance * (self.risk_per_trade_pct / 100.0)
        final_stake = round(max(self.stake_floor, min(self.stake_ceiling, calculated_stake)), 2)

        approved_intent = {
            "symbol": symbol,
            "contract_type": contract_type,
            "stake": final_stake,
            "confidence": decision.confidence,
            "ensemble_score": decision.ensemble_score,
            "barrier": decision.barrier,
            "duration": decision.duration,
            "duration_unit": decision.duration_unit,
            "portfolio_approved": True,
            "portfolio_rationale": f"Approved 1% risk stake ${final_stake:.2f} (balance ${balance:.2f})",
            "participating_agents": [v.agent_name for v in decision.votes],
        }

        logger.info(
            "PortfolioManagerAgent APPROVED trade: symbol=%s ct=%s stake=$%.2f (1%% of $%.2f balance)",
            symbol,
            contract_type,
            final_stake,
            balance,
        )

        return approved_intent

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        """Not used directly; call evaluate_portfolio_decision()."""
        return []
