import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from src.agents.base_agent import BaseAgent, AgentSignal
from src.agents.reputation import AgentReputationEngine

logger = logging.getLogger(__name__)


class ChiefStrategyAgent(BaseAgent):
    """
    Stage 2: Meta-Agent (Chief Strategy Agent)
    
    Responsibilities:
    1. Monitors all specialist strategy agents.
    2. Ranks agents based on accuracy and profit factor.
    3. Promotes strong performers by increasing voting weights.
    4. Auto-disables/quarantines poor performers (WR < 40% after 10 trades) to protect capital.
    5. Evolves the system state dynamically.
    """

    def __init__(
        self,
        reputation_engine: Optional[AgentReputationEngine] = None,
        min_samples_for_quarantine: int = 10,
        quarantine_threshold: float = 0.40,
        name: str = "ChiefStrategyAgent",
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.reputation_engine = reputation_engine or AgentReputationEngine()
        self.min_samples_for_quarantine = min_samples_for_quarantine
        self.quarantine_threshold = quarantine_threshold  # 40% win rate
        self.quarantined_agents: Dict[str, str] = {}  # agent_name -> reason
        self.promoted_agents: Dict[str, float] = {}  # agent_name -> weight

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        """
        Evaluate and audit all registered agents.
        Returns meta-signals detailing agent rank promotions and quarantine actions.
        """
        if not self.enabled:
            return []

        agents_map = context.get("registered_agents", {})
        audit_signals: List[AgentSignal] = []

        for name, agent in agents_map.items():
            if name == self.name:
                continue

            q = self.reputation_engine.history.get(name, [])
            total_samples = len(q)
            acc = self.reputation_engine.get_accuracy(name)
            weight = self.reputation_engine.calculate_weight(name)

            # 1. Check for Auto-Quarantine (Underperformer Disable)
            if total_samples >= self.min_samples_for_quarantine and acc < self.quarantine_threshold:
                if agent.enabled:
                    agent.enabled = False
                    reason = f"Win rate {acc*100:.1f}% < {self.quarantine_threshold*100:.0f}% threshold ({total_samples} trades)"
                    self.quarantined_agents[name] = reason
                    logger.warning("ChiefStrategyAgent QUARANTINED agent %s: %s", name, reason)

                    audit_signals.append(
                        AgentSignal(
                            agent_name=self.name,
                            symbol="ALL",
                            contract_type="META_QUARANTINE",
                            confidence=1.0,
                            rationale=f"Quarantined underperforming agent {name}: {reason}",
                            metadata={"target_agent": name, "action": "disable", "accuracy": acc},
                        )
                    )

            # 2. Check for Auto-Promotion (Strong Performer Weight Increase)
            elif acc >= 0.70 and total_samples >= 5:
                if not agent.enabled:
                    agent.enabled = True
                    self.quarantined_agents.pop(name, None)

                self.promoted_agents[name] = weight
                logger.info("ChiefStrategyAgent PROMOTED agent %s: accuracy=%.1f%% -> weight=%.2fx", name, acc * 100, weight)

                audit_signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol="ALL",
                        contract_type="META_PROMOTION",
                        confidence=1.0,
                        rationale=f"Promoted top-performing agent {name}: accuracy={acc*100:.1f}%, weight={weight:.2f}x",
                        metadata={"target_agent": name, "action": "promote", "weight": weight, "accuracy": acc},
                    )
                )

        return audit_signals

    def get_agent_rankings(self) -> List[Dict[str, Any]]:
        """Retrieve full agent leaderboard ranked by accuracy and profit."""
        core_agents = [
            "TrendAgent",
            "VolatilityAgent",
            "PatternAgent",
            "ScalpingAgent",
            "LearningAgent",
            "ConsensusAgent",
            "RiskAgent",
            "PortfolioManagerAgent",
            "MarketRegimeAgent",
            "RLAgent",
            "ExecutionAgent",
        ]
        history_keys = list(self.reputation_engine.history.keys())
        all_names = list(dict.fromkeys(core_agents + history_keys))

        rankings = []
        for name in all_names:
            q = self.reputation_engine.history.get(name, [])
            wins = sum(1 for x in q if x)
            total = len(q)
            acc = (wins / total * 100) if total > 0 else 50.0
            pnl = self.reputation_engine.pnl_history.get(name, 0.0)
            status = "QUARANTINED" if name in self.quarantined_agents else ("PROMOTED" if acc >= 70 and total >= 5 else "ACTIVE")

            rankings.append(
                {
                    "agent_name": name,
                    "accuracy": round(acc, 1),
                    "trades": total,
                    "wins": wins,
                    "losses": total - wins,
                    "pnl": round(pnl, 2),
                    "weight": self.reputation_engine.calculate_weight(name),
                    "status": status,
                    "reason": self.quarantined_agents.get(name, ""),
                }
            )

        # Sort leaderboard by accuracy DESC then PnL DESC
        rankings.sort(key=lambda x: (x["accuracy"], x["pnl"]), reverse=True)
        return rankings
