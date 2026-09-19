import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent, AgentSignal

logger = logging.getLogger(__name__)


class RLAgent(BaseAgent):
    """
    Stage 4: Reinforcement Learning Agent (The RL Engine)
    
    Implements Q-Learning feedback loop: State -> Action -> Reward
    - State: (symbol, momentum_bin, volatility_bin, consensus_bin)
    - Action: 0 = HOLD / SKIP, 1 = APPROVE_TRADE
    - Reward: +1.0 * (Profit / Stake) on Win, -1.2 * (Loss / Stake) on Loss.
    """

    def __init__(
        self,
        alpha: float = 0.1,  # Learning rate
        gamma: float = 0.9,  # Discount factor
        epsilon: float = 0.1,  # Exploration rate
        storage_path: Optional[Path] = None,
        name: str = "RLAgent",
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.storage_path = storage_path or Path("data/rl_qtable.json")
        # q_table: state_key -> [Q(state, 0), Q(state, 1)]
        self.q_table: Dict[str, List[float]] = {}
        self._load_qtable()

    def _load_qtable(self) -> None:
        """Load stored Q-table from disk if exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.q_table = json.load(f)
                logger.info("Loaded Q-table with %d states", len(self.q_table))
            except Exception as e:
                logger.warning("Failed to load Q-table (%s), starting fresh", e)

    def save_qtable(self) -> None:
        """Persist Q-table to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.q_table, f, indent=2)
        except Exception as e:
            logger.error("Failed to save Q-table: %s", e)

    def _encode_state(self, symbol: str, consensus_score: float, trend_strength: float = 0.0) -> str:
        """Discretize continuous context into state key string."""
        c_bin = "HIGH_CONF" if consensus_score >= 0.85 else ("MED_CONF" if consensus_score >= 0.75 else "LOW_CONF")
        t_bin = "STRONG_TREND" if abs(trend_strength) > 0.5 else "WEAK_TREND"
        return f"{symbol}:{c_bin}:{t_bin}"

    def get_q_values(self, state: str) -> List[float]:
        """Get Q-values [Q(HOLD), Q(TRADE)] for a given state."""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        return self.q_table[state]

    def update_q_value(
        self,
        symbol: str,
        consensus_score: float,
        action: int,  # 0 or 1
        won: bool,
        stake: float = 1.0,
        profit: float = 0.0,
    ) -> float:
        """
        State -> Action -> Reward update step.
        Reward formula: +1.0 * (profit / stake) on win, -1.2 on loss.
        """
        state = self._encode_state(symbol, consensus_score)
        q_vals = self.get_q_values(state)

        # Calculate Reward
        if won:
            reward = 1.0 * (profit / max(0.1, stake))
        else:
            reward = -1.2 * (abs(profit) / max(0.1, stake) if profit < 0 else 1.0)

        # Q-Learning update: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s')) - Q(s,a))
        old_q = q_vals[action]
        new_q = old_q + self.alpha * (reward - old_q)
        q_vals[action] = round(new_q, 4)

        self.save_qtable()
        logger.info("RLAgent update state=%s action=%d reward=%.2f -> Q=%.4f", state, action, reward, new_q)
        return new_q

    def evaluate_rl_boost(self, symbol: str, consensus_score: float) -> float:
        """
        Returns RL confidence boost or penalty based on learned Q-values.
        """
        if not self.enabled:
            return 0.0

        state = self._encode_state(symbol, consensus_score)
        q_vals = self.get_q_values(state)
        q_trade = q_vals[1]

        # Convert Q-value to confidence adjustment (-0.15 to +0.15)
        boost = max(-0.15, min(0.15, q_trade * 0.10))
        return round(boost, 4)

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        """Not used directly; call evaluate_rl_boost() or update_q_value()."""
        return []
