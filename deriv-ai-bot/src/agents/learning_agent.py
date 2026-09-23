import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.agents.base_agent import BaseAgent, AgentSignal
from src.strategy.adaptive_learner import AdaptiveLearner
from config.settings import LEARNING_PATH, LEARNING_ALWAYS

logger = logging.getLogger(__name__)


class LearningAgent(BaseAgent):
    """
    Learning & Adaptation Agent (The Brain):
    - Evaluates historical agent performance.
    - Maintains dynamic agent weights based on win rates and profit contribution.
    - Adjusts confidence scores and historical setup support.
    """

    def __init__(
        self,
        name: str = "LearningAgent",
        enabled: bool = True,
        weights_path: Optional[Path] = None,
    ):
        super().__init__(name=name, enabled=enabled)
        self.learner = AdaptiveLearner(
            path=Path(LEARNING_PATH),
            always_on=LEARNING_ALWAYS,
            min_samples=2,
            cold_streak_skip=2,
        )
        self.weights_path = weights_path or Path("data/agent_weights.json")
        self.agent_stats: Dict[str, Dict[str, Any]] = self._load_weights()

    def _load_weights(self) -> Dict[str, Dict[str, Any]]:
        """Load stored agent performance weights or initialize defaults."""
        defaults = {
            "MarketScannerAgent": {"weight": 1.0, "wins": 0, "losses": 0, "win_rate": 0.50},
            "TrendAgent": {"weight": 1.0, "wins": 0, "losses": 0, "win_rate": 0.50},
            "VolatilityAgent": {"weight": 1.0, "wins": 0, "losses": 0, "win_rate": 0.50},
            "PatternAgent": {"weight": 1.0, "wins": 0, "losses": 0, "win_rate": 0.50},
            "DigitStatAgent": {"weight": 1.0, "wins": 0, "losses": 0, "win_rate": 0.50},
        }
        if self.weights_path.exists():
            try:
                with open(self.weights_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    defaults.update(saved)
            except Exception as e:
                logger.warning("LearningAgent: Failed to load weights (%s), using defaults", e)
        return defaults

    def save_weights(self) -> None:
        """Persist updated agent weights to disk."""
        try:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.weights_path, "w", encoding="utf-8") as f:
                json.dump(self.agent_stats, f, indent=2)
        except Exception as e:
            logger.error("LearningAgent: Failed to save weights: %s", e)

    def get_agent_weight(self, agent_name: str) -> float:
        """Get current dynamic voting weight for an agent (clamped between 0.5 and 2.0)."""
        stats = self.agent_stats.get(agent_name, {})
        w = float(stats.get("weight", 1.0))
        return max(0.5, min(2.0, w))

    def record_trade_outcome(
        self,
        symbol: str,
        contract_type: str,
        won: bool,
        participating_agents: List[str],
        profit: float = 0.0,
    ) -> None:
        """
        Update per-agent weights based on trade settlement outcome.
        Agents that voted for a winning trade get a weight boost; losing trades get a penalty.
        """
        self.learner.record(symbol, contract_type, won, profit=profit)

        for agent_name in participating_agents:
            if agent_name not in self.agent_stats:
                self.agent_stats[agent_name] = {"weight": 1.0, "wins": 0, "losses": 0, "win_rate": 0.50}

            st = self.agent_stats[agent_name]
            if won:
                st["wins"] = st.get("wins", 0) + 1
            else:
                st["losses"] = st.get("losses", 0) + 1

            total = st["wins"] + st["losses"]
            win_rate = st["wins"] / max(1, total)
            st["win_rate"] = round(win_rate, 4)

            # Weight update formula based on win rate & minimum sample size
            if total >= 5:
                # Target win rate is 55%; adjust weight proportionally
                st["weight"] = max(0.5, min(2.0, round(0.5 + win_rate, 4)))

        self.save_weights()
        logger.info(
            "LearningAgent recorded trade outcome symbol=%s won=%s agents=%s weights=%s",
            symbol,
            won,
            participating_agents,
            {a: self.get_agent_weight(a) for a in participating_agents},
        )

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        """
        Evaluates signals from other agents and applies adaptive confidence adjustments.
        """
        if not self.enabled:
            return []

        raw_signals: List[AgentSignal] = context.get("incoming_signals", [])
        symbol = context.get("symbol")

        adjusted_signals: List[AgentSignal] = []

        for sig in raw_signals:
            if sig.contract_type in ("SCAN_OK", "VOLATILITY_RATING"):
                continue

            # Apply adaptive learner confidence adjustment
            adj_conf = self.learner.adjust_confidence(
                sig.symbol, sig.contract_type, sig.confidence
            )

            # Get historical selection bonus
            learn_bonus = self.learner.selection_bonus(sig.symbol, sig.contract_type)

            # Get agent's dynamic weight
            agent_weight = self.get_agent_weight(sig.agent_name)

            # Create enhanced signal
            enhanced = AgentSignal(
                agent_name=sig.agent_name,
                symbol=sig.symbol,
                contract_type=sig.contract_type,
                confidence=adj_conf,
                raw_confidence=sig.raw_confidence or sig.confidence,
                weight=agent_weight,
                barrier=sig.barrier,
                duration=sig.duration,
                duration_unit=sig.duration_unit,
                horizon=sig.horizon,
                family=sig.family,
                rationale=f"{sig.rationale} [Learned Adj: {adj_conf:.2f}, Weight: {agent_weight:.2f}]",
                metadata={
                    **sig.metadata,
                    "learn_bonus": learn_bonus,
                    "confidence_level": self.learner.confidence_level(sig.symbol, sig.contract_type),
                    "historical_support": self.learner.historical_support(sig.symbol, sig.contract_type),
                }
            )
            adjusted_signals.append(enhanced)

        return adjusted_signals
