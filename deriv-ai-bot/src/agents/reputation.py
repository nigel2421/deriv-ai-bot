import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentReputationEngine:
    """
    Stage 1: Agent Intelligence & Reputation Scoring Engine.
    Tracks rolling performance (last 100 trades) for each agent and calculates dynamic influence weights.
    """

    def __init__(self, window_size: int = 100, storage_path: Optional[Path] = None):
        self.window_size = window_size
        self.storage_path = storage_path or Path("data/agent_reputation.json")
        # agent_name -> deque of bool outcomes (True = Win, False = Loss)
        self.history: Dict[str, deque] = {}
        # agent_name -> total PnL generated
        self.pnl_history: Dict[str, float] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load stored reputation history from disk if exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for name, records in data.get("history", {}).items():
                        self.history[name] = deque(records, maxlen=self.window_size)
                    self.pnl_history = data.get("pnl_history", {})
                logger.info("Loaded agent reputation data for %d agents", len(self.history))
            except Exception as e:
                logger.warning("Failed to load agent reputation state (%s), starting fresh", e)

    def save_state(self) -> None:
        """Persist current reputation state to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            export_history = {name: list(q) for name, q in self.history.items()}
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "history": export_history,
                        "pnl_history": self.pnl_history,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error("Failed to save agent reputation state: %s", e)

    def record_outcome(
        self,
        agent_name: str,
        won: bool,
        profit: float = 0.0,
    ) -> float:
        """
        Record trade outcome for an agent and return updated dynamic weight.
        """
        if agent_name not in self.history:
            self.history[agent_name] = deque(maxlen=self.window_size)
            self.pnl_history[agent_name] = 0.0

        self.history[agent_name].append(won)
        self.pnl_history[agent_name] += profit

        self.save_state()
        new_weight = self.calculate_weight(agent_name)
        logger.info(
            "Recorded reputation for %s: won=%s profit=%.2f -> new weight=%.2fx (accuracy=%.1f%%)",
            agent_name,
            won,
            profit,
            new_weight,
            self.get_accuracy(agent_name) * 100,
        )
        return new_weight

    def get_accuracy(self, agent_name: str) -> float:
        """Calculate win rate accuracy over rolling window (0.0 to 1.0)."""
        q = self.history.get(agent_name)
        if not q:
            return 0.50
        return sum(1 for outcome in q if outcome) / len(q)

    def calculate_weight(self, agent_name: str) -> float:
        """
        Calculate dynamic influence weight based on accuracy over last 100 trades.
        - Accuracy >= 75%: 1.8x
        - 70% <= Accuracy < 75%: 1.5x
        - 55% <= Accuracy < 70%: 1.2x
        - 45% <= Accuracy < 55%: 0.9x
        - Accuracy < 45%: 0.5x
        """
        q = self.history.get(agent_name)
        if not q or len(q) < 5:
            return 1.00  # Default baseline before 5 samples

        acc = self.get_accuracy(agent_name)

        if acc >= 0.75:
            return 1.80
        elif acc >= 0.70:
            return 1.50
        elif acc >= 0.55:
            return 1.20
        elif acc >= 0.45:
            return 0.90
        else:
            return 0.50

    def get_reputation_summary() -> Dict[str, Dict[str, Any]]:
        """Retrieve full summary of agent reputation metrics for API & dashboard."""
        summary = {}
        for name, q in self.history.items():
            wins = sum(1 for x in q if x)
            total = len(q)
            acc = (wins / total) * 100 if total > 0 else 50.0
            summary[name] = {
                "trades_analyzed": total,
                "wins": wins,
                "losses": total - wins,
                "accuracy": round(acc, 1),
                "weight": self.calculate_weight(name),
                "pnl": round(self.pnl_history.get(name, 0.0), 2),
            }
        return summary
