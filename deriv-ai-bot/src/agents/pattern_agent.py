import logging
from typing import Any, Dict, List
from src.agents.base_agent import BaseAgent, AgentSignal
from src.ai.predictor import Predictor
from src.strategy.signal_generator import SignalGenerator
from src.strategy.digit_queue import queue_signal
from src.strategy.contract_types import is_digit_contract

logger = logging.getLogger(__name__)


class PatternAgent(BaseAgent):
    """
    Pattern Recognition Agent: Analyzes last-digit ML predictions (XGBoost/LSTM),
    parity streaks (even/odd), digit run queues (over/under), and historical setup matching.
    """

    def __init__(self, name: str = "PatternAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        self.predictor = Predictor()
        self.signal_gen = SignalGenerator(prefer_parity=True)

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        if not self.enabled:
            return []

        symbol = context.get("symbol")
        ticks = context.get("ticks")
        allowed_types = context.get("allowed_types", [])

        if not symbol or not ticks or len(ticks) < 10:
            return []

        digit_allowed = [t for t in allowed_types if is_digit_contract(t)]
        if not digit_allowed and "ALL" not in allowed_types:
            # Default digit contract types if empty & not restricted
            digit_allowed = ["DIGITOVER", "DIGITUNDER", "DIGITEVEN", "DIGITODD"]

        signals: List[AgentSignal] = []

        # 1. ML Predictor (XGBoost / Pattern model)
        pred = self.predictor.predict(ticks)
        pred = {**pred, "recent_ticks": ticks}
        raw_conf = float(pred.get("confidence", 0.5))

        # 2. Parity analysis
        ct_stats, parity_conf = self.signal_gen.parity_confidence(ticks)
        if ("DIGITEVEN" in digit_allowed or "DIGITODD" in digit_allowed) and parity_conf >= 0.60:
            pref_type = "DIGITEVEN" if ct_stats.get("even") else "DIGITODD"
            pred["preferred_type"] = pref_type
            pred["confidence"] = max(raw_conf, parity_conf)
            raw_conf = float(pred["confidence"])

        # 3. Community digit queue (runs -> Over/Under)
        q = queue_signal(ticks)
        if q and (q.get("preferred_type") in digit_allowed or not digit_allowed):
            pred["preferred_type"] = q["preferred_type"]
            pred["confidence"] = min(0.95, max(raw_conf, 0.72) + float(q.get("confidence_boost") or 0))
            if q.get("hint_barrier") is not None:
                pred["barrier"] = q["hint_barrier"]
            raw_conf = float(pred["confidence"])

        # 4. Generate candidate digit signal
        signal_type, signal_barrier, conf = self.signal_gen.generate_signal(
            pred,
            raw_conf,
            min_confidence=0.55,
            allowed_types=digit_allowed,
        )

        if signal_type:
            signals.append(
                AgentSignal(
                    agent_name=self.name,
                    symbol=symbol,
                    contract_type=signal_type,
                    confidence=conf or raw_conf,
                    raw_confidence=conf or raw_conf,
                    barrier=signal_barrier,
                    weight=1.0,
                    family="digits",
                    duration=5,
                    duration_unit="t",
                    rationale=f"Pattern match: {signal_type} (conf={conf or raw_conf:.2f}, barrier={signal_barrier})",
                    metadata={
                        "predicted_digit": pred.get("digit"),
                        "parity_conf": parity_conf,
                        "queue_reason": q.get("reason") if q else None,
                    }
                )
            )

        return signals
