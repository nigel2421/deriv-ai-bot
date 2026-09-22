"""
Step & Skew Step Specialist Agent — Rec #5 & #6

Specialized intelligence agent for symmetric Step Index assets (STPIDX, STEP10)
and asymmetric Skew Step Index assets (SKEWSTEP).

Models:
  - Step direction persistence & short-run imbalance (streaks)
  - Asymmetric probability distributions (80/20 & 90/10 small vs sharp moves)
  - Last-digit parity distribution under low-entropy step regimes
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List
from src.agents.base_agent import BaseAgent, AgentSignal

logger = logging.getLogger(__name__)


def is_step_symbol(symbol: str) -> bool:
    s = str(symbol or "").upper()
    return "STEP" in s or "STPIDX" in s or "SKEW" in s


def is_skew_step_symbol(symbol: str) -> bool:
    s = str(symbol or "").upper()
    return "SKEW" in s or "SKEWSTEP" in s


class StepSpecialistAgent(BaseAgent):
    """Specialist agent for Step Index and Skew Step Index assets."""

    def __init__(self, name: str = "StepSpecialistAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        if not self.enabled:
            return []

        symbol = context.get("symbol", "")
        ticks = context.get("ticks")

        if not symbol or not is_step_symbol(symbol) or not ticks or len(ticks) < 15:
            return []

        quotes = [t.get("quote", 0.0) if isinstance(t, dict) else float(t) for t in ticks[-30:]]
        diffs = [quotes[i] - quotes[i-1] for i in range(1, len(quotes))]
        if not diffs:
            return []

        signals: List[AgentSignal] = []
        is_skew = is_skew_step_symbol(symbol)

        # 1. Directional Imbalance & Streak Persistence
        up_steps = sum(1 for d in diffs if d > 0)
        down_steps = sum(1 for d in diffs if d < 0)
        total_steps = len(diffs)

        up_ratio = up_steps / float(total_steps)

        # Skew Step asymmetric modeling (80/20 or 90/10 distribution)
        if is_skew:
            # Skew Step: sharp moves occur with low probability (~10-20%), small moves ~80-90%
            abs_diffs = [abs(d) for d in diffs]
            avg_step = sum(abs_diffs) / float(len(abs_diffs))
            large_steps = [d for d in diffs if abs(d) > avg_step * 1.5]

            skew_dir = "CALL" if up_ratio > 0.55 else "PUT"
            skew_conf = min(0.88, 0.60 + abs(up_ratio - 0.50) * 1.2)

            signals.append(
                AgentSignal(
                    agent_name=self.name,
                    symbol=symbol,
                    contract_type=skew_dir,
                    confidence=skew_conf,
                    raw_confidence=skew_conf,
                    weight=1.2,
                    duration=5,
                    duration_unit="t",
                    family="rise_fall",
                    horizon="tick",
                    rationale=f"Skew Step asymmetric imbalance: {skew_dir} (up_ratio={up_ratio:.2f}, large_steps={len(large_steps)})",
                    metadata={
                        "is_skew": True,
                        "up_ratio": up_ratio,
                        "avg_step": avg_step,
                        "large_steps_count": len(large_steps),
                    }
                )
            )
        else:
            # Classic Symmetric Step Index: short-run mean-reversion & parity structure
            if up_ratio >= 0.65:
                # Strong upward streak -> Continuation CALL
                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type="CALL",
                        confidence=0.72,
                        raw_confidence=0.72,
                        weight=1.1,
                        duration=5,
                        duration_unit="t",
                        family="rise_fall",
                        horizon="tick",
                        rationale=f"Step Index upward momentum (up_ratio={up_ratio:.2f})",
                        metadata={"up_ratio": up_ratio}
                    )
                )
            elif up_ratio <= 0.35:
                # Strong downward streak -> Continuation PUT
                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type="PUT",
                        confidence=0.72,
                        raw_confidence=0.72,
                        weight=1.1,
                        duration=5,
                        duration_unit="t",
                        family="rise_fall",
                        horizon="tick",
                        rationale=f"Step Index downward momentum (up_ratio={up_ratio:.2f})",
                        metadata={"up_ratio": up_ratio}
                    )
                )

            # Digit parity model on fixed step movements (digits alter predictably)
            last_digits = [int(str(q).split(".")[-1][-1]) if "." in str(q) else int(str(q)[-1]) for q in quotes[-10:]]
            even_count = sum(1 for d in last_digits if d % 2 == 0)
            even_ratio = even_count / float(len(last_digits))

            if even_ratio >= 0.70:
                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type="DIGITEVEN",
                        confidence=0.68,
                        raw_confidence=0.68,
                        weight=1.0,
                        duration=5,
                        duration_unit="t",
                        family="digits",
                        horizon="tick",
                        rationale=f"Step Index parity cluster: DIGITEVEN (even_ratio={even_ratio:.2f})",
                        metadata={"even_ratio": even_ratio}
                    )
                )
            elif even_ratio <= 0.30:
                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type="DIGITODD",
                        confidence=0.68,
                        raw_confidence=0.68,
                        weight=1.0,
                        duration=5,
                        duration_unit="t",
                        family="digits",
                        horizon="tick",
                        rationale=f"Step Index parity cluster: DIGITODD (even_ratio={even_ratio:.2f})",
                        metadata={"even_ratio": even_ratio}
                    )
                )

        return signals
