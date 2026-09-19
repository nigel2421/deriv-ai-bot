import logging
from typing import Any, Dict, List
from src.agents.base_agent import BaseAgent, AgentSignal
from src.strategy.regime_filter import should_skip_digits, should_skip_rise_fall
from src.strategy.session_hours import is_boom_symbol, is_crash_symbol, is_fx_symbol

logger = logging.getLogger(__name__)


class VolatilityAgent(BaseAgent):
    """
    Volatility & Regime Agent: Assesses market stability, chop score, entropy,
    and spike conditions for Boom/Crash/Jump assets.
    """

    def __init__(self, name: str = "VolatilityAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        if not self.enabled:
            return []

        symbol = context.get("symbol")
        ticks = context.get("ticks")

        if not symbol or not ticks or len(ticks) < 15:
            return []

        signals: List[AgentSignal] = []

        skip_d, d_reason, d_reg = should_skip_digits(ticks)
        skip_rf, rf_reason, rf_reg = should_skip_rise_fall(ticks)

        # FX chop adjustment
        if is_fx_symbol(symbol) and skip_rf:
            chop = float(rf_reg.get("chop_score") or 1.0)
            if chop < 0.72:
                skip_rf = False

        # Regime safety evaluation score (1.0 = ideal smooth market, 0.0 = severe chop)
        chop_score = float(rf_reg.get("chop_score") or 0.5)
        efficiency = float(rf_reg.get("efficiency") or 0.5)
        safety_score = max(0.0, min(1.0, (1.0 - chop_score) * 0.7 + efficiency * 0.3))

        # Boom/Crash Spike Detection
        if is_boom_symbol(symbol):
            # Boom index: look for upward spike setups
            signals.append(
                AgentSignal(
                    agent_name=self.name,
                    symbol=symbol,
                    contract_type="CALL",
                    confidence=safety_score * 0.85,
                    raw_confidence=safety_score,
                    family="rise_fall",
                    duration=10,
                    duration_unit="t",
                    rationale=f"Boom index spike setup (safety={safety_score:.2f}, chop={chop_score:.2f})",
                    metadata={"chop_score": chop_score, "efficiency": efficiency, "is_spike": True}
                )
            )
        elif is_crash_symbol(symbol):
            # Crash index: look for downward spike setups
            signals.append(
                AgentSignal(
                    agent_name=self.name,
                    symbol=symbol,
                    contract_type="PUT",
                    confidence=safety_score * 0.85,
                    raw_confidence=safety_score,
                    family="rise_fall",
                    duration=10,
                    duration_unit="t",
                    rationale=f"Crash index spike setup (safety={safety_score:.2f}, chop={chop_score:.2f})",
                    metadata={"chop_score": chop_score, "efficiency": efficiency, "is_spike": True}
                )
            )

        # Emit regime metadata signal for consensus weighting
        signals.append(
            AgentSignal(
                agent_name=self.name,
                symbol=symbol,
                contract_type="VOLATILITY_RATING",
                confidence=safety_score,
                raw_confidence=safety_score,
                weight=1.0,
                rationale=f"Regime safety score={safety_score:.2f} (digits_skip={skip_d}, rf_skip={skip_rf})",
                metadata={
                    "skip_digits": skip_d,
                    "digits_reason": d_reason,
                    "skip_rise_fall": skip_rf,
                    "rf_reason": rf_reason,
                    "chop_score": chop_score,
                    "efficiency": efficiency,
                }
            )
        )

        return signals
