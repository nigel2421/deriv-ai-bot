import logging
from typing import Any, Dict, List
from src.agents.base_agent import BaseAgent, AgentSignal
from src.strategy.trend_analyzer import analyze_trend
from src.strategy.minute_engine import analyze_minute
from src.strategy.session_hours import is_fx_symbol, is_spike_synthetic, preferred_minute_duration

logger = logging.getLogger(__name__)


class TrendAgent(BaseAgent):
    """
    Trend Agent: Analyzes EMA, MACD, RSI, Hurst velocity, and multi-minute trend structure.
    Generates CALL or PUT signals with trend confidence.
    """

    def __init__(self, name: str = "TrendAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        if not self.enabled:
            return []

        symbol = context.get("symbol")
        ticks = context.get("ticks")
        allowed_types = context.get("allowed_types", ["CALL", "PUT"])

        if not symbol or not ticks or len(ticks) < 20:
            return []

        signals: List[AgentSignal] = []

        # 1. Tick-level Trend Analysis (EMA/RSI/MACD)
        rf_allowed = [t for t in allowed_types if t in ("CALL", "PUT")]
        if rf_allowed:
            trend_res = analyze_trend(ticks)
            contract_type = trend_res.get("contract_type")
            conf = float(trend_res.get("confidence") or 0.0)
            strength = float(trend_res.get("strength") or 0.0)

            if contract_type and contract_type in rf_allowed and conf >= 0.55:
                dur = 5
                dur_unit = "t"
                if is_fx_symbol(symbol):
                    dur = preferred_minute_duration(symbol, 30)
                    dur_unit = "m"

                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type=contract_type,
                        confidence=conf,
                        raw_confidence=conf,
                        weight=1.0,
                        duration=dur,
                        duration_unit=dur_unit,
                        family="rise_fall",
                        horizon="minute" if dur_unit == "m" else "tick",
                        rationale=f"Trend {contract_type} (conf={conf:.2f}, strength={strength:.2f})",
                        metadata={
                            "strength": strength,
                            "ema_fast": trend_res.get("ema_fast"),
                            "ema_slow": trend_res.get("ema_slow"),
                            "rsi": trend_res.get("rsi"),
                        }
                    )
                )

        # 2. Minute Candle Analysis (if applicable)
        if not is_spike_synthetic(symbol) and rf_allowed:
            m_dur = preferred_minute_duration(symbol, 2)
            msig = analyze_minute(ticks, period_sec=60, duration_minutes=m_dur, min_confidence=0.65)
            if msig and msig.get("contract_type") in rf_allowed:
                m_type = msig["contract_type"]
                m_conf = float(msig.get("confidence") or 0.0)
                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type=m_type,
                        confidence=m_conf,
                        raw_confidence=m_conf,
                        weight=1.0,
                        duration=msig.get("duration", m_dur),
                        duration_unit="m",
                        family="minute_rise_fall",
                        horizon="minute",
                        rationale=f"Minute Candle Trend {m_type} (conf={m_conf:.2f})",
                        metadata={
                            "minute_details": msig.get("details"),
                        }
                    )
                )

        return signals
