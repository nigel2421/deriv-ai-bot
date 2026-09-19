import logging
from typing import Any, Dict, List, Optional
from src.agents.base_agent import BaseAgent, AgentSignal
from src.strategy.regime_filter import should_skip_digits, should_skip_rise_fall

logger = logging.getLogger(__name__)


class MarketRegimeAgent(BaseAgent):
    """
    Stage 6: Market Regime Agent
    
    Detects market regimes:
    1. TRENDING (High efficiency, low chop score)
    2. SIDEWAYS_CHOP (High chop score > 0.65, whipsaws)
    3. HIGH_VOLATILITY (Spike synthetics, high ATR)
    4. SLOW (Minimal tick movement)
    
    Dynamic Strategy Gating:
    - In SIDEWAYS_CHOP: Disables TrendAgent for that symbol, enables PatternAgent / Parity.
    - In TRENDING: Enables TrendAgent, disables mean-reversion.
    """

    def __init__(self, name: str = "MarketRegimeAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    def detect_regime(self, symbol: str, ticks: List[float]) -> Dict[str, Any]:
        """
        Classifies tick window into market regime.
        """
        if not ticks or len(ticks) < 20:
            return {"symbol": symbol, "regime": "UNKNOWN", "chop_score": 0.50, "efficiency": 0.50}

        skip_d, d_reason, d_reg = should_skip_digits(ticks)
        skip_rf, rf_reason, rf_reg = should_skip_rise_fall(ticks)

        chop_score = float(rf_reg.get("chop_score", 0.50))
        efficiency = float(rf_reg.get("efficiency", 0.50))

        if chop_score > 0.65 or (skip_d and skip_rf):
            regime = "SIDEWAYS_CHOP"
        elif efficiency > 0.60 and chop_score < 0.40:
            regime = "TRENDING"
        elif "BOOM" in symbol or "CRASH" in symbol:
            regime = "HIGH_VOLATILITY"
        elif max(ticks) - min(ticks) < 0.05:
            regime = "SLOW"
        else:
            regime = "NORMAL"

        gating = self.get_strategy_gating(regime)

        logger.debug("MarketRegimeAgent %s regime=%s (chop=%.2f, eff=%.2f)", symbol, regime, chop_score, efficiency)
        return {
            "symbol": symbol,
            "regime": regime,
            "chop_score": round(chop_score, 4),
            "efficiency": round(efficiency, 4),
            "gating": gating,
        }

    def get_strategy_gating(self, regime: str) -> Dict[str, bool]:
        """
        Returns agent enablement flags based on detected regime.
        """
        if regime == "SIDEWAYS_CHOP":
            return {
                "TrendAgent": False,      # Disable Trend Agent during choppy whipsaws
                "PatternAgent": True,     # Enable Pattern Agent
                "VolatilityAgent": True,
                "ScalpingAgent": True,
            }
        elif regime == "TRENDING":
            return {
                "TrendAgent": True,       # Enable Trend Agent
                "PatternAgent": True,
                "VolatilityAgent": False,
                "ScalpingAgent": False,
            }
        elif regime == "HIGH_VOLATILITY":
            return {
                "TrendAgent": True,
                "PatternAgent": False,
                "VolatilityAgent": True,  # Enable Volatility Agent
                "ScalpingAgent": True,
            }
        else:
            return {
                "TrendAgent": True,
                "PatternAgent": True,
                "VolatilityAgent": True,
                "ScalpingAgent": True,
            }

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        """
        Evaluates context ticks and emits regime meta-signals.
        """
        if not self.enabled:
            return []

        symbol = context.get("symbol", "R_100")
        ticks = context.get("ticks", [])
        regime_data = self.detect_regime(symbol, ticks)

        return [
            AgentSignal(
                agent_name=self.name,
                symbol=symbol,
                contract_type="REGIME_INFO",
                confidence=1.0,
                rationale=f"Market regime for {symbol}: {regime_data['regime']} (chop={regime_data['chop_score']:.2f})",
                metadata=regime_data,
            )
        ]
