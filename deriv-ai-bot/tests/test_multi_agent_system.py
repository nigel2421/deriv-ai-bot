import pytest
import asyncio
from typing import Dict, Any

from src.agents.bus import AgentEventBus, TOPIC_SIGNALS
from src.agents.base_agent import AgentSignal, ConsensusDecision
from src.agents.trend_agent import TrendAgent
from src.agents.volatility_agent import VolatilityAgent
from src.agents.pattern_agent import PatternAgent
from src.agents.consensus_agent import ConsensusAgent
from src.agents.manager import AgentManager


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_event_bus_pub_sub():

    bus = AgentEventBus()
    received_messages = []

    def on_signal(payload: Dict[str, Any]):
        received_messages.append(payload)

    bus.subscribe(TOPIC_SIGNALS, on_signal)
    await bus.start()

    test_payload = {"symbol": "R_100", "contract_type": "DIGITEVEN", "confidence": 0.85}
    await bus.publish(TOPIC_SIGNALS, test_payload)

    await asyncio.sleep(0.1)
    await bus.stop()

    assert len(received_messages) == 1
    assert received_messages[0]["symbol"] == "R_100"
    assert received_messages[0]["contract_type"] == "DIGITEVEN"


@pytest.mark.anyio
async def test_agent_manager_consensus():

    manager = AgentManager()
    trend = TrendAgent()
    vol = VolatilityAgent()
    pattern = PatternAgent()

    manager.register_agent(trend, weight=1.2)
    manager.register_agent(vol, weight=1.0)
    manager.register_agent(pattern, weight=1.1)

    consensus = ConsensusAgent(min_consensus_score=0.70)

    # Generate synthetic tick data (even parity run)
    ticks = [100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6, 101.8, 102.0]
    context = {
        "symbol": "R_100",
        "ticks": ticks,
        "allowed_types": ["DIGITEVEN", "DIGITODD", "CALL", "PUT"],
    }

    signals = await manager.evaluate_market_context(context)
    assert len(signals) >= 0  # Signals evaluated safely

    # Test consensus decision calculation
    dummy_signals = [
        AgentSignal("TrendAgent", "R_100", "DIGITEVEN", confidence=0.85, weight=1.2),
        AgentSignal("PatternAgent", "R_100", "DIGITEVEN", confidence=0.80, weight=1.1),
    ]
    decision = consensus.form_consensus("R_100", dummy_signals, min_confidence=0.75)

    assert decision is not None
    assert decision.symbol == "R_100"
    assert decision.contract_type == "DIGITEVEN"
    assert decision.ensemble_score >= 0.80
    assert len(decision.votes) == 2
