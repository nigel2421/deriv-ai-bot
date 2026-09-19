import pytest
import asyncio
from typing import Dict, Any

from src.agents.reputation import AgentReputationEngine
from src.agents.chief_strategy_agent import ChiefStrategyAgent
from src.agents.rl_agent import RLAgent
from src.agents.trend_agent import TrendAgent


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_reputation_engine(tmp_path):
    storage_path = tmp_path / "test_reputation.json"
    engine = AgentReputationEngine(window_size=100, storage_path=storage_path)
    agent_name = "TestAgent"

    # Before min samples, weight is baseline 1.0
    assert engine.calculate_weight(agent_name) == 1.0

    # Record 10 winning trades
    for _ in range(10):
        engine.record_outcome(agent_name, won=True, profit=1.0)

    assert engine.get_accuracy(agent_name) == 1.0
    assert engine.calculate_weight(agent_name) == 1.80  # High accuracy boost


@pytest.mark.anyio
async def test_chief_strategy_agent(tmp_path):
    storage_path = tmp_path / "test_reputation.json"
    engine = AgentReputationEngine(window_size=100, storage_path=storage_path)
    chief = ChiefStrategyAgent(reputation_engine=engine, min_samples_for_quarantine=5)

    trend = TrendAgent()
    agents = {"TrendAgent": trend}

    # Simulate 5 losses for TrendAgent -> WR 0% < 40%
    for _ in range(5):
        engine.record_outcome("TrendAgent", won=False, profit=-1.0)

    signals = await chief.evaluate({"registered_agents": agents})

    assert len(signals) == 1
    assert signals[0].contract_type == "META_QUARANTINE"
    assert trend.enabled is False  # Auto-quarantined!


def test_rl_agent_q_learning(tmp_path):
    storage_path = tmp_path / "test_rl.json"
    rl = RLAgent(alpha=0.1, gamma=0.9, storage_path=storage_path)

    symbol = "R_100"
    score = 0.88

    # Initial Q-values are [0.0, 0.0]
    initial_boost = rl.evaluate_rl_boost(symbol, score)
    assert initial_boost == 0.0

    # Update Q-value on winning trade (action=1)
    new_q = rl.update_q_value(symbol, score, action=1, won=True, stake=1.0, profit=0.95)

    assert new_q > 0.0
    boost_after_win = rl.evaluate_rl_boost(symbol, score)
    assert boost_after_win > 0.0  # Positive boost after profitable trade
