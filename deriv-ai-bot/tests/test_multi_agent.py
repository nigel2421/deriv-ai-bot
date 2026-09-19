import pytest
import asyncio
from pathlib import Path
from src.agents.base_agent import AgentSignal, AgentVote, ConsensusDecision
from src.agents.bus import AgentEventBus
from src.agents.market_scanner_agent import MarketScannerAgent
from src.agents.trend_agent import TrendAgent
from src.agents.volatility_agent import VolatilityAgent
from src.agents.pattern_agent import PatternAgent
from src.agents.learning_agent import LearningAgent
from src.agents.consensus_agent import ConsensusAgent
from src.agents.risk_agent import RiskAgent
from src.api.deriv_client import DerivClient
from src.orchestrator import TradingOrchestrator


def test_agent_event_bus():
    async def _test():
        bus = AgentEventBus()
        received = []

        def handler(payload):
            received.append(payload)

        bus.subscribe("test.topic", handler)
        await bus.publish("test.topic", {"msg": "hello_agent"})

        assert len(received) == 1
        assert received[0]["msg"] == "hello_agent"

    asyncio.run(_test())


def test_market_scanner_agent():
    async def _test():
        scanner = MarketScannerAgent()
        context = {
            "symbols": ["R_100", "frxEURUSD"],
            "ticks_map": {
                "R_100": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
                "frxEURUSD": [1.08, 1.081, 1.082, 1.083, 1.084, 1.085, 1.086, 1.087, 1.088, 1.089, 1.090],
            },
        }
        signals = await scanner.evaluate(context)
        assert len(signals) >= 1
        assert any(s.symbol == "R_100" for s in signals)

    asyncio.run(_test())


def test_trend_agent():
    async def _test():
        trend_agent = TrendAgent()
        ticks = [100.0 + i * 0.5 for i in range(30)]
        context = {
            "symbol": "R_100",
            "ticks": ticks,
            "allowed_types": ["CALL", "PUT"],
        }
        signals = await trend_agent.evaluate(context)
        assert len(signals) > 0
        assert signals[0].contract_type in ("CALL", "PUT")
        assert signals[0].confidence > 0.5

    asyncio.run(_test())


def test_volatility_agent():
    async def _test():
        vol_agent = VolatilityAgent()
        ticks = [100.0, 100.2, 100.1, 100.3, 100.2, 100.4, 100.3, 100.5, 100.4, 100.6, 100.5, 100.7, 100.6, 100.8, 100.7]
        context = {
            "symbol": "BOOM1000",
            "ticks": ticks,
        }
        signals = await vol_agent.evaluate(context)
        assert len(signals) > 0
        assert any(s.contract_type in ("CALL", "VOLATILITY_RATING") for s in signals)

    asyncio.run(_test())


def test_pattern_agent():
    async def _test():
        pattern_agent = PatternAgent()
        ticks = [100.12, 100.14, 100.18, 100.22, 100.26, 100.28, 100.34, 100.36, 100.42, 100.44]
        context = {
            "symbol": "R_100",
            "ticks": ticks,
            "allowed_types": ["DIGITEVEN", "DIGITODD"],
        }
        signals = await pattern_agent.evaluate(context)
        assert len(signals) > 0
        assert signals[0].family == "digits"

    asyncio.run(_test())


def test_learning_agent_weight_adaptation(tmp_path):
    weights_file = tmp_path / "test_weights.json"
    learner_agent = LearningAgent(weights_path=weights_file)

    initial_weight = learner_agent.get_agent_weight("TrendAgent")
    assert initial_weight == 1.0

    for _ in range(5):
        learner_agent.record_trade_outcome("R_100", "CALL", won=True, participating_agents=["TrendAgent"])

    updated_weight = learner_agent.get_agent_weight("TrendAgent")
    assert updated_weight > 1.0


def test_consensus_agent():
    consensus_agent = ConsensusAgent(min_consensus_score=0.60)
    sig1 = AgentSignal(
        agent_name="TrendAgent",
        symbol="R_100",
        contract_type="CALL",
        confidence=0.85,
        weight=1.2,
        rationale="Strong trend",
    )
    sig2 = AgentSignal(
        agent_name="PatternAgent",
        symbol="R_100",
        contract_type="CALL",
        confidence=0.80,
        weight=1.0,
        rationale="Pattern confirmation",
    )

    decision = consensus_agent.form_consensus("R_100", [sig1, sig2], min_confidence=0.70)
    assert decision is not None
    assert decision.contract_type == "CALL"
    assert decision.ensemble_score >= 0.80
    assert len(decision.votes) == 2


def test_risk_agent_evaluation():
    risk_agent = RiskAgent()
    decision = ConsensusDecision(
        symbol="R_100",
        contract_type="CALL",
        confidence=0.82,
        ensemble_score=0.84,
        raw_intent={"symbol": "R_100", "contract_type": "CALL", "family": "digits", "participating_agents": ["TrendAgent"]},
    )
    approved = risk_agent.evaluate_risk(decision, current_balance=100.0, open_trade_count=0)
    assert approved is not None
    assert approved["risk_approved"] is True
    assert approved["stake"] >= 0.35


def test_orchestrator_multi_agent_integration():
    client = DerivClient("test_app", "test_token", mode="demo")
    ticks = [
        {"symbol": "R_100", "quote": 100.0 + i * 0.05, "epoch": 1_700_000_000 + i}
        for i in range(50)
    ]
    client.seed_tick_buffer("R_100", ticks)
    orch = TradingOrchestrator(client, mode="demo")

    assert orch.enable_multi_agent is True
    status = orch.risk_status()
    assert "multi_agent" in status
    assert status["multi_agent"]["enabled"] is True
    assert len(status["multi_agent"]["agents"]) == 8

    async def _test():
        trade = await orch.scan_markets_multi_agent()
        # Returns candidate or None if EV/confidence gate is tight
        assert trade is None or isinstance(trade, dict)

    asyncio.run(_test())
