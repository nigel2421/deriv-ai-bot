import pytest
import asyncio
from typing import Dict, Any

from src.agents.portfolio_manager_agent import PortfolioManagerAgent
from src.agents.market_regime_agent import MarketRegimeAgent
from src.agents.breeding_system import AgentBreedingSystem, StrategyDNA
from src.agents.infrastructure_agent import InfrastructureAgent
from src.agents.base_agent import ConsensusDecision, AgentVote


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_portfolio_manager_agent():
    pm = PortfolioManagerAgent(risk_per_trade_pct=1.0, max_daily_loss_pct=5.0, max_open_trades=3)

    decision = ConsensusDecision(
        symbol="R_100",
        contract_type="DIGITEVEN",
        confidence=0.88,
        ensemble_score=0.88,
        votes=[AgentVote("TrendAgent", "R_100", "DIGITEVEN", 0.88, 1.2, 1.05, "ok")],
    )

    # Test 1: Daily Drawdown Veto (Daily PnL -60 <= -50 max loss on $1000 balance)
    veto_result = pm.evaluate_portfolio_decision(decision, account_balance=1000.0, open_trades=[], daily_pnl=-60.0)
    assert veto_result is None  # Vetoed!

    # Test 2: Max Open Trades Veto (3 active trades >= 3 max limit)
    open_trades_mock = [{"symbol": "R_10"}, {"symbol": "R_25"}, {"symbol": "R_50"}]
    veto_result2 = pm.evaluate_portfolio_decision(decision, account_balance=1000.0, open_trades=open_trades_mock, daily_pnl=0.0)
    assert veto_result2 is None  # Vetoed!

    # Test 3: Approved 1% Risk Sizing ($1,000 balance * 1% = $10.00 stake)
    approved = pm.evaluate_portfolio_decision(decision, account_balance=1000.0, open_trades=[], daily_pnl=0.0)
    assert approved is not None
    assert approved["stake"] == 10.00
    assert approved["portfolio_approved"] is True


def test_market_regime_agent():
    regime_agent = MarketRegimeAgent()

    # Generate synthetic trending ticks
    trending_ticks = [float(100 + i * 0.5) for i in range(30)]
    trending_res = regime_agent.detect_regime("R_100", trending_ticks)
    assert trending_res["regime"] == "TRENDING"
    assert trending_res["gating"]["TrendAgent"] is True

    # Generate synthetic sideways choppy ticks
    choppy_ticks = [100.0, 100.5, 99.8, 100.4, 99.9, 100.6, 99.7, 100.5] * 4
    choppy_res = regime_agent.detect_regime("R_100", choppy_ticks)
    assert choppy_res["regime"] == "SIDEWAYS_CHOP"
    assert choppy_res["gating"]["TrendAgent"] is False  # Disable TrendAgent in chop!


def test_agent_breeding_system():
    breeding = AgentBreedingSystem()

    p1 = StrategyDNA(agent_id="Parent1", ema_fast=9, ema_slow=21, confidence_threshold=0.80)
    p2 = StrategyDNA(agent_id="Parent2", ema_fast=14, ema_slow=28, confidence_threshold=0.70)

    offspring = breeding.crossover(p1, p2, child_id="Child_01")
    assert offspring.agent_id == "Child_01"
    assert offspring.ema_fast in (9, 14)
    assert offspring.ema_slow in (21, 28)

    mutated = breeding.mutate(offspring, mutation_rate=1.0)  # Force mutation
    assert mutated.confidence_threshold >= 0.65


@pytest.mark.anyio
async def test_infrastructure_agent():
    infra = InfrastructureAgent(cpu_threshold=10.0)  # Low threshold to trigger alert
    telemetry = infra.get_system_telemetry()

    assert "cpu_percent" in telemetry
    assert "ram_percent" in telemetry
    assert "disk_percent" in telemetry

    alerts = await infra.evaluate({})
    assert isinstance(alerts, list)
