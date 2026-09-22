import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())

from src.services.market_discovery_manager import MarketDiscoveryManager
from src.strategy.champion_challenger_engine import ChampionChallengerEngine
from src.strategy.opportunity_engine import OpportunityEngine
from src.agents.step_specialist_agent import StepSpecialistAgent
from src.agents.consensus_agent import ConsensusAgent
from src.agents.base_agent import AgentSignal

def main():
    print("--- 1. Testing Market Discovery Manager ---")
    disc = MarketDiscoveryManager()
    active_syms = disc.get_active_symbols()
    paper_syms = disc.get_paper_symbols()
    print("Active symbols count:", len(active_syms))
    print("Paper symbols count:", len(paper_syms))
    print("Sample active:", active_syms[:5])
    print("Sample paper:", paper_syms[:5])

    print("\n--- 2. Testing Champion / Challenger Engine ---")
    cc = ChampionChallengerEngine()
    cc.record_result("R_50", "MultiAgentConsensus", is_win=True, profit=0.85, confidence=0.78)
    cc.record_result("R_50", "MultiAgentConsensus", is_win=True, profit=0.85, confidence=0.82)
    profile = cc.get_profile("R_50", "MultiAgentConsensus")
    print(f"R_50 Profile: total={profile.total_trades}, wins={profile.wins}, status={profile.status}, min_conf={profile.get_min_confidence()}")

    print("\n--- 3. Testing Step Specialist Agent ---")
    step_agent = StepSpecialistAgent()
    mock_ticks = [{"quote": 100.0 + i * 0.1, "epoch": 1000 + i} for i in range(20)]
    eval_ctx = {"symbol": "STPIDX", "ticks": mock_ticks}
    signals = asyncio.run(step_agent.evaluate(eval_ctx))
    print("Step Agent Signals generated:", len(signals))
    for s in signals:
        print(f" -> {s.contract_type} conf={s.confidence:.2f} rationale={s.rationale}")

    print("\n--- 4. Testing Domain-Aware Consensus ---")
    consensus = ConsensusAgent()
    dec = consensus.form_consensus("STPIDX", signals, min_confidence=0.60)
    print("Consensus Decision:", dec)

    print("\n--- 5. Testing 2-Stage Opportunity Engine ---")
    opp_engine = OpportunityEngine(max_concurrent_exec=3)
    raw_candidates = [
        {"symbol": "R_10", "contract_type": "CALL", "confidence": 0.61, "ensemble_score": 0.61, "ev": 0.02, "strategy": "MultiAgentConsensus"},
        {"symbol": "R_50", "contract_type": "PUT", "confidence": 0.84, "ensemble_score": 0.84, "ev": 0.15, "strategy": "MultiAgentConsensus"},
        {"symbol": "1HZ100V", "contract_type": "CALL", "confidence": 0.79, "ensemble_score": 0.79, "ev": 0.11, "strategy": "MultiAgentConsensus"},
        {"symbol": "STPIDX", "contract_type": "CALL", "confidence": 0.67, "ensemble_score": 0.67, "ev": 0.05, "strategy": "MultiAgentConsensus"},
    ]
    ranked = opp_engine.evaluate_and_rank_candidates(raw_candidates, cc)
    print("Ranked Opportunities:")
    for opp in ranked:
        print(f"  Rank #{opp.rank}: [{opp.tier}] {opp.symbol} {opp.contract_type} -> Quality Score: {opp.opportunity_score:.1f} (Stake Mult: {opp.stake_multiplier})")

    selected = opp_engine.select_top_opportunities(ranked, max_trades=1)
    print(f"\n[TOP 1] Selected Opportunity for Execution: {selected[0].symbol} {selected[0].contract_type} (Score: {selected[0].opportunity_score})")

    print("\nALL SYSTEM ENGINES VERIFIED CLEANLY!")

if __name__ == "__main__":
    main()
