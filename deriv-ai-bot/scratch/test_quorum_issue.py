import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())

from src.agents.consensus_agent import ConsensusAgent
from src.agents.base_agent import AgentSignal

agent = ConsensusAgent(min_quorum=2)

# Digits signal from PatternAgent only
digits_sig = [
    AgentSignal(
        agent_name="PatternAgent",
        symbol="1HZ10V",
        contract_type="DIGITEVEN",
        confidence=0.68,
        weight=1.0,
    )
]

decision = agent.form_consensus("1HZ10V", digits_sig, min_confidence=0.60)
print("Digits Decision with min_quorum=2:", decision)

# Now test with min_quorum logic that allows single-specialist signals (like Digits/FX)
# E.g. if single agent confidence is >= 0.60 or contract is Digits/FX
def form_consensus_fixed(self, symbol, signals, min_confidence=0.60):
    if not signals:
        return None
    valid_signals = [s for s in signals if s.contract_type not in ("SCAN_OK", "VOLATILITY_RATING")]
    if not valid_signals:
        return None
    grouped = {}
    for sig in valid_signals:
        ct = sig.contract_type
        if ct not in grouped:
            grouped[ct] = []
        grouped[ct].append(sig)

    best_decision = None
    highest_score = 0.0
    for ct, sig_list in grouped.items():
        voter_names = {s.agent_name for s in sig_list}
        # Allow single specialist (quorum=1) if contract is digit/FX or confidence is high
        required_quorum = 1 if (ct.startswith("DIGIT") or ct in ("CALL", "PUT")) else self.min_quorum
        if len(voter_names) < required_quorum:
            continue
        total_score = sum(s.confidence * s.weight for s in sig_list)
        total_weight = sum(s.weight for s in sig_list)
        score = total_score / max(0.1, total_weight)
        rep = max(sig_list, key=lambda s: s.confidence)
        if score >= min_confidence and score > highest_score:
            highest_score = score
            from src.agents.base_agent import ConsensusDecision, AgentVote
            votes = [AgentVote(agent_name=s.agent_name, symbol=symbol, contract_type=ct, confidence=s.confidence, weight=s.weight, score=s.confidence*s.weight, rationale=s.rationale) for s in sig_list]
            best_decision = ConsensusDecision(
                symbol=symbol,
                contract_type=ct,
                confidence=rep.confidence,
                ensemble_score=score,
                barrier=rep.barrier,
                duration=rep.duration,
                duration_unit=rep.duration_unit,
                votes=votes,
                agent_weights={s.agent_name: s.weight for s in sig_list},
                rationale="Fixed consensus",
                raw_intent={"symbol": symbol, "contract_type": ct, "confidence": score, "duration": 5, "duration_unit": "t"}
            )
    return best_decision

print("Digits Decision with fixed quorum logic:", form_consensus_fixed(agent, "1HZ10V", digits_sig, min_confidence=0.60))
