import pytest
from src.agents.digit_stat_agent import (
    DigitStatAgent,
    compute_digit_hiatus,
    compute_markov_row,
)
from src.agents.consensus_agent import ConsensusAgent
from src.agents.pattern_agent import PatternAgent
from src.agents.base_agent import AgentSignal


def test_compute_digit_hiatus():
    # Sequence of digits ending with last digit
    # Digits 0..8 appeared recently, 9 appeared 25 ticks ago
    digits = [9] + [i % 9 for i in range(25)]
    hiatus = compute_digit_hiatus(digits)
    assert hiatus[9] == 25
    assert hiatus[0] < 10


def test_compute_markov_row():
    # Sequence where digit 2 is always followed by digit 7
    digits = []
    for _ in range(20):
        digits.extend([2, 7, 3, 5])
    row = compute_markov_row(digits, current_digit=2)
    # P(7 | 2) should be highest
    assert max(row, key=row.get) == 7
    assert row[7] > 0.50


import asyncio

def test_digit_stat_agent_cold_digit_diff():
    agent = DigitStatAgent()

    # Generate tick stream where digit 9 is missing for 25 ticks
    ticks = []
    base_price = 1000.0
    for i in range(30):
        digit = i % 9  # Only digits 0-8 used
        quote = base_price + (digit * 0.01)
        ticks.append({"quote": quote, "epoch": 1700000000 + i})

    context = {
        "symbol": "R_100",
        "ticks": ticks,
        "allowed_types": ["DIGITDIFF", "DIGITEVEN", "DIGITODD"],
    }

    signals = asyncio.run(agent.evaluate(context))
    assert len(signals) >= 1

    diff_signals = [s for s in signals if s.contract_type == "DIGITDIFF"]
    assert len(diff_signals) == 1
    sig = diff_signals[0]
    assert sig.barrier == 9
    assert sig.confidence >= 0.88


def test_consensus_agent_digit_quorum():
    consensus = ConsensusAgent(min_quorum=2)

    # 1. Single agent voting DIGITEVEN (should fail quorum)
    sig1 = AgentSignal(
        agent_name="PatternAgent",
        symbol="R_100",
        contract_type="DIGITEVEN",
        confidence=0.75,
        weight=1.0,
    )
    decision_fail = consensus.form_consensus("R_100", [sig1], min_confidence=0.70)
    assert decision_fail is None

    # 2. Dual digit agents voting DIGITEVEN (should pass quorum)
    sig2 = AgentSignal(
        agent_name="DigitStatAgent",
        symbol="R_100",
        contract_type="DIGITEVEN",
        confidence=0.72,
        weight=1.0,
    )
    decision_pass = consensus.form_consensus("R_100", [sig1, sig2], min_confidence=0.70)
    assert decision_pass is not None
    assert decision_pass.contract_type == "DIGITEVEN"
    assert len(decision_pass.votes) == 2
