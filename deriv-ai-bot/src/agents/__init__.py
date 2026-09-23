"""
Multi-Agent AI Trading System Package
"""

from src.agents.base_agent import (
    BaseAgent,
    AgentSignal,
    AgentVote,
    ConsensusDecision,
)
from src.agents.bus import AgentEventBus
from src.agents.digit_stat_agent import DigitStatAgent

__all__ = [
    "BaseAgent",
    "AgentSignal",
    "AgentVote",
    "ConsensusDecision",
    "AgentEventBus",
    "DigitStatAgent",
]

