import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


@dataclass
class AgentSignal:
    """Standard signal structure emitted by specialized intelligence agents."""
    agent_name: str
    symbol: str
    contract_type: str  # e.g. CALL, PUT, DIGITEVEN, DIGITODD, DIGITOVER, DIGITUNDER
    confidence: float   # 0.0 to 1.0
    raw_confidence: float = 0.0
    weight: float = 1.0
    barrier: Optional[Any] = None
    duration: Optional[int] = None
    duration_unit: Optional[str] = None
    horizon: Optional[str] = "tick"
    family: Optional[str] = "digits"
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "symbol": self.symbol,
            "contract_type": self.contract_type,
            "confidence": round(self.confidence, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "weight": round(self.weight, 4),
            "barrier": self.barrier,
            "duration": self.duration,
            "duration_unit": self.duration_unit,
            "horizon": self.horizon,
            "family": self.family,
            "rationale": self.rationale,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentVote:
    """Individual agent vote included in consensus decision."""
    agent_name: str
    symbol: str
    contract_type: str
    confidence: float
    weight: float
    score: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "symbol": self.symbol,
            "contract_type": self.contract_type,
            "confidence": round(self.confidence, 4),
            "weight": round(self.weight, 4),
            "score": round(self.score, 4),
            "rationale": self.rationale,
        }


@dataclass
class ConsensusDecision:
    """Final ensemble decision agreed upon by Consensus Agent."""
    symbol: str
    contract_type: str
    confidence: float
    ensemble_score: float
    barrier: Optional[Any] = None
    duration: Optional[int] = None
    duration_unit: Optional[str] = None
    votes: List[AgentVote] = field(default_factory=list)
    agent_weights: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    raw_intent: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "contract_type": self.contract_type,
            "confidence": round(self.confidence, 4),
            "ensemble_score": round(self.ensemble_score, 4),
            "barrier": self.barrier,
            "duration": self.duration,
            "duration_unit": self.duration_unit,
            "votes": [v.to_dict() for v in self.votes],
            "agent_weights": {k: round(v, 4) for k, v in self.agent_weights.items()},
            "rationale": self.rationale,
            "timestamp": self.timestamp,
        }


class BaseAgent(abc.ABC):
    """Abstract base class for all specialized trading agents."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abc.abstractmethod
    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        """
        Evaluate market context and return a list of signals (can be empty).
        
        context typically contains:
        - symbol: str
        - ticks: List[float] / List[Dict]
        - allowed_types: List[str]
        - market_info: Dict[str, Any]
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "class": self.__class__.__name__,
        }
