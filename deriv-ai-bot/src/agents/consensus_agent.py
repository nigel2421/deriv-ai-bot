import logging
from typing import Any, Dict, List, Optional
from src.agents.base_agent import BaseAgent, AgentSignal, AgentVote, ConsensusDecision

logger = logging.getLogger(__name__)


class ConsensusAgent(BaseAgent):
    """
    Consensus / Decision Agent: Aggregates signals from all intelligence agents,
    computes weighted ensemble voting scores, and selects high-confidence trade decisions.
    """

    def __init__(
        self,
        name: str = "ConsensusAgent",
        min_consensus_score: float = 0.75,
        min_quorum: int = 2,
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.min_consensus_score = min_consensus_score
        self.min_quorum = min_quorum

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        """Not used directly; use form_consensus() for ensemble voting."""
        return []

    def form_consensus(
        self,
        symbol: str,
        signals: List[AgentSignal],
        min_confidence: float = 0.80,
    ) -> Optional[ConsensusDecision]:
        """
        Computes weighted consensus across incoming agent signals for a symbol.
        Enforces min_quorum (minimum unique agreeing specialist agents).
        """
        if not signals:
            return None

        # Filter out meta signals
        valid_signals = [s for s in signals if s.contract_type not in ("SCAN_OK", "VOLATILITY_RATING")]
        if not valid_signals:
            return None

        # Group signals by contract type
        grouped: Dict[str, List[AgentSignal]] = {}
        for sig in valid_signals:
            ct = sig.contract_type
            if ct not in grouped:
                grouped[ct] = []
            grouped[ct].append(sig)

        best_decision: Optional[ConsensusDecision] = None
        highest_score = 0.0

        for ct, sig_list in grouped.items():
            # Optimization #3: Quorum Check — Ensure at least min_quorum unique specialist agents voted
            voter_names = {s.agent_name for s in sig_list}
            if len(voter_names) < self.min_quorum:
                logger.debug(
                    "Skip candidate %s %s: quorum not met (%d/%d unique agents: %s)",
                    symbol,
                    ct,
                    len(voter_names),
                    self.min_quorum,
                    voter_names,
                )
                continue

            votes: List[AgentVote] = []
            total_weighted_conf = 0.0
            total_weight = 0.0

            for sig in sig_list:
                score = sig.confidence * sig.weight
                votes.append(
                    AgentVote(
                        agent_name=sig.agent_name,
                        symbol=symbol,
                        contract_type=ct,
                        confidence=sig.confidence,
                        weight=sig.weight,
                        score=score,
                        rationale=sig.rationale,
                    )
                )
                total_weighted_conf += score
                total_weight += sig.weight

            ensemble_score = total_weighted_conf / max(0.1, total_weight)
            # Pick representative signal for parameters (barrier, duration, etc.)
            rep_sig = max(sig_list, key=lambda s: s.confidence)

            if ensemble_score >= min_confidence and ensemble_score > highest_score:
                highest_score = ensemble_score
                voters_str = ", ".join(f"{v.agent_name} ({v.confidence:.2f})" for v in votes)

                best_decision = ConsensusDecision(
                    symbol=symbol,
                    contract_type=ct,
                    confidence=rep_sig.confidence,
                    ensemble_score=ensemble_score,
                    barrier=rep_sig.barrier,
                    duration=rep_sig.duration,
                    duration_unit=rep_sig.duration_unit,
                    votes=votes,
                    agent_weights={v.agent_name: v.weight for v in votes},
                    rationale=f"Consensus achieved: ensemble_score={ensemble_score:.2f} across [{voters_str}]",
                    raw_intent={
                        "symbol": symbol,
                        "contract_type": ct,
                        "confidence": ensemble_score,
                        "barrier": rep_sig.barrier,
                        "duration": rep_sig.duration,
                        "duration_unit": rep_sig.duration_unit,
                        "family": rep_sig.family,
                        "horizon": rep_sig.horizon,
                        "participating_agents": [v.agent_name for v in votes],
                    }
                )

        if best_decision:
            logger.info(
                "ConsensusAgent decision symbol=%s ct=%s score=%.2f votes=%d rationale=%s",
                best_decision.symbol,
                best_decision.contract_type,
                best_decision.ensemble_score,
                len(best_decision.votes),
                best_decision.rationale,
            )

        return best_decision
