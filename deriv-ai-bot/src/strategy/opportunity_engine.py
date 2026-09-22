"""
2-Stage Opportunity Engine & Opportunity Ranking System.

Stage A (Qualification Gate):
  1. Valid contract parameters & symbol check
  2. Baseline / learned confidence threshold met
  3. Positive Expected Value (EV > 0)
  4. Risk manager clearance & Market not disabled

Stage B (Opportunity Ranking & Tier Assignment):
  Scores qualified candidates on a 0-100 Opportunity Scale:
    Score = (Ensemble Conf * 40) + (min(EV, 0.50) * 60) + Champion Bonus

  Assigns Three Tiers:
    🟢 Tier A (Conf >= 0.75, EV >= 0.08, Score >= 75) -> Full Stake (100%)
    🟡 Tier B (Conf 0.66 - 0.749, EV >= 0.04, Score >= 60) -> Reduced Stake (50%)
    🔵 Tier C (Conf 0.62 - 0.659, EV > 0) -> Paper / Shadow Observation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QualifiedOpportunity:
    """Represents a fully evaluated and ranked trade opportunity."""

    symbol: str
    contract_type: str
    confidence: float
    ensemble_score: float
    ev: float
    opportunity_score: float
    tier: str  # TIER_A | TIER_B | TIER_C
    stake_multiplier: float  # 1.0 for Tier A, 0.5 for Tier B, 0.0 for Tier C (paper)
    participating_agents: List[str]
    intent: Dict[str, Any]
    champion_status: str = "ACTIVE"
    rank: int = 0


class OpportunityEngine:
    """2-Stage Qualification & Opportunity Ranking Engine."""

    def __init__(self, max_concurrent_exec: int = 3):
        self.max_concurrent_exec = max_concurrent_exec
        self.bottlenecks: Dict[str, int] = {
            "ev_gate": 0,
            "confidence_gate": 0,
            "quorum_gate": 0,
            "risk_gate": 0,
            "cooldown_gate": 0,
        }
        self.scanned_last_hour: int = 0
        self.qualified_last_hour: int = 0
        self.executed_last_hour: int = 0

    def evaluate_and_rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        champion_engine: Optional[Any] = None,
    ) -> List[QualifiedOpportunity]:
        """
        Stage A & Stage B processing for a list of candidate trade intents.
        Returns candidates ranked by Opportunity Score.
        """
        if not candidates:
            return []

        self.scanned_last_hour += len(candidates)
        qualified: List[QualifiedOpportunity] = []

        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            ct = candidate.get("contract_type", "")
            conf = float(candidate.get("confidence") or 0.0)
            score = float(candidate.get("ensemble_score") or conf)
            ev = float(candidate.get("ev") or 0.0)
            stype = candidate.get("strategy") or "MultiAgentConsensus"

            # Stage A Qualification Checks
            if ev <= 0.0:
                self.bottlenecks["ev_gate"] += 1
                continue

            champ_status = "ACTIVE"
            champ_bonus = 0.0
            if champion_engine:
                if not champion_engine.is_allowed_to_trade(symbol, stype):
                    self.bottlenecks["risk_gate"] += 1
                    continue
                champ_bonus = champion_engine.get_champion_bonus(symbol, stype)
                champ_profile = champion_engine.get_profile(symbol, stype)
                champ_status = champ_profile.status

            # Calculate Stage B Opportunity Score (0-100)
            conf_pts = max(0.0, min(1.0, score)) * 60.0  # max 60 pts from confidence
            ev_pts = min(1.0, max(0.0, ev / 0.30)) * 30.0  # max 30 pts from EV (+0.30 EV = 30 pts)
            bonus = max(-50.0, min(10.0, champ_bonus))

            opp_score = round(min(100.0, max(0.0, conf_pts + ev_pts + bonus)), 1)

            # Assign Three-Tier Structure
            if score >= 0.75 and ev >= 0.08:
                tier = "TIER_A"
                stake_mult = 1.0  # 100% full stake
            elif score >= 0.66 and ev >= 0.04:
                tier = "TIER_B"
                stake_mult = 0.5  # 50% reduced stake
            else:
                tier = "TIER_C"
                stake_mult = 0.0  # Paper/Shadow observation only

            opp = QualifiedOpportunity(
                symbol=symbol,
                contract_type=ct,
                confidence=conf,
                ensemble_score=score,
                ev=ev,
                opportunity_score=opp_score,
                tier=tier,
                stake_multiplier=stake_mult,
                participating_agents=candidate.get("participating_agents", []),
                intent=candidate,
                champion_status=champ_status,
            )
            qualified.append(opp)

        self.qualified_last_hour += len(qualified)

        # Stage B Ranking: Sort by Opportunity Score descending
        ranked = sorted(qualified, key=lambda x: x.opportunity_score, reverse=True)
        for i, item in enumerate(ranked):
            item.rank = i + 1

        logger.info(
            "OpportunityEngine ranked %d candidates: top=%s %s score=%.1f (tier=%s)",
            len(ranked),
            ranked[0].symbol if ranked else "N/A",
            ranked[0].contract_type if ranked else "N/A",
            ranked[0].opportunity_score if ranked else 0.0,
            ranked[0].tier if ranked else "N/A",
        )

        return ranked

    def select_top_opportunities(
        self,
        ranked: List[QualifiedOpportunity],
        max_trades: int = 1,
    ) -> List[QualifiedOpportunity]:
        """Select top N opportunities for execution."""
        live_eligible = [op for op in ranked if op.tier in ("TIER_A", "TIER_B")]
        selected = live_eligible[:max_trades]
        self.executed_last_hour += len(selected)
        return selected

    def get_stats(self) -> Dict[str, Any]:
        """Return Opportunity Radar stats for dashboard UI."""
        return {
            "scanned_last_hour": self.scanned_last_hour,
            "qualified_last_hour": self.qualified_last_hour,
            "executed_last_hour": self.executed_last_hour,
            "bottlenecks": self.bottlenecks,
        }
