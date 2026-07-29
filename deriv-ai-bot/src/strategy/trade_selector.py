"""
Trade Selector — EV-First Ranking (Rec #9)

Selects best trade across markets / contract families.

NEW scoring formula (EV-first):
    score = ev                           # PRIMARY: EV (confidence * payout - (1-conf))
            + 0.08 * learn_bonus         # HPP quality bonus
            + 0.04 * trend_strength      # Momentum signal
            + persistence_adjustment     # +/- rise/fall persistence (Rec #3)
            + velocity_bonus             # MOR velocity bonus (Rec #4)
            + family_bias                # 0.01 rise_fall preference

Candidates with EV <= 0 are still scored but ranked last.
The selector does NOT filter by EV here — ev_rank() in orchestrator does that.

Changes from original:
  - EV is now the primary component (was: confidence)
  - Added persistence_adjustment for rise/fall from TransitionMatrix
  - Added velocity_bonus from MORTracker
  - family_bias retained (0.01 for rise_fall)
"""
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TradeSelector:
    """
    Selects best trade across markets / contract families.

    Score = EV (primary) + HPP bonus + momentum + persistence + velocity + family_bias
    """

    def select_best_trade(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not signals:
            return None

        def score(s: Dict[str, Any]) -> float:
            # Primary: EV (expected value per unit stake)
            if "ev" in s and s["ev"] is not None:
                ev = float(s["ev"])
            else:
                from src.strategy.ev_engine import compute_ev
                ev = compute_ev(float(s.get("confidence") or 0.0))

            # HPP quality bonus (Rec #2 — already exists, now properly weighted)
            learn_bonus = float(s.get("learn_bonus") or 0.0)

            # Momentum signal strength
            strength = float(s.get("trend_strength") or 0.0)

            # Rise/Fall persistence bonus from TransitionMatrix (Rec #3)
            # Range: [-0.10, +0.10], 0.0 for digit contracts
            persistence_adj = float(s.get("persistence_adjustment") or 0.0)

            # MOR velocity bonus (Rec #4): rapidly improving markets get +0.03
            vel_bonus = float(s.get("velocity_bonus") or 0.0)

            # Slight edge to rise/fall when everything else is equal
            family_bias = 0.01 if s.get("family") in ("rise_fall", "minute_rise_fall") else 0.0

            return (
                ev
                + 0.08 * learn_bonus
                + 0.04 * strength
                + persistence_adj
                + vel_bonus
                + family_bias
            )

        best = max(signals, key=score)
        logger.info(
            "Selected best trade score=%.4f: %s %s conf=%.2f ev=%.4f "
            "label=%s family=%s persist_adj=%.3f vel_bonus=%.3f",
            score(best),
            best.get("symbol"),
            best.get("contract_type"),
            best.get("confidence"),
            best.get("ev", 0.0),
            best.get("ev_label", "?"),
            best.get("family"),
            best.get("persistence_adjustment", 0.0),
            best.get("velocity_bonus", 0.0),
        )
        return best
