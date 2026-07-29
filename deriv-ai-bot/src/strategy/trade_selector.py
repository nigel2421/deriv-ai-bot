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
from typing import Any, Dict, List, Optional, Tuple
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

        def score(s: Dict[str, Any]) -> Tuple[bool, float, float]:
            """
            Phase 7: Opportunity Cost Engine.
            Ranking priority: 1. Positive EV, 2. Highest MPS, 3. Highest Trade Quality
            """
            ev = float(s.get("ev", 0.0))
            is_positive_ev = ev > 0.0
            
            mps = float(s.get("mps", 0.0))
            
            # Trade Quality = EV * Confidence (proxy, or use passed in quality)
            conf = float(s.get("confidence", 0.0))
            trade_quality = float(s.get("trade_quality", ev * conf * 100))
            
            return (is_positive_ev, mps, trade_quality)

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
            best.get("mps", 0.0),
            best.get("trade_quality", 0.0),
        )
        return best
