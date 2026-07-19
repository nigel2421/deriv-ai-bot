from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Soft preference for historically stronger setups (also in AdaptiveLearner)
_WINNER_HINTS = {
    "R_50|PUT",
    "R_25|CALL",
    "1HZ50V|CALL",
    "1HZ50V|PUT",
    "R_10|CALL",
    "R_10|PUT",
}


class TradeSelector:
    """
    Selects best trade across markets / contract families.

    Score heavily prefers learned winners (learn_bonus), decision quality,
    live edge, and known good symbol|type pairs.
    """

    def select_best_trade(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not signals:
            return None

        def score(s: Dict[str, Any]) -> float:
            conf = float(s.get("confidence") or 0.0)
            # Raised weight on learning bonus (was 1.0×, effectively diluted)
            bonus = float(s.get("learn_bonus") or 0.0) * 1.75
            strength = float(s.get("trend_strength") or 0.0)
            family = str(s.get("family") or "")
            family_bias = 0.015 if family == "rise_fall" else 0.0
            if family == "digits":
                family_bias = -0.005  # slightly prefer RF when scores close

            live = float(s.get("live_edge") or 0.0)
            quality = float(s.get("quality_score") or 0.0)
            pattern = float(s.get("pattern_strength") or 0.0)
            decision_q = float(
                s.get("decision_quality")
                if s.get("decision_quality") is not None
                else quality
            )
            ev = float(s.get("ev") or 0.0)
            mp = float(s.get("mp_score") or 0.0)

            edge_bonus = (
                live * 0.0045
                + quality * 0.0035
                + pattern * 0.002
                + decision_q * 0.003
                + max(0.0, ev) * 0.08
                + mp * 0.0015
            )

            key = f"{s.get('symbol')}|{str(s.get('contract_type') or '').upper()}"
            winner_bias = 0.04 if key in _WINNER_HINTS else 0.0

            # Penalize weak EV / rejected no-trade leftovers
            if s.get("no_trade") and not (s.get("no_trade") or {}).get("allow", True):
                return -1.0

            return (
                conf
                + bonus
                + 0.06 * strength
                + family_bias
                + edge_bonus
                + winner_bias
            )

        best = max(signals, key=score)
        logger.info(
            "Selected best trade score=%.3f: %s %s conf=%.2f family=%s "
            "live_edge=%s quality=%s decision_q=%s learn_bonus=%s",
            score(best),
            best.get("symbol"),
            best.get("contract_type"),
            best.get("confidence"),
            best.get("family"),
            best.get("live_edge"),
            best.get("quality_score"),
            best.get("decision_quality"),
            best.get("learn_bonus"),
        )
        return best
