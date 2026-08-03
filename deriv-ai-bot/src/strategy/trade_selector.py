"""
TradeSelector — Dynamic AI-Auditor-Weighted Trade Selection (Rec #9 & Rec #10)

Selects the best trade across markets / contract families using dynamic weights
automatically loaded from the AI Auditor report (data/auditor_report.json).

Features dynamically boosted or penalized based on empirical PnL contribution:
  - trend_strength
  - mor_score
  - learn_bonus
  - persistence_conf
  - parity_conf
  - chop_score (penalized when hurting)
"""
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

AUDITOR_REPORT_PATH = Path("data/auditor_report.json")


class TradeSelector:
    """
    Selects best trade using EV + dynamic feature weights from AI Auditor.
    """

    def __init__(self, auditor_path: Optional[Path] = None):
        self.auditor_path = Path(auditor_path) if auditor_path else AUDITOR_REPORT_PATH
        self.weights = self._load_auditor_weights()

    def _load_auditor_weights(self) -> Dict[str, float]:
        """
        Loads feature contributions from auditor_report.json to dynamically weight signals.
        """
        defaults = {
            "trend_strength": 0.04,
            "mor_score": 0.05,
            "learn_bonus": 0.08,
            "persistence_conf": 0.02,
            "parity_conf": 0.02,
            "chop_score": -0.04,
            "ev": 1.0,
        }
        if not self.auditor_path.is_file():
            return defaults
        try:
            data = json.loads(self.auditor_path.read_text(encoding="utf-8"))
            raw = data.get("raw_contributions") or {}
            for feat, val in raw.items():
                if val is not None:
                    # Convert profit contribution ($/trade) to feature weight multiplier
                    # e.g., +$0.73/trade -> weight 0.073
                    defaults[feat] = round(float(val) * 0.1, 4)
            logger.info("TradeSelector loaded dynamic AI Auditor weights: %s", defaults)
        except Exception as e:
            logger.debug("TradeSelector failed to load auditor report: %s", e)
        return defaults

    def select_best_trade(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not signals:
            return None

        # Refresh weights from auditor report if updated
        self.weights = self._load_auditor_weights()
        w = self.weights

        def compute_trade_score(s: Dict[str, Any]) -> Tuple[bool, float, float]:
            ev = float(s.get("ev", 0.0))
            is_positive_ev = ev > 0.0
            
            mps = float(s.get("mps", 0.0))
            
            # Base features
            trend_str = float(s.get("trend_strength", 0.0))
            mor = float(s.get("mor_score", 0.0)) / 100.0  # normalize 0-1
            learn = float(s.get("learn_bonus", 0.0))
            persist = float(s.get("persistence_conf", 0.0))
            parity = float(s.get("parity_conf", 0.0))
            regime = s.get("regime") or {}
            chop = float(regime.get("chop_score", 0.0))

            # Dynamic AI Auditor scoring formula
            auditor_bonus = (
                (w.get("trend_strength", 0.04) * trend_str)
                + (w.get("mor_score", 0.05) * mor)
                + (w.get("learn_bonus", 0.08) * learn)
                + (w.get("persistence_conf", 0.02) * persist)
                + (w.get("parity_conf", 0.02) * parity)
                + (w.get("chop_score", -0.04) * chop)
            )

            conf = float(s.get("confidence", 0.0))
            trade_quality = float(s.get("trade_quality", (ev * conf * 100))) + (auditor_bonus * 100.0)

            return (is_positive_ev, mps, trade_quality)

        best = max(signals, key=compute_trade_score)
        best_score = compute_trade_score(best)

        logger.info(
            "Selected best trade quality=%.2f (pos_ev=%s, mps=%.1f): %s %s conf=%.2f ev=%.4f",
            best_score[2],
            best_score[0],
            best_score[1],
            best.get("symbol"),
            best.get("contract_type"),
            best.get("confidence"),
            best.get("ev", 0.0),
        )
        return best
