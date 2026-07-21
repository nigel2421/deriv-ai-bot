"""
Expected Value (EV) Engine — Rec #9

True EV per unit stake:
    EV = confidence × payout_rate - (1 - confidence) × 1.0

where payout_rate is the net return on a winning $1 stake
(e.g. payout_rate=0.87 means a $1 stake returns $1.87 on win).

EV > 0 means the trade has positive expected value.
EV is elevated to the PRIMARY ranking criterion in trade_selector.

Trade selection priority:
    1. Positive EV only
    2. Highest EV (not highest confidence)
    3. Highest MOR velocity bonus as tiebreaker
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Fallback payout rate if no live proposal rate is available.
# 0.87 = Deriv standard for most synthetic indices (win returns 1.87x stake).
DEFAULT_PAYOUT_RATE = 0.87


def compute_ev(confidence: float, payout_rate: float = DEFAULT_PAYOUT_RATE) -> float:
    """
    Compute expected value per unit stake.

    Args:
        confidence:   P(win), float in [0, 1]
        payout_rate:  Net profit on $1 stake if win. e.g. 0.87 -> win returns $1.87.

    Returns:
        EV as float. Positive = profitable setup.
        EV = conf * payout - (1 - conf) * 1.0
    """
    p = max(0.0, min(1.0, confidence))
    r = max(0.0, payout_rate)
    return round(p * r - (1.0 - p), 6)


def breakeven_confidence(payout_rate: float = DEFAULT_PAYOUT_RATE) -> float:
    """
    Minimum confidence for EV > 0.
    Derived from:  conf * payout - (1 - conf) = 0
    -> conf = 1 / (1 + payout_rate)
    """
    if payout_rate <= 0:
        return 1.0
    return round(1.0 / (1.0 + payout_rate), 4)


def ev_label(ev: float) -> str:
    """Human-readable EV tier."""
    if ev > 0.30:
        return "STRONG"
    if ev > 0.15:
        return "GOOD"
    if ev > 0.0:
        return "MARGINAL"
    return "NEGATIVE"


def ev_rank(
    candidates: List[Dict[str, Any]],
    *,
    allow_negative: bool = False,
) -> List[Dict[str, Any]]:
    """
    Annotate each candidate with its EV and sort descending.

    Each candidate dict must have:
        - "confidence": float
        - "payout_rate": float (optional, defaults to DEFAULT_PAYOUT_RATE)

    Candidates with EV <= 0 are filtered out unless allow_negative=True.

    Returns:
        Sorted list (highest EV first) with "ev" and "ev_label" fields added.
    """
    annotated = []
    for c in candidates:
        conf = float(c.get("confidence") or 0.0)
        rate = float(c.get("payout_rate") or DEFAULT_PAYOUT_RATE)
        ev = compute_ev(conf, rate)
        c = {**c, "ev": ev, "ev_label": ev_label(ev)}
        if ev > 0 or allow_negative:
            annotated.append(c)
        else:
            logger.debug(
                "EV filter: %s %s conf=%.2f payout=%.2f ev=%.4f (blocked)",
                c.get("symbol"),
                c.get("contract_type"),
                conf,
                rate,
                ev,
            )

    ranked = sorted(annotated, key=lambda x: x["ev"], reverse=True)

    if ranked:
        logger.info(
            "EV ranking: top=%s %s ev=%.4f label=%s  (%d candidates, %d positive-EV)",
            ranked[0].get("symbol"),
            ranked[0].get("contract_type"),
            ranked[0]["ev"],
            ranked[0]["ev_label"],
            len(candidates),
            len(ranked),
        )

    return ranked
