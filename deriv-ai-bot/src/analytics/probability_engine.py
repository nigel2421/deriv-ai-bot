"""
Probability / confidence estimates for each trade type from live digits + trend.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.analytics.digit_analysis import digit_snapshot
from src.strategy.digit_contracts import last_digits_from_ticks
from src.strategy.pro_trend import analyze_pro_trend
from src.strategy.trend_analyzer import analyze_trend


def _clamp01(x: float) -> float:
    return max(0.0, min(0.99, float(x)))


def probability_table(
    ticks: Sequence[Dict[str, Any]],
    *,
    symbol: str = "",
    over_barrier: int = 5,
    under_barrier: int = 5,
    match_digit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute confidence-like scores for UP/DOWN/OVER/UNDER/ODD/EVEN/MATCH/DIFFER.
    These are heuristic live probabilities — not fair odds guarantees.
    """
    snap = digit_snapshot(ticks, primary_window=100)
    digits = last_digits_from_ticks(ticks, n=100)
    n = len(digits) or 1
    heat = (snap.get("heatmap") or {}).get("windows", {}).get("100") or {}
    pct = heat.get("pct") or {d: 10.0 for d in range(10)}

    even_rate = float(snap.get("even_rate") or 0.5)
    odd_rate = 1.0 - even_rate

    # OVER barrier: P(digit > b) from empirical freq
    over_p = sum(pct.get(d, 10.0) for d in range(over_barrier + 1, 10)) / 100.0
    under_p = sum(pct.get(d, 10.0) for d in range(0, under_barrier)) / 100.0

    # Match / differ — cold digits slightly favor MATCH fade? Differ on hot is weak.
    # Use rarity of target digit for MATCH (cold → higher match surprise? Actually
    # match is hard; prefer differ when one digit is hot/overrepresented for "avoid")
    md = match_digit if match_digit is not None else (snap.get("last_digit") or 5)
    match_p = pct.get(int(md), 10.0) / 100.0
    differ_p = 1.0 - match_p

    # Bias: cold digit → slightly higher match surprise is NOT edge; use deviation
    # Differ confidence rises when one digit dominates (avoid matching the mode)
    cold = heat.get("cold") or []
    hot = heat.get("hot") or []
    if hot:
        # Differ away from mode has less edge theoretically; boost differ if mode too hot (mean reversion)
        mode = hot[0]
        mode_share = pct.get(mode, 10.0) / 100.0
        if mode_share > 0.14:
            differ_p = min(0.92, differ_p + (mode_share - 0.10) * 0.5)
            match_p = 1.0 - differ_p

    # Rise/fall from pro + classic
    trend = analyze_trend(ticks)
    pro = analyze_pro_trend(ticks, symbol=symbol, min_confidence=0.5)
    up_conf = 0.5
    down_conf = 0.5
    if pro.get("contract_type") == "CALL":
        up_conf = float(pro.get("confidence") or 0.55)
        down_conf = 1.0 - up_conf * 0.85
    elif pro.get("contract_type") == "PUT":
        down_conf = float(pro.get("confidence") or 0.55)
        up_conf = 1.0 - down_conf * 0.85
    elif trend.get("contract_type") == "CALL":
        up_conf = float(trend.get("confidence") or 0.55)
        down_conf = 1.0 - up_conf * 0.9
    elif trend.get("contract_type") == "PUT":
        down_conf = float(trend.get("confidence") or 0.55)
        up_conf = 1.0 - down_conf * 0.9

    # Convert empirical rates to "confidence" display (distance from 0.5)
    def conf_from_p(p: float, floor: float = 0.52) -> float:
        # Map p to conf: |p-0.5|*2 scaled into 0.5–0.95
        edge = abs(p - 0.5) * 2.0
        return _clamp01(0.50 + edge * 0.45)

    rows = [
        {"trade_type": "UP (CALL)", "key": "CALL", "probability": round(up_conf, 3),
         "confidence": round(up_conf, 3)},
        {"trade_type": "DOWN (PUT)", "key": "PUT", "probability": round(down_conf, 3),
         "confidence": round(down_conf, 3)},
        {"trade_type": f"OVER {over_barrier}", "key": "DIGITOVER",
         "probability": round(over_p, 3), "confidence": conf_from_p(over_p),
         "barrier": over_barrier},
        {"trade_type": f"UNDER {under_barrier}", "key": "DIGITUNDER",
         "probability": round(under_p, 3), "confidence": conf_from_p(under_p),
         "barrier": under_barrier},
        {"trade_type": "EVEN", "key": "DIGITEVEN", "probability": round(even_rate, 3),
         "confidence": conf_from_p(even_rate)},
        {"trade_type": "ODD", "key": "DIGITODD", "probability": round(odd_rate, 3),
         "confidence": conf_from_p(odd_rate)},
        {"trade_type": f"MATCH {md}", "key": "DIGITMATCH", "probability": round(match_p, 3),
         "confidence": conf_from_p(match_p), "barrier": md},
        {"trade_type": f"DIFFER {md}", "key": "DIGITDIFF", "probability": round(differ_p, 3),
         "confidence": conf_from_p(differ_p), "barrier": md},
    ]
    rows_sorted = sorted(rows, key=lambda r: r["confidence"], reverse=True)
    return {
        "symbol": symbol,
        "n": n,
        "rows": rows_sorted,
        "hot": hot,
        "cold": cold,
        "best": rows_sorted[0] if rows_sorted else None,
    }
