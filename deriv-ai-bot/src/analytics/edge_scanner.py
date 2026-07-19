"""
Edge Scanner — rank all markets by live opportunity score.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from src.analytics.probability_engine import probability_table
from src.analytics.trade_filter import evaluate_setup


def scan_markets(
    symbols: Sequence[str],
    get_ticks: Callable[[str], Sequence[Dict[str, Any]]],
    *,
    history_by_key: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Scan each symbol; rank by live_edge * quality blend.
    get_ticks(symbol) -> tick list
    history_by_key: optional "SYMBOL|TYPE" -> trade rows
    """
    history_by_key = history_by_key or {}
    ranked: List[Dict[str, Any]] = []

    for sym in symbols:
        ticks = get_ticks(sym) or []
        if len(ticks) < 30:
            ranked.append(
                {
                    "symbol": sym,
                    "score": 0.0,
                    "status": "NO_DATA",
                    "best_type": None,
                }
            )
            continue

        probs = probability_table(ticks, symbol=sym)
        best = probs.get("best") or {}
        ct = str(best.get("key") or "CALL")
        key = f"{sym}|{ct}"
        hist = history_by_key.get(key) or history_by_key.get(sym) or []
        # Use recent closes for that symbol any type if empty
        if not hist:
            for k, rows in history_by_key.items():
                if k.startswith(f"{sym}|"):
                    hist.extend(rows)

        family = "rise_fall" if ct in {"CALL", "PUT"} else "digits"
        ev = evaluate_setup(
            ticks,
            symbol=sym,
            contract_type=ct,
            family=family,
            history_rows=hist,
            recent_rows=hist[-100:],
        )
        score = float(ev["live_edge"]["live_edge"]) * 0.6 + float(
            ev["quality"]["quality_score"]
        ) * 0.4
        ranked.append(
            {
                "symbol": sym,
                "score": round(score, 1),
                "live_edge": ev["live_edge"]["live_edge"],
                "quality": ev["quality"]["quality_score"],
                "pattern_strength": ev["pattern_strength"]["pattern_strength"],
                "status": ev["live_edge"]["status"],
                "recommendation": ev["recommendation"],
                "best_type": ct,
                "best_confidence": best.get("confidence"),
                "copilot": ev.get("copilot"),
                "allow": ev["allow"],
            }
        )

    ranked.sort(key=lambda r: r["score"], reverse=True)
    return {
        "ranked": ranked[:top_n],
        "all": ranked,
        "best_opportunity": ranked[0] if ranked else None,
    }
