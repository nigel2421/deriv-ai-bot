"""
Edge Scanner — rank all markets by live opportunity score.

Delegates to market_scanner (category-aware + self-optimizing priority).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence


def scan_markets(
    symbols: Sequence[str],
    get_ticks: Callable[[str], Sequence[Dict[str, Any]]],
    *,
    history_by_key: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    top_n: int = 5,
    global_samples: int = 0,
) -> Dict[str, Any]:
    """
    Scan each symbol; rank by scanner score × adaptive priority.
    get_ticks(symbol) -> tick list
    history_by_key: optional "SYMBOL|TYPE" -> trade rows
    """
    from src.analytics.market_scanner import rank_markets

    out = rank_markets(
        symbols,
        get_ticks,
        history_by_key=history_by_key,
        top_n=max(top_n, 10),
        global_samples=global_samples,
    )
    ranked = out.get("ranked") or []
    return {
        **out,
        "top": ranked[:top_n],
        "best_opportunity": out.get("best_tradeable") or out.get("best"),
        "ranked": ranked,
    }
