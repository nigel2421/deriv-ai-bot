"""
Correlation Filter — Rec #7

Prevents trading multiple highly correlated synthetic indices simultaneously.
Within each correlation group, only the highest-EV candidate passes.
Cross-group candidates are unaffected.

Correlation groups (Deriv synthetic index families):
  - volatility_standard: R_10, R_25, R_50, R_75, R_100
  - volatility_hf: 1HZ10V, 1HZ25V, 1HZ50V, 1HZ75V, 1HZ100V

Rationale:
  R_10, R_25, R_50, R_75, R_100 are all generated from similar stochastic
  processes by Deriv. A signal on R_75 likely reflects the same underlying
  condition as a signal on R_50. Taking both wastes stake on correlated risk.

If more symbols are added (BOOM, CRASH, STEP), add their group here.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Define correlation groups. Symbols within a group are treated as highly correlated.
CORRELATION_GROUPS: Dict[str, Set[str]] = {
    "volatility_standard": {"R_10", "R_25", "R_50", "R_75", "R_100"},
    "volatility_hf": {"1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V"},
    # Add future groups here:
    # "boom": {"BOOM500", "BOOM1000"},
    # "crash": {"CRASH500", "CRASH1000"},
    # "step": {"STPRNG"},
}


def get_group(symbol: str) -> Optional[str]:
    """Return the correlation group name for a symbol, or None if uncorrelated."""
    for group_name, members in CORRELATION_GROUPS.items():
        if symbol in members:
            return group_name
    return None


class CorrelationFilter:
    """
    Filter that ensures only one candidate per correlation group passes
    to trade_selector, based on highest EV.

    Usage:
        filtered = correlation_filter.filter_candidates(candidates)
        # Then pass filtered list to TradeSelector.select_best_trade()
    """

    def filter_candidates(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Within each correlation group, keep only the highest-EV candidate.
        Candidates without a group assignment are always kept.

        Requires that each candidate dict has:
            "symbol": str
            "ev": float  (added by ev_engine.ev_rank before this call)

        Returns filtered list.
        """
        if not candidates:
            return candidates

        # Separate by group
        group_best: Dict[str, Dict[str, Any]] = {}  # group_name -> best candidate
        uncorrelated: List[Dict[str, Any]] = []
        blocked_log: List[str] = []

        for cand in candidates:
            symbol = str(cand.get("symbol") or "")
            group = get_group(symbol)

            if group is None:
                uncorrelated.append(cand)
                continue

            ev = float(cand.get("ev") or 0.0)
            existing = group_best.get(group)

            if existing is None:
                group_best[group] = cand
            elif ev > float(existing.get("ev") or 0.0):
                # New candidate beats the current best — block the old one
                blocked_log.append(
                    f"{existing.get('symbol')} {existing.get('contract_type')} "
                    f"ev={existing.get('ev', 0):.4f} (replaced by {symbol})"
                )
                group_best[group] = cand
            else:
                # Current candidate is worse — block it
                blocked_log.append(
                    f"{symbol} {cand.get('contract_type')} "
                    f"ev={ev:.4f} (blocked by {existing.get('symbol')} ev={existing.get('ev', 0):.4f})"
                )

        if blocked_log:
            logger.info(
                "CorrelationFilter blocked %d candidates: %s",
                len(blocked_log),
                " | ".join(blocked_log),
            )

        result = uncorrelated + list(group_best.values())

        logger.info(
            "CorrelationFilter: %d in -> %d out (%d groups active)",
            len(candidates),
            len(result),
            len(group_best),
        )

        return result

    def snapshot(self, candidates_before: List[Dict], candidates_after: List[Dict]) -> Dict:
        """
        Generate a summary for the dashboard showing what was blocked.
        """
        blocked = [c for c in candidates_before if c not in candidates_after]
        passed = candidates_after

        rows = []
        for group_name, members in CORRELATION_GROUPS.items():
            group_passed = [c for c in passed if c.get("symbol") in members]
            group_blocked = [c for c in blocked if c.get("symbol") in members]
            total_signals = len(group_passed) + len(group_blocked)
            if total_signals == 0:
                continue
            selected = group_passed[0] if group_passed else None
            rows.append({
                "group": group_name,
                "signals": total_signals,
                "selected": f"{selected.get('symbol')} EV={selected.get('ev', 0):.4f}" if selected else "—",
                "blocked": [f"{c.get('symbol')} ({c.get('contract_type')})" for c in group_blocked],
            })

        return {"groups": rows}
