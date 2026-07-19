"""
Adaptive stake sizing: lower risk after losses, restore after profits.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def adaptive_risk_pct(
    *,
    base_risk_pct: float = 1.0,
    consecutive_losses: int = 0,
    consecutive_wins: int = 0,
    daily_pnl: float = 0.0,
    session_start_balance: Optional[float] = None,
    min_risk_pct: float = 0.5,
    max_risk_pct: float = 2.0,
) -> Dict[str, Any]:
    """
    After losses → risk 0.5%; after profits → toward base (1%).
    Never exceed max_risk_pct (2%).
    """
    base = max(min_risk_pct, min(max_risk_pct, float(base_risk_pct)))
    risk = base

    if consecutive_losses >= 3:
        risk = min_risk_pct
        reason = f"{consecutive_losses} consecutive losses → min risk"
    elif consecutive_losses >= 1:
        # Step down
        risk = max(min_risk_pct, base * (0.85 ** consecutive_losses))
        reason = f"{consecutive_losses} loss(es) → reduced risk"
    elif consecutive_wins >= 3 and daily_pnl > 0:
        risk = min(max_risk_pct, base * 1.0)  # do not increase above base
        reason = "winning streak — keep base risk (no revenge sizing)"
    else:
        reason = "base risk"

    # Drawdown soft cut
    if session_start_balance and session_start_balance > 0 and daily_pnl < 0:
        dd = abs(daily_pnl) / session_start_balance
        if dd >= 0.03:
            risk = min(risk, min_risk_pct)
            reason = "session drawdown ≥3% → min risk"

    risk = max(min_risk_pct, min(max_risk_pct, risk))
    return {
        "risk_pct": round(risk, 3),
        "base_risk_pct": base,
        "reason": reason,
    }


def stake_from_risk(
    balance: float,
    risk_pct: float,
    *,
    min_stake: float = 0.35,
    max_stake: Optional[float] = None,
) -> float:
    if balance is None or balance <= 0:
        return 0.0
    stake = float(balance) * (float(risk_pct) / 100.0)
    stake = max(min_stake, stake)
    if max_stake is not None:
        stake = min(stake, float(max_stake))
    return round(stake, 2)
