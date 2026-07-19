"""
Martingale danger ladder + account survival probability.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def martingale_ladder(
    base_stake: float,
    max_levels: int = 5,
    multiplier: float = 2.1,
) -> List[Dict[str, Any]]:
    """Show level stakes (slightly super-martingale default 2.1x like user example)."""
    levels = []
    stake = float(base_stake)
    total = 0.0
    for i in range(1, max_levels + 1):
        total += stake
        levels.append(
            {
                "level": i,
                "stake": round(stake, 2),
                "cumulative_risk": round(total, 2),
            }
        )
        stake = stake * float(multiplier)
    return levels


def survival_probability(
    balance: float,
    base_stake: float,
    *,
    win_rate: float = 0.48,
    max_levels: int = 5,
    multiplier: float = 2.0,
) -> Dict[str, Any]:
    """
    Rough P(survive full ladder without wipe) assuming independent losses.
    Danger level from cumulative risk vs balance.
    """
    ladder = martingale_ladder(base_stake, max_levels=max_levels, multiplier=multiplier)
    if not ladder or balance <= 0:
        return {
            "survival_pct": 0.0,
            "danger_level": "CRITICAL",
            "ladder": ladder,
            "can_afford_full_ladder": False,
        }

    full_risk = ladder[-1]["cumulative_risk"]
    can_afford = balance >= full_risk
    # P(hit max_levels consecutive losses) = (1-wr)^max_levels
    loss_p = max(0.01, min(0.99, 1.0 - float(win_rate)))
    wipe_p = loss_p ** max_levels
    # If can't afford ladder, wipe risk higher
    if not can_afford:
        # Find first unaffordable level
        for row in ladder:
            if row["cumulative_risk"] > balance:
                wipe_p = loss_p ** (row["level"] - 1) if row["level"] > 1 else 1.0
                break
        survival = max(0.0, (1.0 - wipe_p) * 0.5)
    else:
        survival = (1.0 - wipe_p) * 100.0
        # scale to percent already
        survival = (1.0 - wipe_p) * 100.0

    ratio = full_risk / balance if balance else 9.0
    if ratio >= 0.5 or not can_afford:
        danger = "CRITICAL"
    elif ratio >= 0.25:
        danger = "HIGH"
    elif ratio >= 0.12:
        danger = "MEDIUM"
    else:
        danger = "LOW"

    return {
        "survival_pct": round(survival if survival <= 100 else survival, 1),
        "danger_level": danger,
        "ladder": ladder,
        "full_ladder_risk": round(full_risk, 2),
        "can_afford_full_ladder": can_afford,
        "wipe_probability": round(wipe_p, 4),
        "recommendation": (
            "Disable martingale — survival risk elevated"
            if danger in {"HIGH", "CRITICAL"}
            else "Ladder within soft limits — prefer flat stake"
        ),
    }
