"""
Market regime filters — skip choppy / hostile conditions early.

Goals:
  - Avoid trading when price is whip-sawing (no clean trend)
  - Avoid oversized martingale in bad regimes
  - Flag "do not trade" before proposal/buy
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategy.chart_tools import atr_proxy, quotes_from_ticks, rsi


def _direction_flips(prices: Sequence[float]) -> int:
    if len(prices) < 3:
        return 0
    flips = 0
    prev = 0
    for i in range(1, len(prices)):
        d = 1 if prices[i] > prices[i - 1] else (-1 if prices[i] < prices[i - 1] else 0)
        if d == 0:
            continue
        if prev and d != prev:
            flips += 1
        prev = d
    return flips


def assess_regime(
    ticks: Sequence[Dict[str, Any]],
    *,
    lookback: int = 40,
) -> Dict[str, Any]:
    """
    Returns regime assessment:
      tradeable: bool
      reason: str
      chop_score: 0..1  (higher = choppier)
      volatility: atr proxy
    """
    prices = quotes_from_ticks(ticks, n=max(lookback + 10, 50))
    if len(prices) < 15:
        return {
            "tradeable": False,
            "reason": "insufficient_ticks",
            "chop_score": 1.0,
            "volatility": 0.0,
            "flips": 0,
            "consistency": 0.5,
        }

    window = prices[-lookback:]
    n = len(window)
    flips = _direction_flips(window)
    max_flips = max(1, n - 2)
    flip_rate = flips / max_flips

    ups = sum(1 for i in range(1, n) if window[i] > window[i - 1])
    downs = n - 1 - ups
    consistency = max(ups, downs) / max(1, n - 1)

    atr = atr_proxy(window, period=min(14, n - 1)) or 0.0
    mid = sum(window) / n
    vol_pct = (atr / abs(mid)) * 100.0 if mid else 0.0

    # Range-bound: large path length vs net move
    net = abs(window[-1] - window[0])
    path = sum(abs(window[i] - window[i - 1]) for i in range(1, n))
    efficiency = (net / path) if path > 0 else 0.0  # 1 = straight, 0 = pure chop

    rsi_v = rsi(window, 14)
    # Extreme RSI + high flip rate = exhaustion / chop edge

    chop_score = (
        0.40 * flip_rate
        + 0.35 * (1.0 - efficiency)
        + 0.25 * (1.0 - consistency)
    )
    chop_score = max(0.0, min(1.0, chop_score))

    tradeable = True
    reasons: List[str] = []

    if chop_score >= 0.62:
        tradeable = False
        reasons.append(f"choppy_{chop_score:.2f}")
    if efficiency < 0.12 and flip_rate > 0.45:
        tradeable = False
        reasons.append("whipsaw")
    if consistency < 0.52 and chop_score > 0.55:
        tradeable = False
        reasons.append("no_direction")
    # Very high short-term noise relative to level
    if vol_pct > 0.15 and efficiency < 0.18:
        tradeable = False
        reasons.append("noisy_vol")

    if not reasons:
        reasons.append("ok")

    return {
        "tradeable": tradeable,
        "reason": "+".join(reasons),
        "chop_score": float(round(chop_score, 4)),
        "volatility": float(round(vol_pct, 5)),
        "flips": flips,
        "consistency": float(round(consistency, 4)),
        "efficiency": float(round(efficiency, 4)),
        "rsi14": rsi_v,
    }


def should_skip_rise_fall(
    ticks: Sequence[Dict[str, Any]],
) -> Tuple[bool, str, Dict[str, Any]]:
    """Skip CALL/PUT when market is choppy."""
    reg = assess_regime(ticks)
    if not reg["tradeable"]:
        return True, reg["reason"], reg
    # Rise/Fall needs cleaner trend than digits
    if reg["chop_score"] >= 0.55:
        return True, f"rf_chop_{reg['chop_score']:.2f}", reg
    if reg["efficiency"] < 0.18:
        return True, "rf_low_efficiency", reg
    return False, "ok", reg


def should_skip_digits(
    ticks: Sequence[Dict[str, Any]],
) -> Tuple[bool, str, Dict[str, Any]]:
    """Digits can trade in mild chop; skip only extreme noise."""
    reg = assess_regime(ticks)
    if reg["reason"] == "insufficient_ticks":
        return True, reg["reason"], reg
    if reg["chop_score"] >= 0.78:
        return True, f"digit_extreme_chop_{reg['chop_score']:.2f}", reg
    return False, "ok", reg
