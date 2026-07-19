"""
Short-horizon trend + chart-tool analysis for Rise (CALL) / Fall (PUT).

Combines classic momentum with EMA/RSI/MACD/structure votes so only
aligned setups reach high confidence.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategy.chart_tools import chart_snapshot, quotes_from_ticks, rise_fall_vote

logger = logging.getLogger(__name__)


def analyze_trend(
    ticks: Sequence[Dict[str, Any]],
    *,
    lookback: int = 50,
    short_window: int = 8,
    long_window: int = 25,
) -> Dict[str, Any]:
    """
    Return trend diagnostics + preferred CALL/PUT signal.

    Confidence is only high when momentum and chart tools agree.
    """
    prices = quotes_from_ticks(ticks, n=max(lookback, long_window + 30, 100))
    empty = {
        "direction": "flat",
        "contract_type": None,
        "confidence": 0.0,
        "momentum_pct": 0.0,
        "slope": 0.0,
        "consistency": 0.0,
        "strength": 0.0,
        "n": len(prices),
        "chart": {},
        "tools": {},
    }
    if len(prices) < max(20, short_window + 2):
        return empty

    window = prices[-lookback:] if len(prices) >= lookback else prices
    n = len(window)
    first = window[0]
    last = window[-1]
    if first == 0:
        return empty

    momentum_pct = (last - first) / abs(first) * 100.0

    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(window) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, window))
    den = sum((x - mean_x) ** 2 for x in xs) or 1.0
    slope = num / den
    slope_norm = slope / (abs(mean_y) or 1.0) * 1000.0

    sw = min(short_window, n)
    lw = min(long_window, n)
    ma_s = sum(window[-sw:]) / sw
    ma_l = sum(window[-lw:]) / lw
    ma_diff = (ma_s - ma_l) / (abs(ma_l) or 1.0)

    ups = sum(1 for i in range(1, n) if window[i] > window[i - 1])
    downs = sum(1 for i in range(1, n) if window[i] < window[i - 1])
    moves = ups + downs
    if momentum_pct >= 0:
        consistency = ups / moves if moves else 0.5
    else:
        consistency = downs / moves if moves else 0.5

    third = max(3, n // 3)
    mid = window[-2 * third : -third] or window[:third]
    end = window[-third:]
    mid_avg = sum(mid) / len(mid)
    end_avg = sum(end) / len(end)
    accel = (end_avg - mid_avg) / (abs(mid_avg) or 1.0)

    mom_score = min(1.0, abs(momentum_pct) / 0.08)
    slope_score = min(1.0, abs(slope_norm) / 0.5)
    cons_score = max(0.0, (consistency - 0.5) * 2.0)
    ma_score = min(1.0, abs(ma_diff) * 80.0)
    accel_score = min(1.0, abs(accel) * 50.0)

    strength = (
        0.28 * mom_score
        + 0.18 * slope_score
        + 0.22 * cons_score
        + 0.17 * ma_score
        + 0.15 * accel_score
    )
    strength = max(0.0, min(1.0, strength))

    bullish = momentum_pct > 0 and ma_diff > 0 and slope >= 0
    bearish = momentum_pct < 0 and ma_diff < 0 and slope <= 0

    if bullish and strength >= 0.32:
        mom_type: Optional[str] = "CALL"
        direction = "up"
    elif bearish and strength >= 0.32:
        mom_type = "PUT"
        direction = "down"
    else:
        mom_type = None
        direction = "flat"
        strength *= 0.5

    mom_conf = (
        0.55 + strength * 0.40
        if mom_type
        else min(0.68, 0.40 + strength * 0.35)
    )
    if mom_type and consistency >= 0.65:
        mom_conf = min(0.95, mom_conf + 0.03)

    # Chart tools (EMA/RSI/MACD/structure)
    chart = chart_snapshot(ticks, n=100)
    tool_type, tool_conf, tool_detail = rise_fall_vote(chart)

    # Require agreement when both present; otherwise take stronger if very high
    contract_type: Optional[str] = None
    confidence = 0.0

    if mom_type and tool_type and mom_type == tool_type:
        # Agreement boosts confidence
        contract_type = mom_type
        confidence = min(0.97, 0.45 * mom_conf + 0.55 * tool_conf + 0.05)
        confidence = min(0.97, confidence + 0.04)  # alignment bonus
    elif mom_type and tool_type and mom_type != tool_type:
        # Conflict → no trade (protect win rate)
        contract_type = None
        confidence = min(mom_conf, tool_conf) * 0.6
        direction = "flat"
    elif tool_type and tool_conf >= 0.82:
        contract_type = tool_type
        confidence = tool_conf * 0.97
        direction = "up" if tool_type == "CALL" else "down"
    elif mom_type and mom_conf >= 0.85:
        contract_type = mom_type
        confidence = mom_conf * 0.95
    else:
        contract_type = None
        confidence = max(mom_conf, tool_conf) * 0.7

    result = {
        "direction": direction,
        "contract_type": contract_type,
        "confidence": float(round(confidence, 4)),
        "momentum_pct": float(round(momentum_pct, 5)),
        "slope": float(round(slope, 8)),
        "consistency": float(round(consistency, 4)),
        "strength": float(round(strength, 4)),
        "ma_diff": float(round(ma_diff, 6)),
        "n": n,
        "mom_type": mom_type,
        "mom_conf": float(round(mom_conf, 4)),
        "tool_type": tool_type,
        "tool_conf": float(round(tool_conf, 4)),
        "tools": tool_detail,
        "chart": {
            "rsi14": chart.get("rsi14"),
            "ema_bull": chart.get("ema_bull"),
            "ema_bear": chart.get("ema_bear"),
            "macd_hist": chart.get("macd_hist"),
            "structure": chart.get("structure"),
        },
    }
    logger.debug("Trend+tools %s", {k: result[k] for k in (
        "contract_type", "confidence", "mom_type", "tool_type", "direction"
    )})
    return result


def pick_rise_fall(
    ticks: Sequence[Dict[str, Any]], min_confidence: float = 0.80
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    t = analyze_trend(ticks)
    ct = t.get("contract_type")
    conf = float(t.get("confidence") or 0.0)
    if not ct or conf < min_confidence:
        return None, conf, t
    return str(ct), conf, t
