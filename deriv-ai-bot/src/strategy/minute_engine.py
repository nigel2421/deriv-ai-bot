"""
Minute-horizon Rise/Fall engine (CALL/PUT).

Uses OHLC candles built from ticks + EMA/RSI/candle patterns
(inspired by community XML bots: EMA 12&26, RSI, five-candle).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.strategy.candles import (
    build_candles,
    candle_closes,
    is_bearish,
    is_bullish,
)
from src.strategy.chart_tools import ema_series, rsi

logger = logging.getLogger(__name__)


def analyze_minute(
    ticks: Sequence[Dict[str, Any]],
    *,
    period_sec: int = 60,
    min_candles: int = 20,
    duration_minutes: int = 2,
    min_confidence: float = 0.78,
) -> Optional[Dict[str, Any]]:
    """
    Return trade intent fields for CALL/PUT on minute horizon, or None.

    Keys: contract_type, confidence, duration, duration_unit, family, details
    """
    candles = build_candles(ticks, period_sec=period_sec, max_candles=80)
    if len(candles) < min_candles:
        return None

    closes = candle_closes(candles)
    if len(closes) < 26:
        return None

    ema12 = ema_series(closes, 12)
    ema26 = ema_series(closes, 26)
    if not ema12 or not ema26:
        return None

    e12, e26 = ema12[-1], ema26[-1]
    e12_prev, e26_prev = ema12[-2], ema26[-2]
    rsi14 = rsi(closes, 14)

    last = candles[-1]
    prev = candles[-2]
    bull_stack = e12 > e26
    bear_stack = e12 < e26
    # Fresh cross
    golden = e12_prev <= e26_prev and e12 > e26
    death = e12_prev >= e26_prev and e12 < e26

    # Last 5 candles majority color (five-candle style)
    last5 = candles[-5:]
    bulls = sum(1 for c in last5 if is_bullish(c))
    bears = sum(1 for c in last5 if is_bearish(c))

    call_pts = 0.0
    put_pts = 0.0
    notes: List[str] = []

    if bull_stack:
        call_pts += 1.2
        notes.append("ema_bull")
    if bear_stack:
        put_pts += 1.2
        notes.append("ema_bear")
    if golden:
        call_pts += 1.0
        notes.append("golden_cross")
    if death:
        put_pts += 1.0
        notes.append("death_cross")

    if bulls >= 4:
        call_pts += 0.9
        notes.append(f"candles_bull_{bulls}")
    if bears >= 4:
        put_pts += 0.9
        notes.append(f"candles_bear_{bears}")

    if is_bullish(last) and is_bullish(prev):
        call_pts += 0.5
        notes.append("2green")
    if is_bearish(last) and is_bearish(prev):
        put_pts += 0.5
        notes.append("2red")

    if rsi14 is not None:
        if rsi14 >= 55:
            call_pts += 0.7 + min(0.4, (rsi14 - 55) / 50)
            notes.append(f"rsi={rsi14:.0f}")
        elif rsi14 <= 45:
            put_pts += 0.7 + min(0.4, (45 - rsi14) / 50)
            notes.append(f"rsi={rsi14:.0f}")
        if rsi14 >= 78:
            call_pts *= 0.7
            notes.append("rsi_ob")
        if rsi14 <= 22:
            put_pts *= 0.7
            notes.append("rsi_os")

    # Momentum over last 5 closes
    mom = (closes[-1] - closes[-5]) / abs(closes[-5] or 1.0)
    if mom > 0.0003:
        call_pts += 0.4
    elif mom < -0.0003:
        put_pts += 0.4

    total = call_pts + put_pts
    if total < 2.0:
        return None

    if call_pts > put_pts * 1.2:
        margin = (call_pts - put_pts) / total
        conf = 0.58 + 0.32 * (call_pts / total) + 0.12 * margin
        conf = min(0.95, conf)
        ct = "CALL"
    elif put_pts > call_pts * 1.2:
        margin = (put_pts - call_pts) / total
        conf = 0.58 + 0.32 * (put_pts / total) + 0.12 * margin
        conf = min(0.95, conf)
        ct = "PUT"
    else:
        return None

    if conf < min_confidence:
        return None

    return {
        "contract_type": ct,
        "confidence": float(round(conf, 4)),
        "duration": int(duration_minutes),
        "duration_unit": "m",
        "family": "minute_rise_fall",
        "horizon": "minute",
        "details": {
            "notes": notes,
            "call_pts": round(call_pts, 2),
            "put_pts": round(put_pts, 2),
            "rsi14": rsi14,
            "ema12": e12,
            "ema26": e26,
            "n_candles": len(candles),
            "period_sec": period_sec,
        },
    }
