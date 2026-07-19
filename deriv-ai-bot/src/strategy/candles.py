"""
Build OHLC candles from tick streams for minute-horizon trading.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _tick_epoch(t: Dict[str, Any]) -> Optional[int]:
    for k in ("epoch", "time", "ut"):
        if t.get(k) is not None:
            try:
                return int(t[k])
            except (TypeError, ValueError):
                pass
    return None


def _tick_quote(t: Dict[str, Any]) -> Optional[float]:
    q = t.get("quote", t.get("price"))
    try:
        return float(q) if q is not None else None
    except (TypeError, ValueError):
        return None


def build_candles(
    ticks: Sequence[Dict[str, Any]],
    *,
    period_sec: int = 60,
    max_candles: int = 80,
) -> List[Dict[str, Any]]:
    """
    Aggregate ticks into OHLC candles of period_sec seconds.
    Returns oldest → newest. Each candle: open, high, low, close, volume, start, end.
    """
    if period_sec <= 0 or not ticks:
        return []

    buckets: Dict[int, List[float]] = {}
    for t in ticks:
        if not isinstance(t, dict):
            continue
        ep = _tick_epoch(t)
        q = _tick_quote(t)
        if ep is None or q is None:
            continue
        bucket = (ep // period_sec) * period_sec
        buckets.setdefault(bucket, []).append(q)

    if not buckets:
        # No epochs — synthesize pseudo-time from order (1 tick ≈ 1s for R_*, 1s for 1HZ)
        prices: List[float] = []
        for t in ticks:
            if isinstance(t, dict):
                q = _tick_quote(t)
                if q is not None:
                    prices.append(q)
        if len(prices) < period_sec:
            return []
        # group every period_sec ticks
        candles = []
        for i in range(0, len(prices) - period_sec + 1, period_sec):
            chunk = prices[i : i + period_sec]
            if not chunk:
                continue
            candles.append(
                {
                    "open": chunk[0],
                    "high": max(chunk),
                    "low": min(chunk),
                    "close": chunk[-1],
                    "volume": len(chunk),
                    "start": i,
                    "end": i + len(chunk),
                }
            )
        return candles[-max_candles:]

    candles: List[Dict[str, Any]] = []
    for start in sorted(buckets.keys()):
        chunk = buckets[start]
        candles.append(
            {
                "open": chunk[0],
                "high": max(chunk),
                "low": min(chunk),
                "close": chunk[-1],
                "volume": len(chunk),
                "start": start,
                "end": start + period_sec,
            }
        )
    return candles[-max_candles:]


def candle_closes(candles: Sequence[Dict[str, Any]]) -> List[float]:
    return [float(c["close"]) for c in candles if c.get("close") is not None]


def is_bullish(c: Dict[str, Any]) -> bool:
    return float(c["close"]) > float(c["open"])


def is_bearish(c: Dict[str, Any]) -> bool:
    return float(c["close"]) < float(c["open"])
