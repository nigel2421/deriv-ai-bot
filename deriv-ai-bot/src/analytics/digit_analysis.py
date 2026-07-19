"""
Real-time digit frequency heatmaps, hot/cold digits, streaks, absence.
Windows: 100 / 500 / 1000 ticks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.strategy.digit_contracts import extract_last_digit, last_digits_from_ticks


def _counts(digits: Sequence[int]) -> Dict[int, int]:
    c = {d: 0 for d in range(10)}
    for d in digits:
        if 0 <= int(d) <= 9:
            c[int(d)] += 1
    return c


def _pct(counts: Dict[int, int], n: int) -> Dict[int, float]:
    if n <= 0:
        return {d: 0.0 for d in range(10)}
    return {d: round(100.0 * counts[d] / n, 2) for d in range(10)}


def digit_heatmap(
    ticks: Sequence[Dict[str, Any]], windows: Sequence[int] = (100, 500, 1000)
) -> Dict[str, Any]:
    """
    Frequency heatmap for multiple lookbacks.
    Returns per-window counts, percentages, hot/cold digits.
    """
    max_w = max(int(w) for w in windows) if windows else 1000
    digits = last_digits_from_ticks(ticks, n=max_w)
    out: Dict[str, Any] = {"n_available": len(digits), "windows": {}}
    for w in windows:
        w = int(w)
        sample = digits[-w:] if len(digits) >= w else digits
        n = len(sample)
        counts = _counts(sample)
        pct = _pct(counts, n)
        # Hot = above fair 10%; cold = below
        fair = 10.0
        ranked = sorted(range(10), key=lambda d: pct[d], reverse=True)
        hot = [d for d in ranked if pct[d] > fair + 1.0][:3]
        cold = [d for d in sorted(range(10), key=lambda d: pct[d]) if pct[d] < fair - 1.0][
            :3
        ]
        out["windows"][str(w)] = {
            "n": n,
            "counts": counts,
            "pct": pct,
            "hot": hot,
            "cold": cold,
            "table": [
                {"digit": d, "count": counts[d], "pct": pct[d]} for d in range(10)
            ],
        }
    return out


def consecutive_streaks(digits: Sequence[int]) -> Dict[str, Any]:
    """Current end-streak and max streaks per digit in window."""
    if not digits:
        return {"current_digit": None, "current_streak": 0, "max_by_digit": {}}
    last = int(digits[-1])
    streak = 1
    for d in reversed(digits[:-1]):
        if int(d) == last:
            streak += 1
        else:
            break
    max_by: Dict[int, int] = {d: 0 for d in range(10)}
    i = 0
    while i < len(digits):
        d = int(digits[i])
        j = i + 1
        while j < len(digits) and int(digits[j]) == d:
            j += 1
        max_by[d] = max(max_by[d], j - i)
        i = j
    return {
        "current_digit": last,
        "current_streak": streak,
        "max_by_digit": max_by,
    }


def ticks_since_last(digits: Sequence[int]) -> Dict[int, Optional[int]]:
    """How many ticks since each digit last appeared (0 = last tick)."""
    out: Dict[int, Optional[int]] = {d: None for d in range(10)}
    for i, d in enumerate(reversed(digits)):
        di = int(d)
        if out[di] is None:
            out[di] = i
        if all(v is not None for v in out.values()):
            break
    return out


def digit_snapshot(
    ticks: Sequence[Dict[str, Any]], *, primary_window: int = 100
) -> Dict[str, Any]:
    """Combined digit analysis for UI + trade filter."""
    digits = last_digits_from_ticks(ticks, n=max(1000, primary_window))
    heat = digit_heatmap(ticks, windows=(100, 500, 1000))
    streaks = consecutive_streaks(digits[-primary_window:] if digits else [])
    since = ticks_since_last(digits[-primary_window:] if digits else [])
    even_n = sum(1 for d in digits[-primary_window:] if d % 2 == 0)
    n = min(primary_window, len(digits))
    even_rate = even_n / n if n else 0.5
    return {
        "heatmap": heat,
        "streaks": streaks,
        "ticks_since": since,
        "even_rate": round(even_rate, 4),
        "odd_rate": round(1.0 - even_rate, 4),
        "last_digit": digits[-1] if digits else None,
        "n": len(digits),
        "recent": digits[-20:],
    }
