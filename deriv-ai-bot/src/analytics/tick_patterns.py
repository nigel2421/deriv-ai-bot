"""
Tick last-digit pattern detection: repeats, alternating, clustering.
Generates pattern alerts for the filter / copilot.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.strategy.digit_contracts import last_digits_from_ticks


def detect_repeat(digits: Sequence[int], min_len: int = 2) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    if len(digits) < min_len:
        return alerts
    # End streak
    last = int(digits[-1])
    streak = 1
    for d in reversed(digits[:-1]):
        if int(d) == last:
            streak += 1
        else:
            break
    if streak >= min_len:
        alerts.append(
            {
                "type": "repeat",
                "digit": last,
                "length": streak,
                "pattern": " ".join(str(last) for _ in range(streak)),
                "strength": min(100, 40 + streak * 15),
            }
        )
    return alerts


def detect_alternating(digits: Sequence[int], min_pairs: int = 3) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    if len(digits) < min_pairs * 2:
        return alerts
    # Check ABABAB at end
    a, b = int(digits[-2]), int(digits[-1])
    if a == b:
        return alerts
    pairs = 0
    i = len(digits) - 1
    expect = b
    while i >= 0:
        if int(digits[i]) != expect:
            break
        pairs += 1
        expect = a if expect == b else b
        i -= 1
        # After first, alternate a/b
        if pairs == 1:
            expect = a
    half = pairs // 1  # number of ticks in alternating run
    run = half
    if run >= min_pairs * 2 - 1:
        alerts.append(
            {
                "type": "alternating",
                "digits": [a, b],
                "length": run,
                "pattern": " ".join(str(digits[j]) for j in range(-run, 0)),
                "strength": min(100, 35 + run * 8),
            }
        )
    return alerts


def detect_clustering(digits: Sequence[int], window: int = 8) -> List[Dict[str, Any]]:
    """High concentration of a small digit set in recent window."""
    alerts: List[Dict[str, Any]] = []
    sample = list(digits[-window:]) if digits else []
    if len(sample) < 6:
        return alerts
    from collections import Counter

    c = Counter(sample)
    top = c.most_common(2)
    if not top:
        return alerts
    d1, n1 = top[0]
    share = n1 / len(sample)
    if share >= 0.45:
        alerts.append(
            {
                "type": "cluster",
                "digits": [d1] + ([top[1][0]] if len(top) > 1 else []),
                "share": round(share, 3),
                "window": len(sample),
                "pattern": " ".join(str(x) for x in sample),
                "strength": min(100, int(share * 120)),
            }
        )
    # High/low band clustering (8,9 or 0,1)
    hi = sum(1 for d in sample if d >= 7)
    lo = sum(1 for d in sample if d <= 2)
    if hi / len(sample) >= 0.55:
        alerts.append(
            {
                "type": "cluster_high",
                "share": round(hi / len(sample), 3),
                "window": len(sample),
                "pattern": " ".join(str(x) for x in sample),
                "strength": min(100, int(hi / len(sample) * 110)),
            }
        )
    if lo / len(sample) >= 0.55:
        alerts.append(
            {
                "type": "cluster_low",
                "share": round(lo / len(sample), 3),
                "window": len(sample),
                "pattern": " ".join(str(x) for x in sample),
                "strength": min(100, int(lo / len(sample) * 110)),
            }
        )
    return alerts


def detect_patterns(
    ticks: Sequence[Dict[str, Any]], lookback: int = 40
) -> Dict[str, Any]:
    digits = last_digits_from_ticks(ticks, n=lookback)
    alerts: List[Dict[str, Any]] = []
    alerts.extend(detect_repeat(digits, min_len=2))
    alerts.extend(detect_alternating(digits, min_pairs=3))
    alerts.extend(detect_clustering(digits, window=min(10, lookback)))
    max_s = max((a.get("strength") or 0) for a in alerts) if alerts else 0
    return {
        "digits": digits[-20:],
        "alerts": alerts,
        "pattern_alert_strength": max_s,
        "has_alert": bool(alerts),
    }
