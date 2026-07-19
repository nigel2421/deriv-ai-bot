"""
Digit-queue patterns inspired by community 'SureBet Queue' / Over-Under bots.

Examples:
  - last N digits all >= threshold → favor DIGITUNDER (mean reversion)
  - last N digits all <= threshold → favor DIGITOVER
  - last N same parity → fade (opposite EVEN/ODD)

Returns optional preferred_type + confidence boost, or None to leave AI path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategy.digit_contracts import clamp_digit, last_digits_from_ticks


def queue_signal(
    ticks: Sequence[Dict[str, Any]],
    *,
    lookback: int = 5,
) -> Optional[Dict[str, Any]]:
    digits = last_digits_from_ticks(ticks, n=max(lookback, 5))
    if len(digits) < lookback:
        return None
    tail = [clamp_digit(d) for d in digits[-lookback:]]

    # Queue high → under
    if all(d >= 7 for d in tail[-2:]):
        return {
            "preferred_type": "DIGITUNDER",
            "confidence_boost": 0.08,
            "reason": "queue_high_2",
            "hint_barrier": 7,  # under 7 style
        }
    if all(d >= 6 for d in tail[-3:]):
        return {
            "preferred_type": "DIGITUNDER",
            "confidence_boost": 0.06,
            "reason": "queue_high_3",
            "hint_barrier": 6,
        }

    # Queue low → over
    if all(d <= 2 for d in tail[-2:]):
        return {
            "preferred_type": "DIGITOVER",
            "confidence_boost": 0.08,
            "reason": "queue_low_2",
            "hint_barrier": 2,
        }
    if all(d <= 3 for d in tail[-3:]):
        return {
            "preferred_type": "DIGITOVER",
            "confidence_boost": 0.06,
            "reason": "queue_low_3",
            "hint_barrier": 3,
        }

    # Parity streak fade (4+ same) — contrarian even/odd
    last_even = tail[-1] % 2 == 0
    streak = 0
    for d in reversed(tail):
        if (d % 2 == 0) == last_even:
            streak += 1
        else:
            break
    if streak >= 4:
        return {
            "preferred_type": "DIGITODD" if last_even else "DIGITEVEN",
            "confidence_boost": 0.05,
            "reason": f"parity_fade_streak_{streak}",
        }

    return None
