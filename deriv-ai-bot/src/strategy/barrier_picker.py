"""
Adaptive digit barrier selection for DIGITOVER / DIGITUNDER.

Instead of fixed OVER@6 / UNDER@4, pick a barrier from recent last-digit
distribution + predicted digit so:
  - predicted digit (if any) stays in the win set when possible
  - empirical hit-rate from recent ticks is maximized (win probability)
  - mild randomization among near-best barriers avoids a single stuck level
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategy.digit_contracts import (
    clamp_digit,
    last_digits_from_ticks,
    normalize_barrier,
    normalize_contract_type,
)

logger = logging.getLogger(__name__)


def digit_histogram(ticks: Sequence[Dict[str, Any]], lookback: int = 40) -> List[int]:
    digits = last_digits_from_ticks(ticks, n=lookback)
    counts = [0] * 10
    for d in digits:
        counts[clamp_digit(d)] += 1
    return counts


def empirical_win_rate(contract_type: str, barrier: int, counts: Sequence[int]) -> float:
    n = sum(counts) or 1
    ct = normalize_contract_type(contract_type)
    if ct == "DIGITOVER":
        hits = sum(counts[d] for d in range(int(barrier) + 1, 10))
        return hits / n
    if ct == "DIGITUNDER":
        hits = sum(counts[d] for d in range(0, int(barrier)))
        return hits / n
    return 0.0


def adaptive_barrier(
    contract_type: str,
    *,
    predicted_digit: Optional[int] = None,
    ticks: Optional[Sequence[Dict[str, Any]]] = None,
    lookback: int = 40,
    mode: str = "adaptive",
    fixed_over: int = 6,
    fixed_under: int = 4,
    min_win_rate: float = 0.35,
    top_k: int = 3,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Choose barrier for OVER/UNDER.

    mode:
      - fixed: always fixed_over / fixed_under
      - adaptive: best empirical + prediction (default)
      - random: uniform random valid barrier (still prediction-aware if provided)
    """
    ct = normalize_contract_type(contract_type)
    if ct not in {"DIGITOVER", "DIGITUNDER"}:
        return None, {"mode": mode, "reason": "not_barrier_type"}

    if mode == "fixed":
        b = fixed_over if ct == "DIGITOVER" else fixed_under
        b = normalize_barrier(ct, b)
        return b, {"mode": "fixed", "barrier": b, "win_rate": None}

    counts = digit_histogram(ticks or [], lookback=lookback)
    n = sum(counts)
    pred = clamp_digit(predicted_digit) if predicted_digit is not None else None
    rnd = rng or random.Random()

    candidates: List[Tuple[float, int, float]] = []  # (score, barrier, emp_wr)

    if ct == "DIGITOVER":
        # barrier 0..8; win if digit > barrier
        for b in range(0, 9):
            emp = empirical_win_rate(ct, b, counts) if n else (9 - b) / 10.0
            if emp < min_win_rate and n >= 15:
                continue
            score = emp
            # Prefer barriers where prediction lands in win set
            if pred is not None:
                if pred > b:
                    score += 0.12
                else:
                    score -= 0.20  # would lose if pred correct
            # Prefer mid barriers: OVER@0 is ~90% win but tiny payout (~+$0.05–0.15)
            if 2 <= b <= 6:
                score += 0.05
            if b <= 1:
                score -= 0.25  # avoid near-certain / junk-payout barriers
            if b >= 7:
                score -= 0.08  # rare win, still ok if prediction strong
            candidates.append((score, b, emp))
    else:
        # UNDER barrier 1..9; win if digit < barrier
        for b in range(1, 10):
            emp = empirical_win_rate(ct, b, counts) if n else b / 10.0
            if emp < min_win_rate and n >= 15:
                continue
            score = emp
            if pred is not None:
                if pred < b:
                    score += 0.12
                else:
                    score -= 0.20
            if 3 <= b <= 7:
                score += 0.05
            if b >= 8:
                score -= 0.25  # UNDER@9 pays almost nothing
            if b <= 2:
                score -= 0.08
            candidates.append((score, b, emp))

    if not candidates:
        # Fallback: prediction-aligned or fixed defaults
        if pred is not None:
            if ct == "DIGITOVER":
                b = normalize_barrier(ct, max(0, pred - 1))
            else:
                b = normalize_barrier(ct, min(9, pred + 1))
        else:
            b = normalize_barrier(
                ct, fixed_over if ct == "DIGITOVER" else fixed_under
            )
        return b, {
            "mode": mode,
            "barrier": b,
            "win_rate": None,
            "reason": "fallback_empty",
        }

    candidates.sort(key=lambda x: x[0], reverse=True)

    if mode == "random":
        # Weighted random over all viable candidates
        weights = [max(0.01, c[0]) for c in candidates]
        pick = rnd.choices(candidates, weights=weights, k=1)[0]
    else:
        # adaptive: random among top-K near-best (diversity, not pure fixed)
        best_score = candidates[0][0]
        pool = [c for c in candidates if c[0] >= best_score - 0.06][: max(1, top_k)]
        pick = rnd.choice(pool)

    barrier = normalize_barrier(ct, pick[1])
    meta = {
        "mode": mode,
        "barrier": barrier,
        "win_rate": round(pick[2], 4),
        "score": round(pick[0], 4),
        "pred": pred,
        "samples": n,
        "top": [(round(s, 3), b, round(w, 3)) for s, b, w in candidates[:5]],
    }
    logger.debug("Barrier pick %s → %s meta=%s", ct, barrier, meta)
    return barrier, meta
