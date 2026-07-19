"""
Hierarchical Clarity Model for professional Deriv analytics.

Level 1: Raw entropy metrics (digit, streak, odd/even, over/under, up/down)
Level 2: Reliability-weighted composite + momentum + stability
Level 3: Pattern Clarity (statistical separation + entropy clarity + …)

Weights are adaptive by contract type (DIGITDIFF vs EVEN/ODD vs OVER/UNDER).
All scores are explainable with contributor checklists.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from src.analytics.edge_score import sample_size_score_100
from src.analytics.pattern_clarity import (
    baseline_separation_score,
    clarity_class,
    simplicity_score,
)
from src.analytics.rolling_entropy import (
    HMAX_DIGITS,
    RollingEntropyEngine,
    compression_bias_label,
    feed_ticks,
    get_engine,
)


# ---------------------------------------------------------------------------
# Adaptive reliability weights by contract family
# ---------------------------------------------------------------------------

# Default: digit-first (generic digit contracts)
DEFAULT_ENTROPY_WEIGHTS = {
    "digit": 0.40,
    "streak": 0.25,
    "odd_even": 0.15,
    "over_under": 0.10,
    "up_down": 0.10,
}

CONTRACT_ENTROPY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "DIGITDIFF": {
        "digit": 0.55,
        "streak": 0.25,
        "odd_even": 0.08,
        "over_under": 0.07,
        "up_down": 0.05,
    },
    "DIGITMATCH": {
        "digit": 0.45,
        "streak": 0.35,
        "odd_even": 0.08,
        "over_under": 0.07,
        "up_down": 0.05,
    },
    "DIGITEVEN": {
        "odd_even": 0.60,
        "digit": 0.20,
        "streak": 0.10,
        "over_under": 0.05,
        "up_down": 0.05,
    },
    "DIGITODD": {
        "odd_even": 0.60,
        "digit": 0.20,
        "streak": 0.10,
        "over_under": 0.05,
        "up_down": 0.05,
    },
    "DIGITOVER": {
        "over_under": 0.50,
        "digit": 0.25,
        "streak": 0.10,
        "odd_even": 0.10,
        "up_down": 0.05,
    },
    "DIGITUNDER": {
        "over_under": 0.50,
        "digit": 0.25,
        "streak": 0.10,
        "odd_even": 0.10,
        "up_down": 0.05,
    },
    # Rise/Fall: directional, not digit-heavy
    "CALL": {
        "up_down": 0.55,
        "streak": 0.25,
        "digit": 0.08,
        "odd_even": 0.06,
        "over_under": 0.06,
    },
    "PUT": {
        "up_down": 0.55,
        "streak": 0.25,
        "digit": 0.08,
        "odd_even": 0.06,
        "over_under": 0.06,
    },
}


def weights_for_contract(contract_type: str = "") -> Dict[str, float]:
    ct = str(contract_type or "").upper()
    w = dict(CONTRACT_ENTROPY_WEIGHTS.get(ct) or DEFAULT_ENTROPY_WEIGHTS)
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}


def confidence_label(score: float) -> str:
    s = float(score)
    if s >= 80:
        return "HIGH"
    if s >= 60:
        return "MEDIUM"
    return "LOW"


# Per-symbol history of composite entropy for stability (last 20 readings)
_composite_history: Dict[str, Deque[float]] = {}


def _push_history(symbol: str, value: float, maxlen: int = 20) -> List[float]:
    if symbol not in _composite_history:
        _composite_history[symbol] = deque(maxlen=maxlen)
    _composite_history[symbol].append(float(value))
    return list(_composite_history[symbol])


def stability_from_series(values: Sequence[float]) -> Tuple[float, float]:
    """
    Low variance of last N entropy/composite readings → high stability.
    Stable ~90, chaotic ~40.
    """
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 3:
        return 55.0, 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    # std 0 → 100, 5 → ~80, 15 → ~50, 30 → ~20
    if std <= 2:
        score = 100.0 - std * 5.0
    elif std <= 8:
        score = 90.0 - (std - 2) * 3.0
    elif std <= 20:
        score = 72.0 - (std - 8) * 2.5
    else:
        score = max(15.0, 42.0 - (std - 20) * 1.5)
    return round(min(100.0, max(0.0, score)), 1), round(std, 3)


def composite_entropy_weighted(
    subscores: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Level 2: weighted sum of Level-1 entropy subscores.

    Default:
      Digit 40% + Streak 25% + OddEven 15% + OverUnder 10% + UpDown 10%
    """
    total = 0.0
    contrib: Dict[str, float] = {}
    for key, w in weights.items():
        val = float(subscores.get(key) or 0.0)
        c = w * val
        contrib[key] = round(c, 2)
        total += c
    return round(total, 1), contrib


def entropy_clarity_engine(
    composite: float,
    momentum: float,
    stability: float,
) -> Dict[str, Any]:
    """
    Level 2→3 entropy clarity:
      60% Composite Entropy + 25% Momentum + 15% Stability
    """
    c = float(composite)
    m = float(momentum)
    s = float(stability)
    total = 0.60 * c + 0.25 * m + 0.15 * s
    total = max(0.0, min(100.0, total))
    return {
        "entropy_clarity": round(total, 1),
        "components": {
            "composite": round(c, 1),
            "momentum": round(m, 1),
            "stability": round(s, 1),
        },
        "weights": {"composite": 0.60, "momentum": 0.25, "stability": 0.15},
        "contributions": {
            "composite": round(0.60 * c, 1),
            "momentum": round(0.25 * m, 1),
            "stability": round(0.15 * s, 1),
        },
    }


def final_pattern_clarity(
    *,
    statistical_separation: float,
    entropy_clarity: float,
    stability: float,
    sample_size_score: float,
    simplicity: float = 85.0,
) -> Dict[str, Any]:
    """
    Final Pattern Clarity:
      40% Statistical Separation
    + 30% Entropy Clarity
    + 15% Stability
    + 10% Sample Size
    +  5% Pattern Simplicity
    """
    sep = float(statistical_separation)
    ent = float(entropy_clarity)
    stab = float(stability)
    samp = float(sample_size_score)
    simp = float(simplicity)
    total = (
        0.40 * sep
        + 0.30 * ent
        + 0.15 * stab
        + 0.10 * samp
        + 0.05 * simp
    )
    total = max(0.0, min(100.0, total))
    return {
        "pattern_clarity": round(total, 1),
        "class": clarity_class(total),
        "auto_ok": total >= 80.0,
        "weights": {
            "statistical_separation": 0.40,
            "entropy_clarity": 0.30,
            "stability": 0.15,
            "sample_size": 0.10,
            "simplicity": 0.05,
        },
        "contributions": {
            "statistical_separation": round(0.40 * sep, 1),
            "entropy_clarity": round(0.30 * ent, 1),
            "stability": round(0.15 * stab, 1),
            "sample_size": round(0.10 * samp, 1),
            "simplicity": round(0.05 * simp, 1),
        },
        "inputs": {
            "statistical_separation": round(sep, 1),
            "entropy_clarity": round(ent, 1),
            "stability": round(stab, 1),
            "sample_size": round(samp, 1),
            "simplicity": round(simp, 1),
        },
    }


def build_hierarchical_clarity(
    ticks: Sequence[Dict[str, Any]],
    *,
    symbol: str = "_default",
    contract_type: str = "",
    pattern_wr: float = 0.5,
    baseline_wr: float = 0.5,
    sample_n: int = 0,
    n_conditions: int = 2,
) -> Dict[str, Any]:
    """
    Full hierarchical stack from rolling entropy → final pattern clarity.

    Returns explainable contributors + regime + confidence.
    """
    # Level 1: feed rolling engine
    roll = feed_ticks(symbol, list(ticks)[-500:] if ticks else [])
    multi = roll.get("multi") or {}
    primary = roll.get("primary") or {}

    # Level-1 subscores (0–100)
    digit_score = float(multi.get("digit") or roll.get("compression_score") or 0)
    streak_score = float(multi.get("streak") or (roll.get("streak") or {}).get("streak_score") or 0)
    odd_even_score = float(multi.get("odd_even") or 0)
    over_under_score = float(multi.get("over_under") or 0)
    up_down_score = float(multi.get("up_down") or 0)

    level1 = {
        "digit": round(digit_score, 1),
        "streak": round(streak_score, 1),
        "odd_even": round(odd_even_score, 1),
        "over_under": round(over_under_score, 1),
        "up_down": round(up_down_score, 1),
    }

    # Adaptive weights for contract
    weights = weights_for_contract(contract_type)
    composite, composite_contrib = composite_entropy_weighted(level1, weights)

    # Momentum from rolling engine
    momentum = float(roll.get("momentum_score") or primary.get("momentum_score") or 50)

    # Stability from last 20 composite readings
    hist = _push_history(symbol, composite, maxlen=20)
    stability, stab_std = stability_from_series(hist)

    # Level 2: Entropy Clarity Engine
    ent_clarity = entropy_clarity_engine(composite, momentum, stability)

    # Statistical separation (baseline edge)
    sep_score, improvement_pp = baseline_separation_score(pattern_wr, baseline_wr)

    # Sample + simplicity
    samp = sample_size_score_100(int(sample_n))
    simp = simplicity_score(n_conditions=n_conditions, explainable=True)

    # Level 3: Final Pattern Clarity
    final = final_pattern_clarity(
        statistical_separation=sep_score,
        entropy_clarity=float(ent_clarity["entropy_clarity"]),
        stability=stability,
        sample_size_score=samp,
        simplicity=simp,
    )

    regime = roll.get("regime") or primary.get("regime") or "RANDOM"
    conf_score = min(
        100.0,
        0.45 * final["pattern_clarity"]
        + 0.30 * float(ent_clarity["entropy_clarity"])
        + 0.25 * (100.0 if sample_n >= 500 else min(100.0, sample_n / 5.0)),
    )
    conf_lab = confidence_label(conf_score)

    # Explainable contributors
    contributors = [
        {
            "name": "Digit Bias Strength",
            "score": level1["digit"],
            "ok": level1["digit"] >= 65,
        },
        {
            "name": "Streak Compression",
            "score": level1["streak"],
            "ok": level1["streak"] >= 65,
        },
        {
            "name": "Odd/Even Entropy",
            "score": level1["odd_even"],
            "ok": level1["odd_even"] >= 55,
        },
        {
            "name": "Over/Under Entropy",
            "score": level1["over_under"],
            "ok": level1["over_under"] >= 55,
        },
        {
            "name": "Up/Down Entropy",
            "score": level1["up_down"],
            "ok": level1["up_down"] >= 50,
        },
        {
            "name": "Entropy Momentum",
            "score": round(momentum, 1),
            "ok": momentum >= 65,
        },
        {
            "name": "Entropy Stability",
            "score": stability,
            "ok": stability >= 60,
        },
        {
            "name": "Statistical Separation",
            "score": round(sep_score, 1),
            "ok": sep_score >= 60,
        },
        {
            "name": "Sample Confidence",
            "score": round(samp, 1),
            "ok": samp >= 45,
        },
    ]

    reasons = [
        f"Pattern Clarity: {final['pattern_clarity']} ({final['class']})",
        "Contributors:",
    ]
    for c in contributors:
        mark = "✓" if c["ok"] else "✗"
        reasons.append(f"{mark} {c['name']:28s} {c['score']}")
    reasons.append(f"Market Regime: {regime}")
    reasons.append(f"Confidence: {conf_lab} ({conf_score:.0f})")
    reasons.append(
        f"Composite entropy {composite} "
        f"(weights for {contract_type or 'default'}: "
        + ", ".join(f"{k}={w:.0%}" for k, w in weights.items())
        + ")"
    )
    reasons.append(
        f"Entropy clarity {ent_clarity['entropy_clarity']} = "
        f"60%×{composite} + 25%×{momentum:.0f} + 15%×{stability}"
    )
    reasons.append(
        f"Final blend: 40% sep {sep_score:.0f} · 30% ent {ent_clarity['entropy_clarity']:.0f} "
        f"· 15% stab {stability:.0f} · 10% n {samp:.0f} · 5% simp {simp:.0f}"
    )
    if improvement_pp is not None:
        reasons.append(f"Baseline separation +{improvement_pp:.1f}pp vs fair odds")

    return {
        "pattern_clarity": final["pattern_clarity"],
        "class": final["class"],
        "auto_ok": final["auto_ok"],
        "formula": "hierarchical",
        "level1_raw": level1,
        "level2_composite": {
            "composite_entropy": composite,
            "weights": weights,
            "contributions": composite_contrib,
            "momentum": round(momentum, 1),
            "stability": stability,
            "stability_stdev": stab_std,
            "entropy_clarity": ent_clarity,
        },
        "level3_final": final,
        "contributors": contributors,
        "regime": regime,
        "confidence": conf_lab,
        "confidence_score": round(conf_score, 1),
        "rolling": roll,
        "reasons": reasons,
        "explain": reasons,
        "display": {
            "pattern_clarity": final["pattern_clarity"],
            "contributors": [
                f"{'✓' if c['ok'] else '✗'} {c['name']} {c['score']}"
                for c in contributors
                if c["ok"] or c["score"] >= 50
            ],
            "market_regime": regime,
            "confidence": conf_lab,
        },
        # aliases for trade_filter / pattern_clarity consumers
        "components": {
            "separation": sep_score,
            "entropy_clarity": ent_clarity["entropy_clarity"],
            "stability": stability,
            "sample_size": samp,
            "simplicity": simp,
            "composite_entropy": composite,
            "momentum": momentum,
            **level1,
        },
        "entropy_strength_detail": {
            "entropy_strength": ent_clarity["entropy_clarity"],
            "regime": regime,
            "rolling": roll,
            "display": roll.get("display"),
        },
    }


def hierarchical_from_rolling(
    roll: Dict[str, Any],
    *,
    symbol: str = "_default",
    contract_type: str = "",
    pattern_wr: float = 0.5,
    baseline_wr: float = 0.5,
    sample_n: int = 0,
) -> Dict[str, Any]:
    """
    Same hierarchy using an existing rolling snapshot (no re-feed).
    Useful when snapshot() was just computed.
    """
    # Temporarily stash multi into a synthetic path via rebuild minimal
    multi = roll.get("multi") or {}
    if not multi and roll.get("compression_score") is not None:
        multi = {
            "digit": roll.get("compression_score"),
            "streak": (roll.get("streak") or {}).get("streak_score") or 30,
            "odd_even": 40,
            "over_under": 40,
            "up_down": 40,
        }
    # Reuse build by faking feed - call internals
    level1 = {
        "digit": float(multi.get("digit") or 0),
        "streak": float(multi.get("streak") or 0),
        "odd_even": float(multi.get("odd_even") or 0),
        "over_under": float(multi.get("over_under") or 0),
        "up_down": float(multi.get("up_down") or 0),
    }
    weights = weights_for_contract(contract_type)
    composite, contrib = composite_entropy_weighted(level1, weights)
    momentum = float(roll.get("momentum_score") or 50)
    hist = _push_history(f"{symbol}|snap", composite)
    stability, _ = stability_from_series(hist)
    ent_c = entropy_clarity_engine(composite, momentum, stability)
    sep, _ = baseline_separation_score(pattern_wr, baseline_wr)
    samp = sample_size_score_100(sample_n)
    final = final_pattern_clarity(
        statistical_separation=sep,
        entropy_clarity=float(ent_c["entropy_clarity"]),
        stability=stability,
        sample_size_score=samp,
        simplicity=85.0,
    )
    return {
        "pattern_clarity": final["pattern_clarity"],
        "class": final["class"],
        "level1_raw": level1,
        "composite_entropy": composite,
        "entropy_clarity": ent_c,
        "final": final,
        "weights": weights,
        "regime": roll.get("regime"),
    }
