"""
Pattern Clarity — how distinct and non-random is this pattern vs market noise?

For Deriv digit trading: clearer patterns imply temporary statistical imbalance
rather than random fluctuation.

Production formula:
  35% Baseline Separation
+ 25% Stability
+ 15% Sample Size
+ 15% Entropy Strength
+ 10% Context Alignment

Entropy Strength =
  50% Compression  (1 − H/Hmax)
+ 50% Entropy Momentum  (long-window H − short-window H)

Composite entropy (richer than digits alone):
  40% Digit + 25% Odd/Even + 20% Up/Down + 15% Streak
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategy.digit_contracts import last_digits_from_ticks
from src.strategy.chart_tools import quotes_from_ticks


# Uniform digit entropy: −log2(0.1) = log2(10) ≈ 3.3219 bits
HMAX_DIGITS = math.log2(10.0)
# Binary sequences (even/odd, up/down, streak-continue): max = 1 bit
HMAX_BINARY = 1.0


def clarity_class(score: float) -> str:
    """
    0–49 Noise · 50–64 Weak · 65–74 Moderate · 75–84 Clear · 85–100 Exceptional
    """
    s = float(score)
    if s >= 85:
        return "Exceptional"
    if s >= 75:
        return "Clear"
    if s >= 65:
        return "Moderate"
    if s >= 50:
        return "Weak"
    return "Noise"


def rarity_score(frequency: float) -> float:
    """
    Rarity = 100 × (1 − Frequency)

    frequency in [0, 1]: how often the pattern fires (e.g. 0.20 → score 80).
    Occurs constantly → weak; rare → strong.
    """
    f = max(0.0, min(1.0, float(frequency)))
    return round(100.0 * (1.0 - f), 1)


def baseline_separation_score(
    pattern_wr: float,
    baseline_wr: float = 0.50,
) -> Tuple[float, float]:
    """
    How much better than random / baseline?

    Improvement = pattern_wr - baseline_wr (e.g. 62% - 50% = 12%)

      0–2% → 10, 2–5% → 30, 5–8% → 60, 8–12% → 80, 12%+ → 100
    """
    p = max(0.0, min(1.0, float(pattern_wr)))
    b = max(0.0, min(1.0, float(baseline_wr)))
    improvement = (p - b) * 100.0  # percentage points

    if improvement < 0:
        # Worse than baseline — clarity collapses
        score = max(0.0, 10.0 + improvement * 2.0)
    elif improvement < 2:
        score = 10.0 + (improvement / 2.0) * 0.0  # stay ~10
        score = 10.0
    elif improvement < 5:
        score = 10.0 + (improvement - 2.0) / 3.0 * 20.0  # → 30
    elif improvement < 8:
        score = 30.0 + (improvement - 5.0) / 3.0 * 30.0  # → 60
    elif improvement < 12:
        score = 60.0 + (improvement - 8.0) / 4.0 * 20.0  # → 80
    else:
        score = 80.0 + min(20.0, (improvement - 12.0) * 2.0)  # → 100

    return round(min(100.0, max(0.0, score)), 1), round(improvement, 2)


def stability_score_from_windows(
    win_rates: Sequence[float],
) -> Tuple[float, float]:
    """
    Lower variance of WR across windows → higher clarity.

    Example stable 61/63/60 → high; 75/42/58 → low.
    Returns (score_0_100, stdev_of_wr_as_fraction).
    """
    rates = [max(0.0, min(1.0, float(x))) for x in win_rates if x is not None]
    if len(rates) < 2:
        return 50.0, 0.0
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / len(rates)
    std = math.sqrt(var)

    # std 0 → 100, 0.05 → ~70, 0.10 → ~40, 0.20 → ~10
    if std <= 0.02:
        score = 100.0 - std / 0.02 * 10.0  # 100–90
    elif std <= 0.05:
        score = 90.0 - (std - 0.02) / 0.03 * 20.0  # 90–70
    elif std <= 0.10:
        score = 70.0 - (std - 0.05) / 0.05 * 30.0  # 70–40
    elif std <= 0.20:
        score = 40.0 - (std - 0.10) / 0.10 * 30.0  # 40–10
    else:
        score = max(0.0, 10.0 - (std - 0.20) * 50.0)

    return round(min(100.0, max(0.0, score)), 1), round(std, 4)


def stability_from_trade_rows(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """WR on last 100 / 500 / 1000 windows → stability score."""

    def _wr(chunk: Sequence[Dict[str, Any]]) -> Optional[float]:
        if not chunk:
            return None
        wins = 0
        n = 0
        for r in chunk:
            n += 1
            try:
                p = float(r.get("profit")) if r.get("profit") is not None else None
            except (TypeError, ValueError):
                p = None
            if p is not None:
                if p > 0:
                    wins += 1
            elif r.get("is_win") is True or str(r.get("status") or "").lower() == "win":
                wins += 1
        return wins / n if n else None

    rows = list(rows or [])
    w100 = _wr(rows[-100:])
    w500 = _wr(rows[-500:])
    w1000 = _wr(rows[-1000:])
    rates = [x for x in (w100, w500, w1000) if x is not None]
    score, std = stability_score_from_windows(rates)
    return score, {
        "wr_100": round(w100, 4) if w100 is not None else None,
        "wr_500": round(w500, 4) if w500 is not None else None,
        "wr_1000": round(w1000, 4) if w1000 is not None else None,
        "stdev": std,
    }


def context_alignment_score(confirmations: int) -> float:
    """
    Multiple signals pointing same direction.

      1 → 30, 2 → 60, 3 → 80, 4+ → 100
    """
    c = max(0, int(confirmations))
    if c <= 0:
        return 0.0
    if c == 1:
        return 30.0
    if c == 2:
        return 60.0
    if c == 3:
        return 80.0
    return 100.0


def simplicity_score(
    *,
    n_conditions: int = 2,
    explainable: bool = True,
) -> float:
    """
    Complex multi-condition rules overfit.

    Prefer 1–3 simple, explainable rules over 5+ brittle filters.
    """
    n = max(0, int(n_conditions))
    if n <= 0:
        return 40.0
    if n == 1:
        base = 85.0
    elif n == 2:
        base = 95.0
    elif n == 3:
        base = 80.0
    elif n == 4:
        base = 55.0
    else:
        base = max(15.0, 50.0 - (n - 4) * 10.0)
    if not explainable:
        base *= 0.7
    return round(min(100.0, max(0.0, base)), 1)


def shannon_entropy(probs: Sequence[float]) -> float:
    """H = −Σ p(x) × log2(p(x))  (skip zeros)."""
    h = 0.0
    for p in probs:
        if p and float(p) > 0:
            pp = float(p)
            h -= pp * math.log2(pp)
    return h


def entropy_from_counts(counts: Sequence[int]) -> float:
    """Shannon entropy from raw counts (JS-compatible)."""
    total = sum(int(c) for c in counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = int(c) / total
        h -= p * math.log2(p)
    return h


def digit_counts(digits: Sequence[int]) -> List[int]:
    counts = [0] * 10
    for d in digits:
        if 0 <= int(d) <= 9:
            counts[int(d)] += 1
    return counts


def compression_from_h(h: float, hmax: float) -> Dict[str, float]:
    """
    Compression = 1 − (H / Hmax)
    Clarity    = 100 × (Hmax − H) / Hmax  = 100 × Compression

    Bias bands on compression %:
      0–5 normal · 5–10 mild · 10–20 strong · 20–30 rare · 30+ exceptional
    """
    hmax = float(hmax) if hmax > 0 else 1.0
    h = max(0.0, float(h))
    # Cap H at Hmax for numerical noise
    h_eff = min(h, hmax)
    compression = 1.0 - (h_eff / hmax)
    compression = max(0.0, min(1.0, compression))
    clarity = 100.0 * compression
    return {
        "h": round(h, 4),
        "hmax": round(hmax, 4),
        "entropy_loss": round(hmax - h_eff, 4),
        "compression": round(compression, 4),
        "compression_pct": round(compression * 100.0, 2),
        "entropy_clarity": round(clarity, 1),
    }


def bias_label(compression_pct: float) -> str:
    c = float(compression_pct)
    if c >= 30:
        return "Exceptional Pattern"
    if c >= 20:
        return "Rare Pattern"
    if c >= 10:
        return "Strong Bias"
    if c >= 5:
        return "Mild Bias"
    return "Normal"


def entropy_clarity_from_digits(
    digits: Sequence[int],
) -> Dict[str, Any]:
    """
    Digit-frequency Shannon entropy vs Hmax = log2(10).

    Clarity = 100 × (Hmax − H) / Hmax
    """
    if not digits:
        base = compression_from_h(HMAX_DIGITS, HMAX_DIGITS)
        return {
            **base,
            "entropy": base["h"],
            "normal_entropy": HMAX_DIGITS,
            "probs": [0.1] * 10,
            "counts": [0] * 10,
            "bias_label": "Normal",
        }
    counts = digit_counts(digits)
    n = sum(counts) or 1
    probs = [c / n for c in counts]
    h = shannon_entropy(probs)
    base = compression_from_h(h, HMAX_DIGITS)
    return {
        **base,
        "entropy": base["h"],
        "normal_entropy": round(HMAX_DIGITS, 4),
        "probs": [round(p, 4) for p in probs],
        "counts": counts,
        "bias_label": bias_label(base["compression_pct"]),
    }


def entropy_clarity_from_ticks(
    ticks: Sequence[Dict[str, Any]], n: int = 100
) -> Dict[str, Any]:
    digits = last_digits_from_ticks(ticks, n=n)
    return entropy_clarity_from_digits(digits)


def binary_entropy_from_bits(bits: Sequence[int]) -> Dict[str, Any]:
    """Entropy of a binary sequence (0/1), Hmax = 1."""
    if not bits:
        return compression_from_h(HMAX_BINARY, HMAX_BINARY)
    n0 = sum(1 for b in bits if int(b) == 0)
    n1 = len(bits) - n0
    h = entropy_from_counts([n0, n1])
    return compression_from_h(h, HMAX_BINARY)


def odd_even_bits(digits: Sequence[int]) -> List[int]:
    return [1 if int(d) % 2 else 0 for d in digits]


def over_under_bits(digits: Sequence[int], barrier: int = 4) -> List[int]:
    """1 = over barrier, 0 = under/equal (binary stream for entropy)."""
    b = int(barrier)
    return [1 if int(d) > b else 0 for d in digits]


def up_down_bits(prices: Sequence[float]) -> List[int]:
    """1 = up tick, 0 = down/flat."""
    bits: List[int] = []
    for i in range(1, len(prices)):
        bits.append(1 if prices[i] > prices[i - 1] else 0)
    return bits


def streak_bits(digits: Sequence[int]) -> List[int]:
    """1 = digit repeats previous, 0 = changes (repeat-streak process)."""
    if len(digits) < 2:
        return []
    bits = []
    for i in range(1, len(digits)):
        bits.append(1 if int(digits[i]) == int(digits[i - 1]) else 0)
    return bits


def sliding_window_entropy(
    digits: Sequence[int],
    windows: Sequence[int] = (50, 100, 200, 500),
) -> Dict[str, Any]:
    """
    Entropy on last N ticks for each window.
    Lower recent H vs longer H → emerging pattern (entropy momentum).
    """
    rows = []
    by_w: Dict[str, float] = {}
    for w in windows:
        w = int(w)
        sample = list(digits[-w:]) if len(digits) >= 1 else []
        if len(sample) < max(10, w // 5):
            h = HMAX_DIGITS
        else:
            h = entropy_from_counts(digit_counts(sample))
        by_w[str(w)] = round(h, 4)
        comp = compression_from_h(h, HMAX_DIGITS)
        rows.append(
            {
                "window": w,
                "entropy": round(h, 4),
                "compression_pct": comp["compression_pct"],
                "n": len(sample),
            }
        )

    # Momentum = long-term H − short-term H (positive → fresh imbalance)
    h_short = by_w.get("50") or by_w.get("100")
    h_long = by_w.get("500") or by_w.get("200")
    if h_short is None or h_long is None:
        # fallback first/last
        if len(rows) >= 2:
            h_short = rows[0]["entropy"]
            h_long = rows[-1]["entropy"]
        else:
            h_short = h_long = HMAX_DIGITS
    momentum_bits = float(h_long) - float(h_short)
    # Normalize momentum to 0–100: 0 → 50, +0.5 → ~90, −0.3 → ~20
    # Max useful gap ~ Hmax
    mom_score = 50.0 + (momentum_bits / HMAX_DIGITS) * 100.0
    mom_score = max(0.0, min(100.0, mom_score))

    return {
        "windows": rows,
        "by_window": by_w,
        "h_short": round(float(h_short), 4),
        "h_long": round(float(h_long), 4),
        "entropy_momentum_bits": round(momentum_bits, 4),
        "momentum_score": round(mom_score, 1),
        "fresh_pattern": momentum_bits >= 0.15,
    }


def composite_entropy_score(
    ticks: Sequence[Dict[str, Any]],
    *,
    lookback: int = 200,
    over_barrier: int = 4,
) -> Dict[str, Any]:
    """
    Richer pattern clarity than digit frequencies alone:

      Composite =
        40% Digit Entropy compression
      + 25% Odd/Even Entropy compression
      + 20% Up/Down Entropy compression
      + 15% Streak Entropy compression
    """
    digits = last_digits_from_ticks(ticks, n=lookback)
    prices = quotes_from_ticks(ticks, n=lookback)

    dig = entropy_clarity_from_digits(digits)
    oe = binary_entropy_from_bits(odd_even_bits(digits))
    ou = binary_entropy_from_bits(over_under_bits(digits, over_barrier))
    ud = binary_entropy_from_bits(up_down_bits(prices))
    st = binary_entropy_from_bits(streak_bits(digits))

    # Prefer up/down for "Up/Down sequences"; fold over/under into digit family
    dig_c = float(dig.get("entropy_clarity") or 0)
    oe_c = float(oe.get("entropy_clarity") or 0)
    ud_c = float(ud.get("entropy_clarity") or 0)
    st_c = float(st.get("entropy_clarity") or 0)
    # Spec weights: 40 digit + 25 O/E + 20 U/D + 15 streak
    # Also note over/under as diagnostic (not in composite weights)
    composite = 0.40 * dig_c + 0.25 * oe_c + 0.20 * ud_c + 0.15 * st_c

    return {
        "composite_entropy_score": round(composite, 1),
        "components": {
            "digit": round(dig_c, 1),
            "odd_even": round(oe_c, 1),
            "up_down": round(ud_c, 1),
            "streak": round(st_c, 1),
            "over_under": round(float(ou.get("entropy_clarity") or 0), 1),
        },
        "weights": {
            "digit": 0.40,
            "odd_even": 0.25,
            "up_down": 0.20,
            "streak": 0.15,
        },
        "digit_detail": dig,
        "odd_even_detail": oe,
        "up_down_detail": ud,
        "streak_detail": st,
        "over_under_detail": ou,
    }


def entropy_strength(
    ticks: Sequence[Dict[str, Any]],
    *,
    lookback: int = 500,
    compression_weight: float = 0.50,
    momentum_weight: float = 0.50,
    use_composite: bool = True,
    symbol: str = "_default",
    use_rolling_engine: bool = True,
) -> Dict[str, Any]:
    """
    EntropyStrength = w_c × CompressionScore + w_m × MomentumScore

    Prefer the real-time RollingEntropyEngine when available so scores
    update every tick rather than a one-shot history scan.
    """
    if use_rolling_engine and ticks:
        try:
            from src.analytics.rolling_entropy import feed_ticks

            roll = feed_ticks(symbol, list(ticks)[-max(lookback, 500) :])
            if roll.get("ready"):
                comp_s = float(roll.get("compression_score") or 0)
                mom_s = float(roll.get("momentum_score") or 50)
                # Prefer realtime pattern strength as entropy strength proxy
                rt = float(roll.get("realtime_pattern_strength") or 0)
                strength = (
                    0.55 * rt
                    + 0.25 * comp_s
                    + 0.20 * mom_s
                )
                dig100 = {
                    "entropy": (roll.get("primary") or {}).get("entropy"),
                    "hmax": HMAX_DIGITS,
                    "compression_pct": (roll.get("primary") or {}).get(
                        "compression_pct"
                    ),
                    "bias_label": (roll.get("primary") or {}).get("bias_label"),
                    "entropy_clarity": (roll.get("primary") or {}).get(
                        "compression_pct"
                    ),
                }
                return {
                    "entropy_strength": round(max(0.0, min(100.0, strength)), 1),
                    "compression_score": round(comp_s, 1),
                    "momentum_score": round(mom_s, 1),
                    "realtime_pattern_strength": rt,
                    "regime": roll.get("regime"),
                    "velocity": (roll.get("primary") or {}).get("velocity"),
                    "composite_entropy": roll.get("composite_entropy"),
                    "triggers": roll.get("triggers"),
                    "rolling": roll,
                    "weights": {
                        "compression": compression_weight,
                        "momentum": momentum_weight,
                    },
                    "digit_100": dig100,
                    "sliding": {
                        "windows": [
                            {
                                "window": int(k),
                                "entropy": (v or {}).get("h"),
                                "compression_pct": (v or {}).get("compression_pct"),
                            }
                            for k, v in (roll.get("windows") or {}).items()
                        ],
                        "entropy_momentum_bits": (roll.get("primary") or {}).get(
                            "momentum_bits"
                        ),
                        "momentum_score": mom_s,
                        "h_short": (roll.get("windows") or {})
                        .get("50", {})
                        .get("h"),
                        "h_long": (roll.get("windows") or {})
                        .get("500", {})
                        .get("h"),
                    },
                    "composite": {
                        "composite_entropy_score": roll.get("composite_entropy"),
                        "components": roll.get("multi") or {},
                    },
                    "bias_label": dig100.get("bias_label"),
                    "display": roll.get("display"),
                    "reasons": [
                        f"Rolling entropy strength {strength:.0f} · regime {roll.get('regime')}",
                        f"Compression {comp_s:.0f} · momentum {mom_s:.0f} · "
                        f"RT pattern {rt:.0f}",
                        f"H50={(roll.get('windows') or {}).get('50', {}).get('h')} "
                        f"H500={(roll.get('windows') or {}).get('500', {}).get('h')} "
                        f"velocity={(roll.get('primary') or {}).get('velocity')}",
                        f"Composite multi-seq {roll.get('composite_entropy')}",
                    ],
                }
        except Exception:
            pass

    digits = last_digits_from_ticks(ticks, n=max(lookback, 500))
    slide = sliding_window_entropy(digits, windows=(50, 100, 200, 500))

    if use_composite:
        comp = composite_entropy_score(ticks, lookback=min(200, lookback))
        compression_score = float(comp["composite_entropy_score"])
    else:
        dig = entropy_clarity_from_digits(digits[-100:] if digits else [])
        compression_score = float(dig.get("entropy_clarity") or 0)
        comp = {"composite_entropy_score": compression_score, "components": {}}

    dig100 = entropy_clarity_from_digits(digits[-100:] if digits else [])
    mom = float(slide.get("momentum_score") or 50)

    wc = max(0.0, float(compression_weight))
    wm = max(0.0, float(momentum_weight))
    s = wc + wm
    if s <= 0:
        wc, wm, s = 0.5, 0.5, 1.0
    wc, wm = wc / s, wm / s

    strength = wc * compression_score + wm * mom
    strength = max(0.0, min(100.0, strength))

    return {
        "entropy_strength": round(strength, 1),
        "compression_score": round(compression_score, 1),
        "momentum_score": round(mom, 1),
        "weights": {"compression": wc, "momentum": wm},
        "digit_100": dig100,
        "sliding": slide,
        "composite": comp,
        "bias_label": dig100.get("bias_label") or bias_label(
            dig100.get("compression_pct") or 0
        ),
        "reasons": [
            f"Entropy strength {strength:.0f}/100 "
            f"(compression {compression_score:.0f}×{wc:.0%} + "
            f"momentum {mom:.0f}×{wm:.0%})",
            f"Digit H100={dig100.get('entropy')} / Hmax={dig100.get('hmax')} "
            f"→ compression {dig100.get('compression_pct')}% ({dig100.get('bias_label')})",
            f"Momentum bits {slide.get('entropy_momentum_bits')} "
            f"(H_long {slide.get('h_long')} − H_short {slide.get('h_short')})",
            f"Composite: digit {comp.get('components', {}).get('digit')} · "
            f"O/E {comp.get('components', {}).get('odd_even')} · "
            f"U/D {comp.get('components', {}).get('up_down')} · "
            f"streak {comp.get('components', {}).get('streak')}",
        ],
    }


def sample_size_clarity_score(n: int) -> float:
    """Map sample size into 0–100 for clarity formula (15% weight)."""
    from src.analytics.edge_score import sample_size_score_100

    return sample_size_score_100(int(n))


def count_context_confirmations(
    ticks: Sequence[Dict[str, Any]],
    *,
    contract_type: str = "",
    family: str = "digits",
) -> Tuple[int, List[str]]:
    """
    Count supporting signals for the intended trade direction.
    Used as context alignment for digit / rise-fall setups.
    """
    from src.analytics.digit_analysis import digit_snapshot
    from src.analytics.tick_patterns import detect_patterns
    from src.strategy.pro_trend import analyze_pro_trend

    notes: List[str] = []
    conf = 0
    ct = str(contract_type or "").upper()
    snap = digit_snapshot(ticks)
    pats = detect_patterns(ticks)
    heat = ((snap.get("heatmap") or {}).get("windows") or {}).get("100") or {}
    cold = heat.get("cold") or []
    hot = heat.get("hot") or []
    streaks = snap.get("streaks") or {}
    even_rate = float(snap.get("even_rate") or 0.5)

    # Pattern alerts present
    if pats.get("has_alert"):
        conf += 1
        notes.append(f"tick_pattern:{pats.get('pattern_alert_strength')}")

    # Cold digit (absence / mean-reversion style)
    if cold:
        conf += 1
        notes.append(f"cold_digits={cold}")

    # Hot digit cluster
    if hot and max((heat.get("pct") or {}).get(h, 10) for h in hot) >= 14:
        conf += 1
        notes.append(f"hot_cluster={hot}")

    # Parity extreme
    if even_rate >= 0.58 or even_rate <= 0.42:
        conf += 1
        notes.append(f"parity_bias={even_rate:.2f}")

    # Streak
    if int(streaks.get("current_streak") or 0) >= 3:
        conf += 1
        notes.append(f"streak={streaks.get('current_streak')}")

    # Entropy drop / compression bias
    ent = entropy_clarity_from_ticks(ticks, 100)
    if float(ent.get("entropy_clarity") or 0) >= 10:
        conf += 1
        notes.append(
            f"entropy_compression={ent.get('compression_pct')}% ({ent.get('bias_label')})"
        )
    # Fresh pattern: short-window entropy much lower than long
    digits = last_digits_from_ticks(ticks, n=500)
    if len(digits) >= 80:
        slide = sliding_window_entropy(digits)
        if slide.get("fresh_pattern"):
            conf += 1
            notes.append(
                f"entropy_momentum={slide.get('entropy_momentum_bits')} bits"
            )

    # Rise/fall pro-trend agreement
    if family in {"rise_fall", "minute_rise_fall"} or ct in {"CALL", "PUT"}:
        pro = analyze_pro_trend(ticks, symbol="", min_confidence=0.5)
        if pro.get("contract_type") == ct or (
            ct == "CALL" and pro.get("direction") == "up"
        ) or (ct == "PUT" and pro.get("direction") == "down"):
            conf += 1
            notes.append("pro_trend_aligned")
        elif pro.get("contract_type") and pro.get("contract_type") != ct:
            conf = max(0, conf - 1)
            notes.append("pro_trend_conflict")

    # Digit type soft alignment
    if ct == "DIGITEVEN" and even_rate >= 0.55:
        conf += 1
        notes.append("even_aligned")
    if ct == "DIGITODD" and even_rate <= 0.45:
        conf += 1
        notes.append("odd_aligned")

    return conf, notes


def pattern_clarity(
    *,
    pattern_wr: float = 0.5,
    baseline_wr: float = 0.5,
    frequency: float = 0.15,
    trade_rows: Optional[Sequence[Dict[str, Any]]] = None,
    window_win_rates: Optional[Sequence[float]] = None,
    confirmations: int = 1,
    n_conditions: int = 2,
    explainable: bool = True,
    ticks: Optional[Sequence[Dict[str, Any]]] = None,
    entropy_blend: float = 0.0,
    sample_n: Optional[int] = None,
    formula: str = "production",
) -> Dict[str, Any]:
    """
    Full Pattern Clarity 0–100.

    **production** (default — entropy-aware):
      35% Baseline Separation
    + 25% Stability
    + 15% Sample Size
    + 15% Entropy Strength   (50% compression + 50% momentum; composite multi-seq)
    + 10% Context Alignment

    **classic**:
      30% Rarity + 25% Separation + 20% Stability + 15% Context + 10% Simplicity

    **legacy**:
      40% Separation + 25% Stability + 20% Rarity + 10% Context + 5% Simplicity
    """
    sep, improvement_pp = baseline_separation_score(pattern_wr, baseline_wr)
    rare = rarity_score(frequency)

    if window_win_rates is not None:
        stab, std = stability_score_from_windows(window_win_rates)
        stab_detail = {"stdev": std, "windows": list(window_win_rates)}
    else:
        stab, stab_detail = stability_from_trade_rows(trade_rows or [])

    ctx = context_alignment_score(confirmations)
    simp = simplicity_score(n_conditions=n_conditions, explainable=explainable)

    n_samp = int(
        sample_n
        if sample_n is not None
        else (len(trade_rows) if trade_rows else 0)
    )
    samp = sample_size_clarity_score(n_samp)

    ent_strength_block: Dict[str, Any] = {}
    ent_strength_val = 40.0  # neutral when no ticks
    if ticks is not None and len(ticks) >= 20:
        ent_strength_block = entropy_strength(
            ticks,
            lookback=500,
            compression_weight=0.50,
            momentum_weight=0.50,
            use_composite=True,
            symbol=str(
                (trade_rows[0].get("symbol") if trade_rows else None) or "_default"
            ),
        )
        ent_strength_val = float(ent_strength_block.get("entropy_strength") or 40)

    mode = (formula or "production").strip().lower()
    if mode == "classic":
        total = (
            0.30 * rare
            + 0.25 * sep
            + 0.20 * stab
            + 0.15 * ctx
            + 0.10 * simp
        )
        weights = {
            "rarity": 0.30,
            "separation": 0.25,
            "stability": 0.20,
            "context": 0.15,
            "simplicity": 0.10,
        }
        contrib = {
            "rarity": round(0.30 * rare, 1),
            "separation": round(0.25 * sep, 1),
            "stability": round(0.20 * stab, 1),
            "context": round(0.15 * ctx, 1),
            "simplicity": round(0.10 * simp, 1),
        }
    elif mode == "legacy":
        total = (
            0.40 * sep
            + 0.25 * stab
            + 0.20 * rare
            + 0.10 * ctx
            + 0.05 * simp
        )
        weights = {
            "separation": 0.40,
            "stability": 0.25,
            "rarity": 0.20,
            "context": 0.10,
            "simplicity": 0.05,
        }
        contrib = {
            "separation": round(0.40 * sep, 1),
            "stability": round(0.25 * stab, 1),
            "rarity": round(0.20 * rare, 1),
            "context": round(0.10 * ctx, 1),
            "simplicity": round(0.05 * simp, 1),
        }
    else:
        # Production: separation + stability + sample + entropy strength + context
        total = (
            0.35 * sep
            + 0.25 * stab
            + 0.15 * samp
            + 0.15 * ent_strength_val
            + 0.10 * ctx
        )
        weights = {
            "separation": 0.35,
            "stability": 0.25,
            "sample_size": 0.15,
            "entropy_strength": 0.15,
            "context": 0.10,
        }
        contrib = {
            "separation": round(0.35 * sep, 1),
            "stability": round(0.25 * stab, 1),
            "sample_size": round(0.15 * samp, 1),
            "entropy_strength": round(0.15 * ent_strength_val, 1),
            "context": round(0.10 * ctx, 1),
        }

    # Optional extra entropy blend (legacy callers)
    if entropy_blend and ticks is not None:
        dig = entropy_clarity_from_ticks(ticks, 100)
        ent_c = float(dig.get("entropy_clarity") or 0)
        blend = max(0.0, min(0.4, float(entropy_blend)))
        total = (1.0 - blend) * total + blend * ent_c
        contrib["entropy_blend"] = round(blend * ent_c, 1)

    total = max(0.0, min(100.0, total))
    label = clarity_class(total)

    dig100 = (ent_strength_block.get("digit_100") if ent_strength_block else None) or {}
    reasons = [
        f"Pattern Clarity {total:.0f}/100 ({label})",
        f"{'✓' if sep >= 60 else '✗'} Baseline separation {sep:.0f} "
        f"(+{improvement_pp:.1f}pp vs {baseline_wr:.0%} baseline)",
        f"{'✓' if stab >= 60 else '✗'} Stability {stab:.0f} "
        f"(window stdev {stab_detail.get('stdev', '—')})",
        f"{'✓' if samp >= 45 else '~'} Sample size score {samp:.0f} (n={n_samp})",
        f"{'✓' if ent_strength_val >= 50 else '~'} Entropy strength {ent_strength_val:.0f} "
        f"(compression + momentum)",
        f"{'✓' if ctx >= 60 else '~'} Context alignment {ctx:.0f} "
        f"({confirmations} confirmations)",
    ]
    if dig100:
        reasons.append(
            f"Digit entropy H={dig100.get('entropy')} / Hmax={dig100.get('hmax')} "
            f"→ compression {dig100.get('compression_pct')}% ({dig100.get('bias_label')})"
        )
    for line in (ent_strength_block.get("reasons") or [])[:2]:
        if line not in reasons:
            reasons.append(line)
    if total >= 80:
        reasons.append("✓ Clarity ≥ 80 — eligible for auto-trade gate")
    else:
        reasons.append("✗ Clarity < 80 — hold auto-execution")

    return {
        "pattern_clarity": round(total, 1),
        "class": label,
        "formula": mode,
        "auto_ok": total >= 80.0,
        "components": {
            "rarity": rare,
            "separation": sep,
            "stability": stab,
            "context": ctx,
            "simplicity": simp,
            "sample_size": samp,
            "entropy_strength": ent_strength_val,
            "improvement_pp": improvement_pp,
        },
        "contributions": contrib,
        "weights": weights,
        "stability_detail": stab_detail,
        "entropy": dig100,
        "entropy_strength_detail": ent_strength_block,
        "reasons": reasons,
        "explain": reasons,
    }


# Default baseline WR by contract family (fair-ish approximations)
BASELINE_WR = {
    "DIGITDIFF": 0.90,  # differ often high payout low? actually digit differ wins 9/10
    "DIGITMATCH": 0.10,
    "DIGITEVEN": 0.50,
    "DIGITODD": 0.50,
    "DIGITOVER": 0.40,  # depends on barrier; approx
    "DIGITUNDER": 0.40,
    "CALL": 0.50,
    "PUT": 0.50,
}


def baseline_for_contract(contract_type: str, barrier: Optional[int] = None) -> float:
    ct = str(contract_type or "").upper()
    if ct == "DIGITOVER" and barrier is not None:
        # P(digit > b) under uniform = (9-b)/10
        b = max(0, min(8, int(barrier)))
        return (9 - b) / 10.0
    if ct == "DIGITUNDER" and barrier is not None:
        b = max(1, min(9, int(barrier)))
        return b / 10.0
    if ct == "DIGITDIFF":
        return 0.90
    if ct == "DIGITMATCH":
        return 0.10
    return BASELINE_WR.get(ct, 0.50)
