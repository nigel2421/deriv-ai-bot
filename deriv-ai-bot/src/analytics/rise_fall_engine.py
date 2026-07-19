"""
Rise/Fall directional engine.

Digit entropy is weak for CALL/PUT. This module scores:
  - Tick momentum (last N up vs down)
  - Persistence (P(continue after up/down))
  - Transition matrix (UP→UP, UP→DOWN, …)
  - Volatility regime (CALM / NORMAL / EXPANDING / CHAOTIC)
  - Directional entropy on {UP, DOWN}

Composite (production RF weights):
  35% Momentum
+ 25% Trend Strength
+ 20% Volatility Regime (favorable = calm/normal)
+ 10% HPP
+ 10% Directional Entropy (low entropy = edge)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategy.chart_tools import quotes_from_ticks

# RF composite weights (user-recommended)
RF_WEIGHTS = {
    "momentum": 0.35,
    "trend_strength": 0.25,
    "volatility_regime": 0.20,
    "hpp": 0.10,
    "directional_entropy": 0.10,
}

# Profile metric keys for CALL/PUT adaptive weights
RF_PROFILE_WEIGHTS = {
    "momentum": 0.35,
    "trend_strength": 0.25,
    "volatility_score": 0.20,
    "persistence": 0.10,  # part of HPP-friendly persistence signal
    "directional_entropy": 0.10,
}


def tick_directions(quotes: Sequence[float]) -> List[int]:
    """
    Direction bits: +1 = up, -1 = down, 0 = flat (dropped from most stats).
    """
    out: List[int] = []
    for i in range(1, len(quotes)):
        a, b = float(quotes[i - 1]), float(quotes[i])
        if b > a:
            out.append(1)
        elif b < a:
            out.append(-1)
        else:
            out.append(0)
    return out


def directional_entropy(dirs: Sequence[int]) -> Dict[str, Any]:
    """
    Bernoulli entropy on UP vs DOWN (flats ignored).
    High entropy (~1 bit) → no edge; low → directional bias.
    Score 0–100: low entropy / strong bias → high score.
    """
    ups = sum(1 for d in dirs if d > 0)
    downs = sum(1 for d in dirs if d < 0)
    n = ups + downs
    if n <= 0:
        return {
            "h": 1.0,
            "h_ratio": 1.0,
            "up_pct": 50.0,
            "down_pct": 50.0,
            "n": 0,
            "score": 40.0,
            "edge": False,
        }
    p_up = ups / n
    p_dn = 1.0 - p_up
    # binary entropy in bits
    def _h(p: float) -> float:
        p = max(1e-12, min(1.0 - 1e-12, p))
        return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

    h = _h(p_up)
    h_ratio = h / 1.0  # max 1 bit
    # Bias strength 0..1 (0.5 share → 0, 100% one side → 1)
    bias = abs(p_up - 0.5) * 2.0
    # Score: bias-led with entropy confirmation
    # 72/28 bias → ~55+, 50/50 → ~0–15
    score = max(0.0, min(100.0, bias * 85.0 + (1.0 - h_ratio) * 15.0))
    return {
        "h": round(h, 4),
        "h_ratio": round(h_ratio, 4),
        "up_pct": round(p_up * 100.0, 1),
        "down_pct": round(p_dn * 100.0, 1),
        "n": n,
        "score": round(score, 1),
        "edge": bias >= 0.20 and score >= 45,  # ≥60/40 share
        "label": (
            "Strong directional"
            if score >= 70
            else ("Biased" if score >= 45 else "No directional edge")
        ),
    }


def tick_momentum_score(
    dirs: Sequence[int],
    *,
    window: int = 20,
) -> Dict[str, Any]:
    """
    Last N non-flat ticks: up_share → momentum %.
    15 up / 5 down → 75% → Strong Bullish.
    """
    nonzero = [d for d in dirs if d != 0]
    chunk = nonzero[-int(window) :] if window else nonzero
    n = len(chunk)
    if n == 0:
        return {
            "window": window,
            "up": 0,
            "down": 0,
            "momentum_pct": 50.0,
            "score": 50.0,
            "direction": "NEUTRAL",
            "label": "No data",
        }
    up = sum(1 for d in chunk if d > 0)
    down = n - up
    mom_pct = up / n * 100.0
    # Score 0–100: 50 = neutral, 100 = all up, 0 = all down
    # For trading CALL we want high; PUT wants low mom_pct mapped high for put
    score_bull = mom_pct  # 0–100 bullish score
    score_bear = 100.0 - mom_pct
    if mom_pct >= 70:
        label, direction = "Strong Bullish", "BULLISH"
    elif mom_pct >= 58:
        label, direction = "Bullish", "BULLISH"
    elif mom_pct <= 30:
        label, direction = "Strong Bearish", "BEARISH"
    elif mom_pct <= 42:
        label, direction = "Bearish", "BEARISH"
    else:
        label, direction = "Neutral", "NEUTRAL"
    return {
        "window": window,
        "up": up,
        "down": down,
        "n": n,
        "momentum_pct": round(mom_pct, 1),
        "score": round(score_bull, 1),  # raw bullish 0–100
        "score_bull": round(score_bull, 1),
        "score_bear": round(score_bear, 1),
        "direction": direction,
        "label": label,
    }


def transition_matrix(dirs: Sequence[int]) -> Dict[str, Any]:
    """
    First-order Markov on UP/DOWN (flats skipped in chain).
      UP→UP, UP→DOWN, DOWN→UP, DOWN→DOWN
    """
    seq = [d for d in dirs if d != 0]
    counts = {"UU": 0, "UD": 0, "DU": 0, "DD": 0}
    for i in range(1, len(seq)):
        a, b = seq[i - 1], seq[i]
        if a > 0 and b > 0:
            counts["UU"] += 1
        elif a > 0 and b < 0:
            counts["UD"] += 1
        elif a < 0 and b > 0:
            counts["DU"] += 1
        else:
            counts["DD"] += 1
    n_from_u = counts["UU"] + counts["UD"]
    n_from_d = counts["DU"] + counts["DD"]
    p_uu = counts["UU"] / n_from_u if n_from_u else 0.5
    p_ud = counts["UD"] / n_from_u if n_from_u else 0.5
    p_du = counts["DU"] / n_from_d if n_from_d else 0.5
    p_dd = counts["DD"] / n_from_d if n_from_d else 0.5
    return {
        "counts": counts,
        "p_uu": round(p_uu, 4),
        "p_ud": round(p_ud, 4),
        "p_du": round(p_du, 4),
        "p_dd": round(p_dd, 4),
        "n_from_up": n_from_u,
        "n_from_down": n_from_d,
        "display": {
            "UP→UP": f"{p_uu * 100:.0f}%",
            "UP→DOWN": f"{p_ud * 100:.0f}%",
            "DOWN→UP": f"{p_du * 100:.0f}%",
            "DOWN→DOWN": f"{p_dd * 100:.0f}%",
        },
    }


def persistence_score(dirs: Sequence[int]) -> Dict[str, Any]:
    """
    P(continue after move):
      After Up → next Up
      After Down → next Down
    Average persistence as 0–100 score (50 = random).
    """
    tm = transition_matrix(dirs)
    p_uu = float(tm["p_uu"])
    p_dd = float(tm["p_dd"])
    # Edge vs 50%
    persist = (p_uu + p_dd) / 2.0
    # Map: 0.5 → 50, 0.6 → 70, 0.7 → 90
    score = max(0.0, min(100.0, 50.0 + (persist - 0.5) * 200.0))
    mean_rev = 1.0 - persist
    return {
        "p_continue_after_up": round(p_uu, 4),
        "p_continue_after_down": round(p_dd, 4),
        "mean_persistence": round(persist, 4),
        "mean_reversion": round(mean_rev, 4),
        "score": round(score, 1),
        "transition": tm,
        "label": (
            "Strong persistence"
            if persist >= 0.60
            else ("Mild persistence" if persist >= 0.54 else "Mean-reverting / random")
        ),
    }


def volatility_regime(
    quotes: Sequence[float],
    *,
    short: int = 20,
    long: int = 60,
) -> Dict[str, Any]:
    """
    Classify: CALM | NORMAL | EXPANDING | CHAOTIC
    from short vs long absolute return volatility.
    Score favors CALM/NORMAL for RF strategies (expansion often kills edge).
    """
    if len(quotes) < 5:
        return {
            "regime": "NORMAL",
            "score": 55.0,
            "short_vol": 0.0,
            "long_vol": 0.0,
            "ratio": 1.0,
            "tradeable": True,
        }
    rets = []
    for i in range(1, len(quotes)):
        a = float(quotes[i - 1])
        if abs(a) < 1e-12:
            continue
        rets.append(abs(float(quotes[i]) - a) / abs(a))
    if not rets:
        return {
            "regime": "NORMAL",
            "score": 55.0,
            "short_vol": 0.0,
            "long_vol": 0.0,
            "ratio": 1.0,
            "tradeable": True,
        }

    def _mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    s_vol = _mean(rets[-short:]) if rets else 0.0
    l_vol = _mean(rets[-long:]) if rets else s_vol
    ratio = s_vol / l_vol if l_vol > 1e-12 else 1.0
    # Absolute scale of short vol (basis points-ish)
    scale = s_vol * 10000.0

    if ratio >= 2.0 or scale >= 25:
        regime = "CHAOTIC"
        score = 15.0
        tradeable = False
    elif ratio >= 1.45 or scale >= 12:
        regime = "EXPANDING"
        score = 35.0
        tradeable = False
    elif ratio <= 0.65 and scale < 4:
        regime = "CALM"
        score = 85.0
        tradeable = True
    else:
        regime = "NORMAL"
        score = 70.0
        tradeable = True

    return {
        "regime": regime,
        "score": round(score, 1),
        "short_vol": round(s_vol, 8),
        "long_vol": round(l_vol, 8),
        "ratio": round(ratio, 3),
        "scale_bps": round(scale, 2),
        "tradeable": tradeable,
        "label": regime,
    }


def trend_strength_score(
    dirs: Sequence[int],
    *,
    windows: Sequence[int] = (10, 20, 40),
) -> Dict[str, Any]:
    """
    Multi-window agreement on direction + magnitude of bias.
    """
    nonzero = [d for d in dirs if d != 0]
    if not nonzero:
        return {"score": 50.0, "direction": "NEUTRAL", "agreement": 0.0}
    signs = []
    biases = []
    for w in windows:
        chunk = nonzero[-int(w) :]
        if not chunk:
            continue
        up = sum(1 for d in chunk if d > 0) / len(chunk)
        bias = up - 0.5
        biases.append(bias)
        signs.append(1 if bias > 0.02 else (-1 if bias < -0.02 else 0))
    if not biases:
        return {"score": 50.0, "direction": "NEUTRAL", "agreement": 0.0}
    # Agreement: fraction of windows with same non-zero sign
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    agree = max(pos, neg) / max(1, len(signs))
    mean_bias = sum(biases) / len(biases)
    direction = (
        "BULLISH"
        if mean_bias > 0.03
        else ("BEARISH" if mean_bias < -0.03 else "NEUTRAL")
    )
    # Score: |bias|*200 + agreement*30, centered
    mag = min(1.0, abs(mean_bias) * 4.0)
    score = max(0.0, min(100.0, 40.0 + mag * 40.0 + agree * 20.0))
    if direction == "NEUTRAL":
        score = min(score, 52.0)
    return {
        "score": round(score, 1),
        "direction": direction,
        "mean_bias": round(mean_bias, 4),
        "agreement": round(agree, 3),
        "windows_used": len(biases),
    }


def oriented_momentum(
    mom: Dict[str, Any],
    contract_type: str,
) -> float:
    """Map bullish momentum score to contract direction (CALL wants bull, PUT bear)."""
    ct = str(contract_type or "").upper()
    if ct in {"PUT", "FALL", "LOWER"}:
        return float(mom.get("score_bear") or (100.0 - float(mom.get("score") or 50)))
    return float(mom.get("score_bull") or mom.get("score") or 50)


def oriented_trend(
    trend: Dict[str, Any],
    contract_type: str,
) -> float:
    ct = str(contract_type or "").upper()
    base = float(trend.get("score") or 50)
    d = str(trend.get("direction") or "NEUTRAL")
    if ct in {"CALL", "RISE", "HIGHER"}:
        if d == "BEARISH":
            return max(10.0, 100.0 - base)
        if d == "NEUTRAL":
            return min(base, 48.0)
        return base
    if ct in {"PUT", "FALL", "LOWER"}:
        if d == "BULLISH":
            return max(10.0, 100.0 - base)
        if d == "NEUTRAL":
            return min(base, 48.0)
        return base
    return base


def composite_rf_score(
    *,
    momentum: float,
    trend_strength: float,
    volatility_score: float,
    hpp: float = 50.0,
    directional_entropy_score: float = 50.0,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    RF decision score:
      35% Momentum + 25% Trend + 20% Vol + 10% HPP + 10% Dir Entropy
    """
    w = dict(weights or RF_WEIGHTS)
    s = sum(w.values()) or 1.0
    w = {k: v / s for k, v in w.items()}
    total = (
        w.get("momentum", 0.35) * float(momentum)
        + w.get("trend_strength", 0.25) * float(trend_strength)
        + w.get("volatility_regime", 0.20) * float(volatility_score)
        + w.get("hpp", 0.10) * float(hpp)
        + w.get("directional_entropy", 0.10) * float(directional_entropy_score)
    )
    total = max(0.0, min(100.0, total))
    return {
        "rf_score": round(total, 1),
        "components": {
            "momentum": round(float(momentum), 1),
            "trend_strength": round(float(trend_strength), 1),
            "volatility_regime": round(float(volatility_score), 1),
            "hpp": round(float(hpp), 1),
            "directional_entropy": round(float(directional_entropy_score), 1),
        },
        "weights": w,
        "auto_ok": total >= 75.0,
        "label": (
            "Strong RF setup"
            if total >= 80
            else ("Tradeable" if total >= 70 else "Weak RF setup")
        ),
    }


def analyze_rise_fall(
    ticks: Sequence[Dict[str, Any]],
    *,
    contract_type: str = "CALL",
    hpp: float = 50.0,
    n_quotes: int = 120,
) -> Dict[str, Any]:
    """
    Full RF analysis for a tick stream + optional HPP blend.
    """
    quotes = quotes_from_ticks(list(ticks), n=n_quotes)
    dirs = tick_directions(quotes)
    mom = tick_momentum_score(dirs, window=20)
    persist = persistence_score(dirs)
    vol = volatility_regime(quotes)
    trend = trend_strength_score(dirs)
    dent = directional_entropy(dirs)
    tm = persist.get("transition") or transition_matrix(dirs)

    o_mom = oriented_momentum(mom, contract_type)
    o_trend = oriented_trend(trend, contract_type)
    # Directional entropy score already favors low entropy bias; orient lightly
    # If bias direction mismatches contract, penalize
    up_pct = float(dent.get("up_pct") or 50)
    ct = str(contract_type or "").upper()
    dent_score = float(dent.get("score") or 50)
    if ct in {"CALL", "RISE", "HIGHER"} and up_pct < 48:
        dent_score *= 0.55
    if ct in {"PUT", "FALL", "LOWER"} and up_pct > 52:
        dent_score *= 0.55

    # Persistence oriented: continuation in trade direction
    p_uu = float((tm or {}).get("p_uu") or 0.5)
    p_dd = float((tm or {}).get("p_dd") or 0.5)
    if ct in {"CALL", "RISE", "HIGHER"}:
        persist_oriented = max(0.0, min(100.0, 50.0 + (p_uu - 0.5) * 200.0))
    else:
        persist_oriented = max(0.0, min(100.0, 50.0 + (p_dd - 0.5) * 200.0))

    comp = composite_rf_score(
        momentum=o_mom,
        trend_strength=o_trend,
        volatility_score=float(vol.get("score") or 50),
        hpp=hpp,
        directional_entropy_score=dent_score,
    )

    # Suggested side from raw (unoriented) momentum + trend
    raw_dir = mom.get("direction") or trend.get("direction") or "NEUTRAL"
    suggested = (
        "CALL"
        if raw_dir == "BULLISH"
        else ("PUT" if raw_dir == "BEARISH" else None)
    )

    # Metric vector for contract profile system
    metrics = {
        "momentum": round(o_mom, 1),
        "trend_strength": round(o_trend, 1),
        "volatility_score": float(vol.get("score") or 50),
        "persistence": round(persist_oriented, 1),
        "directional_entropy": round(dent_score, 1),
        # Keep light digit slots for compatibility (low weight in RF profiles)
        "up_down_entropy": round(dent_score, 1),
        "stability": float(vol.get("score") or 50),
        "digit_entropy": 40.0,  # deliberately weak for RF
        "streak_entropy": round(float(persist.get("score") or 50), 1),
    }

    return {
        "contract_type": str(contract_type).upper(),
        "family": "rise_fall",
        "momentum": mom,
        "oriented_momentum": round(o_mom, 1),
        "persistence": persist,
        "oriented_persistence": round(persist_oriented, 1),
        "transition_matrix": tm,
        "volatility": vol,
        "trend": trend,
        "oriented_trend": round(o_trend, 1),
        "directional_entropy": dent,
        "composite": comp,
        "rf_score": comp["rf_score"],
        "metrics": metrics,
        "suggested_side": suggested,
        "vol_tradeable": bool(vol.get("tradeable")),
        "display": {
            "momentum": mom.get("label"),
            "momentum_pct": mom.get("momentum_pct"),
            "persistence": persist.get("label"),
            "vol_regime": vol.get("regime"),
            "dir_entropy": dent.get("label"),
            "rf_score": comp["rf_score"],
            "transition": (tm or {}).get("display"),
        },
        "ready": len(dirs) >= 15,
    }


def rf_pattern_strength(analysis: Dict[str, Any]) -> float:
    """Map RF composite to pattern-strength scale for filter compatibility."""
    return float((analysis.get("composite") or {}).get("rf_score") or 50)


def rf_pattern_clarity(analysis: Dict[str, Any]) -> float:
    """
    Clarity for RF: agreement of momentum/trend + persistence + non-chaotic vol.
    """
    mom = float(analysis.get("oriented_momentum") or 50)
    trend = float(analysis.get("oriented_trend") or 50)
    persist = float(analysis.get("oriented_persistence") or 50)
    vol_s = float((analysis.get("volatility") or {}).get("score") or 50)
    dent = float((analysis.get("directional_entropy") or {}).get("score") or 50)
    # Alignment of mom & trend
    align = 100.0 - abs(mom - trend)
    clarity = 0.30 * align + 0.25 * persist + 0.25 * vol_s + 0.20 * dent
    if not analysis.get("vol_tradeable", True):
        clarity = min(clarity, 55.0)
    return max(0.0, min(100.0, clarity))
