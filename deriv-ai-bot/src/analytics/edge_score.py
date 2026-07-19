"""
Strategy Edge Score + Pattern Strength + Live Edge.

Goal: not "how often does it win?" but
  "Does this strategy have positive expected value while controlling risk?"

Core:
  EV = (WR × AvgWin) − ((1−WR) × AvgLoss)

Edge Score /100:
  EV (40) + WinRate (20) + ProfitFactor (20) + Drawdown (20)

Live Edge:
  30% Historical + 25% Recent + 20% Pattern + 15% Volatility + 10% Confidence

All scores include explainable ✓/✗ reasons for the dashboard and trade filter.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def bayesian_win_rate(
    wins: int, losses: int, prior_w: float = 50.0, prior_n: float = 100.0
) -> float:
    """
    Smoothed WR: (wins + prior_w) / (total + prior_n).
    Stops 8/10 looking like 80% edge.
    """
    w = max(0, int(wins))
    l = max(0, int(losses))
    total = w + l
    return (w + prior_w) / (total + prior_n)


def expected_value(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """EV = WR*AvgWin − LossRate*AvgLoss (avg_loss as positive magnitude)."""
    wr = max(0.0, min(1.0, float(win_rate)))
    aw = max(0.0, float(avg_win))
    al = abs(float(avg_loss))
    return wr * aw - (1.0 - wr) * al


def wr_score_20(win_rate: float) -> float:
    """
    Win-rate component 0–20 (moderate weight — low WR can still be profitable).

      50% → 10, 60% → 14, 70% → 18, 80%+ → 20
    Linear interpolate between anchors.
    """
    wr = max(0.0, min(1.0, float(win_rate))) * 100.0
    anchors = [
        (0.0, 0.0),
        (40.0, 5.0),
        (50.0, 10.0),
        (60.0, 14.0),
        (70.0, 18.0),
        (80.0, 20.0),
        (100.0, 20.0),
    ]
    for i in range(1, len(anchors)):
        x0, y0 = anchors[i - 1]
        x1, y1 = anchors[i]
        if wr <= x1:
            if x1 == x0:
                return y1
            t = (wr - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return 20.0


def profit_factor_score_20(pf: float) -> float:
    """
    PF < 1 → 0; 1.2 → 8; 1.5 → 12; 2.0 → 18; ≥2.5 → 20
    """
    if pf is None or pf < 1.0:
        return 0.0
    anchors = [
        (1.0, 0.0),
        (1.2, 8.0),
        (1.5, 12.0),
        (2.0, 18.0),
        (2.5, 20.0),
        (10.0, 20.0),
    ]
    p = float(pf)
    for i in range(1, len(anchors)):
        x0, y0 = anchors[i - 1]
        x1, y1 = anchors[i]
        if p <= x1:
            if x1 == x0:
                return y1
            t = (p - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return 20.0


def drawdown_score_20(max_dd_pct: float) -> float:
    """Max DD 5%→20, 10%→15, 20%→10, 30%→5, >40%→0."""
    dd = abs(float(max_dd_pct))
    if dd <= 5:
        return 20.0
    if dd <= 10:
        return 15.0
    if dd <= 20:
        return 10.0
    if dd <= 30:
        return 5.0
    if dd <= 40:
        # linear 30→5 to 40→0
        return 5.0 * (40.0 - dd) / 10.0
    return 0.0


def confidence_score_20(n: int) -> float:
    """
    Statistical significance from sample size (0–20):
      <50 → 0, 100 → 5, 500 → 12, 1000+ → 20
    """
    n = max(0, int(n))
    if n < 50:
        return 0.0
    if n >= 1000:
        return 20.0
    if n >= 500:
        # 500→12, 1000→20
        return 12.0 + (n - 500) / 500.0 * 8.0
    if n >= 100:
        # 100→5, 500→12
        return 5.0 + (n - 100) / 400.0 * 7.0
    # 50→0, 100→5
    return (n - 50) / 50.0 * 5.0


def sample_size_score_100(n: int) -> float:
    """
    Log-style sample quality 0–100 for Pattern Strength.

      Trades → Score
      10 → 10, 50 → 30, 100 → 45, 500 → 75, 1000 → 90, 5000+ → 100

    Example: 1200 trades ≈ 92.
    """
    import math

    n = max(0, int(n))
    if n <= 0:
        return 0.0
    anchors = [
        (1, 0.0),
        (10, 10.0),
        (50, 30.0),
        (100, 45.0),
        (500, 75.0),
        (1000, 90.0),
        (5000, 100.0),
    ]
    if n >= 5000:
        return 100.0
    if n < anchors[0][0]:
        return 0.0
    for i in range(1, len(anchors)):
        x0, y0 = anchors[i - 1]
        x1, y1 = anchors[i]
        if n <= x1:
            # log interpolate between anchors
            if x0 <= 0 or x1 <= 0 or n <= 0:
                t = (n - x0) / max(x1 - x0, 1)
            else:
                t = (math.log(n) - math.log(x0)) / (math.log(x1) - math.log(x0))
            t = max(0.0, min(1.0, t))
            return y0 + t * (y1 - y0)
    return 100.0


def wr_to_pattern_score(win_rate: float) -> float:
    """
    Historical WR → 0–100 for Pattern Strength.

      50% = 0, 60% = 50, 70% = 80, 80% = 100
    Example: 63.3% ≈ 65.
    """
    wr = max(0.0, min(1.0, float(win_rate))) * 100.0
    anchors = [
        (0.0, 0.0),
        (50.0, 0.0),
        (60.0, 50.0),
        (70.0, 80.0),
        (80.0, 100.0),
        (100.0, 100.0),
    ]
    for i in range(1, len(anchors)):
        x0, y0 = anchors[i - 1]
        x1, y1 = anchors[i]
        if wr <= x1:
            if x1 == x0:
                return y1
            t = (wr - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return 100.0


def pattern_wr_score_20(win_rate: float) -> float:
    """
    Digit pattern WR as 0–20 points (legacy display).
    Spec example: 65% WR → 13/20.
    """
    # Prefer mapping from 0–100 WR score / 5
    return wr_to_pattern_score(win_rate) / 5.0


def recency_performance_score(
    recent_wr: float, historical_wr: float
) -> Tuple[float, float, str]:
    """
    Recency Score from Recent WR / Historical WR.

    ratio 1.22 (71/58) → strong momentum ≈ 90.
    Returns (score_0_100, ratio, label).
    """
    hist = max(float(historical_wr), 0.01)
    recent = max(0.0, min(1.0, float(recent_wr)))
    ratio = recent / hist

    # Map ratio → score: 0.7→20, 1.0→55, 1.15→75, 1.22→90, 1.4→100
    if ratio <= 0.7:
        score = max(0.0, ratio / 0.7 * 20.0)
    elif ratio <= 1.0:
        score = 20.0 + (ratio - 0.7) / 0.3 * 35.0  # → 55
    elif ratio <= 1.15:
        score = 55.0 + (ratio - 1.0) / 0.15 * 20.0  # → 75
    elif ratio <= 1.22:
        score = 75.0 + (ratio - 1.15) / 0.07 * 15.0  # → 90
    elif ratio <= 1.4:
        score = 90.0 + (ratio - 1.22) / 0.18 * 10.0  # → 100
    else:
        score = 100.0

    # Also blend absolute recent WR so good recent alone helps
    abs_boost = wr_to_pattern_score(recent) * 0.25
    score = min(100.0, 0.75 * score + abs_boost)

    if ratio >= 1.15 or (recent - hist) >= 0.08:
        label = "High"
    elif ratio <= 0.85 or (hist - recent) >= 0.08:
        label = "Low"
    else:
        label = "Stable"
    return round(score, 1), round(ratio, 4), label


def clarity_score_from_frequency(
    occurrences_per_day: Optional[float] = None,
    *,
    clarity_hint: Optional[float] = None,
) -> float:
    """
    Pattern clarity / rarity 0–100.

    High noise (5000/day) → ~20
    Medium → ~60
    Very clean (50/day) → ~100
    """
    if clarity_hint is not None:
        return max(0.0, min(100.0, float(clarity_hint)))
    if occurrences_per_day is None:
        return 60.0  # medium default
    f = max(0.0, float(occurrences_per_day))
    # 50/day → 100, 500 → 60, 5000 → 20
    if f <= 50:
        return 100.0
    if f >= 5000:
        return 20.0
    if f <= 500:
        # 50→100, 500→60
        return 100.0 - (f - 50) / 450.0 * 40.0
    # 500→60, 5000→20
    return 60.0 - (f - 500) / 4500.0 * 40.0


def edge_label(score: float) -> str:
    s = float(score)
    if s >= 90:
        return "Elite"
    if s >= 80:
        return "Strong"
    if s >= 70:
        return "Tradable"
    if s >= 60:
        return "Weak"
    return "Avoid"


def pattern_class(score: float) -> str:
    """
    Professional pattern tiers:
      0–49 Ignore · 50–64 Weak · 65–74 Tradable · 75–84 Strong · 85–100 Elite
    """
    s = float(score)
    if s >= 85:
        return "Elite"
    if s >= 75:
        return "Strong"
    if s >= 65:
        return "Tradable"
    if s >= 50:
        return "Weak"
    return "Ignore"


def recency_label(recent_wr: float, overall_wr: float) -> str:
    _, _, label = recency_performance_score(recent_wr, overall_wr)
    return label


# ---------------------------------------------------------------------------
# Recency: Last 100 = 50%, Last 500 = 30%, Last 5000 = 20%
# ---------------------------------------------------------------------------

def recency_weighted_performance(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Weight recent windows more heavily so stale strategies decay.
    Returns a 0–100 score + WR diagnostics.
    """
    rows = list(rows or [])
    if not rows:
        return {
            "score": 50.0,
            "label": "Unknown",
            "wr_100": None,
            "wr_500": None,
            "wr_5000": None,
            "weighted_wr": 0.5,
            "n": 0,
        }

    def _wr(chunk: Sequence[Dict[str, Any]]) -> Tuple[float, int]:
        if not chunk:
            return 0.5, 0
        wins = sum(
            1
            for r in chunk
            if float(r.get("profit") or 0) > 0
            or r.get("is_win") is True
            or str(r.get("status") or "").lower() == "win"
        )
        n = len(chunk)
        return wins / n, n

    r100 = rows[-100:]
    r500 = rows[-500:]
    r5000 = rows[-5000:]
    wr100, n100 = _wr(r100)
    wr500, n500 = _wr(r500)
    wr5k, n5k = _wr(r5000)

    # If smaller windows empty, fall back
    w100 = 0.50 if n100 else 0.0
    w500 = 0.30 if n500 else 0.0
    w5k = 0.20 if n5k else 0.0
    wsum = w100 + w500 + w5k
    if wsum <= 0:
        weighted = 0.5
    else:
        weighted = (w100 * wr100 + w500 * wr500 + w5k * wr5k) / wsum

    # Map weighted WR → 0–100 via same curve as historical WR feel
    # 50% → 50, 60% → 70, 70% → 85, 80% → 95
    score = wr_score_20(weighted) / 20.0 * 100.0
    # Prefer slightly higher when n large
    score = min(100.0, score + confidence_score_20(n5k or n500 or n100) * 0.5)

    overall = wr5k if n5k else (wr500 if n500 else wr100)
    label = recency_label(wr100 if n100 else weighted, overall)

    return {
        "score": round(score, 1),
        "label": label,
        "wr_100": round(wr100, 4) if n100 else None,
        "wr_500": round(wr500, 4) if n500 else None,
        "wr_5000": round(wr5k, 4) if n5k else None,
        "weighted_wr": round(weighted, 4),
        "n": n5k or n500 or n100,
        "weights": {"last_100": 0.50, "last_500": 0.30, "last_5000": 0.20},
    }


# ---------------------------------------------------------------------------
# Historical Edge Score (100 pts)
# ---------------------------------------------------------------------------

def historical_edge_score(
    *,
    wins: int,
    losses: int,
    gross_profit: float = 0.0,
    gross_loss: float = 0.0,
    avg_win: Optional[float] = None,
    avg_loss: Optional[float] = None,
    max_dd_pct: float = 15.0,
    ev_multiplier: float = 10.0,
    use_bayesian_wr: bool = True,
) -> Dict[str, Any]:
    """
    Edge Score =
      EV Score (0–40) + WR Score (0–20) + PF Score (0–20) + DD Score (0–20)

    EV Score = Min(40, EV × Multiplier)   # e.g. EV 2.8 → 28
    """
    w, l = max(0, int(wins)), max(0, int(losses))
    n = w + l
    raw_wr = w / n if n else 0.5
    bayes_wr = bayesian_win_rate(w, l)
    wr_for_ev = bayes_wr if use_bayesian_wr else raw_wr

    if avg_win is None:
        avg_win = (float(gross_profit) / w) if w else 1.0
    if avg_loss is None:
        avg_loss = (abs(float(gross_loss)) / l) if l else 1.0

    # Also compute raw EV (user mental model) and bayes EV (production)
    ev_raw = expected_value(raw_wr, float(avg_win), float(avg_loss))
    ev = expected_value(wr_for_ev, float(avg_win), float(avg_loss))

    # Component 1: EV → Min(40, EV × mult); negative EV collapses toward 0
    if ev > 0:
        ev_pts = min(40.0, ev * float(ev_multiplier))
    else:
        # Mild credit only for near-flat; hard zero if deeply negative
        ev_pts = max(0.0, min(8.0, 8.0 + ev * float(ev_multiplier)))

    # Component 2: Win rate (use raw for display feel; bayes for safety blend)
    wr_pts = wr_score_20(0.6 * bayes_wr + 0.4 * raw_wr if n else bayes_wr)

    # Component 3: Profit factor
    if gross_loss and abs(float(gross_loss)) > 1e-9:
        pf = float(gross_profit) / abs(float(gross_loss))
    elif w > l and n:
        pf = 2.0
    elif n:
        pf = 0.8
    else:
        pf = 1.0
    pf_pts = profit_factor_score_20(pf)

    # Component 4: Drawdown
    dd_pts = drawdown_score_20(max_dd_pct)

    # Optional 5th view: statistical significance (not in the 100 total per spec,
    # but exposed for Live Edge confidence input)
    conf_pts = confidence_score_20(n)

    total = ev_pts + wr_pts + pf_pts + dd_pts
    reasons = build_edge_reasons(
        edge_score=total,
        ev=ev,
        ev_raw=ev_raw,
        raw_wr=raw_wr,
        bayes_wr=bayes_wr,
        pf=pf,
        max_dd_pct=max_dd_pct,
        n=n,
        conf_pts=conf_pts,
        wr_pts=wr_pts,
        pf_pts=pf_pts,
        dd_pts=dd_pts,
        ev_pts=ev_pts,
    )

    return {
        "edge_score": round(total, 1),
        "label": edge_label(total),
        "components": {
            "ev": round(ev_pts, 1),
            "win_rate": round(wr_pts, 1),
            "profit_factor": round(pf_pts, 1),
            "drawdown": round(dd_pts, 1),
            # out-of-band: significance for live edge
            "confidence_0_20": round(conf_pts, 1),
        },
        "metrics": {
            "ev": round(ev, 4),
            "ev_raw": round(ev_raw, 4),
            "raw_wr": round(raw_wr, 4),
            "bayes_wr": round(bayes_wr, 4),
            "avg_win": round(float(avg_win), 4),
            "avg_loss": round(float(avg_loss), 4),
            "profit_factor": round(pf, 3),
            "n": n,
            "wins": w,
            "losses": l,
            "max_dd_pct": round(float(max_dd_pct), 2),
            "positive_ev": ev > 0,
        },
        "reasons": reasons,
        "explain": reasons,  # alias for UI
    }


def build_edge_reasons(
    *,
    edge_score: float,
    ev: float,
    ev_raw: float,
    raw_wr: float,
    bayes_wr: float,
    pf: float,
    max_dd_pct: float,
    n: int,
    conf_pts: float,
    wr_pts: float,
    pf_pts: float,
    dd_pts: float,
    ev_pts: float,
    pattern_n: Optional[int] = None,
    pattern_wr: Optional[float] = None,
    recent_wr: Optional[float] = None,
    recency_label_s: Optional[str] = None,
) -> List[str]:
    """Human-readable ✓/✗ lines users trust more than a lone number."""
    lines: List[str] = []

    if ev > 0:
        lines.append(f"✓ Positive expected value (EV {ev:+.2f})")
    else:
        lines.append(f"✗ Non-positive EV ({ev:+.2f}) — no edge")

    if recent_wr is not None and raw_wr > 0:
        if recent_wr > raw_wr + 0.03:
            lines.append(
                f"✓ Win rate increasing ({recent_wr:.0%} recent vs {raw_wr:.0%} overall)"
            )
        elif recent_wr < raw_wr - 0.05:
            lines.append(
                f"✗ Win rate declining ({recent_wr:.0%} recent vs {raw_wr:.0%} overall)"
            )
        else:
            lines.append(f"✓ Win rate stable (~{raw_wr:.0%})")
    else:
        if wr_pts >= 14:
            lines.append(f"✓ Win rate quality solid ({raw_wr:.0%} → {wr_pts:.0f}/20)")
        elif wr_pts >= 10:
            lines.append(f"~ Moderate win rate ({raw_wr:.0%})")
        else:
            lines.append(f"✗ Weak win rate ({raw_wr:.0%})")

    if pf >= 2.0:
        lines.append(f"✓ Profit factor {pf:.2f} (strong)")
    elif pf >= 1.2:
        lines.append(f"✓ Profit factor {pf:.2f} acceptable")
    else:
        lines.append(f"✗ Profit factor {pf:.2f} < 1.2")

    if max_dd_pct <= 10:
        lines.append(f"✓ Drawdown acceptable ({max_dd_pct:.1f}%)")
    elif max_dd_pct <= 20:
        lines.append(f"~ Drawdown elevated ({max_dd_pct:.1f}%)")
    else:
        lines.append(f"✗ Drawdown high ({max_dd_pct:.1f}%)")

    if n >= 1000:
        lines.append(f"✓ Large sample ({n:,} trades) — high significance")
    elif n >= 100:
        lines.append(f"✓ Sample size OK ({n} trades, conf {conf_pts:.0f}/20)")
    elif n >= 50:
        lines.append(f"~ Thin sample ({n} trades) — treat scores cautiously")
    else:
        lines.append(f"✗ Sample too small ({n} trades) — statistical significance low")

    if pattern_n is not None and pattern_n > 0:
        lines.append(f"✓ Pattern seen {pattern_n:,} times")
        if pattern_wr is not None:
            lines.append(f"✓ Similar setups win {pattern_wr:.0%}")

    if recency_label_s:
        if recency_label_s == "High":
            lines.append("✓ Recency score High — strategy not stale")
        elif recency_label_s == "Low":
            lines.append("✗ Recency score Low — recent performance decaying")
        else:
            lines.append(f"~ Recency {recency_label_s}")

    lines.append(
        f"Edge Score {edge_score:.0f}/100 ({edge_label(edge_score)}) "
        f"[EV {ev_pts:.0f}/40 · WR {wr_pts:.0f}/20 · PF {pf_pts:.0f}/20 · DD {dd_pts:.0f}/20]"
    )
    return lines


# ---------------------------------------------------------------------------
# Pattern Strength (digit / setup specific)
# ---------------------------------------------------------------------------

def pattern_strength(
    *,
    wins: int,
    losses: int,
    recent_wins: int = 0,
    recent_losses: int = 0,
    profit_factor: Optional[float] = None,
    rarity_score: float = 60.0,
    clarity_score: Optional[float] = None,
    occurrences_per_day: Optional[float] = None,
    formula: str = "production",
) -> Dict[str, Any]:
    """
    Pattern Strength = reliability historically × strength of appearance now.
    Not win rate alone.

    **classic** (40/25/20/15):
      40% Historical WR + 25% Sample + 20% Recency + 15% Clarity
      Example: 65×.4 + 92×.25 + 90×.2 + 100×.15 = 82

    **production** (default, more robust):
      35% Bayesian WR + 25% Sample + 20% Recency + 10% PF + 10% Rarity

    Bayesian WR = (wins+50)/(total+100) so 8/10 → 52.7%, not 80%.

    Tiers: 0–49 Ignore · 50–64 Weak · 65–74 Tradable · 75–84 Strong · 85–100 Elite
    Auto-execution only when pattern_strength ≥ 75 (and Live Edge ≥ 80 elsewhere).
    """
    w, l = max(0, int(wins)), max(0, int(losses))
    n = w + l
    raw_wr = w / n if n else 0.5
    # Bayesian adjustment (wins+50)/(total+100)
    bayes = bayesian_win_rate(w, l, prior_w=50.0, prior_n=100.0)

    # Use Bayesian WR for scoring so lucky streaks don't dominate
    wr_s = wr_to_pattern_score(bayes)
    sample_s = sample_size_score_100(n)
    pat_20 = pattern_wr_score_20(bayes)

    rw, rl = max(0, int(recent_wins)), max(0, int(recent_losses))
    rn = rw + rl
    if rn > 0:
        recent_wr = rw / rn
        # Recency vs historical (prefer raw historical baseline for ratio)
        hist_base = raw_wr if n >= 20 else bayes
        recency_s, rec_ratio, rlabel = recency_performance_score(recent_wr, hist_base)
    else:
        recent_wr = bayes
        recency_s, rec_ratio, rlabel = wr_s * 0.7, 1.0, "Unknown"

    if profit_factor is None:
        # Implied PF from WR when not supplied (neutral-ish)
        profit_factor = 1.0 + (bayes - 0.5) * 2.0
    pf_s = profit_factor_score_20(float(profit_factor)) * 5.0  # 0–100 scale

    clarity = clarity_score_from_frequency(
        occurrences_per_day,
        clarity_hint=clarity_score if clarity_score is not None else rarity_score,
    )
    rarity = max(0.0, min(100.0, clarity))

    mode = (formula or "production").strip().lower()
    if mode == "classic":
        # 40% WR + 25% Sample + 20% Recency + 15% Clarity
        total = (
            0.40 * wr_s
            + 0.25 * sample_s
            + 0.20 * recency_s
            + 0.15 * rarity
        )
        weights = {
            "historical_wr": 0.40,
            "sample_size": 0.25,
            "recency": 0.20,
            "clarity": 0.15,
        }
        contrib = {
            "historical_wr": round(0.40 * wr_s, 1),
            "sample_size": round(0.25 * sample_s, 1),
            "recency": round(0.20 * recency_s, 1),
            "clarity": round(0.15 * rarity, 1),
        }
    else:
        # Production: 35% Bayes WR + 25% Sample + 20% Recency + 10% PF + 10% Rarity
        total = (
            0.35 * wr_s
            + 0.25 * sample_s
            + 0.20 * recency_s
            + 0.10 * pf_s
            + 0.10 * rarity
        )
        weights = {
            "bayesian_wr": 0.35,
            "sample_size": 0.25,
            "recency": 0.20,
            "profit_factor": 0.10,
            "rarity": 0.10,
        }
        contrib = {
            "bayesian_wr": round(0.35 * wr_s, 1),
            "sample_size": round(0.25 * sample_s, 1),
            "recency": round(0.20 * recency_s, 1),
            "profit_factor": round(0.10 * pf_s, 1),
            "rarity": round(0.10 * rarity, 1),
        }

    reasons = []
    if n >= 1000:
        reasons.append(f"✓ Pattern seen {n:,} times (strong sample)")
    elif n >= 100:
        reasons.append(f"✓ Pattern seen {n:,} times")
    else:
        reasons.append(
            f"✗ Thin pattern sample ({n}) — Bayesian WR {bayes:.0%} "
            f"(raw would be {raw_wr:.0%})"
        )
    reasons.append(
        f"✓ Historical reliability: bayesian WR {bayes:.0%} "
        f"(raw {raw_wr:.0%}) → WR score {wr_s:.0f}/100"
    )
    reasons.append(f"Sample score {sample_s:.0f}/100 (n={n})")
    reasons.append(
        f"Recency {rlabel}: recent WR {recent_wr:.0%} / hist {raw_wr:.0%} "
        f"= ratio {rec_ratio:.2f} → {recency_s:.0f}/100"
    )
    reasons.append(f"Pattern clarity/rarity {rarity:.0f}/100")
    if mode == "production":
        reasons.append(f"Profit factor score {pf_s:.0f}/100 (PF≈{float(profit_factor):.2f})")
    if total >= 75:
        reasons.append(
            f"✓ Pattern strength {total:.0f}/100 ({pattern_class(total)}) — auto-ok ≥75"
        )
    else:
        reasons.append(
            f"✗ Pattern strength {total:.0f}/100 ({pattern_class(total)}) < 75 — no auto-exec"
        )

    return {
        "pattern_strength": round(total, 1),
        "class": pattern_class(total),
        "formula": mode,
        "pattern_wr_score_20": round(pat_20, 1),
        "components": {
            "wr_score": round(wr_s, 1),
            "sample_score": round(sample_s, 1),
            "recency_score": round(recency_s, 1),
            "clarity_score": round(rarity, 1),
            "profit_factor_score": round(pf_s, 1),
            "bayes_wr": round(wr_s, 1),  # alias for older callers
            "sample": round(sample_s, 1),
            "recency": round(recency_s, 1),
            "profit_factor": round(pf_s, 1),
            "rarity": round(rarity, 1),
        },
        "contributions": contrib,
        "weights": weights,
        "bayes_wr": round(bayes, 4),
        "raw_wr": round(raw_wr, 4),
        "recent_wr": round(recent_wr, 4),
        "recency_ratio": rec_ratio,
        "recency_label": rlabel,
        "n": n,
        "auto_ok": total >= 75.0,
        "reasons": reasons,
        "explain": reasons,
    }


# ---------------------------------------------------------------------------
# Live Edge
# ---------------------------------------------------------------------------

def live_edge_score(
    *,
    historical_edge: float,
    recent_score: float,
    pattern_strength_val: float,
    volatility_match: float = 70.0,
    confidence: float = 50.0,
    extra_reasons: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Live Edge =
      30% Historical Edge
    + 25% Recent Performance
    + 20% Pattern Strength
    + 15% Volatility Match
    + 10% Confidence

    Display:
      LIVE EDGE: 84 | STRONG BUY | Confidence HIGH | EV POSITIVE | Risk MEDIUM
    """
    h = max(0.0, min(100.0, float(historical_edge)))
    r = max(0.0, min(100.0, float(recent_score)))
    p = max(0.0, min(100.0, float(pattern_strength_val)))
    v = max(0.0, min(100.0, float(volatility_match)))
    c = max(0.0, min(100.0, float(confidence)))

    total = 0.30 * h + 0.25 * r + 0.20 * p + 0.15 * v + 0.10 * c

    if total >= 85:
        status = "STRONG BUY"
    elif total >= 80:
        status = "BUY"
    elif total >= 70:
        status = "WATCH"
    elif total >= 60:
        status = "WEAK"
    else:
        status = "SKIP"

    conf_label = "HIGH" if c >= 70 else ("MEDIUM" if c >= 40 else "LOW")
    risk = (
        "LOW"
        if total >= 85 and h >= 75
        else ("MEDIUM" if total >= 70 else "HIGH")
    )
    ev_state = "POSITIVE" if total >= 70 and h >= 60 else (
        "NEGATIVE" if h < 50 else "UNCERTAIN"
    )

    reasons: List[str] = [
        f"LIVE EDGE: {total:.0f}",
        f"Status: {status}",
        f"Confidence: {conf_label}",
        f"Expected Value: {ev_state}",
        f"Risk: {risk}",
        f"Mix: Hist {h:.0f}×30% + Recent {r:.0f}×25% + Pattern {p:.0f}×20% "
        f"+ Vol {v:.0f}×15% + Conf {c:.0f}×10%",
    ]
    if extra_reasons:
        reasons.extend(list(extra_reasons)[:8])

    return {
        "live_edge": round(total, 1),
        "status": status,
        "confidence": conf_label,
        "expected_value": ev_state,
        "risk": risk,
        "auto_ok": total >= 80.0,
        "label": edge_label(total),
        "components": {
            "historical": round(h, 1),
            "recent": round(r, 1),
            "pattern": round(p, 1),
            "volatility": round(v, 1),
            "confidence": round(c, 1),
            "weights": {
                "historical": 0.30,
                "recent": 0.25,
                "pattern": 0.20,
                "volatility": 0.15,
                "confidence": 0.10,
            },
        },
        "reasons": reasons,
        "explain": reasons,
    }


# ---------------------------------------------------------------------------
# Trade-row aggregation
# ---------------------------------------------------------------------------

def stats_from_trade_rows(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate win/loss/pnl rows into edge inputs."""
    wins = losses = 0
    gp = gl = 0.0
    pnls: List[float] = []
    for r in rows:
        # honor is_win / status when profit missing
        try:
            p = float(r.get("profit")) if r.get("profit") is not None else None
        except (TypeError, ValueError):
            p = None
        if p is None:
            if r.get("is_win") is True or str(r.get("status") or "").lower() == "win":
                p = 1.0
            elif r.get("is_win") is False or str(r.get("status") or "").lower() == "loss":
                p = -1.0
            else:
                continue
        pnls.append(float(p))
        if p > 0:
            wins += 1
            gp += p
        elif p < 0:
            losses += 1
            gl += abs(p)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    denom = peak if peak > 1e-6 else max(sum(abs(x) for x in pnls), 1.0)
    max_dd_pct = 100.0 * max_dd / denom if denom else 0.0

    return {
        "wins": wins,
        "losses": losses,
        "gross_profit": gp,
        "gross_loss": gl,
        "max_dd_pct": min(100.0, max_dd_pct),
        "n": wins + losses,
        "avg_win": (gp / wins) if wins else 0.0,
        "avg_loss": (gl / losses) if losses else 0.0,
    }


def compute_full_edge_report(
    rows: Sequence[Dict[str, Any]],
    *,
    pattern_wins: Optional[int] = None,
    pattern_losses: Optional[int] = None,
    volatility_match: float = 70.0,
) -> Dict[str, Any]:
    """
    One-shot report: historical edge + recency + pattern + live edge + explain.
    Used by dashboard and filter.
    """
    rows = list(rows or [])
    hist_stats = stats_from_trade_rows(rows)
    recent = recency_weighted_performance(rows)

    hist = historical_edge_score(
        wins=hist_stats["wins"],
        losses=hist_stats["losses"],
        gross_profit=hist_stats["gross_profit"],
        gross_loss=hist_stats["gross_loss"],
        avg_win=hist_stats.get("avg_win") or None,
        avg_loss=hist_stats.get("avg_loss") or None,
        max_dd_pct=hist_stats["max_dd_pct"],
    )

    # Enrich hist reasons with recency
    r100 = recent.get("wr_100")
    hist_reasons = build_edge_reasons(
        edge_score=float(hist["edge_score"]),
        ev=float(hist["metrics"]["ev"]),
        ev_raw=float(hist["metrics"]["ev_raw"]),
        raw_wr=float(hist["metrics"]["raw_wr"]),
        bayes_wr=float(hist["metrics"]["bayes_wr"]),
        pf=float(hist["metrics"]["profit_factor"]),
        max_dd_pct=float(hist["metrics"]["max_dd_pct"]),
        n=int(hist["metrics"]["n"]),
        conf_pts=float(hist["components"]["confidence_0_20"]),
        wr_pts=float(hist["components"]["win_rate"]),
        pf_pts=float(hist["components"]["profit_factor"]),
        dd_pts=float(hist["components"]["drawdown"]),
        ev_pts=float(hist["components"]["ev"]),
        pattern_n=(
            (pattern_wins or 0) + (pattern_losses or 0)
            if pattern_wins is not None
            else hist_stats["n"]
        ),
        pattern_wr=(
            (pattern_wins / max(1, (pattern_wins or 0) + (pattern_losses or 0)))
            if pattern_wins is not None
            else float(hist["metrics"]["raw_wr"])
        ),
        recent_wr=r100,
        recency_label_s=str(recent.get("label")),
    )
    hist = {**hist, "reasons": hist_reasons, "explain": hist_reasons}

    pw = pattern_wins if pattern_wins is not None else hist_stats["wins"]
    pl = pattern_losses if pattern_losses is not None else hist_stats["losses"]
    # recent window for pattern
    rstats = stats_from_trade_rows(rows[-100:])
    ps = pattern_strength(
        wins=pw,
        losses=pl,
        recent_wins=rstats["wins"],
        recent_losses=rstats["losses"],
        profit_factor=float(hist["metrics"]["profit_factor"]),
    )

    conf_100 = confidence_score_20(hist_stats["n"]) / 20.0 * 100.0
    live = live_edge_score(
        historical_edge=float(hist["edge_score"]),
        recent_score=float(recent["score"]),
        pattern_strength_val=float(ps["pattern_strength"]),
        volatility_match=volatility_match,
        confidence=conf_100,
        extra_reasons=hist_reasons[:5] + list(ps.get("reasons") or [])[:3],
    )

    return {
        "historical_edge": hist,
        "recency": recent,
        "pattern_strength": ps,
        "live_edge": live,
        "edge_score": hist["edge_score"],
        "live_edge_score": live["live_edge"],
        "label": hist["label"],
        "status": live["status"],
        "reasons": live.get("reasons") or hist_reasons,
        "explain": live.get("explain") or hist_reasons,
    }
