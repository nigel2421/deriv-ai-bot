"""
HPP Velocity — how fast is a metric's predictive power changing?

HPP = how good a signal is.
Velocity = whether that signal is improving or deteriorating.

Multi-window (trade-based, preferred for Deriv):
  Short  = hpp - hpp_20
  Medium = hpp - hpp_100
  Long   = hpp - hpp_500

Velocity Score = 50% Short + 30% Medium + 20% Long
Smoothed with EMA (α=0.2).
Effective velocity = velocity × sample confidence.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def velocity_state(velocity: float) -> str:
    """
    Velocity > 10  → RAPIDLY IMPROVING
    5–10           → IMPROVING
    -5–5           → STABLE
    -10–-5         → DECLINING
    < -10          → RAPID DECAY
    """
    v = float(velocity)
    if v > 10:
        return "RAPIDLY IMPROVING"
    if v >= 5:
        return "IMPROVING"
    if v > -5:
        return "STABLE"
    if v >= -10:
        return "DECLINING"
    return "RAPID DECAY"


def momentum_score_from_velocity(velocity: float) -> float:
    """
    Map velocity to 0–100 score for edge models.

      >15 → 100, 10 → 80, 5 → 60, 0 → 50, -5 → 30, -10 → 10
    """
    v = float(velocity)
    anchors = [
        (-20.0, 0.0),
        (-10.0, 10.0),
        (-5.0, 30.0),
        (0.0, 50.0),
        (5.0, 60.0),
        (10.0, 80.0),
        (15.0, 100.0),
        (30.0, 100.0),
    ]
    if v <= anchors[0][0]:
        return anchors[0][1]
    for i in range(1, len(anchors)):
        x0, y0 = anchors[i - 1]
        x1, y1 = anchors[i]
        if v <= x1:
            t = (v - x0) / (x1 - x0) if x1 != x0 else 1.0
            return y0 + t * (y1 - y0)
    return 100.0


def percentage_velocity(current: float, previous: float) -> float:
    """(Current - Previous) / Previous × 100."""
    prev = float(previous)
    if abs(prev) < 1e-9:
        return 0.0
    return (float(current) - prev) / abs(prev) * 100.0


def sample_confidence_trades(n: int) -> float:
    """min(SampleSize / 500, 1)."""
    return min(1.0, max(0.0, int(n) / 500.0))


def ema_update(current: float, prev_ema: Optional[float], alpha: float = 0.2) -> float:
    """Velocity EMA = 0.2 × current + 0.8 × previous EMA."""
    a = max(0.0, min(1.0, float(alpha)))
    if prev_ema is None:
        return float(current)
    return a * float(current) + (1.0 - a) * float(prev_ema)


def multi_window_velocity(
    hpp: float,
    hpp_20: float,
    hpp_100: float,
    hpp_500: float,
) -> Dict[str, float]:
    """
    Short = hpp - hpp_20
    Medium = hpp - hpp_100
    Long = hpp - hpp_500
    Score = 50% short + 30% medium + 20% long
    """
    short = float(hpp) - float(hpp_20)
    medium = float(hpp) - float(hpp_100)
    long_ = float(hpp) - float(hpp_500)
    score = 0.50 * short + 0.30 * medium + 0.20 * long_
    return {
        "short": round(short, 2),
        "medium": round(medium, 2),
        "long": round(long_, 2),
        "velocity_score": round(score, 2),
    }


def classify_edge_flag(
    velocities: Sequence[float],
) -> str:
    """
    Consistently negative → Strategy Decaying
    Increasing positive → Emerging Edge
    """
    vels = [float(v) for v in velocities if v is not None]
    if len(vels) < 3:
        return "INSUFFICIENT_DATA"
    if all(v <= -3 for v in vels[-4:]):
        return "STRATEGY_DECAYING"
    # increasing velocities
    if len(vels) >= 3 and vels[-1] > vels[-2] > vels[-3] and vels[-1] > 2:
        return "EMERGING_EDGE"
    if all(v >= 3 for v in vels[-3:]):
        return "EMERGING_EDGE"
    if all(abs(v) < 3 for v in vels[-3:]):
        return "STABLE_EDGE"
    return "MIXED"


def compute_metric_velocity(
    *,
    hpp: float,
    hpp_20: Optional[float] = None,
    hpp_100: Optional[float] = None,
    hpp_500: Optional[float] = None,
    previous_hpp: Optional[float] = None,
    ma7: Optional[float] = None,
    prev_velocity_ema: Optional[float] = None,
    sample_n: int = 0,
) -> Dict[str, Any]:
    """
    Full velocity package for one metric.
    """
    h = float(hpp)
    h20 = float(hpp_20 if hpp_20 is not None else h)
    h100 = float(hpp_100 if hpp_100 is not None else h)
    h500 = float(hpp_500 if hpp_500 is not None else h)

    multi = multi_window_velocity(h, h20, h100, h500)
    raw = multi["velocity_score"]

    # Day-to-day
    day_vel = (h - float(previous_hpp)) if previous_hpp is not None else 0.0

    # Rolling MA velocity
    ma_vel = (h - float(ma7)) if ma7 is not None else day_vel

    # Trade-based primary = multi-window score
    velocity = raw

    # Percentage
    vel_pct = percentage_velocity(h, h100 if h100 > 0 else (previous_hpp or h))

    # EMA smooth
    vel_ema = ema_update(velocity, prev_velocity_ema, alpha=0.2)

    conf = sample_confidence_trades(sample_n)
    effective = velocity * conf
    effective_ema = vel_ema * conf

    state = velocity_state(vel_ema)
    mom_score = momentum_score_from_velocity(vel_ema)

    return {
        "hpp": round(h, 1),
        "hpp_20": round(h20, 1),
        "hpp_100": round(h100, 1),
        "hpp_500": round(h500, 1),
        "short_velocity": multi["short"],
        "medium_velocity": multi["medium"],
        "long_velocity": multi["long"],
        "velocity": round(velocity, 2),
        "velocity_pct": round(vel_pct, 2),
        "day_velocity": round(day_vel, 2),
        "ma_velocity": round(ma_vel, 2),
        "velocity_ema": round(vel_ema, 2),
        "effective_velocity": round(effective, 2),
        "effective_velocity_ema": round(effective_ema, 2),
        "sample_confidence": round(conf, 3),
        "sample_n": int(sample_n),
        "status": state,
        "momentum_score": round(mom_score, 1),
        "arrow": "▲" if vel_ema > 1 else ("▼" if vel_ema < -1 else "→"),
    }


def weighted_engine_velocity(
    metric_velocities: Dict[str, float],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Overall Velocity = Σ(MetricVelocity × CurrentWeight)
    """
    total_w = 0.0
    acc = 0.0
    parts = []
    for m, vel in metric_velocities.items():
        w = float(weights.get(m) or 0.0)
        # try alias keys
        if w == 0 and m in weights:
            w = float(weights[m])
        contrib = float(vel) * w
        acc += contrib
        total_w += w
        parts.append({"metric": m, "velocity": round(float(vel), 2), "weight": round(w, 4), "contrib": round(contrib, 3)})
    overall = acc  # weights should already sum ~1
    return {
        "overall_velocity": round(overall, 2),
        "status": velocity_state(overall),
        "momentum_score": round(momentum_score_from_velocity(overall), 1),
        "parts": parts,
    }


def attach_velocities_to_snapshot(
    contract_data: Dict[str, Any],
    *,
    previous_contract: Optional[Dict[str, Any]] = None,
    series_hpp: Optional[Sequence[float]] = None,
    sample_n: int = 0,
    prev_ema_by_metric: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Enrich a contract snapshot from HPPTimeSeries with full velocity objects.
    Uses windows short/mid/long as hpp_20/100/500 proxies when present.
    """
    prev_ema_by_metric = prev_ema_by_metric or {}
    metrics = contract_data.get("metrics") or {}
    windows = contract_data.get("windows") or {}
    weights = contract_data.get("weights") or {}

    # MA7 of contract HPP series
    ma7 = None
    if series_hpp and len(series_hpp) >= 2:
        chunk = list(series_hpp)[-7:]
        ma7 = sum(chunk) / len(chunk)

    prev_metrics = (previous_contract or {}).get("metrics") or {}
    prev_vel_map = (previous_contract or {}).get("metric_velocity_detail") or {}

    detail: Dict[str, Any] = {}
    vel_for_weight: Dict[str, float] = {}

    for m, hpp in metrics.items():
        w = windows.get(m) or {}
        # Map: short window ~20-100 trades, mid ~500, long ~1000
        # Use short as current-ish, but primary hpp is the time-decay HPP
        h20 = float(w.get("short") or hpp)
        h100 = float(w.get("mid") or hpp)
        h500 = float(w.get("long") or hpp)
        # For "current" use the headline metric HPP; windows are trade-based levels
        # Short velocity = current - short_window is noisy; use:
        # multi_window: hpp - each window baseline where hpp is mid-weighted current
        current = float(hpp)
        prev_h = float(prev_metrics.get(m) or current)
        prev_ema = prev_ema_by_metric.get(m)
        if prev_ema is None and m in prev_vel_map:
            prev_ema = (prev_vel_map[m] or {}).get("velocity_ema")

        pack = compute_metric_velocity(
            hpp=current,
            # invert: if short window HPP is higher, recent is better → positive vel
            # short_vel = hpp_short_window_style: use current vs older
            # Store: hpp_20 ≈ previous block; use h20 as "recent block avg"
            # Spec: Short = hpp - hpp_20 where hpp_20 is HPP of previous 20-trade regime
            # We approximate: hpp_20 = h20 if h20 is "last 100 short", better:
            # short = h20 - h100, medium = h100 - h500, and also day vel
            hpp_20=h100,  # vs medium as previous regime
            hpp_100=h500,
            hpp_500=h500 if h500 else h100,
            previous_hpp=prev_h,
            ma7=ma7,
            prev_velocity_ema=prev_ema,
            sample_n=sample_n,
        )
        # Override multi-window to match spec more closely:
        # Short = current - mid(100), Medium = current - long(500), Long = current - long
        # Actually: Short = hpp - hpp_20, with hpp_20 = short window reading
        # Prefer: short = short_window_hpp - mid (recent vs older)
        short_v = float(h20) - float(h100)
        med_v = float(h100) - float(h500)
        long_v = float(current) - float(h500)
        score = 0.50 * short_v + 0.30 * med_v + 0.20 * long_v
        pack["short_velocity"] = round(short_v, 2)
        pack["medium_velocity"] = round(med_v, 2)
        pack["long_velocity"] = round(long_v, 2)
        pack["velocity"] = round(score, 2)
        pack["velocity_ema"] = round(
            ema_update(score, prev_ema, 0.2), 2
        )
        pack["status"] = velocity_state(pack["velocity_ema"])
        pack["momentum_score"] = round(
            momentum_score_from_velocity(pack["velocity_ema"]), 1
        )
        pack["arrow"] = (
            "▲" if pack["velocity_ema"] > 1 else ("▼" if pack["velocity_ema"] < -1 else "→")
        )
        pack["effective_velocity_ema"] = round(
            pack["velocity_ema"] * pack["sample_confidence"], 2
        )
        pack["velocity_pct"] = round(percentage_velocity(current, h100 or prev_h or current), 2)
        pack["display"] = (
            f"HPP: {current:.0f}  Velocity: {pack['velocity_ema']:+.1f}  "
            f"Status: {pack['status']}"
        )
        detail[m] = pack
        vel_for_weight[m] = pack["velocity_ema"]

    overall = weighted_engine_velocity(vel_for_weight, weights)

    # Edge flags from recent EMA velocities
    flag = classify_edge_flag([detail[m]["velocity_ema"] for m in detail])

    # Contract-level velocity from overall + day
    contract_hpp = float(contract_data.get("hpp") or 50)
    prev_ch = float((previous_contract or {}).get("hpp") or contract_hpp)
    day_vel = contract_hpp - prev_ch
    prev_c_ema = (previous_contract or {}).get("velocity_ema")
    c_ema = ema_update(
        overall["overall_velocity"],
        float(prev_c_ema) if prev_c_ema is not None else day_vel,
        0.2,
    )

    return {
        **contract_data,
        "metric_velocity_detail": detail,
        "metric_velocity": {m: detail[m]["velocity_ema"] for m in detail},
        "overall_velocity": overall,
        "velocity": round(day_vel, 2),
        "velocity_ema": round(c_ema, 2),
        "velocity_pct": round(percentage_velocity(contract_hpp, prev_ch), 2),
        "effective_velocity": round(c_ema * sample_confidence_trades(sample_n), 2),
        "status": velocity_state(c_ema),
        "momentum_score": round(momentum_score_from_velocity(c_ema), 1),
        "edge_flag": flag,
        "arrow": "▲" if c_ema > 1 else ("▼" if c_ema < -1 else "→"),
    }
