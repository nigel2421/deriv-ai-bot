"""
Momentum + Persistence + Transition engines (second system).

Does NOT replace Entropy / Pattern / HPP stack. Runs in parallel and
feeds Final Trade Quality:

  Final Quality =
    30% Pattern Strength
  + 20% Pattern Clarity
  + 15% HPP
  + 10% HPP Velocity
  + 15% Momentum Persistence
  + 10% Confidence

Also supports dual blend:
  50% Existing Edge Model + 30% Momentum + 20% Persistence

Persistence velocity + sample confidence block late / untrusted entries.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategy.chart_tools import quotes_from_ticks

DEFAULT_PERSIST_PATH = Path("data/persistence_history.json")
REQUIRED_TRANSITION_SAMPLES = 200  # confidence = count / required
# Velocity sample confidence uses 500 transitions (spec)
VELOCITY_CONFIDENCE_SAMPLES = 500
# Multi-timeframe lookbacks (history samples, one per calculation tick)
FAST_LOOKBACK = 20
MEDIUM_LOOKBACK = 100
SLOW_LOOKBACK = 500
MAX_HISTORY = 600


# ---------------------------------------------------------------------------
# Core direction stream
# ---------------------------------------------------------------------------

def tick_directions(quotes: Sequence[float]) -> List[int]:
    """+1 up, -1 down, 0 flat."""
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


# ---------------------------------------------------------------------------
# Momentum Engine
# ---------------------------------------------------------------------------

def momentum_engine(
    dirs: Sequence[int],
    *,
    window: int = 20,
) -> Dict[str, Any]:
    """
    Last N non-flat ticks:
      Momentum = (Up - Down) / N
      Score    = 50 + (Momentum × 50)

    50 = Neutral · 70 = Bullish · 90 = Strong Bullish
    30 = Bearish · 10 = Strong Bearish
    """
    nonzero = [d for d in dirs if d != 0]
    chunk = nonzero[-int(window) :] if window else nonzero
    n = len(chunk)
    if n == 0:
        return {
            "window": window,
            "up": 0,
            "down": 0,
            "n": 0,
            "raw_momentum": 0.0,
            "momentum_score": 50.0,
            "direction": "NEUTRAL",
            "label": "No data",
        }
    up = sum(1 for d in chunk if d > 0)
    down = n - up
    raw = (up - down) / float(n)  # in [-1, 1]
    score = max(0.0, min(100.0, 50.0 + raw * 50.0))
    if score >= 85:
        label, direction = "Strong Bullish", "BULLISH"
    elif score >= 65:
        label, direction = "Bullish", "BULLISH"
    elif score <= 15:
        label, direction = "Strong Bearish", "BEARISH"
    elif score <= 35:
        label, direction = "Bearish", "BEARISH"
    else:
        label, direction = "Neutral", "NEUTRAL"
    return {
        "window": window,
        "up": up,
        "down": down,
        "n": n,
        "raw_momentum": round(raw, 4),
        "momentum_score": round(score, 1),
        "direction": direction,
        "label": label,
        # aliases for older RF code
        "score": round(score, 1),
        "score_bull": round(score, 1),
        "score_bear": round(100.0 - score, 1),
        "momentum_pct": round((up / n) * 100.0, 1),
    }


# ---------------------------------------------------------------------------
# Transition Engine
# ---------------------------------------------------------------------------

def transition_engine(dirs: Sequence[int]) -> Dict[str, Any]:
    """
    Live first-order matrix on UP/DOWN:

              NEXT
            UP    DOWN
    UP      p_uu  p_ud
    DOWN    p_du  p_dd
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
    n_u = counts["UU"] + counts["UD"]
    n_d = counts["DU"] + counts["DD"]
    p_uu = counts["UU"] / n_u if n_u else 0.5
    p_ud = counts["UD"] / n_u if n_u else 0.5
    p_du = counts["DU"] / n_d if n_d else 0.5
    p_dd = counts["DD"] / n_d if n_d else 0.5
    total_trans = n_u + n_d
    return {
        "counts": counts,
        "p_uu": round(p_uu, 4),
        "p_ud": round(p_ud, 4),
        "p_du": round(p_du, 4),
        "p_dd": round(p_dd, 4),
        "n_from_up": n_u,
        "n_from_down": n_d,
        "n_transitions": total_trans,
        "matrix": {
            "UP": {"UP": round(p_uu * 100, 1), "DOWN": round(p_ud * 100, 1)},
            "DOWN": {"UP": round(p_du * 100, 1), "DOWN": round(p_dd * 100, 1)},
        },
        "display": {
            "UP→UP": f"{p_uu * 100:.0f}%",
            "UP→DOWN": f"{p_ud * 100:.0f}%",
            "DOWN→UP": f"{p_du * 100:.0f}%",
            "DOWN→DOWN": f"{p_dd * 100:.0f}%",
        },
        # If last tick UP, P(next UP)
        "p_next_up_given_up": round(p_uu, 4),
        "p_next_down_given_down": round(p_dd, 4),
    }


# ---------------------------------------------------------------------------
# Persistence Engine
# ---------------------------------------------------------------------------

def persistence_confidence(
    transition_count: int,
    required: int = REQUIRED_TRANSITION_SAMPLES,
) -> Dict[str, Any]:
    """
    Confidence = TransitionCount / RequiredSample (capped 1.0).
    3 occurrences at 100% persistence → worthless (low conf).
    """
    req = max(1, int(required))
    n = max(0, int(transition_count))
    c = min(1.0, n / float(req))
    if c >= 0.75:
        label = "HIGH"
    elif c >= 0.40:
        label = "MEDIUM"
    else:
        label = "LOW"
    return {
        "confidence": round(c, 4),
        "confidence_pct": round(c * 100.0, 1),
        "n": n,
        "required": req,
        "label": label,
        "trustworthy": c >= 0.40,
    }


def persistence_engine(
    dirs: Sequence[int],
    *,
    contract_type: str = "",
    required_samples: int = REQUIRED_TRANSITION_SAMPLES,
) -> Dict[str, Any]:
    """
    When a move starts, how often does it continue?

    For CALL/RISE: use UP→UP persistence
    For PUT/FALL:  use DOWN→DOWN
    Generic: average of both

    Score is 0–100 (62% → 62).
    Effective score = raw × sample confidence (avoids n=3 @ 100%).
    """
    tm = transition_engine(dirs)
    p_uu = float(tm["p_uu"])
    p_dd = float(tm["p_dd"])
    ct = str(contract_type or "").upper()

    if ct in {"CALL", "RISE", "HIGHER"}:
        raw = p_uu * 100.0
        n_side = int(tm["n_from_up"])
        side = "UP→UP"
    elif ct in {"PUT", "FALL", "LOWER"}:
        raw = p_dd * 100.0
        n_side = int(tm["n_from_down"])
        side = "DOWN→DOWN"
    else:
        raw = ((p_uu + p_dd) / 2.0) * 100.0
        n_side = int(tm["n_transitions"])
        side = "AVG"

    conf = persistence_confidence(n_side, required_samples)
    # Shrink toward 50 when untrusted
    effective = conf["confidence"] * raw + (1.0 - conf["confidence"]) * 50.0
    return {
        "persistence": round(raw, 1),
        "persistence_score": round(raw, 1),
        "effective_persistence": round(effective, 1),
        "side": side,
        "p_uu": tm["p_uu"],
        "p_dd": tm["p_dd"],
        "transition": tm,
        "sample_confidence": conf,
        "valuable_for_rise": raw > 55 and ct in {"CALL", "RISE", "HIGHER", ""},
        "label": (
            "Strong persistence"
            if raw >= 62 and conf["trustworthy"]
            else (
                "Mild persistence"
                if raw >= 55
                else "Weak / mean-reverting"
            )
        ),
    }


# ---------------------------------------------------------------------------
# Momentum Persistence Score
# ---------------------------------------------------------------------------

def momentum_persistence_score(
    momentum_score: float,
    persistence_score: float,
    *,
    sample_confidence: float = 1.0,
) -> Dict[str, Any]:
    """
    MP = 60% Persistence + 40% Momentum
    Then dampen toward 50 if sample confidence is low.
    """
    m = float(momentum_score)
    p = float(persistence_score)
    raw = 0.60 * p + 0.40 * m
    c = max(0.0, min(1.0, float(sample_confidence)))
    damped = c * raw + (1.0 - c) * 50.0
    return {
        "momentum_persistence": round(damped, 1),
        "raw": round(raw, 1),
        "components": {
            "persistence": round(p, 1),
            "momentum": round(m, 1),
        },
        "weights": {"persistence": 0.60, "momentum": 0.40},
        "sample_confidence": round(c, 4),
    }


# ---------------------------------------------------------------------------
# Persistence Velocity / Acceleration / Engine Score
# ---------------------------------------------------------------------------

def velocity_momentum_score(velocity: float) -> float:
    """
    Score = 50 + (Velocity × 3), clamp 0–100.

      ≤ -15 → 0 · -10 → 20 · 0 → 50 · +10 → 80 · ≥ +15 → 100
    """
    return max(0.0, min(100.0, 50.0 + float(velocity) * 3.0))


def acceleration_score(acceleration: float) -> float:
    """Map acceleration (Δ velocity) to 0–100 with same slope scale."""
    return max(0.0, min(100.0, 50.0 + float(acceleration) * 5.0))


def transition_confidence(
    transition_count: int,
    required: int = VELOCITY_CONFIDENCE_SAMPLES,
) -> float:
    """min(TransitionCount / 500, 1)."""
    return min(1.0, max(0.0, int(transition_count) / float(max(1, required))))


def persistence_engine_score(
    *,
    persistence: float,
    velocity_score: float,
    acceleration_score_val: float,
    weight_persistence: float = 0.50,
    weight_velocity: float = 0.30,
    weight_acceleration: float = 0.20,
) -> Dict[str, Any]:
    """
    Persistence Engine =
      50% Persistence + 30% Velocity score + 20% Acceleration score
    """
    w_p = float(weight_persistence)
    w_v = float(weight_velocity)
    w_a = float(weight_acceleration)
    s = w_p + w_v + w_a or 1.0
    w_p, w_v, w_a = w_p / s, w_v / s, w_a / s
    total = (
        w_p * float(persistence)
        + w_v * float(velocity_score)
        + w_a * float(acceleration_score_val)
    )
    total = max(0.0, min(100.0, total))
    return {
        "persistence_engine_score": round(total, 1),
        "components": {
            "persistence": round(float(persistence), 1),
            "velocity_score": round(float(velocity_score), 1),
            "acceleration_score": round(float(acceleration_score_val), 1),
        },
        "weights": {
            "persistence": w_p,
            "velocity": w_v,
            "acceleration": w_a,
        },
        "auto_ok": total >= 70.0,
    }


class PersistenceVelocityTracker:
    """
    Track persistence history per symbol/contract.

    Computes:
      - raw velocity (current − previous)
      - smoothed (current − mean of prior window)
      - multi-timeframe: fast(20) / medium(100) / slow(500)
      - acceleration (Δ velocity)
      - confidence-adjusted velocity
      - Persistence Engine composite score
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        max_points: int = MAX_HISTORY,
        # Adaptive weight: reduced if validation fails large sample
        velocity_weight: float = 0.30,
        acceleration_weight: float = 0.20,
    ):
        self.path = Path(path) if path else DEFAULT_PERSIST_PATH
        self.max_points = max_points
        # key -> list of {ts, persistence, velocity?, n_trans?}
        self.series: Dict[str, List[Dict[str, Any]]] = {}
        self.velocity_weight = float(velocity_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.series = data.get("series") or {}
            if data.get("velocity_weight") is not None:
                self.velocity_weight = float(data["velocity_weight"])
            if data.get("acceleration_weight") is not None:
                self.acceleration_weight = float(data["acceleration_weight"])
        except Exception:
            self.series = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "series": self.series,
                        "velocity_weight": self.velocity_weight,
                        "acceleration_weight": self.acceleration_weight,
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def set_weights(
        self,
        *,
        velocity_weight: Optional[float] = None,
        acceleration_weight: Optional[float] = None,
    ) -> None:
        if velocity_weight is not None:
            self.velocity_weight = max(0.0, min(0.5, float(velocity_weight)))
        if acceleration_weight is not None:
            self.acceleration_weight = max(
                0.0, min(0.4, float(acceleration_weight))
            )
        self.save()

    def reduce_velocity_weight(self, factor: float = 0.5) -> float:
        """Auto-downweight if validation shows no edge."""
        self.velocity_weight = max(0.05, self.velocity_weight * float(factor))
        self.save()
        return self.velocity_weight

    def note(
        self,
        key: str,
        persistence: float,
        *,
        n_transitions: int = 0,
    ) -> Dict[str, Any]:
        k = str(key or "_default").upper()
        row = {
            "ts": time.time(),
            "persistence": float(persistence),
            "n_transitions": int(n_transitions),
        }
        arr = self.series.setdefault(k, [])
        arr.append(row)
        if len(arr) > self.max_points:
            self.series[k] = arr[-self.max_points :]
        self.save()
        return self.compute(k)

    def compute(self, key: str) -> Dict[str, Any]:
        """
        Full multi-timeframe velocity + acceleration package.
        """
        k = str(key or "_default").upper()
        arr = list(self.series.get(k) or [])
        n = len(arr)
        if n == 0:
            return _empty_pvel()

        values = [float(x["persistence"]) for x in arr]
        current = values[-1]
        previous = values[-2] if n >= 2 else current
        n_trans = int(arr[-1].get("n_transitions") or 0)

        # Step 3: raw velocity
        raw_vel = current - previous if n >= 2 else 0.0

        # Step 4: smoothed vs previous average (exclude current)
        if n >= 5:
            prev_avg = sum(values[-5:-1]) / 4.0
            smooth_vel = current - prev_avg
        elif n >= 2:
            prev_avg = sum(values[:-1]) / (n - 1)
            smooth_vel = current - prev_avg
        else:
            prev_avg = current
            smooth_vel = 0.0

        # Step 5: multi-timeframe
        def _mtf(lookback: int) -> float:
            if n <= lookback:
                # Use earliest available
                if n < 2:
                    return 0.0
                return current - values[0]
            return current - values[-(lookback + 1)]

        fast = _mtf(FAST_LOOKBACK)
        medium = _mtf(MEDIUM_LOOKBACK)
        slow = _mtf(SLOW_LOOKBACK)

        # Composite raw velocity: prefer smooth, blend MTF
        # 50% smooth + 30% fast + 15% medium + 5% slow
        composite_raw = (
            0.50 * smooth_vel
            + 0.30 * fast
            + 0.15 * medium
            + 0.05 * slow
        )

        # Step 8: confidence adjustment
        conf = transition_confidence(n_trans if n_trans > 0 else min(n * 10, 500))
        # If we don't have transition count, use history depth as weak proxy
        if n_trans <= 0:
            conf = min(1.0, n / 50.0)  # ~50 history points → full
        adj_vel = composite_raw * conf

        # Step 6: velocity → score
        vel_score = velocity_momentum_score(adj_vel)

        # Step 7: acceleration (Δ of successive raw velocities)
        velocities_hist: List[float] = []
        for i in range(1, n):
            velocities_hist.append(values[i] - values[i - 1])
        if len(velocities_hist) >= 2:
            accel = velocities_hist[-1] - velocities_hist[-2]
        else:
            accel = 0.0
        # Confidence-adjust acceleration too
        adj_accel = accel * conf
        accel_sc = acceleration_score(adj_accel)

        # Status labels
        if adj_vel > 5:
            status = "STRENGTHENING"
        elif adj_vel > 0.5:
            status = "IMPROVING"
        elif adj_vel < -5:
            status = "WEAKENING"
        elif adj_vel < -0.5:
            status = "DECLINING"
        else:
            status = "STABLE"

        # Persistence Engine composite
        eng = persistence_engine_score(
            persistence=current,
            velocity_score=vel_score,
            acceleration_score_val=accel_sc,
            weight_persistence=0.50,
            weight_velocity=self.velocity_weight,
            weight_acceleration=self.acceleration_weight,
        )

        # Path for display (last 8)
        path = [round(v, 1) for v in values[-8:]]

        return {
            "velocity": round(adj_vel, 2),
            "raw_velocity": round(raw_vel, 2),
            "smooth_velocity": round(smooth_vel, 2),
            "fast_velocity": round(fast, 2),
            "medium_velocity": round(medium, 2),
            "slow_velocity": round(slow, 2),
            "composite_raw_velocity": round(composite_raw, 2),
            "velocity_score": round(vel_score, 1),
            "acceleration": round(adj_accel, 2),
            "raw_acceleration": round(accel, 2),
            "acceleration_score": round(accel_sc, 1),
            "confidence": round(conf, 4),
            "status": status,
            "current": round(current, 1),
            "previous": round(previous, 1),
            "previous_average": round(prev_avg, 2),
            "path": path,
            "n": n,
            "n_transitions": n_trans,
            "strengthening": adj_vel > 0 and adj_accel >= 0,
            "premium": (
                current > 60 and adj_vel > 5 and adj_accel > 2
            ),
            "block_late_entry": status in {"WEAKENING", "DECLINING"}
            or (adj_vel < -1.0),
            "persistence_engine": eng,
            "persistence_engine_score": eng["persistence_engine_score"],
            "weights": {
                "velocity": self.velocity_weight,
                "acceleration": self.acceleration_weight,
            },
            "interpretation": (
                f"Persistence {current:.0f}% "
                + (
                    f"rising from ~{prev_avg:.0f}"
                    if adj_vel > 0
                    else (
                        f"falling from ~{prev_avg:.0f}"
                        if adj_vel < 0
                        else "flat"
                    )
                )
                + f" · vel {adj_vel:+.1f} · accel {adj_accel:+.1f}"
            ),
        }

    # Back-compat alias
    def velocity(self, key: str, lookback: int = 4) -> Dict[str, Any]:
        return self.compute(key)


def _empty_pvel() -> Dict[str, Any]:
    return {
        "velocity": 0.0,
        "raw_velocity": 0.0,
        "smooth_velocity": 0.0,
        "fast_velocity": 0.0,
        "medium_velocity": 0.0,
        "slow_velocity": 0.0,
        "velocity_score": 50.0,
        "acceleration": 0.0,
        "acceleration_score": 50.0,
        "confidence": 0.0,
        "status": "STABLE",
        "current": None,
        "previous": None,
        "path": [],
        "n": 0,
        "strengthening": False,
        "premium": False,
        "block_late_entry": False,
        "persistence_engine_score": 50.0,
        "interpretation": "No persistence history yet",
    }


_pvel: Optional[PersistenceVelocityTracker] = None


def get_persistence_velocity_tracker() -> PersistenceVelocityTracker:
    global _pvel
    if _pvel is None:
        _pvel = PersistenceVelocityTracker()
    return _pvel


# ---------------------------------------------------------------------------
# Final Trade Quality (recommended formula)
# ---------------------------------------------------------------------------

def final_trade_quality(
    *,
    pattern_strength: float,
    pattern_clarity: float,
    hpp: float,
    hpp_velocity: float,
    momentum_persistence: float,
    confidence: float,
) -> Dict[str, Any]:
    """
    Final Quality =
      30% Pattern Strength
    + 20% Pattern Clarity
    + 15% HPP
    + 10% HPP Velocity (mapped 0–100)
    + 15% Momentum Persistence
    + 10% Confidence
    """
    vel = float(hpp_velocity)
    vel_score = max(0.0, min(100.0, 50.0 + vel * 3.33))
    total = (
        0.30 * float(pattern_strength)
        + 0.20 * float(pattern_clarity)
        + 0.15 * float(hpp)
        + 0.10 * vel_score
        + 0.15 * float(momentum_persistence)
        + 0.10 * float(confidence)
    )
    total = max(0.0, min(100.0, total))
    return {
        "trade_quality": round(total, 1),
        "final_quality": round(total, 1),
        "components": {
            "pattern_strength": round(float(pattern_strength), 1),
            "pattern_clarity": round(float(pattern_clarity), 1),
            "hpp": round(float(hpp), 1),
            "hpp_velocity_score": round(vel_score, 1),
            "momentum_persistence": round(float(momentum_persistence), 1),
            "confidence": round(float(confidence), 1),
        },
        "weights": {
            "pattern_strength": 0.30,
            "pattern_clarity": 0.20,
            "hpp": 0.15,
            "hpp_velocity": 0.10,
            "momentum_persistence": 0.15,
            "confidence": 0.10,
        },
        "auto_ok": total >= 80.0,
    }


def dual_system_blend(
    *,
    existing_edge: float,
    momentum_score: float,
    persistence_score: float,
) -> Dict[str, Any]:
    """
    50% Existing Edge Model + 30% Momentum + 20% Persistence
    """
    total = (
        0.50 * float(existing_edge)
        + 0.30 * float(momentum_score)
        + 0.20 * float(persistence_score)
    )
    total = max(0.0, min(100.0, total))
    return {
        "dual_score": round(total, 1),
        "components": {
            "existing_edge": round(float(existing_edge), 1),
            "momentum": round(float(momentum_score), 1),
            "persistence": round(float(persistence_score), 1),
        },
        "weights": {
            "existing_edge": 0.50,
            "momentum": 0.30,
            "persistence": 0.20,
        },
    }


# ---------------------------------------------------------------------------
# Full analysis for a tick stream
# ---------------------------------------------------------------------------

def analyze_momentum_persistence(
    ticks: Sequence[Dict[str, Any]],
    *,
    symbol: str = "",
    contract_type: str = "",
    window: int = 20,
    n_quotes: int = 120,
    note_velocity: bool = True,
) -> Dict[str, Any]:
    """
    Run Momentum + Persistence + Transition engines together.
    """
    quotes = quotes_from_ticks(list(ticks), n=n_quotes)
    dirs = tick_directions(quotes)
    mom = momentum_engine(dirs, window=window)
    pers = persistence_engine(dirs, contract_type=contract_type)
    tm = pers.get("transition") or transition_engine(dirs)

    # For RF, orient momentum score to contract direction
    ct = str(contract_type or "").upper()
    mom_score = float(mom["momentum_score"])
    if ct in {"PUT", "FALL", "LOWER"}:
        # Fall wants bearish: invert so high = good for PUT
        oriented_mom = float(mom["score_bear"])
    elif ct in {"CALL", "RISE", "HIGHER"}:
        oriented_mom = mom_score
    else:
        # Digits: absolute directional pressure (instability if near 50)
        # Use distance from neutral as "stability of pressure"
        oriented_mom = mom_score  # raw; digit path uses MP as confirmation

    conf = float((pers.get("sample_confidence") or {}).get("confidence") or 0)
    # Use effective persistence (sample-damped)
    p_score = float(pers.get("effective_persistence") or pers.get("persistence") or 50)
    mp = momentum_persistence_score(
        oriented_mom if ct in {"CALL", "PUT", "RISE", "FALL", "HIGHER", "LOWER"} else mom_score,
        p_score,
        sample_confidence=max(conf, 0.15),  # never fully zero
    )

    # Persistence velocity (multi-TF + acceleration + engine score)
    pvel: Dict[str, Any] = _empty_pvel()
    key = f"{symbol}|{ct or 'ALL'}"
    n_trans = int((tm or {}).get("n_transitions") or 0)
    try:
        tracker = get_persistence_velocity_tracker()
        if note_velocity:
            pvel = tracker.note(
                key,
                float(pers.get("persistence") or 50),
                n_transitions=n_trans,
            )
        else:
            pvel = tracker.compute(key)
    except Exception:
        pass

    # Digit confirmation interpretation
    mp_val = float(mp["momentum_persistence"])
    if ct.startswith("DIGIT") or ct in {"", "DIGITS"}:
        if mp_val >= 70:
            digit_note = "Directional pressure stable — confidence increased"
            conf_delta = +5.0
        elif mp_val <= 35:
            digit_note = "Pattern exists but directional pressure unstable — reduce confidence"
            conf_delta = -8.0
        else:
            digit_note = "Neutral momentum-persistence confirmation"
            conf_delta = 0.0
    else:
        digit_note = None
        conf_delta = 0.0

    # RF hard-gate preview (base + premium)
    p_raw = float(pers.get("persistence") or 0)
    p_vel = float(pvel.get("velocity") or 0)
    p_acc = float(pvel.get("acceleration") or 0)
    rf_gates = None
    if ct in {"CALL", "RISE", "HIGHER"}:
        rf_gates = {
            "momentum_ok": mom_score > 65,
            "persistence_ok": p_raw > 55,
            "pvel_ok": p_vel > 0 and not pvel.get("block_late_entry"),
            "accel_ok": p_acc > 0 or pvel.get("n", 0) < 3,
            "premium": p_raw > 60 and p_vel > 5 and p_acc > 2,
            "require_hpp_velocity_positive": True,
        }
    elif ct in {"PUT", "FALL", "LOWER"}:
        rf_gates = {
            "momentum_ok": mom_score < 35,
            "persistence_ok": p_raw > 55,
            "pvel_ok": p_vel > 0 and not pvel.get("block_late_entry"),
            "accel_ok": p_acc > 0 or pvel.get("n", 0) < 3,
            "premium": p_raw > 60 and p_vel > 5 and p_acc > 2,
            "require_hpp_velocity_positive": True,
        }

    return {
        "symbol": symbol,
        "contract_type": ct,
        "momentum": mom,
        "persistence": pers,
        "transition": tm,
        "momentum_persistence": mp,
        "mp_score": mp["momentum_persistence"],
        "persistence_velocity": pvel,
        "persistence_engine_score": pvel.get("persistence_engine_score"),
        "oriented_momentum": round(oriented_mom, 1),
        "digit_confirmation": {
            "note": digit_note,
            "confidence_delta": conf_delta,
            "mp_score": mp_val,
        },
        "rf_gates": rf_gates,
        "display": {
            "momentum": f"{mom.get('label')} ({mom_score:.0f})",
            "persistence": f"{pers.get('label')} ({pers.get('persistence')})",
            "mp": mp["momentum_persistence"],
            "transition": tm.get("display"),
            "p_velocity": pvel.get("status"),
            "p_vel": pvel.get("velocity"),
            "p_accel": pvel.get("acceleration"),
            "fast_med_slow": (
                f"F{pvel.get('fast_velocity', 0):+.0f} "
                f"M{pvel.get('medium_velocity', 0):+.0f} "
                f"S{pvel.get('slow_velocity', 0):+.0f}"
            ),
            "engine": pvel.get("persistence_engine_score"),
            "interpretation": pvel.get("interpretation"),
            "sample_conf": (pers.get("sample_confidence") or {}).get("label"),
        },
        "ready": len([d for d in dirs if d != 0]) >= 15,
    }


def rf_mp_gates(
    analysis: Dict[str, Any],
    *,
    trade_quality: float,
    hpp_velocity: float,
    contract_type: str,
    premium: bool = False,
) -> Dict[str, Any]:
    """
    Rise base:
      Persistence > 55% AND Persistence Velocity > 0 AND Acceleration > 0
      (+ Momentum > 65 · TQ > 80 · HPP Vel > 0)

    Premium:
      Persistence > 60 · Velocity > +5 · Acceleration > +2
    """
    ct = str(contract_type or analysis.get("contract_type") or "").upper()
    mom = float((analysis.get("momentum") or {}).get("momentum_score") or 50)
    pers = float((analysis.get("persistence") or {}).get("persistence") or 50)
    conf = (analysis.get("persistence") or {}).get("sample_confidence") or {}
    pvel = analysis.get("persistence_velocity") or {}
    p_vel = float(pvel.get("velocity") or 0)
    p_acc = float(pvel.get("acceleration") or 0)
    n_hist = int(pvel.get("n") or 0)
    checks = []

    def _c(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    if ct in {"CALL", "RISE", "HIGHER"}:
        _c("momentum", mom > 65, f"Momentum {mom:.0f} > 65")
    elif ct in {"PUT", "FALL", "LOWER"}:
        _c("momentum", mom < 35, f"Momentum {mom:.0f} < 35 (bearish)")
    else:
        _c("momentum", True, "n/a")

    min_pers = 60.0 if premium else 55.0
    min_vel = 5.0 if premium else 0.0
    min_acc = 2.0 if premium else 0.0

    _c("persistence", pers > min_pers, f"Persistence {pers:.0f} > {min_pers:.0f}")
    _c("trade_quality", float(trade_quality) > 80, f"TQ {trade_quality:.0f} > 80")
    _c("hpp_velocity", float(hpp_velocity) > 0, f"HPP Vel {hpp_velocity:+.1f} > 0")

    # Velocity > 0 (or > +5 premium); skip strict if insufficient history
    if n_hist >= 3:
        _c(
            "persistence_velocity",
            p_vel > min_vel and not pvel.get("block_late_entry"),
            f"P-Vel {p_vel:+.1f} > {min_vel:.0f} ({pvel.get('status')})",
        )
        _c(
            "persistence_acceleration",
            p_acc > min_acc,
            f"P-Accel {p_acc:+.1f} > {min_acc:.0f}",
        )
    else:
        _c(
            "persistence_velocity",
            not pvel.get("block_late_entry"),
            f"P-Vel history n={n_hist} (soft)",
        )

    _c(
        "sample_confidence",
        bool(conf.get("trustworthy")) or int(conf.get("n") or 0) >= 40,
        f"Persist sample conf {conf.get('label')}",
    )

    ok = all(c["ok"] for c in checks)
    return {
        "allow": ok,
        "checks": checks,
        "status": "PASS" if ok else "FAIL",
        "premium": premium,
        "reason": next(
            (c["detail"] for c in checks if not c["ok"]),
            "RF MP + velocity gates passed",
        ),
    }


def validate_persistence_velocity_edge(
    outcomes: Sequence[Dict[str, Any]],
    *,
    min_samples: int = 1000,
    auto_reduce: bool = True,
) -> Dict[str, Any]:
    """
    Group A: Persistence Velocity > 0  vs  Group B: < 0

    Expected: WR(A) > WR(B) and PF(A) > PF(B).
    If not after large sample, auto-reduce velocity weight.
    """
    a = [
        r
        for r in outcomes
        if r.get("persistence_velocity") is not None
        and float(r["persistence_velocity"]) > 0
    ]
    b = [
        r
        for r in outcomes
        if r.get("persistence_velocity") is not None
        and float(r["persistence_velocity"]) < 0
    ]

    def _stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not rows:
            return {"n": 0, "wr": 0.0, "pf": 1.0}
        wins = sum(1 for r in rows if r.get("is_win"))
        gp = sum(float(r.get("profit") or 0) for r in rows if float(r.get("profit") or 0) > 0)
        gl = abs(
            sum(float(r.get("profit") or 0) for r in rows if float(r.get("profit") or 0) < 0)
        )
        pf = gp / gl if gl > 1e-9 else (10.0 if gp > 0 else 1.0)
        return {"n": len(rows), "wr": wins / len(rows), "pf": pf}

    sa, sb = _stats(a), _stats(b)
    enough = sa["n"] >= min_samples // 2 and sb["n"] >= min_samples // 2
    wr_ok = sa["wr"] > sb["wr"] if enough else True
    pf_ok = sa["pf"] >= sb["pf"] if enough else True
    passed = wr_ok and pf_ok
    weight_action = None
    if enough and not passed and auto_reduce:
        try:
            new_w = get_persistence_velocity_tracker().reduce_velocity_weight(0.5)
            weight_action = f"reduced velocity weight → {new_w:.2f}"
        except Exception as e:
            weight_action = f"reduce failed: {e}"

    return {
        "group_a": {
            "label": "P-Vel > 0",
            "n": sa["n"],
            "wr": round(sa["wr"] * 100, 1),
            "pf": round(sa["pf"], 3),
        },
        "group_b": {
            "label": "P-Vel < 0",
            "n": sb["n"],
            "wr": round(sb["wr"] * 100, 1),
            "pf": round(sb["pf"], 3),
        },
        "enough_sample": enough,
        "min_samples": min_samples,
        "pass": passed,
        "weight_action": weight_action,
        "display": (
            f"P-Vel>0 WR={sa['wr']*100:.0f}% PF={sa['pf']:.2f} (n={sa['n']}) vs "
            f"P-Vel<0 WR={sb['wr']*100:.0f}% PF={sb['pf']:.2f} (n={sb['n']}) · "
            f"{'PASS' if passed else 'FAIL'}"
            + (f" · {weight_action}" if weight_action else "")
        ),
    }
