"""
No-Trade / Decision Engine — block bad trades even when a signal looks good.

Elite systems optimize: "When should I NOT trade?"

Gates:
  - Pattern Clarity < 75 → BLOCK
  - HPP Velocity < -5 → BLOCK (pattern degrading)
  - Entropy Stability < 60 → BLOCK
  - Trade Quality Score < 80 → BLOCK
  - Expected Value ≤ 0 → BLOCK
  - Ensemble disagreement → BLOCK
  - Regime incompatible with contract → BLOCK
  - Edge decay past threshold → BLOCK / retire setup
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# --- Default hard blocks ---
MIN_CLARITY = 75.0
MIN_HPP_VELOCITY = -5.0
MIN_ENTROPY_STABILITY = 60.0
MIN_TRADE_QUALITY = 80.0
MIN_EV = 0.0
MAX_EDGE_DECAY_PCT = 20.0  # retire if peak→current decay exceeds this


# Regime → allowed contract families / types
REGIME_ALLOWED: Dict[str, Set[str]] = {
    "RANDOM": set(),
    "NORMAL": {"DIGITEVEN", "DIGITODD"},  # only soft parity if anything
    "BALANCED": {"DIGITEVEN", "DIGITODD", "CALL", "PUT"},
    "EMERGING PATTERN": {
        "DIGITDIFF",
        "DIGITOVER",
        "DIGITUNDER",
        "DIGITEVEN",
        "DIGITODD",
        "CALL",
        "PUT",
    },
    "STRONG PATTERN": {
        "DIGITDIFF",
        "DIGITOVER",
        "DIGITUNDER",
        "DIGITEVEN",
        "DIGITODD",
        "DIGITMATCH",
        "CALL",
        "PUT",
    },
    "HIGH CLUSTERING": {
        "DIGITMATCH",
        "DIGITDIFF",
        "DIGITEVEN",
        "DIGITODD",
    },
    # map engine regimes
    "BIASED": {
        "DIGITDIFF",
        "DIGITOVER",
        "DIGITUNDER",
        "DIGITEVEN",
        "DIGITODD",
        "CALL",
        "PUT",
    },
    "EXTREME ANOMALY": {
        "DIGITDIFF",
        "DIGITMATCH",
        "DIGITEVEN",
        "DIGITODD",
    },
}


def normalize_regime(raw: Optional[str]) -> str:
    r = str(raw or "RANDOM").strip().upper().replace("_", " ")
    aliases = {
        "STRONG PATTERN": "STRONG PATTERN",
        "EXTREME ANOMALY": "EXTREME ANOMALY",
        "EMERGING PATTERN": "EMERGING PATTERN",
        "HIGH CLUSTERING": "HIGH CLUSTERING",
        "BALANCED": "BALANCED",
        "BIASED": "BIASED",
        "NORMAL": "NORMAL",
        "RANDOM": "RANDOM",
    }
    return aliases.get(r, r if r in REGIME_ALLOWED else "RANDOM")


def map_engine_regime(rolling_regime: Optional[str], rt_strength: float = 0.0) -> str:
    """Map rolling entropy regime labels to decision regimes."""
    r = str(rolling_regime or "RANDOM").upper()
    if r in {"RANDOM"}:
        return "RANDOM"
    if r in {"NORMAL"}:
        return "NORMAL" if rt_strength < 55 else "BALANCED"
    if r in {"BIASED"}:
        return "EMERGING PATTERN" if rt_strength < 70 else "BIASED"
    if r in {"STRONG PATTERN"}:
        return "STRONG PATTERN"
    if r in {"EXTREME ANOMALY"}:
        return "HIGH CLUSTERING" if rt_strength >= 60 else "EXTREME ANOMALY"
    return normalize_regime(r)


def expected_value(
    p_win: float,
    reward: float,
    risk: float = 1.0,
) -> float:
    """
    EV = (Pwin × Reward) − (Ploss × Risk)
    Reward = net win multiple of stake (e.g. 0.92 for payout 1.92)
    Risk = 1.0 for full stake loss
    """
    p = max(0.0, min(1.0, float(p_win)))
    rew = max(0.0, float(reward))
    risk_f = max(0.0, float(risk))
    return p * rew - (1.0 - p) * risk_f


def trade_quality_score(
    *,
    pattern_strength: float,
    pattern_clarity: float,
    hpp: float,
    hpp_velocity: float,
    confidence: float,
    momentum_persistence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Final Trade Quality (dual system):

      30% Pattern Strength
    + 20% Pattern Clarity
    + 15% HPP
    + 10% HPP Velocity (mapped 0–100)
    + 15% Momentum Persistence   ← second engine
    + 10% Confidence

    If momentum_persistence is omitted, classic 5-factor weights are used.
    """
    vel = float(hpp_velocity)
    vel_score = max(0.0, min(100.0, 50.0 + vel * 3.33))

    if momentum_persistence is not None:
        mp = float(momentum_persistence)
        total = (
            0.30 * float(pattern_strength)
            + 0.20 * float(pattern_clarity)
            + 0.15 * float(hpp)
            + 0.10 * vel_score
            + 0.15 * mp
            + 0.10 * float(confidence)
        )
        weights = {
            "pattern_strength": 0.30,
            "pattern_clarity": 0.20,
            "hpp": 0.15,
            "hpp_velocity": 0.10,
            "momentum_persistence": 0.15,
            "confidence": 0.10,
        }
        components = {
            "pattern_strength": round(float(pattern_strength), 1),
            "pattern_clarity": round(float(pattern_clarity), 1),
            "hpp": round(float(hpp), 1),
            "hpp_velocity_score": round(vel_score, 1),
            "momentum_persistence": round(mp, 1),
            "confidence": round(float(confidence), 1),
        }
    else:
        total = (
            0.30 * float(pattern_strength)
            + 0.25 * float(pattern_clarity)
            + 0.20 * float(hpp)
            + 0.15 * vel_score
            + 0.10 * float(confidence)
        )
        weights = {
            "pattern_strength": 0.30,
            "pattern_clarity": 0.25,
            "hpp": 0.20,
            "hpp_velocity": 0.15,
            "confidence": 0.10,
        }
        components = {
            "pattern_strength": round(float(pattern_strength), 1),
            "pattern_clarity": round(float(pattern_clarity), 1),
            "hpp": round(float(hpp), 1),
            "hpp_velocity_score": round(vel_score, 1),
            "confidence": round(float(confidence), 1),
        }

    total = max(0.0, min(100.0, total))
    return {
        "trade_quality": round(total, 1),
        "final_quality": round(total, 1),
        "components": components,
        "weights": weights,
        "auto_ok": total >= MIN_TRADE_QUALITY,
    }


def risk_pct_from_quality(quality: float) -> float:
    """
    90+ → 1.0% · 80–90 → 0.5% · below 80 → 0% (no trade)
    """
    q = float(quality)
    if q >= 90:
        return 1.0
    if q >= 80:
        return 0.5
    return 0.0


def ensemble_votes(
    *,
    entropy_buy: bool,
    pattern_buy: bool,
    hpp_buy: bool,
    probability_buy: bool,
) -> Dict[str, Any]:
    """Trade only when all engines agree BUY."""
    votes = {
        "entropy": entropy_buy,
        "pattern": pattern_buy,
        "hpp": hpp_buy,
        "probability": probability_buy,
    }
    n_yes = sum(1 for v in votes.values() if v)
    return {
        "votes": votes,
        "agree": n_yes == 4,
        "n_yes": n_yes,
        "n_engines": 4,
    }


def edge_decay_pct(peak_hpp: float, current_hpp: float) -> float:
    peak = max(1e-6, float(peak_hpp))
    cur = float(current_hpp)
    if cur >= peak:
        return 0.0
    return max(0.0, (peak - cur) / peak * 100.0)


def evaluate_no_trade(
    *,
    contract_type: str,
    family: str = "digits",
    pattern_clarity: float = 50.0,
    pattern_strength: float = 50.0,
    hpp: float = 50.0,
    hpp_velocity: float = 0.0,
    entropy_stability: float = 55.0,
    confidence: float = 50.0,  # 0–100
    signal_confidence: float = 0.5,  # 0–1 live conf
    p_win: Optional[float] = None,
    reward: float = 0.92,  # net on win / stake
    risk: float = 1.0,
    regime_raw: Optional[str] = None,
    realtime_pattern_strength: float = 0.0,
    peak_hpp: Optional[float] = None,
    cold_start: bool = False,
    momentum_persistence: Optional[float] = None,
    min_confidence: float = 70.0,
    # ensemble inputs (optional; inferred if None)
    entropy_buy: Optional[bool] = None,
    pattern_buy: Optional[bool] = None,
    hpp_buy: Optional[bool] = None,
    probability_buy: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Central decision: ALLOW or REJECT with explicit reasons.

    Final Quality ≥ 80 AND EV > 0 AND Regime Compatible AND Confidence ≥ 70.
    """
    ct = str(contract_type or "").upper()
    regime = map_engine_regime(regime_raw, realtime_pattern_strength)
    allowed = REGIME_ALLOWED.get(regime, set())

    # Soften slightly in cold-start so learning can begin (still EV-gated)
    min_clarity = 65.0 if cold_start else MIN_CLARITY
    min_stability = 50.0 if cold_start else MIN_ENTROPY_STABILITY
    min_quality = 70.0 if cold_start else MIN_TRADE_QUALITY
    min_vel = -8.0 if cold_start else MIN_HPP_VELOCITY
    min_conf = 60.0 if cold_start else float(min_confidence)

    tq = trade_quality_score(
        pattern_strength=pattern_strength,
        pattern_clarity=pattern_clarity,
        hpp=hpp,
        hpp_velocity=hpp_velocity,
        confidence=confidence,
        momentum_persistence=momentum_persistence,
    )
    quality = float(tq["trade_quality"])

    # P(win) estimate
    if p_win is None:
        # blend signal conf with quality
        p_win = 0.55 * float(signal_confidence) + 0.45 * (quality / 100.0)
    p_win = max(0.01, min(0.99, float(p_win)))
    ev = expected_value(p_win, reward=reward, risk=risk)

    # Ensemble defaults from scores
    if entropy_buy is None:
        entropy_buy = realtime_pattern_strength >= 55 or regime in {
            "STRONG PATTERN",
            "EMERGING PATTERN",
            "BIASED",
            "HIGH CLUSTERING",
        }
    if pattern_buy is None:
        pattern_buy = float(pattern_strength) >= 70 and float(pattern_clarity) >= min_clarity
    if hpp_buy is None:
        hpp_buy = float(hpp) >= 65 and float(hpp_velocity) >= min_vel
    if probability_buy is None:
        probability_buy = float(signal_confidence) >= 0.78 and p_win >= 0.55

    ens = ensemble_votes(
        entropy_buy=bool(entropy_buy),
        pattern_buy=bool(pattern_buy),
        hpp_buy=bool(hpp_buy),
        probability_buy=bool(probability_buy),
    )

    decay = 0.0
    retired = False
    if peak_hpp is not None and float(peak_hpp) > 0:
        decay = edge_decay_pct(float(peak_hpp), float(hpp))
        retired = decay >= MAX_EDGE_DECAY_PCT and not cold_start

    reasons: List[str] = []
    blocks: List[str] = []

    def _block(cond: bool, msg: str) -> None:
        if cond:
            blocks.append(msg)
            reasons.append(f"✗ {msg}")
        else:
            reasons.append(f"✓ pass: {msg.split(':')[0]}")

    _block(pattern_clarity < min_clarity, f"Pattern Clarity {pattern_clarity:.0f} < {min_clarity:.0f}")
    _block(hpp_velocity < min_vel, f"HPP Velocity {hpp_velocity:+.1f} < {min_vel:.0f} (pattern degrading)")
    _block(
        entropy_stability < min_stability,
        f"Entropy Stability {entropy_stability:.0f} < {min_stability:.0f}",
    )
    _block(quality < min_quality, f"Trade Quality {quality:.0f} < {min_quality:.0f}")
    _block(
        float(confidence) < min_conf,
        f"Confidence {confidence:.0f} < {min_conf:.0f}",
    )
    _block(ev <= MIN_EV, f"EV {ev:+.3f} ≤ 0 (no positive expectancy)")
    _block(not ens["agree"], f"Ensemble disagreement ({ens['n_yes']}/4 engines BUY)")
    _block(
        ct not in allowed and regime == "RANDOM",
        f"Regime {regime}: no contracts allowed",
    )
    if ct and allowed and ct not in allowed and regime != "RANDOM":
        _block(True, f"Regime {regime} does not allow {ct}")
    _block(retired, f"Edge decay {decay:.0f}% ≥ {MAX_EDGE_DECAY_PCT:.0f}% — retire strategy")

    risk_pct = risk_pct_from_quality(quality)
    if risk_pct <= 0 and quality < min_quality:
        _block(True, f"Risk sizing 0% (quality {quality:.0f})")

    # During cold-start, ensemble/regime may be too strict — require EV + quality soft
    # RANDOM is normally blocked, but very high conf + positive EV may still learn.
    def _cold_soft_ok() -> bool:
        conf_ok = float(signal_confidence) >= 0.80
        if regime == "RANDOM":
            conf_ok = float(signal_confidence) >= 0.85
        return (
            ev > 0
            and quality >= 65
            and conf_ok
            and hpp_velocity >= -10
            and pattern_clarity >= 55
            and not retired
        )

    if cold_start and blocks and _cold_soft_ok():
        # Drop ensemble + regime soft blocks; keep clarity/EV/velocity critical
        drop_keys = ("Ensemble", "Regime", "Risk sizing")
        blocks = [b for b in blocks if not any(k in b for k in drop_keys)]
        # Still hard-block pure clarity/EV failures
        if pattern_clarity < 55 or ev <= 0:
            pass
        else:
            reasons.append("✓ Cold-start soft path (EV+conf) while learning")

    allowed_trade = len(blocks) == 0
    if cold_start and not allowed_trade and _cold_soft_ok():
        allowed_trade = True
        blocks = []
        reasons.append("✓ Cold-start allow: positive EV + high conf")

    status = "ALLOWED" if allowed_trade else "REJECTED"
    primary_reason = blocks[0] if blocks else "All no-trade gates passed"

    return {
        "status": status,
        "allow": allowed_trade,
        "reason": primary_reason,
        "reasons": reasons,
        "blocks": blocks,
        "regime": regime,
        "regime_allowed_contracts": sorted(allowed),
        "trade_quality": tq,
        "ev": round(ev, 4),
        "p_win": round(p_win, 4),
        "reward": reward,
        "risk_pct": risk_pct,
        "ensemble": ens,
        "edge_decay_pct": round(decay, 1),
        "retired": retired,
        "cold_start": cold_start,
        "display": {
            "signal": "FOUND",
            "status": status,
            "reason": primary_reason if not allowed_trade else "OK",
            "trade_quality": quality,
            "ev": round(ev, 4),
            "regime": regime,
            "risk_pct": risk_pct,
        },
    }
