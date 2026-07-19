"""
Meta-Validator — all key signals must agree before a trade is APPROVED.

Blocks decaying or partial-agreement setups even when some scores look strong.

Required agreement:
  Pattern Strength ≥ min
  Pattern Clarity  ≥ min
  HPP              ≥ min
  Velocity         ≥ min_vel  (negative velocity = decaying edge → BLOCK)
  EV               > 0
  Confidence       ≥ min
  Regime           allows contract
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from src.analytics.no_trade_engine import (
    MIN_CLARITY,
    MIN_EV,
    MIN_HPP_VELOCITY,
    MIN_TRADE_QUALITY,
    expected_value,
    map_engine_regime,
    REGIME_ALLOWED,
    trade_quality_score,
)

MIN_STRENGTH = 75.0
MIN_HPP = 65.0
MIN_CONFIDENCE = 75.0  # 0–100 scale


def meta_validate(
    *,
    contract_type: str,
    pattern_strength: float,
    pattern_clarity: float,
    hpp: float,
    hpp_velocity: float,
    confidence: float,  # 0–100
    p_win: float,
    reward: float = 0.95,
    risk: float = 1.0,
    regime_raw: Optional[str] = None,
    realtime_pattern_strength: float = 0.0,
    family: str = "digits",
    min_strength: float = MIN_STRENGTH,
    min_clarity: float = MIN_CLARITY,
    min_hpp: float = MIN_HPP,
    min_velocity: float = MIN_HPP_VELOCITY,
    min_confidence: float = 70.0,  # Final formula: Confidence ≥ 70
    momentum_persistence: Optional[float] = None,
    # Optional RF extras
    rf_score: Optional[float] = None,
    vol_tradeable: Optional[bool] = None,
    mp_analysis: Optional[Dict[str, Any]] = None,
    cold_start: bool = False,
) -> Dict[str, Any]:
    """
    Returns status APPROVED | BLOCKED with per-check votes.
    """
    ct = str(contract_type or "").upper()
    regime = map_engine_regime(regime_raw, realtime_pattern_strength)
    allowed: Set[str] = REGIME_ALLOWED.get(regime, set())

    # Soft floors during cold-start (still require EV and non-decaying velocity)
    if cold_start:
        min_strength = min(min_strength, 65.0)
        min_clarity = min(min_clarity, 60.0)
        min_hpp = min(min_hpp, 55.0)
        min_velocity = min(min_velocity, -8.0)
        min_confidence = min(min_confidence, 70.0)

    ev = expected_value(p_win, reward=reward, risk=risk)
    tq = trade_quality_score(
        pattern_strength=pattern_strength,
        pattern_clarity=pattern_clarity,
        hpp=hpp,
        hpp_velocity=hpp_velocity,
        confidence=confidence,
        momentum_persistence=momentum_persistence,
    )
    quality = float(tq["trade_quality"])

    checks: List[Dict[str, Any]] = []

    def _check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    _check(
        "pattern_strength",
        float(pattern_strength) >= min_strength,
        f"{pattern_strength:.0f} ≥ {min_strength:.0f}",
    )
    _check(
        "pattern_clarity",
        float(pattern_clarity) >= min_clarity,
        f"{pattern_clarity:.0f} ≥ {min_clarity:.0f}",
    )
    _check(
        "hpp",
        float(hpp) >= min_hpp,
        f"{hpp:.0f} ≥ {min_hpp:.0f}",
    )
    _check(
        "velocity",
        float(hpp_velocity) >= min_velocity,
        f"{hpp_velocity:+.1f} ≥ {min_velocity:.0f} (not decaying)",
    )
    _check(
        "ev",
        float(ev) > MIN_EV,
        f"EV {ev:+.3f} > 0",
    )
    _check(
        "confidence",
        float(confidence) >= min_confidence,
        f"{confidence:.0f} ≥ {min_confidence:.0f}",
    )
    regime_ok = ct in allowed if allowed is not None else False
    if regime == "RANDOM":
        regime_ok = False
    if cold_start and regime == "RANDOM" and float(confidence) >= 85:
        # Allow learning path noted in no-trade engine
        regime_ok = True
    _check(
        "regime",
        regime_ok,
        f"Regime {regime} allows {ct}" if regime_ok else f"Regime {regime} blocks {ct}",
    )
    _check(
        "trade_quality",
        quality >= (70.0 if cold_start else MIN_TRADE_QUALITY),
        f"TQ {quality:.0f}",
    )

    # Rise/Fall: Momentum/Persistence gates + vol
    if family in {"rise_fall", "minute_rise_fall"} or ct in {
        "CALL",
        "PUT",
        "RISE",
        "FALL",
    }:
        if rf_score is not None:
            _check(
                "rf_score",
                float(rf_score) >= (65.0 if cold_start else 75.0),
                f"RF score {float(rf_score):.0f}",
            )
        if vol_tradeable is not None:
            _check(
                "vol_regime",
                bool(vol_tradeable) or cold_start,
                "Vol tradeable" if vol_tradeable else "Vol EXPANDING/CHAOTIC — block",
            )
        # Exact RF MP gates when analysis provided
        if mp_analysis and not cold_start:
            try:
                from src.analytics.momentum_persistence_engine import rf_mp_gates

                g = rf_mp_gates(
                    mp_analysis,
                    trade_quality=quality,
                    hpp_velocity=hpp_velocity,
                    contract_type=ct,
                )
                _check("rf_mp_gates", bool(g.get("allow")), g.get("reason") or "RF MP")
            except Exception:
                pass
        # HPP velocity must be positive for RF (strict)
        if not cold_start:
            _check(
                "rf_hpp_velocity",
                float(hpp_velocity) > 0,
                f"RF requires HPP Vel > 0 (got {hpp_velocity:+.1f})",
            )

    n_ok = sum(1 for c in checks if c["ok"])
    n_tot = len(checks)
    all_agree = n_ok == n_tot
    status = "APPROVED" if all_agree else "BLOCKED"
    blocked = [c for c in checks if not c["ok"]]
    primary = blocked[0]["detail"] if blocked else "All meta-validator checks passed"

    # Explicit decaying-edge message
    if float(hpp_velocity) < min_velocity:
        primary = f"Edge decaying (velocity {hpp_velocity:+.1f})"

    return {
        "status": status,
        "allow": all_agree,
        "reason": primary,
        "checks": checks,
        "n_ok": n_ok,
        "n_total": n_tot,
        "ev": round(ev, 4),
        "trade_quality": tq,
        "regime": regime,
        "display": {
            "status": status,
            "reason": primary if not all_agree else "APPROVED",
            "agreement": f"{n_ok}/{n_tot}",
            "strength": round(float(pattern_strength), 1),
            "clarity": round(float(pattern_clarity), 1),
            "hpp": round(float(hpp), 1),
            "velocity": round(float(hpp_velocity), 2),
            "ev": round(ev, 4),
            "confidence": round(float(confidence), 1),
            "regime": regime,
        },
    }
