"""
Market Opportunity Ranking (MOR)

Shift from:  "Is this a good trade?"
         to:  "Across all markets, where is the BEST edge right now?"

Production formula:

  Opportunity =
    20% Pattern Strength
  + 15% Pattern Clarity
  + 15% HPP
  + 10% HPP Velocity (mapped 0–100)
  + 15% Momentum Persistence
  + 10% Regime Match
  + 10% Expected Value (mapped)
  +  5% Confidence

  Final Score = Opportunity − Risk Penalties

Tiers: ELITE 90+ · STRONG 80–89 · WATCHLIST 70–79 · IGNORE <70

Tracks opportunity velocity & acceleration for emerging edges.
Validates top vs bottom after every 1000 trades.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_MOR_PATH = Path("data/mor_history.json")
VALIDATE_EVERY_N = 1000

# Weights (production)
W_STRENGTH = 0.20
W_CLARITY = 0.15
W_HPP = 0.15
W_HPP_VEL = 0.10
W_MP = 0.15
W_REGIME = 0.10
W_EV = 0.10
W_CONF = 0.05


def hpp_velocity_to_score(velocity: float) -> float:
    """Map HPP velocity (~−15..+15) → 0–100. +15 → 100, 0 → 50, −15 → 0."""
    return max(0.0, min(100.0, 50.0 + float(velocity) * 3.33))


def ev_to_score(ev: float) -> float:
    """Map EV (typical −0.2..+0.3) → 0–100."""
    # EV 0 → 50, +0.15 → ~95, −0.1 → ~20
    return max(0.0, min(100.0, 50.0 + float(ev) * 300.0))


def opportunity_score(
    *,
    pattern_strength: float,
    pattern_clarity: float,
    hpp: float,
    hpp_velocity: float,
    momentum_persistence: float,
    regime_match: float,
    expected_value: float = 0.0,
    confidence: float = 50.0,  # 0–100
) -> Dict[str, Any]:
    """
    Weighted opportunity before penalties.
    """
    vel_s = hpp_velocity_to_score(hpp_velocity)
    ev_s = ev_to_score(expected_value)
    conf = max(0.0, min(100.0, float(confidence)))

    raw = (
        W_STRENGTH * float(pattern_strength)
        + W_CLARITY * float(pattern_clarity)
        + W_HPP * float(hpp)
        + W_HPP_VEL * vel_s
        + W_MP * float(momentum_persistence)
        + W_REGIME * float(regime_match)
        + W_EV * ev_s
        + W_CONF * conf
    )
    raw = max(0.0, min(100.0, raw))
    return {
        "opportunity_raw": round(raw, 1),
        "components": {
            "pattern_strength": round(float(pattern_strength), 1),
            "pattern_clarity": round(float(pattern_clarity), 1),
            "hpp": round(float(hpp), 1),
            "hpp_velocity_score": round(vel_s, 1),
            "momentum_persistence": round(float(momentum_persistence), 1),
            "regime_match": round(float(regime_match), 1),
            "ev_score": round(ev_s, 1),
            "confidence": round(conf, 1),
        },
        "weights": {
            "pattern_strength": W_STRENGTH,
            "pattern_clarity": W_CLARITY,
            "hpp": W_HPP,
            "hpp_velocity": W_HPP_VEL,
            "momentum_persistence": W_MP,
            "regime_match": W_REGIME,
            "expected_value": W_EV,
            "confidence": W_CONF,
        },
    }


def risk_penalties(
    *,
    drawdown_high: bool = False,
    drawdown_pct: float = 0.0,
    hpp_unstable: bool = False,
    sample_n: int = 0,
    min_samples_full: int = 50,
) -> Dict[str, Any]:
    """
    Subtract risk from opportunity.
    High DD → −10, unstable HPP → −5, low samples → up to −15.
    """
    pen = 0.0
    parts = []
    if drawdown_high or float(drawdown_pct) >= 0.05:
        pen += 10.0
        parts.append({"name": "drawdown", "penalty": 10.0})
    elif float(drawdown_pct) >= 0.03:
        pen += 5.0
        parts.append({"name": "drawdown_mild", "penalty": 5.0})

    if hpp_unstable:
        pen += 5.0
        parts.append({"name": "hpp_unstable", "penalty": 5.0})

    n = int(sample_n)
    if n < 20:
        p = 15.0
        pen += p
        parts.append({"name": "low_confidence_samples", "penalty": p, "n": n})
    elif n < min_samples_full:
        # linear 15 → 0 from 20..50
        p = 15.0 * (1.0 - (n - 20) / max(1, min_samples_full - 20))
        pen += p
        parts.append({"name": "sample_penalty", "penalty": round(p, 1), "n": n})

    return {"total_penalty": round(pen, 1), "parts": parts}


def opportunity_tier(score: float) -> str:
    s = float(score)
    if s >= 90:
        return "ELITE"
    if s >= 80:
        return "STRONG"
    if s >= 70:
        return "WATCHLIST"
    return "IGNORE"


def regime_match_score(
    *,
    family: str,
    market_regime: str,
    chop_score: float = 0.0,
    strategy_path: str = "",
) -> float:
    """
    Does current condition fit the strategy?
    Trend RF likes non-choppy; digits like stable/biased.
    """
    reg = str(market_regime or "RANDOM").upper().replace("_", " ")
    chop = float(chop_score or 0)
    path = str(strategy_path or "")
    fam = str(family or "")

    base = 55.0
    if fam in {"rise_fall", "minute_rise_fall"} or path in {
        "directional",
        "spike",
    }:
        if reg in {"STRONG PATTERN", "BIASED", "EMERGING PATTERN"}:
            base = 90.0
        elif reg in {"BALANCED", "NORMAL"}:
            base = 75.0
        elif reg == "RANDOM":
            base = 35.0
        # chop hurts RF
        base = max(15.0, base - chop * 50.0)
    else:
        # digits
        if reg in {"STRONG PATTERN", "EXTREME ANOMALY", "HIGH CLUSTERING", "BIASED"}:
            base = 88.0
        elif reg in {"BALANCED", "NORMAL"}:
            base = 70.0
        elif reg == "RANDOM":
            base = 40.0
    return max(0.0, min(100.0, base))


def compute_mor(
    *,
    pattern_strength: float,
    pattern_clarity: float,
    hpp: float,
    hpp_velocity: float = 0.0,
    momentum_persistence: float = 50.0,
    regime_match: float = 50.0,
    expected_value: float = 0.0,
    confidence: float = 50.0,
    sample_n: int = 0,
    drawdown_pct: float = 0.0,
    hpp_unstable: bool = False,
) -> Dict[str, Any]:
    """
    Full MOR for one market.
    Final = Opportunity − Penalties, then × confidence factor for effective rank.
    """
    opp = opportunity_score(
        pattern_strength=pattern_strength,
        pattern_clarity=pattern_clarity,
        hpp=hpp,
        hpp_velocity=hpp_velocity,
        momentum_persistence=momentum_persistence,
        regime_match=regime_match,
        expected_value=expected_value,
        confidence=confidence,
    )
    pen = risk_penalties(
        drawdown_pct=drawdown_pct,
        hpp_unstable=hpp_unstable,
        sample_n=sample_n,
    )
    final = max(0.0, min(100.0, float(opp["opportunity_raw"]) - float(pen["total_penalty"])))
    conf01 = max(0.05, min(1.0, float(confidence) / 100.0))
    # Effective score = Final × confidence (Step 10)
    effective = final * conf01
    # Also store "rank score" blend: don't fully destroy high quality with mid conf
    rank_score = 0.65 * final + 0.35 * (final * conf01)

    return {
        "opportunity_score": round(final, 1),
        "opportunity_raw": opp["opportunity_raw"],
        "effective_score": round(effective, 1),
        "rank_score": round(rank_score, 1),
        "tier": opportunity_tier(final),
        "penalties": pen,
        "components": opp["components"],
        "weights": opp["weights"],
        "confidence": round(float(confidence), 1),
        "sample_n": int(sample_n),
    }


# ---------------------------------------------------------------------------
# History: velocity & acceleration of opportunity scores
# ---------------------------------------------------------------------------

class OpportunityHistory:
    """Track opportunity score path per market for velocity/acceleration."""

    def __init__(self, path: Optional[Path] = None, max_points: int = 100):
        self.path = Path(path) if path else DEFAULT_MOR_PATH
        self.max_points = max_points
        # symbol -> list of {ts, score, tier}
        self.series: Dict[str, List[Dict[str, Any]]] = {}
        self.validation: Optional[Dict[str, Any]] = None
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.series = data.get("series") or {}
            self.validation = data.get("validation")
        except Exception:
            self.series = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "series": self.series,
                        "validation": self.validation,
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def note(self, symbol: str, score: float, tier: str = "") -> Dict[str, Any]:
        k = str(symbol).upper()
        arr = self.series.setdefault(k, [])
        arr.append({"ts": time.time(), "score": float(score), "tier": tier})
        if len(arr) > self.max_points:
            self.series[k] = arr[-self.max_points :]
        self.save()
        return self.velocity_pack(k)

    def velocity_pack(self, symbol: str) -> Dict[str, Any]:
        k = str(symbol).upper()
        arr = list(self.series.get(k) or [])
        if len(arr) < 2:
            return {
                "velocity": 0.0,
                "acceleration": 0.0,
                "path": [a["score"] for a in arr[-5:]],
                "emerging": False,
            }
        scores = [float(a["score"]) for a in arr]
        vel = scores[-1] - scores[-2]
        if len(scores) >= 3:
            prev_vel = scores[-2] - scores[-3]
            acc = vel - prev_vel
        else:
            acc = 0.0
        # Multi-horizon
        short = scores[-1]
        medium = sum(scores[-5:]) / min(5, len(scores))
        long = sum(scores[-15:]) / min(15, len(scores)) if len(scores) >= 3 else medium
        emerging = short > medium > long and vel > 0
        return {
            "velocity": round(vel, 2),
            "acceleration": round(acc, 2),
            "path": [round(s, 1) for s in scores[-8:]],
            "short_term": round(short, 1),
            "medium_term": round(medium, 1),
            "long_term": round(long, 1),
            "emerging": emerging,
            "interpretation": (
                "Fresh opportunity emerging"
                if emerging
                else (
                    "Strengthening"
                    if vel > 2
                    else ("Weakening" if vel < -2 else "Stable")
                )
            ),
        }


_hist: Optional[OpportunityHistory] = None


def get_opportunity_history() -> OpportunityHistory:
    global _hist
    if _hist is None:
        _hist = OpportunityHistory()
    return _hist


def correlation_filter(
    ranked: Sequence[Dict[str, Any]],
    *,
    corr_threshold: float = 0.90,
    max_per_cluster: int = 1,
) -> List[Dict[str, Any]]:
    """
    Drop highly correlated synthetics when both rank high.
    Heuristic: same category + adjacent volatility labels.
    """
    kept: List[Dict[str, Any]] = []
    seen_clusters: Dict[str, int] = {}

    def cluster_id(row: Dict[str, Any]) -> str:
        cat = str(row.get("category") or "unknown")
        sym = str(row.get("symbol") or "").upper()
        # Group R_* and 1HZ* as synthetic cluster families
        if cat == "synthetic_vol":
            if sym.startswith("R_"):
                return "synth_r"
            if sym.startswith("1HZ"):
                return "synth_1hz"
            return "synth_other"
        if cat in {"boom", "crash"}:
            return cat
        return f"{cat}:{sym}"

    for row in ranked:
        cid = cluster_id(row)
        n = seen_clusters.get(cid, 0)
        # Always keep ELITE first of cluster; skip lower correlated copies
        if n >= max_per_cluster and str(row.get("tier") or "") != "ELITE":
            row = {**row, "correlation_filtered": True, "tradeable": False}
            # still include for transparency but mark filtered
            kept.append(row)
            continue
        if not row.get("correlation_filtered"):
            seen_clusters[cid] = n + 1
        kept.append(row)
    return kept


def multi_horizon_label(pack: Dict[str, Any]) -> str:
    s = float(pack.get("short_term") or 0)
    m = float(pack.get("medium_term") or 0)
    lo = float(pack.get("long_term") or 0)
    if s >= 90 and m < 85:
        return "Fresh opportunity emerging"
    if s > m > lo and pack.get("velocity", 0) > 0:
        return "Building edge"
    if s < m < lo:
        return "Fading edge"
    return pack.get("interpretation") or "Stable"


def validate_mor_ranking(
    outcomes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    After many trades: top-ranked markets must beat bottom on WR/PF/DD.
    Each outcome: {symbol, is_win, profit, mor_score_at_entry?, tier?}
    """
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    for o in outcomes:
        by_sym.setdefault(str(o.get("symbol") or "").upper(), []).append(o)

    def stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        if not rows:
            return {"n": 0, "wr": 0.0, "pf": 1.0, "pnl": 0.0}
        n = len(rows)
        wins = sum(1 for r in rows if r.get("is_win"))
        gp = sum(float(r.get("profit") or 0) for r in rows if float(r.get("profit") or 0) > 0)
        gl = abs(sum(float(r.get("profit") or 0) for r in rows if float(r.get("profit") or 0) < 0))
        pf = gp / gl if gl > 1e-9 else (10.0 if gp > 0 else 1.0)
        return {"n": n, "wr": wins / n, "pf": pf, "pnl": sum(float(r.get("profit") or 0) for r in rows)}

    # Split by average mor_score if present
    elite_rows, strong_rows, low_rows = [], [], []
    vel_pos, vel_neg = [], []
    conf_hi, conf_lo = [], []
    for o in outcomes:
        tier = str(o.get("tier") or "")
        score = o.get("mor_score")
        if score is not None:
            if float(score) >= 90:
                elite_rows.append(o)
            elif float(score) >= 80:
                strong_rows.append(o)
            elif float(score) < 70:
                low_rows.append(o)
        if o.get("opp_velocity") is not None:
            if float(o["opp_velocity"]) > 0:
                vel_pos.append(o)
            else:
                vel_neg.append(o)
        if o.get("confidence") is not None:
            if float(o["confidence"]) >= 80:
                conf_hi.append(o)
            elif float(o["confidence"]) < 50:
                conf_lo.append(o)

    se, ss, sl = stats(elite_rows), stats(strong_rows), stats(low_rows)
    checks = {
        "elite_beats_strong": (
            se["n"] < 20
            or ss["n"] < 20
            or (se["wr"] >= ss["wr"] - 0.02 and se["pf"] >= ss["pf"] * 0.95)
        ),
        "top_beats_bottom": (
            se["n"] < 20
            or sl["n"] < 20
            or (se["wr"] > sl["wr"] and se["pf"] > sl["pf"])
        ),
        "velocity_positive_better": (
            len(vel_pos) < 20
            or len(vel_neg) < 20
            or stats(vel_pos)["wr"] >= stats(vel_neg)["wr"] - 0.02
        ),
        "high_confidence_better": (
            len(conf_hi) < 20
            or len(conf_lo) < 20
            or stats(conf_hi)["wr"] >= stats(conf_lo)["wr"] - 0.02
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "elite": se,
        "strong": ss,
        "ignore": sl,
        "n_outcomes": len(outcomes),
    }
