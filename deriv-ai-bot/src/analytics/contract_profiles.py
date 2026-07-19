"""
Contract Profile System — no hard-coded single weight vector.

Layer 1: Metric Engine (signal strengths 0–100)
Layer 2: Contract Profiles (relevance of each metric per contract)
Layer 3: Adaptive Weight Engine
  - base profile weights
  - × current metric strength (dynamic)
  - × sample confidence (reliability)
  - × historical predictive power (self-learning)
  → normalize → weighted clarity

Adding a new Deriv contract = define a new profile, not redesign the engine.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_LEARNING_PATH = Path("data/contract_profile_learning.json")


# ---------------------------------------------------------------------------
# Layer 2 — Contract Weight Registry (base relevance)
# ---------------------------------------------------------------------------

CONTRACT_PROFILES: Dict[str, Dict[str, float]] = {
    # Digit Match — concentration & streaks matter
    "DIGITMATCH": {
        "digit_entropy": 0.35,
        "streak_entropy": 0.30,
        "repetition_bias": 0.20,
        "odd_even_entropy": 0.05,
        "up_down_entropy": 0.10,
    },
    # Digit Differ — dispersion / high entropy / rarity
    "DIGITDIFF": {
        "digit_entropy": 0.50,
        "streak_entropy": 0.20,
        "rarity_score": 0.20,
        "odd_even_entropy": 0.05,
        "up_down_entropy": 0.05,
    },
    # Even / Odd — parity imbalance
    "DIGITEVEN": {
        "parity_entropy": 0.50,
        "parity_momentum": 0.25,
        "digit_entropy": 0.10,
        "streak_entropy": 0.10,
        "stability": 0.05,
    },
    "DIGITODD": {
        "parity_entropy": 0.50,
        "parity_momentum": 0.25,
        "digit_entropy": 0.10,
        "streak_entropy": 0.10,
        "stability": 0.05,
    },
    # Over / Under — threshold imbalance
    "DIGITOVER": {
        "threshold_entropy": 0.45,
        "threshold_momentum": 0.25,
        "digit_entropy": 0.15,
        "streak_entropy": 0.10,
        "stability": 0.05,
    },
    "DIGITUNDER": {
        "threshold_entropy": 0.45,
        "threshold_momentum": 0.25,
        "digit_entropy": 0.15,
        "streak_entropy": 0.10,
        "stability": 0.05,
    },
    # Rise / Fall — directional model (NOT digit-entropy heavy)
    # Momentum 35% · Trend 25% · Vol regime 20% · Persistence 10% · Dir entropy 10%
    "CALL": {
        "momentum": 0.35,
        "trend_strength": 0.25,
        "volatility_score": 0.20,
        "persistence": 0.10,
        "directional_entropy": 0.10,
    },
    "PUT": {
        "momentum": 0.35,
        "trend_strength": 0.25,
        "volatility_score": 0.20,
        "persistence": 0.10,
        "directional_entropy": 0.10,
    },
    # Aliases
    "RISE": {
        "momentum": 0.35,
        "trend_strength": 0.25,
        "volatility_score": 0.20,
        "persistence": 0.10,
        "directional_entropy": 0.10,
    },
    "FALL": {
        "momentum": 0.35,
        "trend_strength": 0.25,
        "volatility_score": 0.20,
        "persistence": 0.10,
        "directional_entropy": 0.10,
    },
}

# Friendly aliases → registry keys
_ALIASES = {
    "MATCH": "DIGITMATCH",
    "DIFFER": "DIGITDIFF",
    "EVEN": "DIGITEVEN",
    "ODD": "DIGITODD",
    "EVEN_ODD": "DIGITEVEN",
    "OVER": "DIGITOVER",
    "UNDER": "DIGITUNDER",
    "OVER_UNDER": "DIGITOVER",
    "RISE_FALL": "CALL",
    "HIGHER": "CALL",
    "LOWER": "PUT",
}


def normalize_contract_key(contract_type: str) -> str:
    ct = str(contract_type or "").strip().upper()
    if ct in CONTRACT_PROFILES:
        return ct
    if ct in _ALIASES:
        return _ALIASES[ct]
    # DIGIT* already upper
    return ct if ct in CONTRACT_PROFILES else "DIGITDIFF"


def get_base_profile(contract_type: str) -> Dict[str, float]:
    key = normalize_contract_key(contract_type)
    prof = dict(CONTRACT_PROFILES.get(key) or CONTRACT_PROFILES["DIGITDIFF"])
    s = sum(prof.values()) or 1.0
    return {k: v / s for k, v in prof.items()}


def list_profiles() -> List[str]:
    return sorted(CONTRACT_PROFILES.keys())


# ---------------------------------------------------------------------------
# Layer 1 helpers — map rolling/hierarchical metrics → profile metric keys
# ---------------------------------------------------------------------------

def build_metric_vector(
    *,
    rolling: Optional[Dict[str, Any]] = None,
    level1: Optional[Dict[str, float]] = None,
    momentum: float = 50.0,
    stability: float = 55.0,
    rarity: float = 50.0,
    repetition_bias: Optional[float] = None,
) -> Dict[str, float]:
    """
    Normalize available signals into the full metric vocabulary (0–100).
    Missing signals get neutral mid scores so profiles still evaluate.
    """
    roll = rolling or {}
    multi = roll.get("multi") or roll.get("level1") or {}
    l1 = level1 or {}
    primary = roll.get("primary") or {}
    streak = roll.get("streak") or {}

    digit = float(
        l1.get("digit")
        or multi.get("digit")
        or roll.get("compression_score")
        or 40.0
    )
    streak_e = float(
        l1.get("streak")
        or multi.get("streak")
        or streak.get("streak_score")
        or 40.0
    )
    oe = float(l1.get("odd_even") or multi.get("odd_even") or 40.0)
    ou = float(l1.get("over_under") or multi.get("over_under") or 40.0)
    ud = float(l1.get("up_down") or multi.get("up_down") or 40.0)

    mom = float(
        momentum
        if momentum is not None
        else roll.get("momentum_score") or primary.get("momentum_score") or 50.0
    )
    stab = float(
        stability
        if stability is not None
        else roll.get("stability_score") or 55.0
    )

    # Repetition bias from streak length
    cur_streak = int(streak.get("current_streak") or 0)
    if repetition_bias is None:
        repetition_bias = min(100.0, 25.0 + cur_streak * 20.0)

    # Parity momentum: how extreme odd/even is (from OE compression as proxy)
    parity_momentum = min(100.0, oe * 0.7 + mom * 0.3)
    threshold_momentum = min(100.0, ou * 0.7 + mom * 0.3)

    return {
        "digit_entropy": _clamp(digit),
        "streak_entropy": _clamp(streak_e),
        "odd_even_entropy": _clamp(oe),
        "parity_entropy": _clamp(oe),  # alias for EVEN_ODD profile
        "threshold_entropy": _clamp(ou),
        "over_under_entropy": _clamp(ou),
        "up_down_entropy": _clamp(ud),
        "directional_entropy": _clamp(ud),  # RF alias until RF engine overwrites
        "repetition_bias": _clamp(float(repetition_bias)),
        "rarity_score": _clamp(float(rarity)),
        "parity_momentum": _clamp(parity_momentum),
        "threshold_momentum": _clamp(threshold_momentum),
        "momentum": _clamp(mom),
        "trend_strength": _clamp(mom),  # proxy; RF engine overwrites
        "volatility_score": _clamp(stab),
        "persistence": _clamp(streak_e),
        "stability": _clamp(stab),
    }


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(x)))


def sample_confidence(n: int) -> float:
    """
    Statistical reliability 0–1 from sample size.
    n=30 → ~0.20, n=100 → ~0.45, n=500 → ~0.75, n=1000+ → ~0.95–1.0
    """
    n = max(0, int(n))
    if n <= 0:
        return 0.05
    if n >= 5000:
        return 0.99
    if n >= 1000:
        return 0.95
    if n >= 500:
        return 0.75 + (n - 500) / 500.0 * 0.20
    if n >= 100:
        return 0.45 + (n - 100) / 400.0 * 0.30
    if n >= 30:
        return 0.20 + (n - 30) / 70.0 * 0.25
    return max(0.05, n / 30.0 * 0.20)


# ---------------------------------------------------------------------------
# Layer 3 — Adaptive Weight Engine
# ---------------------------------------------------------------------------

class AdaptiveWeightEngine:
    """
    Evolves metric importance from Historical Predictive Power (HPP).

    HPP = 35% Lift + 25% Profit Factor + 20% Stability
        + 10% Information Gain + 10% Sample Confidence
    with time-decay (100/500/1000 windows).

    Weights: BaseProfile × MetricStrength × HPP → normalize × sample conf.

    Outcomes stored for attribution via HPPTracker (data/hpp_outcomes.json).
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_LEARNING_PATH
        # legacy stats (kept for backward compatibility)
        self.stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.load()
        # HPP outcome attribution
        from src.analytics.historical_predictive_power import get_hpp_tracker

        self.hpp = get_hpp_tracker()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.stats = data.get("stats") or {}
        except Exception:
            self.stats = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {"stats": self.stats, "updated_at": time.time()},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def record_outcome(
        self,
        contract_type: str,
        metrics_used: Dict[str, float],
        is_win: bool,
        profit: float = 0.0,
        symbol: str = "",
        clarity: Optional[float] = None,
    ) -> None:
        """Store full outcome attribution + update legacy counters."""
        key = normalize_contract_key(contract_type)
        bucket = self.stats.setdefault(key, {})
        for metric, score in (metrics_used or {}).items():
            row = bucket.setdefault(
                metric,
                {"wins": 0.0, "losses": 0.0, "score_sum": 0.0, "n": 0.0},
            )
            if is_win:
                row["wins"] += 1.0
            else:
                row["losses"] += 1.0
            row["score_sum"] += float(score)
            row["n"] += 1.0
        self.save()
        # HPP tracker — full metric vector for P(Win | High Metric)
        try:
            self.hpp.record(
                contract=key,
                metrics=metrics_used or {},
                is_win=is_win,
                profit=profit,
                symbol=symbol,
                clarity=clarity,
            )
        except Exception:
            pass

    def predictive_power(self, contract_type: str, metric: str) -> float:
        """
        HPP as 0–1.15 multiplier from composite historical predictive power.
        Neutral ~0.5 when insufficient data.
        """
        key = normalize_contract_key(contract_type)
        try:
            report = self.hpp.metric_hpp(key, metric)
            if report.get("insufficient"):
                return self._legacy_predictive(key, metric)
            # HPP 0–100 → multiplier 0.25–1.15
            hpp = float(report.get("hpp") or 50.0)
            return max(0.25, min(1.15, 0.25 + (hpp / 100.0) * 0.90))
        except Exception:
            return self._legacy_predictive(key, metric)

    def _legacy_predictive(self, key: str, metric: str) -> float:
        row = (self.stats.get(key) or {}).get(metric) or {}
        w = float(row.get("wins") or 0)
        l = float(row.get("losses") or 0)
        n = w + l
        if n < 8:
            return 0.50
        wr = (w + 5.0) / (n + 10.0)
        return max(0.25, min(1.15, (wr - 0.35) / 0.35))

    def hpp_report(self, contract_type: str, metrics: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Full HPP breakdown for a contract (for dashboard / explain)."""
        key = normalize_contract_key(contract_type)
        base = get_base_profile(key)
        mets = list(metrics) if metrics else list(base.keys())
        return self.hpp.hpp_weights(
            key,
            mets,
            blend_base=base,
            base_blend=0.25,
        )

    def compute_weights(
        self,
        contract_type: str,
        metrics: Dict[str, float],
        *,
        sample_n: int = 0,
        use_dynamic_strength: bool = True,
        use_learning: bool = True,
        use_sample_confidence: bool = True,
        use_hpp_weights: bool = True,
    ) -> Dict[str, Any]:
        """
        Preferred path when learning is on and enough outcomes exist:
          weights ∝ normalized HPP (blended with base profile)

        Otherwise:
          BaseWeight × MetricStrength × HPP_multiplier → normalize
        """
        key = normalize_contract_key(contract_type)
        base = get_base_profile(key)
        conf = sample_confidence(sample_n) if use_sample_confidence else 1.0
        metric_keys = list(base.keys())

        hpp_pack: Dict[str, Any] = {}
        if use_learning and use_hpp_weights:
            try:
                hpp_pack = self.hpp.hpp_weights(
                    key,
                    metric_keys,
                    blend_base=base,
                    base_blend=0.30,
                )
            except Exception:
                hpp_pack = {}

        use_pure_hpp = bool(
            use_learning
            and use_hpp_weights
            and hpp_pack.get("weights")
            and not any(
                (hpp_pack.get("details") or {}).get(m, {}).get("insufficient")
                for m in metric_keys
            )
            and sum(
                int((hpp_pack.get("details") or {}).get(m, {}).get("n") or 0)
                for m in metric_keys
            )
            >= 40
        )

        raw: Dict[str, float] = {}
        detail: Dict[str, Dict[str, float]] = {}

        if use_pure_hpp:
            # Self-learning weights from HPP normalization
            for metric in metric_keys:
                w = float((hpp_pack.get("weights") or {}).get(metric) or 0)
                strength = float(metrics.get(metric) or 50.0) / 100.0
                if not use_dynamic_strength:
                    strength = 1.0
                # mild strength tilt on top of HPP weights
                adj = w * (0.5 + 0.5 * max(0.05, strength))
                raw[metric] = adj
                hpp_m = float((hpp_pack.get("hpp_by_metric") or {}).get(metric) or 50)
                detail[metric] = {
                    "base_weight": round(float(base.get(metric) or 0), 4),
                    "strength_01": round(strength, 4),
                    "hpp": round(hpp_m, 1),
                    "predictive_power": round(hpp_m / 100.0, 4),
                    "raw_adjusted": round(adj, 4),
                    "source": "hpp_normalized",
                }
        else:
            for metric, base_w in base.items():
                strength = float(metrics.get(metric) or 50.0) / 100.0
                if not use_dynamic_strength:
                    strength = 1.0
                pred = (
                    self.predictive_power(contract_type, metric)
                    if use_learning
                    else 1.0
                )
                hpp_val = pred * 100.0 if use_learning else 50.0
                adj = base_w * max(0.05, strength) * max(0.2, pred)
                raw[metric] = adj
                detail[metric] = {
                    "base_weight": round(base_w, 4),
                    "strength_01": round(strength, 4),
                    "hpp": round(hpp_val, 1),
                    "predictive_power": round(pred, 4),
                    "raw_adjusted": round(adj, 4),
                    "source": "base_x_strength_x_hpp",
                }

        total = sum(raw.values()) or 1.0
        normalized = {k: v / total for k, v in raw.items()}
        for k, w in normalized.items():
            detail[k]["normalized_weight"] = round(w, 4)
            detail[k]["effective_weight"] = round(w * conf, 4)

        strongest = max(normalized.items(), key=lambda x: x[1])[0] if normalized else ""

        return {
            "contract": key,
            "base_profile": base,
            "normalized_weights": {k: round(v, 4) for k, v in normalized.items()},
            "sample_confidence": round(conf, 4),
            "detail": detail,
            "hpp_pack": {
                "hpp_by_metric": (hpp_pack or {}).get("hpp_by_metric"),
                "strongest": (hpp_pack or {}).get("strongest") or strongest,
                "insight": (hpp_pack or {}).get("insight"),
            },
            "learning_mode": "hpp_normalized" if use_pure_hpp else "base_x_hpp",
            "strongest_metric": strongest,
        }


# Process singleton for learning
_weight_engine: Optional[AdaptiveWeightEngine] = None


def get_weight_engine() -> AdaptiveWeightEngine:
    global _weight_engine
    if _weight_engine is None:
        _weight_engine = AdaptiveWeightEngine()
    return _weight_engine


# ---------------------------------------------------------------------------
# Meta-model: Contract Clarity
# ---------------------------------------------------------------------------

def contract_clarity(
    contract_type: str,
    metrics: Dict[str, float],
    *,
    sample_n: int = 0,
    weight_engine: Optional[AdaptiveWeightEngine] = None,
    use_dynamic_strength: bool = True,
    use_learning: bool = True,
) -> Dict[str, Any]:
    """
    Contract Clarity = Σ(metric × weight × confidence)

    weight  = contract relevance (profile) × strength × predictive power → normalized
    confidence = sample reliability
    metric  = current signal strength 0–100
    """
    eng = weight_engine or get_weight_engine()
    wt = eng.compute_weights(
        contract_type,
        metrics,
        sample_n=sample_n,
        use_dynamic_strength=use_dynamic_strength,
        use_learning=use_learning,
        use_sample_confidence=True,
    )
    conf = float(wt["sample_confidence"])
    weights = wt["normalized_weights"]

    clarity = 0.0
    contributors: List[Dict[str, Any]] = []
    for metric, w in weights.items():
        m_score = float(metrics.get(metric) or 0.0)
        # Contribution in 0–100 space: score * weight * conf
        part = m_score * w * conf
        # Without conf the max is 100; with conf it scales down for thin samples
        clarity += part
        contributors.append(
            {
                "metric": metric,
                "score": round(m_score, 1),
                "weight": round(w, 4),
                "confidence": round(conf, 4),
                "contribution": round(part, 2),
                "ok": m_score >= 60 and w >= 0.08,
            }
        )

    # If conf < 1, clarity max is 100*conf — rescale display to 0–100 of full potential
    # Keep raw product as "raw_clarity", display = raw / conf when conf>0 so scores readable
    # Spec: Score × Confidence in contribution; total should still be interpretable 0–100
    # Using Σ(score * w * conf) → max 100*conf. Map back: display_clarity = Σ(score*w) * conf
    display = max(0.0, min(100.0, clarity))
    # Also compute unconfounded for ranking when samples large
    unconf = sum(
        float(metrics.get(m) or 0) * float(weights.get(m) or 0) for m in weights
    )

    contributors.sort(key=lambda x: -x["contribution"])

    # Recommendation
    if display >= 85 and conf >= 0.5:
        rec = "STRONG SETUP"
    elif display >= 75 and conf >= 0.35:
        rec = "TRADEABLE"
    elif display >= 60:
        rec = "WATCH"
    else:
        rec = "SKIP"

    conf_label = (
        "HIGH" if conf >= 0.75 else ("MEDIUM" if conf >= 0.45 else "LOW")
    )

    reasons = [
        f"Contract: {wt['contract']}",
        f"Clarity Score: {display:.0f}",
        "Contributors",
        "------------",
    ]
    for c in contributors[:6]:
        mark = "✓" if c["ok"] else "·"
        reasons.append(
            f"{mark} {c['metric']:22s} {c['score']:.0f}  "
            f"(w={c['weight']:.0%} · contrib {c['contribution']:.1f})"
        )
    reasons.append(f"Confidence: {conf_label} ({conf:.0%} sample reliability)")
    reasons.append(f"Recommendation: {rec}")
    if wt.get("hpp_pack", {}).get("insight"):
        reasons.append(f"HPP: {wt['hpp_pack']['insight']}")
    if wt.get("learning_mode"):
        reasons.append(f"Weight mode: {wt['learning_mode']}")

    return {
        "contract": wt["contract"],
        "clarity_score": round(display, 1),
        "clarity_unweighted_conf": round(unconf, 1),
        "recommendation": rec,
        "confidence": conf_label,
        "sample_confidence": conf,
        "sample_n": sample_n,
        "weights": weights,
        "weight_detail": wt["detail"],
        "base_profile": wt["base_profile"],
        "hpp": wt.get("hpp_pack"),
        "learning_mode": wt.get("learning_mode"),
        "metrics": {k: round(float(v), 1) for k, v in metrics.items()},
        "contributors": contributors,
        "reasons": reasons,
        "explain": reasons,
        "auto_ok": display >= 80 and conf >= 0.45,
    }


def evaluate_contract_setup(
    ticks: Sequence[Dict[str, Any]],
    *,
    symbol: str = "_default",
    contract_type: str = "DIGITDIFF",
    sample_n: int = 0,
    pattern_wr: float = 0.5,
    baseline_wr: float = 0.5,
) -> Dict[str, Any]:
    """
    End-to-end: rolling metrics → profile → adaptive weights → contract clarity.
    """
    from src.analytics.rolling_entropy import feed_ticks
    from src.analytics.pattern_clarity import baseline_separation_score, rarity_score

    roll = feed_ticks(symbol, list(ticks)[-500:] if ticks else [])
    sep, _ = baseline_separation_score(pattern_wr, baseline_wr)
    # rarity from compression (high compression → rarer structure)
    comp_pct = float((roll.get("primary") or {}).get("compression_pct") or 5)
    rarity = min(100.0, 40.0 + comp_pct * 2.5)

    metrics = build_metric_vector(
        rolling=roll,
        momentum=float(roll.get("momentum_score") or 50),
        stability=float(roll.get("stability_score") or 55),
        rarity=rarity,
    )
    # Blend statistical separation into momentum-ish slot for differ
    metrics["momentum"] = max(
        metrics["momentum"], min(100.0, sep * 0.9 + metrics["momentum"] * 0.1)
    )

    # Rise/Fall: overwrite with directional engine (momentum/persistence/vol/dir-entropy)
    rf_analysis: Dict[str, Any] = {}
    ct_key = normalize_contract_key(contract_type)
    if ct_key in {"CALL", "PUT", "RISE", "FALL"}:
        try:
            from src.analytics.rise_fall_engine import analyze_rise_fall

            rf_analysis = analyze_rise_fall(
                ticks,
                contract_type=ct_key,
                hpp=50.0,
            )
            for k, v in (rf_analysis.get("metrics") or {}).items():
                metrics[k] = float(v)
        except Exception:
            rf_analysis = {}

    result = contract_clarity(
        contract_type,
        metrics,
        sample_n=sample_n,
        use_dynamic_strength=True,
        use_learning=True,
    )
    result["rolling"] = roll
    result["regime"] = roll.get("regime")
    result["statistical_separation"] = sep
    result["metrics"] = metrics
    if rf_analysis:
        result["rise_fall"] = rf_analysis
        result["rf_score"] = rf_analysis.get("rf_score")
        # Prefer RF clarity blend when directional
        if rf_analysis.get("rf_score") is not None:
            from src.analytics.rise_fall_engine import rf_pattern_clarity

            rf_c = rf_pattern_clarity(rf_analysis)
            # Blend profile clarity with RF clarity
            result["clarity_score"] = round(
                0.40 * float(result.get("clarity_score") or 50) + 0.60 * rf_c, 1
            )
            result["rf_clarity"] = round(rf_c, 1)
    result["display"] = {
        "contract": result["contract"],
        "clarity_score": result["clarity_score"],
        "contributors": [
            f"{'✓' if c['ok'] else '·'} {c['metric']} {c['score']}"
            for c in result["contributors"][:5]
        ],
        "confidence": f"{result['confidence']} ({result['sample_confidence']:.0%})",
        "recommendation": result["recommendation"],
        "regime": roll.get("regime"),
        "rf_score": result.get("rf_score"),
        "vol_regime": (rf_analysis.get("volatility") or {}).get("regime")
        if rf_analysis
        else None,
    }
    return result


def register_profile(name: str, weights: Dict[str, float]) -> None:
    """Register or override a contract profile (extensibility)."""
    key = str(name).upper()
    s = sum(float(v) for v in weights.values()) or 1.0
    CONTRACT_PROFILES[key] = {k: float(v) / s for k, v in weights.items()}
