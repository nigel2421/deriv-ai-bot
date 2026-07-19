"""
Historical Predictive Power (HPP)

Answers: "How much does this metric improve prediction vs random guessing?"

Used to auto-adjust contract profile weights over time.

Composite (production):
  35% Lift Score
+ 25% Profit Factor Power
+ 20% Stability
+ 10% Information Gain
+ 10% Sample Confidence

Time-decay on windows:
  Last 100  = 50%
  Last 500  = 30%
  Last 1000 = 20%
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_PATH = Path("data/hpp_outcomes.json")
HIGH_THRESHOLD = 65.0  # metric considered "high" for attribution
MAX_EDGE = 0.25  # practical max edge for Method 1 scaling
DEFAULT_BASELINE = 0.50


def binary_entropy(p: float) -> float:
    """Bernoulli entropy in bits for win probability p."""
    p = max(1e-9, min(1.0 - 1e-9, float(p)))
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def lift_score(signal_wr: float, baseline_wr: float = DEFAULT_BASELINE) -> float:
    """
    Lift = Signal WR / Baseline WR
    HPP component: (Lift - 1) × 100  clamped 0–100
    """
    b = max(0.05, float(baseline_wr))
    s = max(0.0, min(1.0, float(signal_wr)))
    lift = s / b
    return max(0.0, min(100.0, (lift - 1.0) * 100.0))


def edge_based_hpp(
    signal_wr: float,
    baseline_wr: float = DEFAULT_BASELINE,
    max_edge: float = MAX_EDGE,
) -> float:
    """
    Edge = WR - baseline; HPP = Edge / MaxPossibleEdge × 100
    """
    edge = max(0.0, float(signal_wr) - float(baseline_wr))
    me = max(0.05, float(max_edge))
    return max(0.0, min(100.0, (edge / me) * 100.0))


def profit_factor_power(gross_profit: float, gross_loss: float) -> float:
    """ProfitPower = min(100, PF × 40)."""
    gl = abs(float(gross_loss))
    gp = max(0.0, float(gross_profit))
    if gl < 1e-9:
        return 100.0 if gp > 0 else 0.0
    pf = gp / gl
    return max(0.0, min(100.0, pf * 40.0))


def information_gain(
    baseline_wr: float,
    signal_wr: float,
) -> float:
    """
    IG = H(baseline) - H(signal_wr) in bits, scaled to 0–100.
    Max useful IG for binary ~1 bit → map × 100.
    """
    h0 = binary_entropy(baseline_wr)
    h1 = binary_entropy(signal_wr)
    ig = max(0.0, h0 - h1)
    return max(0.0, min(100.0, ig * 100.0))


def stability_power(win_rates: Sequence[float]) -> float:
    """
    Low variance of WR across windows → high score (~95 stable, ~40 unstable).
    """
    rates = [max(0.0, min(1.0, float(r))) for r in win_rates if r is not None]
    if len(rates) < 2:
        return 50.0
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / len(rates)
    std = math.sqrt(var)
    # std 0 → 100, 0.02 → ~90, 0.08 → ~50, 0.15 → ~25
    if std <= 0.02:
        score = 100.0 - std / 0.02 * 10.0
    elif std <= 0.05:
        score = 90.0 - (std - 0.02) / 0.03 * 25.0
    elif std <= 0.10:
        score = 65.0 - (std - 0.05) / 0.05 * 25.0
    else:
        score = max(10.0, 40.0 - (std - 0.10) * 200.0)
    return max(0.0, min(100.0, score))


def sample_size_hpp(n: int) -> float:
    """0–100 from sample size for HPP blend."""
    n = max(0, int(n))
    if n >= 5000:
        return 100.0
    if n >= 1000:
        return 95.0
    if n >= 500:
        return 80.0
    if n >= 100:
        return 55.0
    if n >= 50:
        return 35.0
    if n >= 20:
        return 20.0
    return max(5.0, n * 1.0)


def composite_hpp(
    *,
    lift: float,
    profit_power: float,
    stability: float,
    info_gain: float,
    sample: float,
) -> float:
    """
    Production HPP:
      35% Lift + 25% PF + 20% Stability + 10% Info Gain + 10% Sample
    """
    total = (
        0.35 * float(lift)
        + 0.25 * float(profit_power)
        + 0.20 * float(stability)
        + 0.10 * float(info_gain)
        + 0.10 * float(sample)
    )
    return max(0.0, min(100.0, total))


def time_decay_hpp(
    recent: float,
    medium: float,
    long_: float,
) -> float:
    """Last 100 = 50%, Last 500 = 30%, Last 1000 = 20%."""
    return 0.50 * float(recent) + 0.30 * float(medium) + 0.20 * float(long_)


def stats_from_outcomes(
    outcomes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    wins = losses = 0
    gp = gl = 0.0
    for o in outcomes:
        is_win = bool(o.get("is_win") or o.get("result") == "WIN")
        try:
            profit = float(o.get("profit") if o.get("profit") is not None else (1.0 if is_win else -1.0))
        except (TypeError, ValueError):
            profit = 1.0 if is_win else -1.0
        if profit > 0 or is_win:
            wins += 1
            gp += abs(profit)
        else:
            losses += 1
            gl += abs(profit)
    n = wins + losses
    wr = wins / n if n else 0.5
    return {
        "wins": wins,
        "losses": losses,
        "n": n,
        "wr": wr,
        "gross_profit": gp,
        "gross_loss": gl,
    }


def window_win_rates(outcomes: Sequence[Dict[str, Any]]) -> List[float]:
    """WR on last 100 / 500 / 1000 / 5000 slices."""
    rows = list(outcomes)
    out = []
    for w in (100, 500, 1000, 5000):
        chunk = rows[-w:] if rows else []
        if not chunk:
            continue
        s = stats_from_outcomes(chunk)
        if s["n"] >= 5:
            out.append(s["wr"])
    return out


class HPPTracker:
    """
    Outcome attribution store + per-metric HPP computation.

    Each trade:
      {
        "digit_entropy": 82, "streak_entropy": 75, "momentum": 91,
        "contract": "DIGITDIFF", "result": "WIN", "profit": 1.2, "ts": ...
      }
    """

    def __init__(self, path: Optional[Path] = None, max_outcomes: int = 8000):
        self.path = Path(path) if path else DEFAULT_PATH
        self.max_outcomes = max_outcomes
        # outcomes list (newest last)
        self.outcomes: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.outcomes = list(data.get("outcomes") or [])[-self.max_outcomes :]
        except Exception:
            self.outcomes = []

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "outcomes": self.outcomes[-self.max_outcomes :],
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def record(
        self,
        *,
        contract: str,
        metrics: Dict[str, float],
        is_win: bool,
        profit: float = 0.0,
        symbol: str = "",
        clarity: Optional[float] = None,
    ) -> None:
        row = {
            **{k: float(v) for k, v in (metrics or {}).items()},
            "contract": str(contract).upper(),
            "symbol": symbol,
            "result": "WIN" if is_win else "LOSS",
            "is_win": bool(is_win),
            "profit": float(profit),
            "clarity": float(clarity) if clarity is not None else None,
            "ts": time.time(),
        }
        self.outcomes.append(row)
        if len(self.outcomes) > self.max_outcomes:
            self.outcomes = self.outcomes[-self.max_outcomes :]
        self.save()

    def filter_outcomes(
        self,
        *,
        contract: Optional[str] = None,
        metric: Optional[str] = None,
        high_only: bool = False,
        last_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rows = list(self.outcomes)
        if contract:
            ct = str(contract).upper()
            rows = [r for r in rows if str(r.get("contract") or "").upper() == ct]
        if last_n is not None:
            rows = rows[-int(last_n) :]
        if metric and high_only:
            rows = [
                r
                for r in rows
                if float(r.get(metric) or 0) >= HIGH_THRESHOLD
            ]
        return rows

    def metric_hpp(
        self,
        contract: str,
        metric: str,
        *,
        baseline_wr: float = DEFAULT_BASELINE,
    ) -> Dict[str, Any]:
        """
        Full composite HPP for one metric under one contract, with time-decay.
        """
        # Attribution: only when metric was "high"
        all_c = self.filter_outcomes(contract=contract)
        if not all_c:
            return {
                "hpp": 50.0,
                "hpp_01": 0.50,
                "n": 0,
                "insufficient": True,
                "components": {},
            }

        def _hpp_for(rows: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
            high = [
                r
                for r in rows
                if float(r.get(metric) or 0) >= HIGH_THRESHOLD
            ]
            # Fallback: use all rows if few high signals
            use = high if len(high) >= 8 else rows
            st = stats_from_outcomes(use)
            if st["n"] < 5:
                return 50.0, {"n": st["n"], "insufficient": True}

            wr = st["wr"]
            lift = lift_score(wr, baseline_wr)
            edge = edge_based_hpp(wr, baseline_wr)
            pf_pow = profit_factor_power(st["gross_profit"], st["gross_loss"])
            ig = information_gain(baseline_wr, wr)
            stab = stability_power(window_win_rates(use))
            samp = sample_size_hpp(st["n"])
            # Blend edge lightly into lift channel (edge is related)
            lift_blend = 0.7 * lift + 0.3 * edge
            hpp = composite_hpp(
                lift=lift_blend,
                profit_power=pf_pow,
                stability=stab,
                info_gain=ig,
                sample=samp,
            )
            return hpp, {
                "n": st["n"],
                "wr": round(wr, 4),
                "lift": round(lift, 1),
                "edge_hpp": round(edge, 1),
                "profit_power": round(pf_pow, 1),
                "stability": round(stab, 1),
                "info_gain": round(ig, 1),
                "sample": round(samp, 1),
                "high_signals": len(high),
            }

        h100, d100 = _hpp_for(all_c[-100:])
        h500, d500 = _hpp_for(all_c[-500:])
        h1000, d1000 = _hpp_for(all_c[-1000:])

        # If medium/long empty, fall back
        if d500.get("insufficient") and not d100.get("insufficient"):
            h500, d500 = h100, d100
        if d1000.get("insufficient"):
            h1000, d1000 = h500, d500

        weighted = time_decay_hpp(h100, h500, h1000)
        n_total = stats_from_outcomes(all_c)["n"]

        return {
            "hpp": round(weighted, 1),
            "hpp_01": round(weighted / 100.0, 4),
            "n": n_total,
            "insufficient": n_total < 8,
            "windows": {
                "last_100": round(h100, 1),
                "last_500": round(h500, 1),
                "last_1000": round(h1000, 1),
            },
            "components": d100 if not d100.get("insufficient") else d500,
            "detail_100": d100,
            "detail_500": d500,
            "detail_1000": d1000,
            "metric": metric,
            "contract": str(contract).upper(),
            "attribution": f"P(Win | {metric} ≥ {HIGH_THRESHOLD})",
        }

    def all_metric_hpp(
        self,
        contract: str,
        metrics: Sequence[str],
        *,
        baseline_wr: float = DEFAULT_BASELINE,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            m: self.metric_hpp(contract, m, baseline_wr=baseline_wr) for m in metrics
        }

    def hpp_weights(
        self,
        contract: str,
        metrics: Sequence[str],
        *,
        baseline_wr: float = DEFAULT_BASELINE,
        blend_base: Optional[Dict[str, float]] = None,
        base_blend: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Normalize HPP values into weights; optionally blend with base profile.

        Final weight_i ∝ (1 - base_blend) * HPP_i + base_blend * Base_i * 100
        """
        hpps = self.all_metric_hpp(contract, metrics, baseline_wr=baseline_wr)
        scores = {m: float(hpps[m]["hpp"]) for m in metrics}

        if blend_base:
            for m in metrics:
                b = float(blend_base.get(m) or 0) * 100.0
                scores[m] = (1.0 - base_blend) * scores[m] + base_blend * b

        total = sum(max(1.0, v) for v in scores.values()) or 1.0
        weights = {m: max(1.0, scores[m]) / total for m in metrics}

        # Which is strongest?
        top = max(scores.items(), key=lambda x: x[1]) if scores else ("", 0.0)

        return {
            "contract": str(contract).upper(),
            "hpp_by_metric": {m: round(scores[m], 1) for m in metrics},
            "weights": {m: round(w, 4) for m, w in weights.items()},
            "strongest": top[0],
            "strongest_hpp": round(float(top[1]), 1),
            "details": hpps,
            "insight": (
                f"{top[0]} is currently the strongest predictor "
                f"(HPP {top[1]:.0f})"
                if top[0]
                else "Insufficient data"
            ),
        }

    def conditional_win_rate(
        self,
        contract: str,
        metric: str,
        *,
        high_threshold: float = HIGH_THRESHOLD,
    ) -> Dict[str, Any]:
        """P(Win | High Metric) vs overall — true attribution."""
        all_c = self.filter_outcomes(contract=contract)
        if not all_c:
            return {"p_win_high": 0.5, "p_win_all": 0.5, "n_high": 0, "n_all": 0}
        high = [
            r for r in all_c if float(r.get(metric) or 0) >= high_threshold
        ]
        st_all = stats_from_outcomes(all_c)
        st_hi = stats_from_outcomes(high) if high else {"wr": 0.5, "n": 0}
        return {
            "p_win_high": round(st_hi["wr"], 4),
            "p_win_all": round(st_all["wr"], 4),
            "n_high": st_hi["n"],
            "n_all": st_all["n"],
            "lift": round(st_hi["wr"] / max(0.05, st_all["wr"]), 3)
            if st_hi["n"]
            else 1.0,
        }


# Singleton
_tracker: Optional[HPPTracker] = None


def get_hpp_tracker() -> HPPTracker:
    global _tracker
    if _tracker is None:
        _tracker = HPPTracker()
    return _tracker
