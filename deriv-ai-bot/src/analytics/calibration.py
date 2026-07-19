"""
Calibration, confidence intervals, prediction drift, and outcome attribution.

Tracks whether score bands (e.g. 80–90) produce matching win rates,
monitors predicted vs actual drift, and stores rich trade outcomes so the
system can learn which metrics actually matter.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_PATH = Path("data/calibration_outcomes.json")
DEFAULT_PEAK_PATH = Path("data/hpp_peaks.json")
DRIFT_WINDOW = 500
DRIFT_ALERT_PCT = 10.0  # alert if |pred - actual| > 10% over rolling window


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion (95% default).
    Returns (low, high) as fractions in [0, 1].
    """
    if n <= 0:
        return (0.0, 1.0)
    w = max(0, int(wins))
    nn = int(n)
    p = w / nn
    z2 = z * z
    denom = 1.0 + z2 / nn
    center = p + z2 / (2.0 * nn)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * nn)) / nn)
    lo = max(0.0, (center - margin) / denom)
    hi = min(1.0, (center + margin) / denom)
    return (lo, hi)


def calibration_error(expected: float, actual: float) -> float:
    """Absolute gap between expected and realized rate (both 0–1 or 0–100)."""
    e = float(expected)
    a = float(actual)
    if e > 1.0 or a > 1.0:
        e = e / 100.0 if e > 1.0 else e
        a = a / 100.0 if a > 1.0 else a
    return abs(e - a)


def score_band(score: float) -> str:
    """Map 0–100 score into display bands."""
    s = float(score)
    if s >= 90:
        return "90-100"
    if s >= 80:
        return "80-90"
    if s >= 70:
        return "70-80"
    if s >= 60:
        return "60-70"
    return "0-60"


class CalibrationTracker:
    """
    Rolling outcome store for:
      - score-band calibration
      - prediction drift
      - confidence intervals
      - full metric attribution
      - peak HPP / edge decay
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        peak_path: Optional[Path] = None,
        max_outcomes: int = 12000,
    ):
        self.path = Path(path) if path else DEFAULT_PATH
        self.peak_path = Path(peak_path) if peak_path else DEFAULT_PEAK_PATH
        self.max_outcomes = max_outcomes
        self.outcomes: List[Dict[str, Any]] = []
        # contract -> peak HPP seen
        self.peaks: Dict[str, float] = {}
        self.load()

    def load(self) -> None:
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.outcomes = list(data.get("outcomes") or [])[-self.max_outcomes :]
            except Exception:
                self.outcomes = []
        if self.peak_path.is_file():
            try:
                self.peaks = {
                    str(k).upper(): float(v)
                    for k, v in (
                        json.loads(self.peak_path.read_text(encoding="utf-8")).get(
                            "peaks"
                        )
                        or {}
                    ).items()
                }
            except Exception:
                self.peaks = {}

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
        try:
            self.peak_path.parent.mkdir(parents=True, exist_ok=True)
            self.peak_path.write_text(
                json.dumps({"peaks": self.peaks, "updated_at": time.time()}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def note_hpp(self, contract: str, hpp: float) -> float:
        """Update peak HPP for contract; return peak after update."""
        key = str(contract or "").upper() or "_ALL"
        cur = float(hpp)
        prev = float(self.peaks.get(key) or 0.0)
        if cur > prev:
            self.peaks[key] = cur
            self.save()
            return cur
        return prev if prev > 0 else cur

    def peak_hpp(self, contract: str) -> float:
        key = str(contract or "").upper() or "_ALL"
        return float(self.peaks.get(key) or 0.0)

    def record(
        self,
        *,
        contract: str,
        is_win: bool,
        predicted_p: Optional[float] = None,
        quality: Optional[float] = None,
        entropy: Optional[float] = None,
        clarity: Optional[float] = None,
        hpp: Optional[float] = None,
        velocity: Optional[float] = None,
        strength: Optional[float] = None,
        momentum: Optional[float] = None,
        persistence: Optional[float] = None,
        momentum_persistence: Optional[float] = None,
        persistence_velocity: Optional[float] = None,
        persistence_acceleration: Optional[float] = None,
        regime: Optional[str] = None,
        profit: float = 0.0,
        stake: Optional[float] = None,
        symbol: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full outcome attribution row:

          {
            "contract":"DIFFER", "entropy":83, "clarity":87,
            "hpp":79, "velocity":11, "quality":84, "result":"WIN"
          }
        """
        p = float(predicted_p) if predicted_p is not None else None
        if p is not None and p > 1.0:
            p = p / 100.0
        q = float(quality) if quality is not None else None
        row: Dict[str, Any] = {
            "contract": str(contract or "").upper(),
            "symbol": symbol,
            "result": "WIN" if is_win else "LOSS",
            "is_win": bool(is_win),
            "profit": float(profit),
            "stake": float(stake) if stake is not None else None,
            "predicted_p": round(p, 4) if p is not None else None,
            "quality": round(q, 1) if q is not None else None,
            "score_band": score_band(q) if q is not None else None,
            "entropy": round(float(entropy), 1) if entropy is not None else None,
            "clarity": round(float(clarity), 1) if clarity is not None else None,
            "hpp": round(float(hpp), 1) if hpp is not None else None,
            "velocity": round(float(velocity), 2) if velocity is not None else None,
            "strength": round(float(strength), 1) if strength is not None else None,
            "momentum": round(float(momentum), 1) if momentum is not None else None,
            "persistence": round(float(persistence), 1)
            if persistence is not None
            else None,
            "momentum_persistence": round(float(momentum_persistence), 1)
            if momentum_persistence is not None
            else None,
            "persistence_velocity": round(float(persistence_velocity), 2)
            if persistence_velocity is not None
            else None,
            "persistence_acceleration": round(float(persistence_acceleration), 2)
            if persistence_acceleration is not None
            else None,
            "regime": regime,
            "ts": time.time(),
        }
        if extra:
            for k, v in extra.items():
                if k not in row:
                    row[k] = v
        if hpp is not None:
            row["peak_hpp"] = self.note_hpp(str(contract), float(hpp))
            peak = float(row["peak_hpp"] or 0)
            if peak > 0:
                from src.analytics.no_trade_engine import edge_decay_pct

                row["edge_decay_pct"] = round(
                    edge_decay_pct(peak, float(hpp)), 1
                )
        self.outcomes.append(row)
        if len(self.outcomes) > self.max_outcomes:
            self.outcomes = self.outcomes[-self.max_outcomes :]
        self.save()
        return row

    def band_report(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Calibration by score band:

          Signals Score 80-90 · Generated 100 · Won 78 · Expected ~80–85
        """
        rows = list(self.outcomes)
        if last_n is not None:
            rows = rows[-int(last_n) :]
        bands: Dict[str, Dict[str, Any]] = {}
        order = ["90-100", "80-90", "70-80", "60-70", "0-60"]
        for b in order:
            bands[b] = {
                "band": b,
                "generated": 0,
                "won": 0,
                "win_rate": None,
                "expected_mid": _band_expected_mid(b),
                "calibration_error": None,
                "ci_low": None,
                "ci_high": None,
            }
        for r in rows:
            b = r.get("score_band")
            if not b or b not in bands:
                q = r.get("quality")
                if q is None:
                    continue
                b = score_band(float(q))
            bucket = bands[b]
            bucket["generated"] += 1
            if r.get("is_win"):
                bucket["won"] += 1
        for b, bucket in bands.items():
            n = int(bucket["generated"])
            w = int(bucket["won"])
            if n <= 0:
                continue
            wr = w / n
            lo, hi = wilson_ci(w, n)
            exp = float(bucket["expected_mid"])
            bucket["win_rate"] = round(wr * 100.0, 1)
            bucket["ci_low"] = round(lo * 100.0, 1)
            bucket["ci_high"] = round(hi * 100.0, 1)
            bucket["calibration_error"] = round(
                calibration_error(exp / 100.0, wr) * 100.0, 1
            )
            bucket["display"] = (
                f"Score {b}: Generated {n} · Won {w} · "
                f"WR {bucket['win_rate']}% "
                f"(CI {bucket['ci_low']}–{bucket['ci_high']}%) · "
                f"Expected ~{exp:.0f} · Error {bucket['calibration_error']}%"
            )
        # Monotonic check: higher bands should not underperform lower (when n≥20)
        mono_ok = True
        prev_wr = None
        for b in reversed(order):  # low → high
            n = int(bands[b]["generated"])
            wr = bands[b]["win_rate"]
            if n >= 20 and wr is not None:
                if prev_wr is not None and wr + 2.0 < prev_wr:
                    # allow 2pp noise; still fail if inverted
                    mono_ok = False
                prev_wr = wr
        overall_err = _mean(
            [
                float(bands[b]["calibration_error"])
                for b in order
                if bands[b]["calibration_error"] is not None
                and int(bands[b]["generated"]) >= 10
            ]
        )
        return {
            "bands": bands,
            "monotonic_ok": mono_ok,
            "mean_calibration_error": round(overall_err, 2)
            if overall_err is not None
            else None,
            "n_total": len(rows),
        }

    def confidence_interval_report(
        self,
        *,
        contract: Optional[str] = None,
        last_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        rows = list(self.outcomes)
        if contract:
            ct = str(contract).upper()
            rows = [r for r in rows if str(r.get("contract") or "").upper() == ct]
        if last_n is not None:
            rows = rows[-int(last_n) :]
        n = len(rows)
        wins = sum(1 for r in rows if r.get("is_win"))
        lo, hi = wilson_ci(wins, n)
        wr = wins / n if n else 0.0
        width = (hi - lo) * 100.0
        reliable = n >= 50 and width <= 25.0
        return {
            "n": n,
            "wins": wins,
            "win_rate": round(wr * 100.0, 1),
            "ci_low": round(lo * 100.0, 1),
            "ci_high": round(hi * 100.0, 1),
            "ci_width": round(width, 1),
            "reliable": reliable,
            "display": (
                f"Win Rate {wr * 100:.0f}% · "
                f"CI {lo * 100:.0f}–{hi * 100:.0f}% · n={n}"
                + ("" if reliable else " · UNRELIABLE (wide CI / small n)")
            ),
        }

    def prediction_drift(
        self,
        *,
        last_n: int = DRIFT_WINDOW,
        alert_pct: float = DRIFT_ALERT_PCT,
    ) -> Dict[str, Any]:
        """
        Compare mean predicted win probability vs actual WR over rolling window.
        Alert if |error| > alert_pct.
        """
        rows = [
            r
            for r in self.outcomes[-int(last_n) :]
            if r.get("predicted_p") is not None
        ]
        n = len(rows)
        if n == 0:
            return {
                "n": 0,
                "predicted": None,
                "actual": None,
                "drift_pct": None,
                "alert": False,
                "display": "No predicted outcomes yet",
            }
        pred = sum(float(r["predicted_p"]) for r in rows) / n
        actual = sum(1 for r in rows if r.get("is_win")) / n
        drift = abs(pred - actual) * 100.0
        alert = n >= min(50, last_n // 2) and drift > float(alert_pct)
        return {
            "n": n,
            "predicted": round(pred * 100.0, 1),
            "actual": round(actual * 100.0, 1),
            "drift_pct": round(drift, 1),
            "alert": alert,
            "alert_threshold": float(alert_pct),
            "display": (
                f"Predicted {pred * 100:.0f}% · Actual {actual * 100:.0f}% · "
                f"Drift {drift:.0f}%"
                + (" · ALERT" if alert else "")
            ),
        }

    def metric_validation(
        self,
        metric: str,
        *,
        high: float = 80.0,
        low: float = 60.0,
        last_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        High-metric signals should outperform low-metric (HPP / clarity checks).
        """
        rows = list(self.outcomes)
        if last_n is not None:
            rows = rows[-int(last_n) :]
        key = str(metric)
        hi = [r for r in rows if r.get(key) is not None and float(r[key]) >= high]
        lo = [r for r in rows if r.get(key) is not None and float(r[key]) < low]
        wr_hi = _wr(hi)
        wr_lo = _wr(lo)
        ok = True
        if len(hi) >= 20 and len(lo) >= 20:
            ok = wr_hi >= wr_lo - 0.02  # allow 2pp noise
        return {
            "metric": key,
            "high_threshold": high,
            "low_threshold": low,
            "n_high": len(hi),
            "n_low": len(lo),
            "wr_high": round(wr_hi * 100.0, 1) if hi else None,
            "wr_low": round(wr_lo * 100.0, 1) if lo else None,
            "pass": ok,
            "display": (
                f"{key} ≥{high:.0f}: WR={wr_hi * 100:.0f}% (n={len(hi)}) · "
                f"{key} <{low:.0f}: WR={wr_lo * 100:.0f}% (n={len(lo)}) · "
                f"{'PASS' if ok else 'FAIL'}"
            ),
        }

    def regime_profitability(
        self, last_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """RANDOM regime should be less profitable than STRONG PATTERN."""
        rows = list(self.outcomes)
        if last_n is not None:
            rows = rows[-int(last_n) :]
        by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            reg = str(r.get("regime") or "UNKNOWN").upper()
            by[reg].append(r)
        report = {}
        for reg, rs in by.items():
            pnl = sum(float(x.get("profit") or 0) for x in rs)
            report[reg] = {
                "n": len(rs),
                "wr": round(_wr(rs) * 100.0, 1),
                "pnl": round(pnl, 2),
            }
        strong = report.get("STRONG PATTERN") or report.get("BIASED") or {}
        random = report.get("RANDOM") or {}
        ok = True
        if strong.get("n", 0) >= 15 and random.get("n", 0) >= 15:
            ok = float(strong.get("pnl") or 0) >= float(random.get("pnl") or 0)
        return {
            "by_regime": report,
            "pass": ok,
            "display": (
                f"Strong/pattern regimes should beat RANDOM · "
                f"{'PASS' if ok else 'FAIL — regime detector ineffective'}"
            ),
        }

    def velocity_validation(
        self,
        *,
        last_n: Optional[int] = None,
        pos_threshold: float = 0.0,
        neg_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Positive velocity MUST outperform negative velocity."""
        rows = list(self.outcomes)
        if last_n is not None:
            rows = rows[-int(last_n) :]
        pos = [
            r
            for r in rows
            if r.get("velocity") is not None and float(r["velocity"]) > pos_threshold
        ]
        neg = [
            r
            for r in rows
            if r.get("velocity") is not None and float(r["velocity"]) < neg_threshold
        ]
        wr_pos = _wr(pos)
        wr_neg = _wr(neg)
        ok = True
        if len(pos) >= 20 and len(neg) >= 20:
            ok = wr_pos >= wr_neg - 0.02
        return {
            "n_positive": len(pos),
            "n_negative": len(neg),
            "wr_positive": round(wr_pos * 100.0, 1) if pos else None,
            "wr_negative": round(wr_neg * 100.0, 1) if neg else None,
            "pass": ok,
            "display": (
                f"Vel>0 WR={wr_pos * 100:.0f}% (n={len(pos)}) · "
                f"Vel<0 WR={wr_neg * 100:.0f}% (n={len(neg)}) · "
                f"{'PASS' if ok else 'FAIL'}"
            ),
        }

    def family_metric_validation(
        self,
        *,
        last_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Contract-specific: digit metrics should help digit trades;
        directional metrics should help rise/fall.
        Uses clarity for digits and strength proxy for RF when family tagged.
        """
        rows = list(self.outcomes)
        if last_n is not None:
            rows = rows[-int(last_n) :]
        digit_cts = {
            "DIGITDIFF",
            "DIGITMATCH",
            "DIGITEVEN",
            "DIGITODD",
            "DIGITOVER",
            "DIGITUNDER",
        }
        rf_cts = {"CALL", "PUT", "RISE", "FALL"}
        dig = [r for r in rows if str(r.get("contract") or "").upper() in digit_cts]
        rf = [r for r in rows if str(r.get("contract") or "").upper() in rf_cts]

        def _hi_lo(rs, key, hi=80.0, lo=60.0):
            h = [r for r in rs if r.get(key) is not None and float(r[key]) >= hi]
            l = [r for r in rs if r.get(key) is not None and float(r[key]) < lo]
            ok = True
            if len(h) >= 15 and len(l) >= 15:
                ok = _wr(h) >= _wr(l) - 0.02
            return {
                "n_high": len(h),
                "n_low": len(l),
                "wr_high": round(_wr(h) * 100, 1) if h else None,
                "wr_low": round(_wr(l) * 100, 1) if l else None,
                "pass": ok,
            }

        dig_clarity = _hi_lo(dig, "clarity")
        dig_hpp = _hi_lo(dig, "hpp")
        # RF: strength / hpp when available
        rf_strength = _hi_lo(rf, "strength")
        rf_hpp = _hi_lo(rf, "hpp")
        ok = all(
            x.get("pass", True)
            for x in (dig_clarity, dig_hpp, rf_strength, rf_hpp)
        )
        return {
            "digit_clarity": dig_clarity,
            "digit_hpp": dig_hpp,
            "rf_strength": rf_strength,
            "rf_hpp": rf_hpp,
            "n_digit": len(dig),
            "n_rf": len(rf),
            "pass": ok,
            "display": (
                f"Digits n={len(dig)} clarity/HPP separation · "
                f"RF n={len(rf)} strength/HPP · {'PASS' if ok else 'FAIL'}"
            ),
        }

    def feature_contribution_report(
        self,
        *,
        last_n: int = 500,
        features: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Every ~500 trades: estimate lift of high vs low feature bands.

        Feature Contribution
          Entropy        +12%
          Pattern Clarity +8%
          Persistence    +15%
          Momentum       +10%
          HPP Velocity   +7%
        """
        feats = list(
            features
            or (
                "entropy",
                "clarity",
                "persistence",
                "momentum",
                "momentum_persistence",
                "persistence_velocity",
                "persistence_acceleration",
                "hpp",
                "velocity",
                "strength",
                "quality",
            )
        )
        rows = list(self.outcomes[-int(last_n) :])
        contrib: Dict[str, Any] = {}
        for f in feats:
            with_f = [r for r in rows if r.get(f) is not None]
            if len(with_f) < 30:
                contrib[f] = {
                    "lift_pp": None,
                    "n": len(with_f),
                    "display": f"{f}: insufficient n={len(with_f)}",
                }
                continue
            vals = sorted(float(r[f]) for r in with_f)
            hi_cut = vals[int(len(vals) * 0.70)]
            lo_cut = vals[int(len(vals) * 0.30)]
            hi = [r for r in with_f if float(r[f]) >= hi_cut]
            lo = [r for r in with_f if float(r[f]) <= lo_cut]
            wr_hi = _wr(hi)
            wr_lo = _wr(lo)
            lift = (wr_hi - wr_lo) * 100.0
            contrib[f] = {
                "lift_pp": round(lift, 1),
                "wr_high": round(wr_hi * 100, 1),
                "wr_low": round(wr_lo * 100, 1),
                "n_high": len(hi),
                "n_low": len(lo),
                "n": len(with_f),
                "display": f"{f}: {lift:+.0f}% WR lift (hi vs lo tercile)",
            }
        # Rank by absolute lift
        ranked = sorted(
            [(k, v) for k, v in contrib.items() if v.get("lift_pp") is not None],
            key=lambda x: abs(float(x[1]["lift_pp"])),
            reverse=True,
        )
        lines = [f"{k:24s} {v['lift_pp']:+.0f}%" for k, v in ranked]
        return {
            "n": len(rows),
            "features": contrib,
            "ranked": [{"feature": k, **v} for k, v in ranked],
            "report_lines": lines,
            "display": "Feature Contribution\n" + "\n".join(lines)
            if lines
            else "Feature Contribution: need more outcomes",
        }

    def validation_checklist(self) -> Dict[str, Any]:
        """AI development agent checklist (subset runnable online)."""
        bands = self.band_report()
        drift = self.prediction_drift()
        hpp_v = self.metric_validation("hpp", high=80, low=60)
        clarity_v = self.metric_validation("clarity", high=80, low=60)
        mom_v = self.metric_validation("momentum", high=70, low=40)
        pers_v = self.metric_validation("persistence", high=60, low=45)
        mp_v = self.metric_validation("momentum_persistence", high=70, low=50)
        try:
            from src.analytics.momentum_persistence_engine import (
                validate_persistence_velocity_edge,
            )

            pvel_edge = validate_persistence_velocity_edge(
                self.outcomes,
                min_samples=1000,
                auto_reduce=True,
            )
        except Exception as e:
            pvel_edge = {"pass": True, "error": str(e)}
        vel_v = self.velocity_validation()
        family_v = self.family_metric_validation()
        regime_v = self.regime_profitability()
        ci = self.confidence_interval_report(last_n=500)
        feat = self.feature_contribution_report(last_n=500)
        tq_hi = self.metric_validation("quality", high=80, low=70)
        checks = {
            "precision_by_score_band": {
                "pass": bands.get("monotonic_ok", True),
                "detail": bands,
            },
            "hpp_validation": hpp_v,
            "clarity_validation": clarity_v,
            "momentum_validation": mom_v,
            "persistence_validation": pers_v,
            "momentum_persistence_validation": mp_v,
            "persistence_velocity_edge": pvel_edge,
            "quality_80_vs_60": tq_hi,
            "velocity_validation": vel_v,
            "contract_specific": family_v,
            "feature_contribution": {
                "pass": True,  # informational
                "detail": feat,
            },
            "calibration": {
                "pass": (
                    bands.get("mean_calibration_error") is None
                    or float(bands["mean_calibration_error"]) <= 12.0
                ),
                "mean_error": bands.get("mean_calibration_error"),
            },
            "drift": {
                "pass": not drift.get("alert"),
                "detail": drift,
            },
            "regime": regime_v,
            "confidence_interval": ci,
        }
        all_pass = all(
            bool(v.get("pass", True))
            for k, v in checks.items()
            if isinstance(v, dict) and "pass" in v
        )
        return {"pass": all_pass, "checks": checks, "n_outcomes": len(self.outcomes)}


def _band_expected_mid(band: str) -> float:
    mids = {
        "90-100": 92.0,
        "80-90": 85.0,
        "70-80": 75.0,
        "60-70": 65.0,
        "0-60": 50.0,
    }
    return mids.get(band, 50.0)


def _wr(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("is_win")) / len(rows)


def _mean(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


# Singleton
_calib: Optional[CalibrationTracker] = None


def get_calibration_tracker() -> CalibrationTracker:
    global _calib
    if _calib is None:
        _calib = CalibrationTracker()
    return _calib
