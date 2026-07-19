"""
HPP as a time series — not a static score.

Tracks: trend lines, multi-metric comparison, weight evolution,
rolling windows, velocity/acceleration, lifecycle, waterfall, Meta-HPP.

Snapshots stored daily and every N trades under data/hpp_timeseries.json.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.analytics.historical_predictive_power import (
    get_hpp_tracker,
    time_decay_hpp,
)
from src.analytics.contract_profiles import (
    get_base_profile,
    get_weight_engine,
    list_profiles,
    normalize_contract_key,
)

DEFAULT_PATH = Path("data/hpp_timeseries.json")
SNAPSHOT_EVERY_N_TRADES = 10


def _day_key(ts: Optional[float] = None) -> str:
    dt = (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        if ts
        else datetime.now(timezone.utc)
    )
    return dt.strftime("%Y-%m-%d")


def moving_average(series: Sequence[float], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i in range(len(series)):
        if i + 1 < window:
            out.append(None)
        else:
            chunk = series[i + 1 - window : i + 1]
            out.append(round(sum(chunk) / len(chunk), 2))
    return out


def lifecycle_stage(hpp: float) -> str:
    """
    85+ Peak · 70–85 Mature · 55–70 Declining · <55 Retire
    """
    h = float(hpp)
    if h >= 85:
        return "Peak"
    if h >= 70:
        return "Mature"
    if h >= 55:
        return "Declining"
    return "Retire"


def trend_label(velocity: float, acceleration: float = 0.0) -> str:
    if velocity >= 5 and acceleration >= 0:
        return "STRONGLY IMPROVING"
    if velocity >= 2:
        return "UPWARD"
    if velocity <= -5:
        return "STRONGLY DECLINING"
    if velocity <= -2:
        return "DOWNWARD"
    return "STABLE"


class HPPTimeSeries:
    """
    Persist and analyze HPP snapshots over time.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        # daily[date][contract][metric] = hpp
        self.daily: Dict[str, Dict[str, Dict[str, float]]] = {}
        # points: chronological snapshots
        self.points: List[Dict[str, Any]] = []
        # weight history
        self.weight_history: List[Dict[str, Any]] = []
        self._trades_since_snap = 0
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.daily = data.get("daily") or {}
            self.points = list(data.get("points") or [])[-2000:]
            self.weight_history = list(data.get("weight_history") or [])[-1000:]
            self._trades_since_snap = int(data.get("trades_since_snap") or 0)
        except Exception:
            pass

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "daily": self.daily,
                        "points": self.points[-2000:],
                        "weight_history": self.weight_history[-1000:],
                        "trades_since_snap": self._trades_since_snap,
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def note_trade(self) -> bool:
        """Call on each closed trade; returns True if a snapshot was taken."""
        self._trades_since_snap += 1
        if self._trades_since_snap >= SNAPSHOT_EVERY_N_TRADES:
            self._trades_since_snap = 0
            self.capture_snapshot(reason="trades")
            return True
        # Always ensure daily point exists
        today = _day_key()
        if today not in self.daily or not self.points or self.points[-1].get("day") != today:
            self.capture_snapshot(reason="daily")
            return True
        self.save()
        return False

    def capture_snapshot(
        self,
        *,
        reason: str = "manual",
        contracts: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Compute HPP for all metrics/contracts and append time series point."""
        tracker = get_hpp_tracker()
        eng = get_weight_engine()
        day = _day_key()
        ts = time.time()

        cts = list(contracts) if contracts else list_profiles()
        # Only contracts that have outcomes + always common ones
        seen = {str(o.get("contract") or "").upper() for o in tracker.outcomes}
        if seen:
            cts = sorted(set(cts) | seen)

        by_contract: Dict[str, Any] = {}
        for ct in cts:
            key = normalize_contract_key(ct)
            base = get_base_profile(key)
            metrics = list(base.keys())
            if not metrics:
                continue
            pack = tracker.hpp_weights(key, metrics, blend_base=base, base_blend=0.25)
            # Rolling windows per metric
            windows: Dict[str, Dict[str, float]] = {}
            for m in metrics:
                rep = tracker.metric_hpp(key, m)
                windows[m] = {
                    "short": float((rep.get("windows") or {}).get("last_100") or rep.get("hpp") or 50),
                    "mid": float((rep.get("windows") or {}).get("last_500") or rep.get("hpp") or 50),
                    "long": float((rep.get("windows") or {}).get("last_1000") or rep.get("hpp") or 50),
                    "hpp": float(rep.get("hpp") or 50),
                }
            # Aggregate contract HPP = mean of metric HPPs
            hpps = [windows[m]["hpp"] for m in metrics]
            contract_hpp = sum(hpps) / len(hpps) if hpps else 50.0

            # Current adaptive weights
            try:
                dummy_metrics = {m: 70.0 for m in metrics}
                wt = eng.compute_weights(
                    key, dummy_metrics, sample_n=200, use_learning=True
                )
                weights = wt.get("normalized_weights") or pack.get("weights") or {}
            except Exception:
                weights = pack.get("weights") or {}

            by_contract[key] = {
                "hpp": round(contract_hpp, 1),
                "metrics": {m: round(windows[m]["hpp"], 1) for m in metrics},
                "windows": windows,
                "weights": {k: round(float(v), 4) for k, v in weights.items()},
                "strongest": pack.get("strongest"),
                "lifecycle": lifecycle_stage(contract_hpp),
            }

            # daily matrix
            self.daily.setdefault(day, {})
            self.daily[day][key] = {
                m: round(windows[m]["hpp"], 1) for m in metrics
            }
            self.daily[day][key]["_contract_hpp"] = round(contract_hpp, 1)

        point = {
            "ts": ts,
            "day": day,
            "reason": reason,
            "contracts": by_contract,
        }
        # Full multi-window HPP velocity (trade-based + EMA + state machine)
        from src.analytics.hpp_velocity import (
            attach_velocities_to_snapshot,
            classify_edge_flag,
        )
        from src.analytics.historical_predictive_power import get_hpp_tracker

        n_out = len(get_hpp_tracker().outcomes)
        prev_point = self.points[-1] if self.points else None
        series_by_ct: Dict[str, List[float]] = {}
        for p in self.points:
            for ct, cd in (p.get("contracts") or {}).items():
                series_by_ct.setdefault(ct, []).append(float(cd.get("hpp") or 50))

        for ct, data in by_contract.items():
            prev_c = (
                ((prev_point or {}).get("contracts") or {}).get(ct)
                if prev_point
                else None
            )
            enriched = attach_velocities_to_snapshot(
                data,
                previous_contract=prev_c,
                series_hpp=series_by_ct.get(ct) or [],
                sample_n=n_out,
            )
            # Acceleration of EMA velocity
            prev_ema = float((prev_c or {}).get("velocity_ema") or 0)
            cur_ema = float(enriched.get("velocity_ema") or 0)
            enriched["acceleration"] = round(cur_ema - prev_ema, 2)
            enriched["trend"] = trend_label(cur_ema, enriched["acceleration"])
            by_contract[ct] = enriched

        point["contracts"] = by_contract
        self.points.append(point)
        if len(self.points) > 2000:
            self.points = self.points[-2000:]

        # Weight evolution sample (first contract with data)
        for ct, data in by_contract.items():
            self.weight_history.append(
                {
                    "ts": ts,
                    "day": day,
                    "contract": ct,
                    "weights": data.get("weights") or {},
                    "hpp": data.get("hpp"),
                }
            )
            break
        if len(self.weight_history) > 1000:
            self.weight_history = self.weight_history[-1000:]

        self.save()
        return point

    # ----- Series extractors for charts -----

    def contract_hpp_series(self, contract: str) -> Dict[str, Any]:
        key = normalize_contract_key(contract)
        days, values = [], []
        for p in self.points:
            c = (p.get("contracts") or {}).get(key)
            if not c:
                continue
            days.append(p.get("day"))
            values.append(float(c.get("hpp") or 50))
        ma7 = moving_average(values, 7)
        ma30 = moving_average(values, 30)
        return {
            "contract": key,
            "days": days,
            "hpp": values,
            "ma7": ma7,
            "ma30": ma30,
            "current": values[-1] if values else None,
            "lifecycle": lifecycle_stage(values[-1]) if values else "Retire",
        }

    def multi_metric_series(self, contract: str) -> Dict[str, Any]:
        key = normalize_contract_key(contract)
        series: Dict[str, List[float]] = defaultdict(list)
        days: List[str] = []
        for p in self.points:
            c = (p.get("contracts") or {}).get(key)
            if not c:
                continue
            days.append(p.get("day") or "")
            mets = c.get("metrics") or {}
            for m, v in mets.items():
                # pad missing
                pass
            # align all metrics seen
            all_m = set()
            for pp in self.points:
                cc = (pp.get("contracts") or {}).get(key) or {}
                all_m |= set((cc.get("metrics") or {}).keys())
            for m in sorted(all_m):
                series[m].append(float((c.get("metrics") or {}).get(m) or 50))
        # rebuild cleanly
        series = defaultdict(list)
        days = []
        all_m: set = set()
        for p in self.points:
            c = (p.get("contracts") or {}).get(key) or {}
            all_m |= set((c.get("metrics") or {}).keys())
        for p in self.points:
            c = (p.get("contracts") or {}).get(key)
            if not c:
                continue
            days.append(p.get("day") or "")
            for m in sorted(all_m):
                series[m].append(float((c.get("metrics") or {}).get(m) or 50))
        return {"contract": key, "days": days, "series": dict(series)}

    def weight_evolution(self, contract: str) -> Dict[str, Any]:
        key = normalize_contract_key(contract)
        days, by_metric = [], defaultdict(list)
        for wh in self.weight_history:
            if normalize_contract_key(str(wh.get("contract") or "")) != key:
                continue
            days.append(wh.get("day"))
            w = wh.get("weights") or {}
            for m, v in w.items():
                by_metric[m].append(round(float(v) * 100.0, 2))  # as %
            # pad metrics not in this snap
        # rebuild aligned
        days = []
        by_metric = defaultdict(list)
        all_m: set = set()
        rows = [
            wh
            for wh in self.weight_history
            if normalize_contract_key(str(wh.get("contract") or "")) == key
        ]
        for wh in rows:
            all_m |= set((wh.get("weights") or {}).keys())
        for wh in rows:
            days.append(wh.get("day"))
            w = wh.get("weights") or {}
            for m in sorted(all_m):
                by_metric[m].append(round(float(w.get(m) or 0) * 100.0, 2))
        return {"contract": key, "days": days, "weights_pct": dict(by_metric)}

    def rolling_windows_table(self, contract: str) -> Dict[str, Any]:
        key = normalize_contract_key(contract)
        if not self.points:
            return {"contract": key, "rows": []}
        c = (self.points[-1].get("contracts") or {}).get(key) or {}
        windows = c.get("windows") or {}
        rows = []
        for m, w in windows.items():
            short = float(w.get("short") or 50)
            mid = float(w.get("mid") or 50)
            long_ = float(w.get("long") or 50)
            if short > mid + 3 and short > long_ + 3:
                interp = "improving"
            elif short < mid - 3 and short < long_ - 3:
                interp = "weakening"
            else:
                interp = "stable"
            rows.append(
                {
                    "metric": m,
                    "short": round(short, 1),
                    "mid": round(mid, 1),
                    "long": round(long_, 1),
                    "interpretation": interp,
                }
            )
        return {"contract": key, "rows": rows}

    def heatmap(self, contract: str, max_months: int = 6) -> Dict[str, Any]:
        """Month × metric grid of average HPP."""
        key = normalize_contract_key(contract)
        # group points by YYYY-MM
        by_month: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for p in self.points:
            day = p.get("day") or ""
            month = day[:7] if len(day) >= 7 else day
            c = (p.get("contracts") or {}).get(key) or {}
            for m, v in (c.get("metrics") or {}).items():
                by_month[month][m].append(float(v))
        months = sorted(by_month.keys())[-max_months:]
        metrics = sorted(
            {m for mo in months for m in by_month[mo].keys()}
        )
        cells = []
        for m in metrics:
            row = {"metric": m, "months": {}}
            for mo in months:
                vals = by_month[mo].get(m) or []
                avg = sum(vals) / len(vals) if vals else None
                if avg is None:
                    band = "none"
                elif avg >= 75:
                    band = "strong"
                elif avg >= 55:
                    band = "neutral"
                else:
                    band = "weak"
                row["months"][mo] = {
                    "hpp": round(avg, 1) if avg is not None else None,
                    "band": band,
                }
            cells.append(row)
        return {"contract": key, "months": months, "rows": cells}

    def radar(self, contract: str) -> Dict[str, Any]:
        key = normalize_contract_key(contract)
        if not self.points:
            return {"contract": key, "axes": {}}
        c = (self.points[-1].get("contracts") or {}).get(key) or {}
        mets = c.get("metrics") or {}
        axes = {
            "entropy": float(mets.get("digit_entropy") or mets.get("parity_entropy") or 50),
            "momentum": float(mets.get("momentum") or mets.get("parity_momentum") or 50),
            "streak": float(mets.get("streak_entropy") or 50),
            "sample": min(100.0, len(get_hpp_tracker().outcomes) / 10.0),
            "stability": float(c.get("hpp") or 50),  # proxy
        }
        # pull stability from windows variance if possible
        return {"contract": key, "axes": {k: round(v, 1) for k, v in axes.items()}}

    def waterfall(self, contract: str) -> Dict[str, Any]:
        """
        Explain HPP change vs previous snapshot via metric deltas.
        """
        key = normalize_contract_key(contract)
        if len(self.points) < 2:
            cur = (self.points[-1].get("contracts") or {}).get(key) if self.points else {}
            base = float((cur or {}).get("hpp") or 50)
            return {
                "contract": key,
                "base": base,
                "current": base,
                "steps": [],
                "total_delta": 0.0,
            }
        prev = (self.points[-2].get("contracts") or {}).get(key) or {}
        cur = (self.points[-1].get("contracts") or {}).get(key) or {}
        base = float(prev.get("hpp") or 50)
        current = float(cur.get("hpp") or 50)
        prev_m = prev.get("metrics") or {}
        cur_m = cur.get("metrics") or {}
        # contribution = equal share of metric deltas scaled to total delta
        deltas = {
            m: float(cur_m.get(m) or 0) - float(prev_m.get(m) or 0)
            for m in set(prev_m) | set(cur_m)
        }
        raw_sum = sum(deltas.values()) or 1.0
        total_delta = current - base
        steps = []
        for m, d in sorted(deltas.items(), key=lambda x: -abs(x[1])):
            # scale metric delta to contract HPP delta proportionally
            contrib = (d / raw_sum) * total_delta if abs(raw_sum) > 1e-9 else 0.0
            steps.append(
                {
                    "metric": m,
                    "delta": round(contrib, 2),
                    "metric_hpp_delta": round(d, 2),
                }
            )
        # residual noise
        explained = sum(s["delta"] for s in steps)
        noise = round(total_delta - explained, 2)
        if abs(noise) >= 0.05:
            steps.append({"metric": "noise", "delta": noise, "metric_hpp_delta": 0.0})
        return {
            "contract": key,
            "base": round(base, 1),
            "current": round(current, 1),
            "total_delta": round(total_delta, 2),
            "steps": steps,
        }

    def meta_hpp(self, contract: str) -> Dict[str, Any]:
        """
        Meta-HPP = 40% Current + 25% Trend Strength + 20% Velocity + 15% Stability
        """
        key = normalize_contract_key(contract)
        series = self.contract_hpp_series(key)
        values = series.get("hpp") or []
        current = float(values[-1]) if values else 50.0
        # Trend strength: slope of last up to 10 points → 0–100
        if len(values) >= 3:
            n = min(10, len(values))
            recent = values[-n:]
            slope = (recent[-1] - recent[0]) / max(1, n - 1)
            trend_strength = max(0.0, min(100.0, 50.0 + slope * 15.0))
        else:
            trend_strength = 50.0
            slope = 0.0
        # Velocity from last point
        vel = 0.0
        acc = 0.0
        if self.points:
            c = (self.points[-1].get("contracts") or {}).get(key) or {}
            vel = float(c.get("velocity") or 0)
            acc = float(c.get("acceleration") or 0)
        # map velocity ±10 → 0–100
        velocity_score = max(0.0, min(100.0, 50.0 + vel * 5.0))
        # stability of last 10 HPP
        if len(values) >= 3:
            chunk = values[-10:]
            mean = sum(chunk) / len(chunk)
            var = sum((x - mean) ** 2 for x in chunk) / len(chunk)
            std = math.sqrt(var)
            stability = max(0.0, min(100.0, 100.0 - std * 8.0))
        else:
            stability = 55.0

        meta = (
            0.40 * current
            + 0.25 * trend_strength
            + 0.20 * velocity_score
            + 0.15 * stability
        )
        meta = max(0.0, min(100.0, meta))
        status = trend_label(vel, acc)
        conf = (
            "HIGH"
            if len(values) >= 15 and stability >= 60
            else ("MEDIUM" if len(values) >= 5 else "LOW")
        )
        # Recommended weight increase when strongly improving
        weight_adj = 0.0
        if vel >= 5 and meta >= 75:
            weight_adj = min(15.0, 5.0 + vel)
        elif vel <= -5:
            weight_adj = max(-15.0, -5.0 + vel)

        return {
            "contract": key,
            "meta_hpp": round(meta, 1),
            "status": status,
            "confidence": conf,
            "recommended_weight_change_pct": round(weight_adj, 1),
            "components": {
                "current_hpp": round(current, 1),
                "trend_strength": round(trend_strength, 1),
                "velocity_score": round(velocity_score, 1),
                "stability": round(stability, 1),
                "velocity": round(vel, 2),
                "acceleration": round(acc, 2),
            },
            "lifecycle": lifecycle_stage(current),
        }

    def contract_dashboard(self, contract: str) -> Dict[str, Any]:
        """Full package for one contract (dashboard card)."""
        key = normalize_contract_key(contract)
        if not self.points:
            self.capture_snapshot(reason="bootstrap")
        cur = {}
        if self.points:
            cur = (self.points[-1].get("contracts") or {}).get(key) or {}
        metrics = cur.get("metrics") or {}
        mvel = cur.get("metric_velocity") or {}
        mdetail = cur.get("metric_velocity_detail") or {}
        lines = []
        for m, v in sorted(metrics.items(), key=lambda x: -x[1]):
            det = mdetail.get(m) or {}
            d = float(det.get("velocity_ema") if det else mvel.get(m) or 0)
            arrow = det.get("arrow") or (
                "▲" if d > 1 else ("▼" if d < -1 else "→")
            )
            lines.append(
                {
                    "metric": m,
                    "hpp": v,
                    "delta": d,
                    "arrow": arrow,
                    "status": det.get("status") or velocity_state_safe(d),
                    "velocity_pct": det.get("velocity_pct"),
                    "short": det.get("short_velocity"),
                    "medium": det.get("medium_velocity"),
                    "long": det.get("long_velocity"),
                    "effective": det.get("effective_velocity_ema"),
                    "display": det.get("display"),
                }
            )
        overall = cur.get("overall_velocity") or {}
        return {
            "contract": key,
            "current_hpp": cur.get("hpp"),
            "trend": cur.get("trend") or "STABLE",
            "velocity": cur.get("velocity_ema", cur.get("velocity")),
            "velocity_pct": cur.get("velocity_pct"),
            "effective_velocity": cur.get("effective_velocity"),
            "velocity_status": cur.get("status"),
            "edge_flag": cur.get("edge_flag"),
            "acceleration": cur.get("acceleration"),
            "momentum_score": cur.get("momentum_score"),
            "arrow": cur.get("arrow"),
            "lifecycle": cur.get("lifecycle") or lifecycle_stage(float(cur.get("hpp") or 50)),
            "metrics": lines,
            "overall_velocity": overall,
            "strongest": cur.get("strongest"),
            "series": self.contract_hpp_series(key),
            "multi_metric": self.multi_metric_series(key),
            "weights": self.weight_evolution(key),
            "windows": self.rolling_windows_table(key),
            "heatmap": self.heatmap(key),
            "radar": self.radar(key),
            "waterfall": self.waterfall(key),
            "meta": self.meta_hpp(key),
        }

    def dashboard_bundle(
        self, contracts: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """Multi-contract dashboard payload."""
        if not self.points:
            self.capture_snapshot(reason="bootstrap")
        cts = list(contracts) if contracts else []
        if not cts and self.points:
            cts = list((self.points[-1].get("contracts") or {}).keys())
        if not cts:
            cts = ["DIGITDIFF", "DIGITEVEN", "CALL"]
        boards = {c: self.contract_dashboard(c) for c in cts[:6]}
        return {
            "updated_at": time.time(),
            "n_points": len(self.points),
            "contracts": boards,
            "primary": boards.get(cts[0]) if cts else {},
        }


def velocity_state_safe(v: float) -> str:
    from src.analytics.hpp_velocity import velocity_state

    return velocity_state(v)


# Singleton
_ts: Optional[HPPTimeSeries] = None


def get_hpp_timeseries() -> HPPTimeSeries:
    global _ts
    if _ts is None:
        _ts = HPPTimeSeries()
    return _ts
