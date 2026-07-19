"""
Self-optimizing Market Scanner.

Continuously ranks markets:

  V75     Score 91
  V50     Score 88
  Boom500 Score 83
  ...

Trade only when (all markets):

  Edge Score          >= 80
  Pattern Clarity     >= 75
  HPP                 >= 75
  Momentum Persistence >= 70
  EV                  >  0

Every 500 trades → Top/Worst markets by Profit Factor report.

Scan priority auto-adjusts:
  reduce: PF < 1, HPP velocity < 0, drawdown rising
  boost:  PF > 1.5, HPP velocity > 0, clarity improving
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.strategy.market_categories import classify_market, market_profile

DEFAULT_PRIORITY_PATH = Path("data/market_scan_priority.json")
REPORT_EVERY_N = 500

# Hard trade gates for scanner "tradeable" flag
MIN_EDGE = 80.0
MIN_CLARITY = 75.0
MIN_HPP = 75.0
MIN_MP = 70.0
MIN_EV = 0.0


def _key(symbol: str) -> str:
    return str(symbol or "").strip().upper()


class MarketPriorityBook:
    """
    Persistent per-symbol priority weights for scan ordering.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PRIORITY_PATH
        # symbol -> {priority, pf, n, wins, losses, pnl, peak_pnl, dd, hpp_vel, clarity_trend}
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.total_trades = 0
        self.last_report: Optional[Dict[str, Any]] = None
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.stats = data.get("stats") or {}
            self.total_trades = int(data.get("total_trades") or 0)
            self.last_report = data.get("last_report")
        except Exception:
            self.stats = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "stats": self.stats,
                        "total_trades": self.total_trades,
                        "last_report": self.last_report,
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _row(self, symbol: str) -> Dict[str, Any]:
        k = _key(symbol)
        return self.stats.setdefault(
            k,
            {
                "priority": 1.0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
                "peak_pnl": 0.0,
                "max_dd": 0.0,
                "hpp_vel_sum": 0.0,
                "hpp_vel_n": 0,
                "clarity_sum": 0.0,
                "clarity_n": 0,
                "clarity_prev": None,
            },
        )

    def record_trade(
        self,
        symbol: str,
        *,
        is_win: bool,
        profit: float = 0.0,
        hpp_velocity: Optional[float] = None,
        clarity: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        row = self._row(symbol)
        if is_win:
            row["wins"] = int(row["wins"]) + 1
        else:
            row["losses"] = int(row["losses"]) + 1
        row["pnl"] = float(row["pnl"]) + float(profit)
        peak = float(row.get("peak_pnl") or 0)
        if row["pnl"] > peak:
            row["peak_pnl"] = row["pnl"]
        dd = max(0.0, float(row["peak_pnl"]) - float(row["pnl"]))
        row["max_dd"] = max(float(row.get("max_dd") or 0), dd)

        if hpp_velocity is not None:
            row["hpp_vel_sum"] = float(row["hpp_vel_sum"]) + float(hpp_velocity)
            row["hpp_vel_n"] = int(row["hpp_vel_n"]) + 1
        if clarity is not None:
            prev = row.get("clarity_prev")
            row["clarity_sum"] = float(row["clarity_sum"]) + float(clarity)
            row["clarity_n"] = int(row["clarity_n"]) + 1
            row["clarity_prev"] = float(clarity)
            if prev is not None:
                row["clarity_delta"] = float(clarity) - float(prev)

        self.total_trades += 1
        self._recompute_priority(symbol)
        report = None
        if self.total_trades > 0 and self.total_trades % REPORT_EVERY_N == 0:
            report = self.build_performance_report()
            self.last_report = report
        self.save()
        return report

    def profit_factor(self, symbol: str) -> float:
        row = self._row(symbol)
        # Approximate PF from wins/losses if stakes ~1
        w, l = int(row["wins"]), int(row["losses"])
        gp = max(0.0, float(row["pnl"])) if float(row["pnl"]) > 0 else float(w) * 0.9
        gl = abs(min(0.0, float(row["pnl"]))) if float(row["pnl"]) < 0 else float(l)
        # Better: use win count * avg win vs losses
        if l == 0:
            return 10.0 if w > 0 else 1.0
        if w == 0:
            return 0.0
        # Use pnl decomposition if possible
        avg_win = 0.95
        avg_loss = 1.0
        gp = w * avg_win
        gl = l * avg_loss
        # Blend with realized pnl sign
        if float(row["pnl"]) != 0 and (w + l) >= 3:
            # scale PF toward realized
            realized = float(row["pnl"])
            # crude: PF from W/L counts
            pass
        return max(0.0, gp / gl)

    def _recompute_priority(self, symbol: str) -> float:
        """
        reduce: PF < 1, HPP vel < 0, drawdown increasing
        boost:  PF > 1.5, HPP vel > 0, clarity improving
        """
        row = self._row(symbol)
        n = int(row["wins"]) + int(row["losses"])
        pf = self.profit_factor(symbol)
        vel = (
            float(row["hpp_vel_sum"]) / int(row["hpp_vel_n"])
            if int(row["hpp_vel_n"]) > 0
            else 0.0
        )
        clarity_delta = float(row.get("clarity_delta") or 0)
        dd = float(row.get("max_dd") or 0)
        pnl = float(row.get("pnl") or 0)
        dd_rising = dd > 0 and pnl < float(row.get("peak_pnl") or 0) * 0.5

        p = 1.0
        if n >= 5:
            if pf < 1.0:
                p *= 0.55
            if vel < 0:
                p *= 0.75
            if dd_rising:
                p *= 0.70
            if pf > 1.5:
                p *= 1.35
            if vel > 0:
                p *= 1.15
            if clarity_delta > 0:
                p *= 1.10
        p = max(0.15, min(2.5, p))
        row["priority"] = round(p, 3)
        row["pf"] = round(pf, 3)
        row["avg_hpp_velocity"] = round(vel, 2)
        return p

    def priority(self, symbol: str) -> float:
        return float(self._row(symbol).get("priority") or 1.0)

    def ordered_symbols(self, symbols: Sequence[str]) -> List[str]:
        """Higher priority first; stable for ties."""
        return sorted(
            list(symbols),
            key=lambda s: (-self.priority(s), s),
        )

    def build_performance_report(self) -> Dict[str, Any]:
        """
        Top / Worst markets by profit factor (min samples).
        """
        rows = []
        for sym, st in self.stats.items():
            n = int(st.get("wins") or 0) + int(st.get("losses") or 0)
            if n < 5:
                continue
            pf = self.profit_factor(sym)
            rows.append(
                {
                    "symbol": sym,
                    "category": classify_market(sym),
                    "n": n,
                    "wins": st.get("wins"),
                    "losses": st.get("losses"),
                    "pnl": round(float(st.get("pnl") or 0), 2),
                    "pf": round(pf, 3),
                    "priority": st.get("priority"),
                    "max_dd": round(float(st.get("max_dd") or 0), 2),
                    "hpp_velocity": st.get("avg_hpp_velocity"),
                }
            )
        rows.sort(key=lambda r: r["pf"], reverse=True)
        top = rows[:10]
        worst = list(reversed(rows[-10:])) if rows else []
        lines = ["Top Markets By Profit Factor"]
        for i, r in enumerate(top[:5], 1):
            lines.append(f"{i}. {r['symbol']}  PF = {r['pf']:.2f}  (n={r['n']})")
        lines.append("")
        lines.append("Worst Markets")
        for i, r in enumerate(worst[:5], 1):
            lines.append(f"{i}. {r['symbol']}  PF = {r['pf']:.2f}  (n={r['n']})")
        return {
            "total_trades": self.total_trades,
            "top": top,
            "worst": worst,
            "report_lines": lines,
            "display": "\n".join(lines),
            "ts": time.time(),
        }


_book: Optional[MarketPriorityBook] = None


def get_priority_book() -> MarketPriorityBook:
    global _book
    if _book is None:
        _book = MarketPriorityBook()
    return _book


def scanner_gates_pass(
    *,
    edge_score: float,
    pattern_clarity: float,
    hpp: float,
    momentum_persistence: float,
    ev: float,
    cold_start: bool = False,
) -> Dict[str, Any]:
    """
    Edge >= 80 · Clarity >= 75 · HPP >= 75 · MP >= 70 · EV > 0
    Softened slightly in cold-start.
    """
    min_edge = 70.0 if cold_start else MIN_EDGE
    min_cl = 65.0 if cold_start else MIN_CLARITY
    min_hpp = 60.0 if cold_start else MIN_HPP
    min_mp = 55.0 if cold_start else MIN_MP

    checks = {
        "edge": float(edge_score) >= min_edge,
        "clarity": float(pattern_clarity) >= min_cl,
        "hpp": float(hpp) >= min_hpp,
        "momentum_persistence": float(momentum_persistence) >= min_mp,
        "ev": float(ev) > MIN_EV,
    }
    return {
        "allow": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "edge": min_edge,
            "clarity": min_cl,
            "hpp": min_hpp,
            "mp": min_mp,
            "ev": MIN_EV,
        },
    }


def rank_markets(
    symbols: Sequence[str],
    get_ticks: Callable[[str], Sequence[Dict[str, Any]]],
    *,
    history_by_key: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    top_n: int = 15,
    global_samples: int = 0,
) -> Dict[str, Any]:
    """
    Full scanner: category-aware evaluate_setup + priority-weighted rank.
    """
    from src.analytics.trade_filter import evaluate_setup
    from src.analytics.probability_engine import probability_table

    history_by_key = history_by_key or {}
    book = get_priority_book()
    ordered = book.ordered_symbols(symbols)
    ranked: List[Dict[str, Any]] = []
    cold = global_samples < 50

    for sym in ordered:
        ticks = list(get_ticks(sym) or [])
        prof = market_profile(sym)
        if len(ticks) < 30:
            ranked.append(
                {
                    "symbol": sym,
                    "score": 0.0,
                    "status": "NO_DATA",
                    "category": prof["category"],
                    "priority": book.priority(sym),
                    "best_type": None,
                    "tradeable": False,
                }
            )
            continue

        probs = probability_table(ticks, symbol=sym)
        best = probs.get("best") or {}
        # Prefer RF if category disallows digits
        ct = str(best.get("key") or "CALL").upper()
        allowed = set(prof.get("allowed_contracts") or [])
        if ct not in allowed:
            ct = "CALL" if "CALL" in allowed else next(iter(allowed), "CALL")

        key = f"{sym}|{ct}"
        hist = list(history_by_key.get(key) or [])
        if not hist:
            for k, rows in history_by_key.items():
                if k.startswith(f"{sym}|"):
                    hist.extend(rows)

        family = "rise_fall" if ct in {"CALL", "PUT"} else "digits"
        try:
            ev = evaluate_setup(
                ticks,
                symbol=sym,
                contract_type=ct,
                family=family,
                history_rows=hist,
                recent_rows=hist[-100:],
                global_samples=global_samples,
            )
        except Exception as e:
            ranked.append(
                {
                    "symbol": sym,
                    "score": 0.0,
                    "status": f"ERROR:{e}",
                    "category": prof["category"],
                    "priority": book.priority(sym),
                    "tradeable": False,
                }
            )
            continue

        live = float((ev.get("live_edge") or {}).get("live_edge") or 0)
        quality = float((ev.get("quality") or {}).get("quality_score") or 0)
        edge = float((ev.get("historical_edge") or {}).get("edge_score") or live)
        clarity = float((ev.get("pattern_clarity") or {}).get("pattern_clarity") or 0)
        strength = float((ev.get("pattern_strength") or {}).get("pattern_strength") or 0)
        hpp = float(ev.get("hpp") or 50)
        mp = float(ev.get("mp_score") or 50)
        exp_v = float(ev.get("ev") or 0)
        hpp_vel = float(ev.get("hpp_velocity") or 0)
        conf100 = float(ev.get("p_win") or best.get("confidence") or 0.5)
        if conf100 <= 1.0:
            conf100 *= 100.0
        conf100 = max(conf100, float(ev.get("decision_quality") or quality) * 0.5)

        # Regime match for MOR
        from src.analytics.market_opportunity_ranking import (
            compute_mor,
            correlation_filter,
            get_opportunity_history,
            multi_horizon_label,
            regime_match_score,
        )

        reg = str(ev.get("regime") or (ev.get("rolling_entropy") or {}).get("regime") or "RANDOM")
        chop = 0.0
        try:
            chop = float(
                ((ev.get("filter") or {}) if False else 0)
            )
        except Exception:
            chop = 0.0
        # chop from last filter reasons not always present — use entropy regime
        regime_m = regime_match_score(
            family=family,
            market_regime=reg,
            chop_score=0.35 if reg == "RANDOM" else 0.15,
            strategy_path=prof.get("scoring_path") or "",
        )
        sample_n = int(ev.get("sample_size") or len(hist) or 0)
        mor = compute_mor(
            pattern_strength=strength,
            pattern_clarity=clarity,
            hpp=hpp,
            hpp_velocity=hpp_vel,
            momentum_persistence=mp,
            regime_match=regime_m,
            expected_value=exp_v,
            confidence=conf100,
            sample_n=sample_n,
            drawdown_pct=0.0,
            hpp_unstable=abs(hpp_vel) > 12 and sample_n < 30,
        )
        # Opportunity velocity / acceleration
        hist_pack = get_opportunity_history().note(
            sym, float(mor["opportunity_score"]), mor["tier"]
        )
        mor_vel = hist_pack
        # Priority amplifies rank (self-optimizing scanner)
        score = float(mor["rank_score"]) * book.priority(sym)

        gates = scanner_gates_pass(
            edge_score=max(edge, live),
            pattern_clarity=clarity,
            hpp=hpp,
            momentum_persistence=mp,
            ev=exp_v,
            cold_start=cold or bool(ev.get("cold_start")),
        )
        # Only ELITE/STRONG for attention; tradeable still needs gates
        tier = mor["tier"]
        attention = tier in {"ELITE", "STRONG", "WATCHLIST"}

        ranked.append(
            {
                "symbol": sym,
                "score": round(score, 1),
                "opportunity_score": mor["opportunity_score"],
                "opportunity_raw": mor["opportunity_raw"],
                "effective_score": mor["effective_score"],
                "rank_score": mor["rank_score"],
                "tier": tier,
                "mor": mor,
                "opportunity_velocity": mor_vel.get("velocity"),
                "opportunity_acceleration": mor_vel.get("acceleration"),
                "opportunity_horizons": {
                    "short": mor_vel.get("short_term"),
                    "medium": mor_vel.get("medium_term"),
                    "long": mor_vel.get("long_term"),
                    "label": multi_horizon_label(mor_vel),
                },
                "live_edge": live,
                "edge_score": edge,
                "quality": quality,
                "pattern_strength": strength,
                "pattern_clarity": clarity,
                "hpp": hpp,
                "hpp_velocity": hpp_vel,
                "mp_score": mp,
                "ev": exp_v,
                "regime_match": regime_m,
                "status": (ev.get("live_edge") or {}).get("status"),
                "recommendation": ev.get("recommendation"),
                "best_type": ct,
                "best_confidence": best.get("confidence"),
                "allow": bool(ev.get("allow")) and gates["allow"] and tier != "IGNORE",
                "tradeable": gates["allow"] and tier in {"ELITE", "STRONG"},
                "attention": attention,
                "scanner_gates": gates,
                "category": prof["category"],
                "scoring_path": prof["scoring_path"],
                "priority": book.priority(sym),
                "copilot": ev.get("copilot"),
            }
        )

    ranked.sort(key=lambda r: r.get("score") or 0, reverse=True)
    # Correlation filter: avoid stacking highly related synthetics
    ranked = correlation_filter(ranked, max_per_cluster=1)
    ranked.sort(key=lambda r: r.get("score") or 0, reverse=True)
    tradeable = [r for r in ranked if r.get("tradeable") and not r.get("correlation_filtered")]
    elite = [r for r in ranked if r.get("tier") == "ELITE"]
    strong = [r for r in ranked if r.get("tier") == "STRONG"]
    return {
        "ranked": ranked[: max(top_n, len(ranked))],
        "best": ranked[0] if ranked else None,
        "best_tradeable": tradeable[0] if tradeable else None,
        "elite": elite[:5],
        "strong": strong[:5],
        "n_scanned": len(ranked),
        "n_tradeable": len(tradeable),
        "priority_book": {s: book.priority(s) for s in symbols},
        "last_report": book.last_report,
        "display": [
            f"{r['symbol']:12s}  Score {r.get('opportunity_score') or r.get('score')}  "
            f"{r.get('tier')}  {r.get('category')}  "
            f"{'TRADE' if r.get('tradeable') else r.get('recommendation') or '—'}"
            for r in ranked[:10]
        ],
        "tiers": {
            "ELITE": [r["symbol"] for r in elite],
            "STRONG": [r["symbol"] for r in strong],
            "WATCHLIST": [r["symbol"] for r in ranked if r.get("tier") == "WATCHLIST"],
            "IGNORE": [r["symbol"] for r in ranked if r.get("tier") == "IGNORE"],
        },
    }
