"""
AI Auditor — Rec #10

Self-improving system that analyzes feature contributions every 100 trades.
Identifies what's helping, what's hurting, and generates weight recommendations.

Minor audit: every 100 closed trades (persistent across restarts)
Major audit: every 1000 closed trades (comprehensive deep analysis)

Approach — Bootstrap feature quartile attribution:
  For each feature tracked in trade_history.jsonl:
    - Split trades into Q1 (bottom 25%) and Q4 (top 25%) by feature value
    - Contribution = Q4_win_rate - Q1_win_rate
    - Positive = feature helps when high
    - Negative = feature hurts when high (or at best, doesn't correlate)

Persistence:
  - data/auditor_report.json  (latest report)
  - data/auditor_history.jsonl (all past reports, appended)

Trade history source: data/trade_history.jsonl
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HISTORY_PATH = Path("data/trade_history.jsonl")
REPORT_PATH = Path("data/auditor_report.json")
HISTORY_LOG_PATH = Path("data/auditor_history.jsonl")

MINOR_INTERVAL = 100
MAJOR_INTERVAL = 1000
MIN_SAMPLES_PER_QUARTILE = 5  # need at least 5 per quartile to be meaningful

# Features tracked in trade_history.jsonl
TRACKED_FEATURES = [
    "confidence",
    "trend_strength",
    "learn_bonus",
    "parity_conf",
    "mor_score",
    "persistence_conf",
    "ev",
    "chop_score",
]


class AIAuditor:
    """
    Analyzes feature contributions and generates self-improvement recommendations.
    """

    def __init__(
        self,
        history_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
    ):
        self.history_path = Path(history_path) if history_path else HISTORY_PATH
        self.report_path = Path(report_path) if report_path else REPORT_PATH
        self.history_log_path = HISTORY_LOG_PATH
        self._last_report: Optional[Dict[str, Any]] = self._load_report()

    def _load_report(self) -> Optional[Dict[str, Any]]:
        if self.report_path.is_file():
            try:
                return json.loads(self.report_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def _load_trade_history(self, last_n: int = 1000) -> List[Dict[str, Any]]:
        """Load the last N trades from trade_history.jsonl."""
        if not self.history_path.is_file():
            return []
        trades = []
        try:
            with self.history_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return trades[-last_n:] if len(trades) > last_n else trades
        except Exception as e:
            logger.warning("AIAuditor: failed to load trade history: %s", e)
            return []

    def _quartile_contribution(
        self, trades: List[Dict], feature: str
    ) -> Optional[float]:
        """
        Compute Q4_win_rate - Q1_win_rate for a feature.
        Returns None if insufficient data.
        """
        # Filter trades that have this feature
        valid = [
            t for t in trades
            if t.get(feature) is not None and t.get("is_win") is not None
        ]
        if len(valid) < MIN_SAMPLES_PER_QUARTILE * 4:
            return None

        # Sort by feature value
        valid.sort(key=lambda t: float(t.get(feature) or 0))
        n = len(valid)
        q1_size = max(MIN_SAMPLES_PER_QUARTILE, n // 4)

        q1 = valid[:q1_size]
        q4 = valid[-q1_size:]

        q1_wr = sum(1 for t in q1 if t.get("is_win")) / len(q1)
        q4_wr = sum(1 for t in q4 if t.get("is_win")) / len(q4)

        return round((q4_wr - q1_wr) * 100, 2)  # as percentage points

    def _generate_recommendations(
        self,
        contributions: Dict[str, Optional[float]],
        overall_wr: float,
    ) -> List[str]:
        """Generate plain-language recommendations from feature contributions."""
        recs = []
        sorted_contribs = sorted(
            [(k, v) for k, v in contributions.items() if v is not None],
            key=lambda x: x[1],
            reverse=True,
        )

        for feature, contrib in sorted_contribs:
            if contrib > 10:
                recs.append(
                    f"Increase '{feature}' weight in trade_selector — "
                    f"strong positive correlation (+{contrib:.1f}%)"
                )
            elif contrib < -5:
                recs.append(
                    f"Consider reducing '{feature}' weight — "
                    f"negative contribution ({contrib:.1f}%)"
                )
            elif contrib < -10:
                recs.append(
                    f"WARNING: '{feature}' is actively hurting performance "
                    f"({contrib:.1f}%). Consider removing."
                )

        if overall_wr < 0.50:
            recs.append(
                "Overall win rate below 50% — review min_confidence threshold "
                "and consider raising it by 5%"
            )
        elif overall_wr > 0.65:
            recs.append(
                "Win rate is strong. Consider slightly increasing stake size "
                "within risk limits."
            )

        return recs if recs else ["No significant adjustments recommended at this time."]

    def run_minor_audit(self, trades_analyzed: int = MINOR_INTERVAL) -> Dict[str, Any]:
        """100-trade standard audit."""
        trades = self._load_trade_history(last_n=trades_analyzed)
        if not trades:
            return {"status": "no_data", "trades_analyzed": 0}

        overall_wr = sum(1 for t in trades if t.get("is_win")) / len(trades)
        contributions: Dict[str, Optional[float]] = {}

        for feature in TRACKED_FEATURES:
            contributions[feature] = self._quartile_contribution(trades, feature)

        helping = sorted(
            [(f, v) for f, v in contributions.items() if v is not None and v > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        hurting = sorted(
            [(f, v) for f, v in contributions.items() if v is not None and v < 0],
            key=lambda x: x[1],
        )

        recs = self._generate_recommendations(contributions, overall_wr)

        report = {
            "type": "minor",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trades_analyzed": len(trades),
            "overall_win_rate": round(overall_wr * 100, 1),
            "helping": [{"feature": f, "contribution": f"+{v:.1f}%"} for f, v in helping],
            "hurting": [{"feature": f, "contribution": f"{v:.1f}%"} for f, v in hurting],
            "neutral": [
                f for f, v in contributions.items()
                if v is None or abs(v) < 1.0
            ],
            "recommendations": recs,
            "raw_contributions": {k: v for k, v in contributions.items() if v is not None},
        }

        self._save_report(report)
        return report

    def run_major_audit(self) -> Dict[str, Any]:
        """1000-trade deep audit with full validation."""
        trades = self._load_trade_history(last_n=MAJOR_INTERVAL)
        if not trades:
            return {"status": "no_data", "trades_analyzed": 0}

        # Run minor audit as foundation
        minor = self.run_minor_audit(trades_analyzed=MAJOR_INTERVAL)
        minor["type"] = "major"

        # MOR validation
        mor_buckets: Dict[str, Dict] = {
            "90+": {"wins": 0, "n": 0},
            "80-89": {"wins": 0, "n": 0},
            "70-79": {"wins": 0, "n": 0},
            "<70": {"wins": 0, "n": 0},
        }
        for t in trades:
            mor = t.get("mor_score")
            if mor is None:
                continue
            if mor >= 90:
                b = "90+"
            elif mor >= 80:
                b = "80-89"
            elif mor >= 70:
                b = "70-79"
            else:
                b = "<70"
            mor_buckets[b]["n"] += 1
            if t.get("is_win"):
                mor_buckets[b]["wins"] += 1

        mor_validation = {}
        prev_wr = None
        mor_valid = True
        for bucket in ["90+", "80-89", "70-79", "<70"]:
            n = mor_buckets[bucket]["n"]
            wr = round(mor_buckets[bucket]["wins"] / n * 100, 1) if n > 0 else None
            mor_validation[bucket] = {"n": n, "win_rate": wr}
            if wr is not None and prev_wr is not None and wr > prev_wr:
                mor_valid = False  # MOR order violated
            if wr is not None:
                prev_wr = wr

        minor["mor_validation"] = mor_validation
        minor["mor_ordering_valid"] = mor_valid
        if not mor_valid:
            minor["recommendations"].append(
                "MOR ordering violated — higher MOR is NOT producing higher win rates. "
                "MOR scoring formula needs recalibration."
            )

        # Contract-type breakdown
        ct_stats: Dict[str, Dict] = {}
        for t in trades:
            ct = str(t.get("contract_type") or "UNKNOWN")
            sym = str(t.get("symbol") or "?")
            key = f"{sym}|{ct}"
            if key not in ct_stats:
                ct_stats[key] = {"wins": 0, "n": 0, "pnl": 0.0}
            ct_stats[key]["n"] += 1
            ct_stats[key]["pnl"] += float(t.get("profit") or 0)
            if t.get("is_win"):
                ct_stats[key]["wins"] += 1

        best_setups = sorted(
            [
                {
                    "key": k,
                    "wr": round(v["wins"] / v["n"] * 100, 1),
                    "pnl": round(v["pnl"], 2),
                    "n": v["n"],
                }
                for k, v in ct_stats.items()
                if v["n"] >= 10
            ],
            key=lambda x: x["pnl"],
            reverse=True,
        )[:5]

        worst_setups = sorted(
            [
                {
                    "key": k,
                    "wr": round(v["wins"] / v["n"] * 100, 1),
                    "pnl": round(v["pnl"], 2),
                    "n": v["n"],
                }
                for k, v in ct_stats.items()
                if v["n"] >= 10
            ],
            key=lambda x: x["pnl"],
        )[:5]

        minor["best_setups"] = best_setups
        minor["worst_setups"] = worst_setups

        self._save_report(minor)
        return minor

    def check_and_run(self, cumulative_trade_count: int) -> Optional[Dict[str, Any]]:
        """
        Called after each trade settlement.
        Triggers minor or major audit based on cumulative count.
        Returns the report if an audit ran, else None.
        """
        if cumulative_trade_count > 0 and cumulative_trade_count % MAJOR_INTERVAL == 0:
            logger.info("AIAuditor: MAJOR audit triggered at %d trades", cumulative_trade_count)
            return self.run_major_audit()
        elif cumulative_trade_count > 0 and cumulative_trade_count % MINOR_INTERVAL == 0:
            logger.info("AIAuditor: minor audit triggered at %d trades", cumulative_trade_count)
            return self.run_minor_audit()
        return None

    def _save_report(self, report: Dict[str, Any]) -> None:
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            # Append to history log
            with self.history_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(report) + "\n")
            self._last_report = report
            logger.info(
                "AIAuditor report saved: %s audit, wr=%.1f%%, %d trades",
                report.get("type"),
                report.get("overall_win_rate", 0),
                report.get("trades_analyzed", 0),
            )
        except Exception as e:
            logger.warning("AIAuditor: failed to save report: %s", e)

    def latest_report(self) -> Optional[Dict[str, Any]]:
        """Return the most recent audit report."""
        return self._last_report
