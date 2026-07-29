"""
DeepSeek Advisor — AI-powered per-market strategy analysis.

Trigger: every 100 closed trades on a specific market symbol.
Data:    reads ALL available trade history from data/trade_history.jsonl
         (backed by GCS bucket — persistent across restarts).
Output:  structured JSON recommendation stored in data/deepseek_report.json
         + exposed in /status as 'deepseek' key.
         + optional Telegram notification.

API:     DeepSeek Chat API (OpenAI-compatible format).
         Env:  DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL
               DEEPSEEK_ENABLED, DEEPSEEK_ANALYZE_EVERY (per-symbol, default 100)
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Paths (all under GCS-mounted /app/data) ──────────────────────────────────
HISTORY_PATH = Path("data/trade_history.jsonl")
REPORT_PATH = Path("data/deepseek_report.json")
STATE_PATH = Path("data/deepseek_state.json")  # per-symbol close counters

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_ANALYZE_EVERY = 100     # trades per market before triggering
MAX_HISTORY_TRADES = 500        # most-recent trades per symbol sent to LLM
MAX_GLOBAL_TRADES = 2000        # max rows scanned from JSONL for context

_SYSTEM_PROMPT = """\
You are an expert quantitative analyst for a Deriv binary options AI trading bot.
You receive structured JSON trade performance data for a specific market symbol.
Your job is to:
  1. Identify which contract types (DIGITOVER, DIGITUNDER, DIGITEVEN, DIGITODD, CALL, PUT) are profitable vs losing.
  2. Identify which confidence ranges actually produce wins vs losses.
  3. Detect any patterns in losing trades (time of day, specific barriers, consecutive losses).
  4. Recommend specific, actionable changes to improve win rate and expected value.
  5. Suggest whether the bot should raise/lower min_confidence for this symbol, ban any contract types, or change stake sizing.
  6. Rate the overall health of this symbol's trading as: HEALTHY / WATCH / STRUGGLING / BAN.

Respond with a JSON object (no markdown) with these keys:
{
  "symbol": "...",
  "health": "HEALTHY|WATCH|STRUGGLING|BAN",
  "overall_win_rate": 0.0,
  "summary": "2-3 sentence plain-english summary",
  "contract_recommendations": [
    {"contract_type": "...", "action": "KEEP|BOOST|REDUCE|BAN", "reason": "..."}
  ],
  "duration_recommendations": [
    {"contract_type": "...", "action": "CHANGE", "suggested_duration": 10, "unit": "t", "reason": "..."}
  ],
  "confidence_recommendation": {"action": "RAISE|LOWER|KEEP", "suggested_threshold": 0.82, "reason": "..."},
  "learning_hints": ["hint1", "hint2"],
  "ban_setups": ["SYMBOL|CONTRACT_TYPE", ...],
  "boost_setups": ["SYMBOL|CONTRACT_TYPE", ...],
  "stake_recommendation": {"action": "INCREASE|DECREASE|KEEP", "reason": "..."}
}
"""


class DeepSeekAdvisor:
    """
    Per-market DeepSeek analysis engine.

    Call `record_close(symbol)` after each trade settles.
    The advisor auto-triggers when any symbol crosses ANALYZE_EVERY closes.
    Results are stored persistently and returned via `snapshot()`.
    """

    def __init__(
        self,
        history_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
    ) -> None:
        self.history_path = Path(history_path or HISTORY_PATH)
        self.report_path = Path(report_path or REPORT_PATH)
        self.state_path = Path(state_path or STATE_PATH)

        self.api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
        self.enabled: bool = os.getenv("DEEPSEEK_ENABLED", "true").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.model: str = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.base_url: str = os.getenv(
            "DEEPSEEK_BASE_URL", DEFAULT_BASE_URL
        ).rstrip("/")
        self.analyze_every: int = max(
            10, int(os.getenv("DEEPSEEK_ANALYZE_EVERY", str(DEFAULT_ANALYZE_EVERY)))
        )

        # Per-symbol close counters since last analysis (in-memory + persisted)
        self._counters: Dict[str, int] = self._load_state()
        # Latest per-symbol report
        self._last_reports: Dict[str, Dict[str, Any]] = self._load_reports()

        if self.enabled and not self.api_key:
            logger.warning(
                "DeepSeekAdvisor: DEEPSEEK_ENABLED=true but DEEPSEEK_API_KEY is not set. "
                "Analysis will be skipped."
            )

    # ── State persistence ─────────────────────────────────────────────────────

    def _load_state(self) -> Dict[str, int]:
        if self.state_path.is_file():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                return {k: int(v) for k, v in (data.get("counters") or {}).items()}
            except Exception as e:
                logger.debug("DeepSeek load state failed: %s", e)
        return {}

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({"counters": self._counters, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("DeepSeek save state failed: %s", e)

    def _load_reports(self) -> Dict[str, Dict[str, Any]]:
        if self.report_path.is_file():
            try:
                data = json.loads(self.report_path.read_text(encoding="utf-8"))
                # Support both legacy (single report) and new multi-symbol format
                if isinstance(data, dict) and "reports" in data:
                    return dict(data["reports"])
                elif isinstance(data, dict) and "symbol" in data:
                    return {data["symbol"]: data}
            except Exception as e:
                logger.debug("DeepSeek load reports failed: %s", e)
        return {}

    def _save_reports(self) -> None:
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(
                json.dumps(
                    {
                        "reports": self._last_reports,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("DeepSeek save reports failed: %s", e)

    # ── Trade history loading ─────────────────────────────────────────────────

    def _load_symbol_history(self, symbol: str, max_trades: int = MAX_HISTORY_TRADES) -> List[Dict[str, Any]]:
        """Load the most-recent `max_trades` closed trades for `symbol` from GCS-backed JSONL."""
        if not self.history_path.is_file():
            return []
        trades: List[Dict[str, Any]] = []
        try:
            with self.history_path.open("r", encoding="utf-8") as f:
                all_lines = f.readlines()
            # Scan last MAX_GLOBAL_TRADES rows (avoid reading entire file for huge histories)
            for line in all_lines[-MAX_GLOBAL_TRADES:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    if t.get("symbol") == symbol:
                        trades.append(t)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.warning("DeepSeek: failed to load trade history: %s", e)
        return trades[-max_trades:]

    # ── Payload builder ───────────────────────────────────────────────────────

    def _build_analysis_payload(self, symbol: str, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarise trade history into a rich JSON payload for the LLM prompt."""
        if not trades:
            return {}

        total = len(trades)
        wins = sum(1 for t in trades if t.get("is_win"))
        overall_wr = round(wins / total * 100, 1)
        total_pnl = round(sum(float(t.get("profit") or 0) for t in trades), 2)

        # Per-contract-type breakdown
        ct_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0, "confs": []})
        for t in trades:
            ct = str(t.get("contract_type") or "UNKNOWN")
            ct_stats[ct]["n"] += 1
            ct_stats[ct]["pnl"] += float(t.get("profit") or 0)
            if t.get("is_win"):
                ct_stats[ct]["wins"] += 1
            conf = t.get("confidence")
            if conf is not None:
                ct_stats[ct]["confs"].append(float(conf))

        ct_summary = []
        for ct, s in ct_stats.items():
            wr = round(s["wins"] / s["n"] * 100, 1) if s["n"] > 0 else 0
            avg_conf = round(sum(s["confs"]) / len(s["confs"]) * 100, 1) if s["confs"] else None
            ct_summary.append({
                "contract_type": ct,
                "n": s["n"],
                "win_rate": wr,
                "pnl": round(s["pnl"], 2),
                "avg_confidence": avg_conf,
            })
        ct_summary.sort(key=lambda x: x["pnl"], reverse=True)

        # Confidence bucket breakdown (80-82, 82-85, 85-90, 90+)
        conf_buckets: Dict[str, Dict[str, Any]] = {
            "80-82%": {"n": 0, "wins": 0},
            "82-85%": {"n": 0, "wins": 0},
            "85-90%": {"n": 0, "wins": 0},
            "90%+":   {"n": 0, "wins": 0},
        }
        for t in trades:
            conf = t.get("confidence")
            if conf is None:
                continue
            c = float(conf) * 100
            if c < 82:
                b = "80-82%"
            elif c < 85:
                b = "82-85%"
            elif c < 90:
                b = "85-90%"
            else:
                b = "90%+"
            conf_buckets[b]["n"] += 1
            if t.get("is_win"):
                conf_buckets[b]["wins"] += 1

        conf_analysis = {}
        for b, d in conf_buckets.items():
            conf_analysis[b] = {
                "n": d["n"],
                "win_rate": round(d["wins"] / d["n"] * 100, 1) if d["n"] > 0 else None,
            }

        # EV stats
        evs = [float(t.get("ev") or 0) for t in trades if t.get("ev") is not None]
        ev_stats = {
            "avg_ev": round(sum(evs) / len(evs), 4) if evs else None,
            "positive_ev_trades": sum(1 for e in evs if e > 0),
            "negative_ev_trades": sum(1 for e in evs if e <= 0),
        }

        # Loss streak analysis
        max_streak = 0
        cur_streak = 0
        for t in trades:
            if not t.get("is_win"):
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 0

        # Most recent 10 trades summary
        recent_10 = [
            {
                "contract_type": t.get("contract_type"),
                "is_win": t.get("is_win"),
                "confidence": round(float(t.get("confidence") or 0) * 100, 1),
                "profit": t.get("profit"),
                "ev": t.get("ev"),
                "ts": str(t.get("ts") or "")[:19],
            }
            for t in trades[-10:]
        ]

        return {
            "symbol": symbol,
            "total_trades_analyzed": total,
            "overall_win_rate": overall_wr,
            "total_pnl": total_pnl,
            "max_loss_streak": max_streak,
            "ev_stats": ev_stats,
            "contract_type_breakdown": ct_summary,
            "confidence_bucket_analysis": conf_analysis,
            "recent_10_trades": recent_10,
        }

    # ── DeepSeek API call ─────────────────────────────────────────────────────

    def _call_deepseek(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        POST to DeepSeek Chat Completions endpoint (OpenAI-compatible).
        Uses stdlib urllib only — no extra dependencies.
        """
        import urllib.request
        import urllib.error

        user_message = (
            f"Analyze this trading data for symbol {payload.get('symbol')} "
            f"and return your JSON recommendation:\n\n"
            f"{json.dumps(payload, indent=2)}"
        )

        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")

        url = f"{self.base_url}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
        except urllib.error.HTTPError as e:
            body_err = e.read().decode("utf-8", errors="replace")[:300]
            logger.error("DeepSeek API error %s: %s", e.code, body_err)
        except urllib.error.URLError as e:
            logger.error("DeepSeek connection error: %s", e.reason)
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error("DeepSeek response parse error: %s", e)
        except Exception as e:
            logger.error("DeepSeek unexpected error: %s", e)
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def record_close(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Called after every trade settles on `symbol`.
        Triggers a DeepSeek analysis when the per-symbol counter hits `analyze_every`.
        Returns the new report if analysis ran, else None.
        """
        self._counters[symbol] = self._counters.get(symbol, 0) + 1
        self._save_state()

        if self._counters[symbol] < self.analyze_every:
            return None

        # Reset counter for this symbol
        self._counters[symbol] = 0
        self._save_state()

        return self.run_analysis(symbol)

    def run_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Force a DeepSeek analysis for `symbol` right now.
        Loads full history from GCS bucket, builds payload, calls API.
        """
        if not self.enabled or not self.api_key:
            logger.info("DeepSeek: skipped (disabled or no API key) for %s", symbol)
            return None

        logger.info("DeepSeek: starting analysis for %s (reads from GCS-backed JSONL)...", symbol)

        trades = self._load_symbol_history(symbol)
        if len(trades) < 10:
            logger.info("DeepSeek: not enough trades for %s (%d < 10)", symbol, len(trades))
            return None

        payload = self._build_analysis_payload(symbol, trades)
        if not payload:
            return None

        t0 = time.time()
        recommendation = self._call_deepseek(payload)
        elapsed = round(time.time() - t0, 1)

        if recommendation is None:
            logger.warning("DeepSeek: no recommendation returned for %s", symbol)
            return None

        report = {
            "symbol": symbol,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trades_analyzed": len(trades),
            "api_latency_s": elapsed,
            "payload_summary": {
                "overall_win_rate": payload.get("overall_win_rate"),
                "total_pnl": payload.get("total_pnl"),
                "max_loss_streak": payload.get("max_loss_streak"),
            },
            "recommendation": recommendation,
        }

        self._last_reports[symbol] = report
        self._save_reports()

        logger.info(
            "DeepSeek: analysis complete for %s | health=%s | wr=%.1f%% | latency=%.1fs",
            symbol,
            recommendation.get("health", "?"),
            float(payload.get("overall_win_rate") or 0),
            elapsed,
        )
        return report

    def closes_until_next(self, symbol: str) -> int:
        """How many more closes until the next analysis for this symbol."""
        current = self._counters.get(symbol, 0)
        return max(0, self.analyze_every - current)

    def get_duration_override(self, symbol: str, contract_type: str) -> Optional[Dict[str, Any]]:
        """Phase 4: Extracts active duration overrides from the latest DeepSeek report."""
        report = self._last_reports.get(symbol)
        if not report:
            return None
        recs = report.get("recommendation", {}).get("duration_recommendations", [])
        for rec in recs:
            if str(rec.get("contract_type")).upper() == str(contract_type).upper() and str(rec.get("action")).upper() == "CHANGE":
                dur = int(rec.get("suggested_duration", 0))
                if dur > 0:
                    return {
                        "duration": dur,
                        "duration_unit": rec.get("unit", "t")
                    }
        return None

    def snapshot(self) -> Dict[str, Any]:
        """Return the full state for /status JSON and dashboard."""
        symbol_status = {}
        for sym, count in self._counters.items():
            report = self._last_reports.get(sym)
            symbol_status[sym] = {
                "closes_since_analysis": count,
                "closes_until_next": max(0, self.analyze_every - count),
                "last_analysis": report.get("generated_at") if report else None,
                "health": (report or {}).get("recommendation", {}).get("health"),
                "win_rate": (report or {}).get("payload_summary", {}).get("overall_win_rate"),
                "trades_analyzed": (report or {}).get("trades_analyzed"),
            }

        # Most recent report across all symbols
        latest = None
        if self._last_reports:
            latest = max(
                self._last_reports.values(),
                key=lambda r: r.get("generated_at", ""),
            )

        return {
            "enabled": self.enabled,
            "analyze_every": self.analyze_every,
            "model": self.model,
            "symbol_status": symbol_status,
            "latest_report": latest,
            "total_symbols_analyzed": len(self._last_reports),
            # Legacy flat fields for audit_live_status.py compatibility
            "recommendation": (latest or {}).get("recommendation"),
            "closes_since_analysis": sum(self._counters.values()),
        }
