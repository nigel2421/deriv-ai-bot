"""
Market Opportunity Tracker — Recs #4 and #5

Rec #4: Market Opportunity Velocity
  - Tracks MOR score per symbol over time (rolling 48h)
  - Computes velocity = current_score - yesterday_avg
  - Rapidly improving markets get a velocity bonus in trade_selector

Rec #5: Track Opportunity Success
  - Records {mor, is_win, contract_type} for each settled trade
  - Computes win rate by MOR bucket (90+, 80-90, 70-80, <70)
  - Every 500 trades generates a MOR validation table

Persistence: data/mor_state.json

Score normalization:
  Raw trade_selector score (typically 0.6..1.4) is mapped to 0..100.
  max_raw_score = 1.4 (conf=0.95 + bonus=0.08 + strength*0.04 + 0.01)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/mor_state.json")

MAX_RAW_SCORE = 1.40       # theoretical max of trade_selector raw score
SCORE_HISTORY_HOURS = 48   # keep 48h of score readings
MAX_OUTCOMES = 500         # ring buffer for trade outcome history


def normalize_score(raw: float) -> float:
    """Map raw trade_selector score [0..MAX_RAW_SCORE] to [0..100]."""
    return round(min(100.0, max(0.0, (raw / MAX_RAW_SCORE) * 100.0)), 1)


def _bucket(mor: float) -> str:
    if mor >= 90:
        return "90+"
    if mor >= 80:
        return "80-89"
    if mor >= 70:
        return "70-79"
    return "<70"


class MORTracker:
    """
    Market Opportunity Ranking tracker with velocity and outcome validation.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        # data[symbol] = {score_history: [(ts, score), ...], trade_outcomes: [...]}
        self.data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            logger.info("MORTracker loaded %d symbols from %s", len(self.data), self.path)
        except Exception as e:
            logger.warning("MORTracker load failed: %s", e)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("MORTracker save failed: %s", e)

    def _init_symbol(self, symbol: str) -> Dict[str, Any]:
        return self.data.setdefault(
            symbol,
            {
                "score_history": [],
                "trade_outcomes": [],
                "current_score": 0.0,
            },
        )

    def update_score(self, symbol: str, raw_score: float) -> float:
        """
        Called each scan cycle for a symbol that produced a candidate signal.
        Returns the normalized MOR score (0-100).
        """
        mor = normalize_score(raw_score)
        sym = self._init_symbol(symbol)

        now = time.time()
        cutoff = now - SCORE_HISTORY_HOURS * 3600
        # Prune old entries
        sym["score_history"] = [
            (ts, s) for ts, s in sym.get("score_history", []) if ts > cutoff
        ]
        sym["score_history"].append((now, mor))
        sym["current_score"] = mor
        # No save here — save happens in record_outcome (to reduce I/O)
        return mor

    def get_velocity(self, symbol: str) -> float:
        """
        Velocity = current_score - 24h-ago avg.
        Positive = improving, negative = declining.
        Returns 0.0 if insufficient history.
        """
        sym = self.data.get(symbol, {})
        history = sym.get("score_history", [])
        if len(history) < 2:
            return 0.0

        now = time.time()
        yesterday_cutoff = now - 24 * 3600
        older = [s for ts, s in history if ts < yesterday_cutoff]
        recent = [s for ts, s in history if ts >= yesterday_cutoff]

        if not older or not recent:
            return 0.0

        yesterday_avg = sum(older) / len(older)
        current = recent[-1]  # most recent reading
        return round(current - yesterday_avg, 1)

    def get_velocity_bonus(self, symbol: str) -> float:
        """
        Score bonus for trade_selector based on MOR velocity.
        Velocity > +20 -> +0.03 bonus
        Velocity > +10 -> +0.015 bonus
        Velocity < -20 -> -0.02 penalty
        """
        vel = self.get_velocity(symbol)
        if vel > 20:
            return 0.03
        if vel > 10:
            return 0.015
        if vel < -20:
            return -0.02
        if vel < -10:
            return -0.01
        return 0.0

    def record_outcome(
        self,
        symbol: str,
        mor: float,
        is_win: bool,
        contract_type: str,
    ) -> None:
        """Record the MOR score at trade time and the outcome."""
        sym = self._init_symbol(symbol)
        outcomes = sym.setdefault("trade_outcomes", [])
        outcomes.append({
            "ts": time.time(),
            "mor": round(mor, 1),
            "is_win": is_win,
            "contract_type": str(contract_type).upper(),
            "bucket": _bucket(mor),
        })
        # Ring buffer: keep last MAX_OUTCOMES
        if len(outcomes) > MAX_OUTCOMES:
            sym["trade_outcomes"] = outcomes[-MAX_OUTCOMES:]
        self.save()

    def win_rate_by_bucket(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns win rate per MOR bucket, optionally for a single symbol.
        If symbol=None, aggregates all symbols.
        """
        buckets: Dict[str, Dict[str, int]] = {
            "90+": {"wins": 0, "n": 0},
            "80-89": {"wins": 0, "n": 0},
            "70-79": {"wins": 0, "n": 0},
            "<70": {"wins": 0, "n": 0},
        }

        symbols = [symbol] if symbol else list(self.data.keys())
        for sym in symbols:
            for outcome in self.data.get(sym, {}).get("trade_outcomes", []):
                b = outcome.get("bucket") or _bucket(outcome.get("mor", 0))
                if b not in buckets:
                    continue
                buckets[b]["n"] += 1
                if outcome.get("is_win"):
                    buckets[b]["wins"] += 1

        result = {}
        for b, d in buckets.items():
            n = d["n"]
            result[b] = {
                "n": n,
                "win_rate": round(d["wins"] / n * 100, 1) if n > 0 else None,
            }
        return result

    def ranked_symbols(self) -> List[Tuple[str, float, float, float]]:
        """
        Returns list of (symbol, current_score, yesterday_avg, velocity) sorted by score desc.
        """
        rows = []
        for sym, d in self.data.items():
            history = d.get("score_history", [])
            current = d.get("current_score", 0.0)
            now = time.time()
            yesterday_cutoff = now - 24 * 3600
            older = [s for ts, s in history if ts < yesterday_cutoff]
            yesterday_avg = round(sum(older) / len(older), 1) if older else None
            velocity = self.get_velocity(sym)
            rows.append((sym, current, yesterday_avg, velocity))
        return sorted(rows, key=lambda x: x[1], reverse=True)

    def snapshot(self) -> Dict[str, Any]:
        ranked = self.ranked_symbols()
        symbol_rows = []
        for sym, score, yesterday, velocity in ranked:
            vel = self.get_velocity(sym)
            vel_arrow = "↑↑" if vel > 20 else "↑" if vel > 5 else "↓↓" if vel < -20 else "↓" if vel < -5 else "→"
            outcomes = self.data.get(sym, {}).get("trade_outcomes", [])
            # Only high-MOR trades
            hi = [o for o in outcomes if o.get("mor", 0) >= 90]
            hi_wr = round(sum(1 for o in hi if o.get("is_win")) / len(hi) * 100, 1) if hi else None

            symbol_rows.append({
                "symbol": sym,
                "score": score,
                "yesterday": yesterday,
                "velocity": vel,
                "velocity_arrow": vel_arrow,
                "high_mor_wr": hi_wr,
                "total_outcomes": len(outcomes),
            })
        return {
            "ranked": symbol_rows,
            "bucket_analysis": self.win_rate_by_bucket(),
        }
