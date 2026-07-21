"""
Transition Matrix Engine — Rec #3

Tracks UP->UP, UP->DOWN, DOWN->UP, DOWN->DOWN transitions for CALL/PUT trades
per symbol. Provides persistence_confidence() which influences rise/fall scoring
more heavily than entropy.

Persistence: data/transition_matrix.json

Schema per symbol:
{
  "R_75": {
    "UP_UP": 63, "UP_DOWN": 37,
    "DOWN_UP": 41, "DOWN_DOWN": 59,
    "last_direction": "UP",
    "last_ts": 1750000000.0,
    "total": 200
  }
}

Confidence levels:
  < 20 samples  -> return 0.5 (neutral, no data)
  >= 20 samples -> conditional P(same direction repeats)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/transition_matrix.json")
MIN_SAMPLES_FOR_SIGNAL = 20


def _contract_to_direction(contract_type: str) -> Optional[str]:
    ct = str(contract_type).upper()
    if ct in ("CALL", "RISE", "HIGHER"):
        return "UP"
    if ct in ("PUT", "FALL", "LOWER"):
        return "DOWN"
    return None


class TransitionMatrix:
    """
    Markov-style direction transition tracker for rise/fall trades.
    Only operates on CALL/PUT family trades; digit trades are ignored.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        # data[symbol] = {UP_UP, UP_DOWN, DOWN_UP, DOWN_DOWN, last_direction, last_ts}
        self.data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            logger.info(
                "TransitionMatrix loaded %d symbols from %s", len(self.data), self.path
            )
        except Exception as e:
            logger.warning("TransitionMatrix load failed: %s", e)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("TransitionMatrix save failed: %s", e)

    def _init_symbol(self, symbol: str) -> Dict[str, Any]:
        return self.data.setdefault(
            symbol,
            {
                "UP_UP": 0,
                "UP_DOWN": 0,
                "DOWN_UP": 0,
                "DOWN_DOWN": 0,
                "last_direction": None,
                "last_ts": 0.0,
                "total": 0,
            },
        )

    def record_outcome(
        self, symbol: str, contract_type: str, is_win: bool
    ) -> None:
        """
        Call after a CALL/PUT trade settles.

        The outcome direction is: CALL->UP, PUT->DOWN, regardless of win/loss.
        Transition recorded = (last_direction -> current_direction).
        """
        direction = _contract_to_direction(contract_type)
        if direction is None:
            return  # Not a rise/fall contract — skip

        sym = self._init_symbol(symbol)
        prev = sym.get("last_direction")

        if prev is not None:
            key = f"{prev}_{direction}"
            sym[key] = int(sym.get(key, 0)) + 1
            sym["total"] = int(sym.get("total", 0)) + 1
            logger.debug(
                "Transition %s: %s->%s (totals: UU=%s UD=%s DU=%s DD=%s)",
                symbol,
                prev,
                direction,
                sym["UP_UP"],
                sym["UP_DOWN"],
                sym["DOWN_UP"],
                sym["DOWN_DOWN"],
            )

        sym["last_direction"] = direction
        sym["last_ts"] = time.time()
        self.save()

    def total_transitions(self, symbol: str) -> int:
        sym = self.data.get(symbol, {})
        return int(sym.get("total", 0))

    def persistence_probability(self, symbol: str) -> float:
        """
        Overall P(direction repeats) = (UP_UP + DOWN_DOWN) / total transitions.
        Returns 0.5 if insufficient data.
        """
        sym = self.data.get(symbol, {})
        total = int(sym.get("total", 0))
        if total < MIN_SAMPLES_FOR_SIGNAL:
            return 0.5
        same = int(sym.get("UP_UP", 0)) + int(sym.get("DOWN_DOWN", 0))
        return round(same / total, 4)

    def persistence_confidence(
        self, symbol: str, current_direction: str
    ) -> float:
        """
        Conditional P(next = same | current = current_direction).

        If current = UP:   returns UP_UP / (UP_UP + UP_DOWN)
        If current = DOWN: returns DOWN_DOWN / (DOWN_DOWN + DOWN_UP)

        Returns 0.5 (neutral) if < MIN_SAMPLES_FOR_SIGNAL total.
        """
        sym = self.data.get(symbol, {})
        total = int(sym.get("total", 0))
        if total < MIN_SAMPLES_FOR_SIGNAL:
            return 0.5

        direction = _contract_to_direction(current_direction) or current_direction.upper()

        if direction == "UP":
            same = int(sym.get("UP_UP", 0))
            opposite = int(sym.get("UP_DOWN", 0))
        else:
            same = int(sym.get("DOWN_DOWN", 0))
            opposite = int(sym.get("DOWN_UP", 0))

        denom = same + opposite
        if denom == 0:
            return 0.5
        return round(same / denom, 4)

    def persistence_score_adjustment(
        self, symbol: str, current_direction: str
    ) -> float:
        """
        Score delta for trade_selector. Range: [-0.10, +0.10].

        > 0.5 persistence confidence -> positive adjustment (market is persistent)
        < 0.5 -> negative adjustment (market tends to reverse)
        Neutral at exactly 0.5 -> 0.0 adjustment.
        """
        pc = self.persistence_confidence(symbol, current_direction)
        # Scale: 0.5 -> 0.0, 0.6 -> +0.04, 0.7 -> +0.08, 0.4 -> -0.04
        return round((pc - 0.5) * 0.4, 4)

    def snapshot(self) -> Dict[str, Any]:
        result = {}
        for sym, d in self.data.items():
            total = int(d.get("total", 0))
            uu = int(d.get("UP_UP", 0))
            ud = int(d.get("UP_DOWN", 0))
            du = int(d.get("DOWN_UP", 0))
            dd = int(d.get("DOWN_DOWN", 0))
            persist = self.persistence_probability(sym)
            result[sym] = {
                "UP_UP_pct": round(uu / total * 100, 1) if total else 0,
                "UP_DOWN_pct": round(ud / total * 100, 1) if total else 0,
                "DOWN_UP_pct": round(du / total * 100, 1) if total else 0,
                "DOWN_DOWN_pct": round(dd / total * 100, 1) if total else 0,
                "persistence_pct": round(persist * 100, 1),
                "total": total,
                "last_direction": d.get("last_direction"),
                "sufficient_data": total >= MIN_SAMPLES_FOR_SIGNAL,
            }
        return result
