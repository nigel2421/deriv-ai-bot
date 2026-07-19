"""
Simple no-code strategy rules: save / load / evaluate / export.

Example:
  IF digit 8 absent for 30 ticks THEN DIGITOVER barrier 7
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.strategy.digit_contracts import last_digits_from_ticks

DEFAULT_DIR = Path("data/strategies")


def _absent_ticks(digits: Sequence[int], digit: int) -> int:
    for i, d in enumerate(reversed(digits)):
        if int(d) == int(digit):
            return i
    return len(digits)


class StrategyBuilder:
    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory) if directory else DEFAULT_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def list_strategies(self) -> List[Dict[str, Any]]:
        out = []
        for p in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["_file"] = p.name
                out.append(data)
            except Exception:
                continue
        return out

    def save(self, strategy: Dict[str, Any]) -> Path:
        sid = strategy.get("id") or str(uuid.uuid4())[:8]
        strategy = {**strategy, "id": sid, "updated_at": time.time()}
        path = self.directory / f"{sid}.json"
        path.write_text(json.dumps(strategy, indent=2), encoding="utf-8")
        return path

    def load(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        path = self.directory / f"{strategy_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def export_json(self, strategy_id: str) -> Optional[str]:
        data = self.load(strategy_id)
        return json.dumps(data, indent=2) if data else None

    def evaluate(
        self, strategy: Dict[str, Any], ticks: Sequence[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate rule set against live ticks.
        Supported rules (all must pass if present):
          - digit_absent: {digit, ticks}
          - min_volatility_score: float 0-100 (optional, needs volatility_score on strategy eval context)
          - last_won: bool (caller injects via strategy['_last_won'])
        Action:
          - contract_type, barrier?, stake?
        """
        digits = last_digits_from_ticks(ticks, n=200)
        rules = strategy.get("rules") or strategy.get("if") or {}
        if isinstance(rules, list):
            # list of dict conditions
            conds = rules
        else:
            conds = [rules] if rules else []

        notes = []
        for cond in conds:
            if not isinstance(cond, dict):
                continue
            if "digit_absent" in cond:
                spec = cond["digit_absent"]
                d = int(spec.get("digit", 8))
                need = int(spec.get("ticks", 20))
                absent = _absent_ticks(digits, d)
                notes.append(f"digit_{d}_absent={absent}")
                if absent < need:
                    return None
            if "consecutive_repeat" in cond:
                spec = cond["consecutive_repeat"]
                d = int(spec.get("digit", -1))
                need = int(spec.get("length", 3))
                if not digits:
                    return None
                last = int(digits[-1])
                if d >= 0 and last != d:
                    return None
                streak = 1
                for x in reversed(digits[:-1]):
                    if int(x) == last:
                        streak += 1
                    else:
                        break
                notes.append(f"repeat_{last}={streak}")
                if streak < need:
                    return None
            if cond.get("last_won") is True and not strategy.get("_last_won"):
                return None

        action = strategy.get("then") or strategy.get("action") or {}
        ct = action.get("contract_type") or action.get("type")
        if not ct:
            return None
        return {
            "strategy_id": strategy.get("id"),
            "strategy_name": strategy.get("name"),
            "contract_type": str(ct).upper(),
            "barrier": action.get("barrier"),
            "stake": action.get("stake"),
            "notes": notes,
            "source": "strategy_builder",
        }

    def evaluate_all(
        self, ticks: Sequence[Dict[str, Any]], *, last_won: bool = False
    ) -> List[Dict[str, Any]]:
        hits = []
        for s in self.list_strategies():
            s = {**s, "_last_won": last_won}
            r = self.evaluate(s, ticks)
            if r:
                hits.append(r)
        return hits

    def create_example_cold_digit(self) -> Path:
        return self.save(
            {
                "name": "Cold Digit Recovery",
                "description": "If digit 8 absent for 30 ticks, buy Over 7",
                "rules": [{"digit_absent": {"digit": 8, "ticks": 30}}],
                "then": {"contract_type": "DIGITOVER", "barrier": 7},
            }
        )
