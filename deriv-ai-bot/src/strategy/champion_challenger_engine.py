"""
Champion / Challenger Performance Engine — Rec #14

Tracks empirical performance for every (Market, Strategy / Agent) combination
over trailing window of observations.

Statuses:
  CHAMPION   : Win Rate >= 58%, Profit Factor >= 1.25, 20+ trades -> Full stake, priority ranking.
  ACTIVE     : Win Rate >= 52%, Profit Factor >= 1.05 -> Standard trade execution.
  CHALLENGER : Experimental / paper validation -> Paper or reduced stake.
  DISABLED   : Win Rate < 48% over 20+ trades -> Automatically paused.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StrategyProfile:
    """Performance profile for a specific (symbol, strategy) pair."""

    def __init__(
        self,
        symbol: str,
        strategy_type: str,
        status: str = "CHALLENGER",
        min_confidence_override: Optional[float] = None,
    ):
        self.symbol = symbol
        self.strategy_type = strategy_type
        self.status = status
        self.min_confidence_override = min_confidence_override
        self.trades_history: List[Dict[str, Any]] = []
        self.max_history = 200

    @property
    def total_trades(self) -> int:
        return len(self.trades_history)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades_history if t.get("is_win"))

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades_history if not t.get("is_win"))

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.50
        return self.wins / float(self.total_trades)

    @property
    def net_profit(self) -> float:
        return sum(float(t.get("profit") or 0.0) for t in self.trades_history)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(float(t.get("profit") or 0.0) for t in self.trades_history if float(t.get("profit") or 0.0) > 0)
        gross_loss = abs(sum(float(t.get("profit") or 0.0) for t in self.trades_history if float(t.get("profit") or 0.0) < 0))
        if gross_loss == 0.0:
            return 2.5 if gross_profit > 0 else 1.0
        return round(gross_profit / gross_loss, 2)

    def record_result(self, is_win: bool, profit: float = 0.0, confidence: float = 0.0) -> None:
        self.trades_history.append({
            "is_win": is_win,
            "profit": profit,
            "confidence": confidence,
        })
        if len(self.trades_history) > self.max_history:
            self.trades_history.pop(0)

        self._update_status()

    def _update_status(self) -> None:
        n = self.total_trades
        wr = self.win_rate
        pf = self.profit_factor

        if n >= 20 and wr >= 0.58 and pf >= 1.20:
            self.status = "CHAMPION"
        elif n >= 10 and wr >= 0.52 and pf >= 1.02:
            self.status = "ACTIVE"
        elif n >= 20 and wr < 0.48:
            self.status = "DISABLED"
        else:
            self.status = "CHALLENGER"

    def get_min_confidence(self, global_default: float = 0.62) -> float:
        if self.min_confidence_override is not None:
            return self.min_confidence_override
        if self.status == "CHAMPION":
            return 0.60  # Champion setup requires lower threshold to capture edge
        if self.status == "ACTIVE":
            return 0.64
        if self.status == "CHALLENGER":
            return 0.68  # Higher threshold for challengers
        return 0.75  # Disabled / unproven require strict conf

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy_type": self.strategy_type,
            "status": self.status,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "net_profit": self.net_profit,
            "profit_factor": self.profit_factor,
            "min_confidence_override": self.min_confidence_override,
            "trades_history": self.trades_history[-50:],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyProfile":
        p = cls(
            symbol=data["symbol"],
            strategy_type=data["strategy_type"],
            status=data.get("status", "CHALLENGER"),
            min_confidence_override=data.get("min_confidence_override"),
        )
        p.trades_history = data.get("trades_history") or []
        p._update_status()
        return p


class ChampionChallengerEngine:
    """Manages performance tracking & status evaluation across all market/strategy pairs."""

    def __init__(self, state_path: Path = Path("data/champion_challenger_state.json")):
        self.state_path = state_path
        self.profiles: Dict[str, StrategyProfile] = {}  # key: "symbol:strategy_type"
        self._load_state()

    def _key(self, symbol: str, strategy_type: str) -> str:
        return f"{symbol}:{strategy_type}"

    def get_profile(self, symbol: str, strategy_type: str) -> StrategyProfile:
        k = self._key(symbol, strategy_type)
        if k not in self.profiles:
            self.profiles[k] = StrategyProfile(symbol, strategy_type)
        return self.profiles[k]

    def record_result(self, symbol: str, strategy_type: str, is_win: bool, profit: float = 0.0, confidence: float = 0.0) -> None:
        p = self.get_profile(symbol, strategy_type)
        p.record_result(is_win, profit, confidence)
        self.save_state()

    def get_min_confidence(self, symbol: str, strategy_type: str, default_conf: float = 0.62) -> float:
        p = self.get_profile(symbol, strategy_type)
        return p.get_min_confidence(default_conf)

    def is_allowed_to_trade(self, symbol: str, strategy_type: str) -> bool:
        p = self.get_profile(symbol, strategy_type)
        return p.status != "DISABLED"

    def get_champion_bonus(self, symbol: str, strategy_type: str) -> float:
        p = self.get_profile(symbol, strategy_type)
        if p.status == "CHAMPION":
            return 15.0  # +15 points opportunity score bonus
        if p.status == "ACTIVE":
            return 5.0
        if p.status == "DISABLED":
            return -50.0
        return 0.0

    def _load_state(self) -> None:
        if self.state_path.is_file():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for k, data in raw.items():
                    self.profiles[k] = StrategyProfile.from_dict(data)
                logger.info("Loaded %d Champion/Challenger profiles from state file.", len(self.profiles))
            except Exception as e:
                logger.error("Failed to load Champion/Challenger state: %s", e)

    def save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: p.to_dict() for k, p in self.profiles.items()}
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save Champion/Challenger state: %s", e)

    def get_rankings(self) -> List[Dict[str, Any]]:
        """Return all profiles sorted by win_rate and profit_factor."""
        ranked = [p.to_dict() for p in self.profiles.values()]
        return sorted(ranked, key=lambda x: (x["status"] == "CHAMPION", x["win_rate"], x["profit_factor"]), reverse=True)
