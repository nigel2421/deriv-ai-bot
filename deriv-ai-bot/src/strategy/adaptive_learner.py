"""
Online learning from closed trade outcomes.

Tracks win-rate and streak per (symbol, contract_type) and produces:
  - confidence multipliers (boost winners, cut losers)
  - market preference weights for multi-symbol selection
  - temporary skip flags after cold streaks

State is persisted under data/learning_state.json (ephemeral on Cloud Run
unless volume attached — still useful within a session / warm instance).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/learning_state.json")

# Prefer proven winners (soft prior until real samples dominate)
PREFERRED_SETUPS = {
    "R_50|PUT": 0.06,
    "R_25|CALL": 0.06,
    "R_10|CALL": 0.03,
    "R_10|PUT": 0.03,
    "1HZ50V|CALL": 0.04,
    "1HZ50V|PUT": 0.04,
    "R_25|PUT": 0.03,
    "R_50|CALL": 0.03,
}


def _key(symbol: str, contract_type: str) -> str:
    return f"{symbol}|{str(contract_type).upper()}"


class AdaptiveLearner:
    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        min_samples: int = 2,
        cold_streak_skip: int = 2,  # soft-ban after 2 losses
        decay: float = 0.995,
        always_on: bool = True,
        max_selection_bonus: float = 0.15,  # was 0.08 — prefer winners harder
    ):
        self.path = Path(path) if path else DEFAULT_PATH
        self.min_samples = min_samples
        self.cold_streak_skip = cold_streak_skip
        self.decay = decay
        self.always_on = always_on
        self.max_selection_bonus = max_selection_bonus
        # stats[key] = {wins, losses, streak_loss, streak_win, pnl, last_ts, conf_sum}
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.global_wins = 0
        self.global_losses = 0
        self.total_recorded = 0
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.stats = data.get("stats") or {}
            self.global_wins = int(data.get("global_wins") or 0)
            self.global_losses = int(data.get("global_losses") or 0)
            self.total_recorded = int(
                data.get("total_recorded")
                or (self.global_wins + self.global_losses)
            )
            logger.info(
                "AdaptiveLearner loaded %d keys (W=%s L=%s total=%s) from %s",
                len(self.stats),
                self.global_wins,
                self.global_losses,
                self.total_recorded,
                self.path,
            )
        except Exception as e:
            logger.warning("AdaptiveLearner load failed: %s", e)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "stats": self.stats,
                "global_wins": self.global_wins,
                "global_losses": self.global_losses,
                "total_recorded": self.total_recorded,
                "updated_at": time.time(),
            }
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            # Optional GCS sync (no-op if LEARNING_GCS_URI unset)
            try:
                from src.strategy.learning_persistence import push_to_gcs

                push_to_gcs()
            except Exception:
                pass
        except Exception as e:
            logger.debug("AdaptiveLearner save failed: %s", e)

    def global_samples(self) -> int:
        """Total closed trades observed (session + restored)."""
        return max(
            self.total_recorded,
            self.global_wins + self.global_losses,
        )

    def record(
        self,
        symbol: str,
        contract_type: str,
        is_win: bool,
        profit: float = 0.0,
        *,
        confidence: Optional[float] = None,
        family: Optional[str] = None,
    ) -> None:
        if not self.always_on and not symbol:
            return
        k = _key(symbol, contract_type)
        s = self.stats.setdefault(
            k,
            {
                "wins": 0,
                "losses": 0,
                "streak_loss": 0,
                "streak_win": 0,
                "pnl": 0.0,
                "last_ts": 0.0,
                "conf_sum": 0.0,
                "conf_n": 0,
                "family": family or "",
            },
        )
        if is_win:
            s["wins"] = int(s["wins"]) + 1
            s["streak_win"] = int(s["streak_win"]) + 1
            s["streak_loss"] = 0
            self.global_wins += 1
        else:
            s["losses"] = int(s["losses"]) + 1
            s["streak_loss"] = int(s["streak_loss"]) + 1
            s["streak_win"] = 0
            self.global_losses += 1
        s["pnl"] = float(s["pnl"]) + float(profit)
        s["last_ts"] = time.time()
        if confidence is not None:
            s["conf_sum"] = float(s.get("conf_sum") or 0) + float(confidence)
            s["conf_n"] = int(s.get("conf_n") or 0) + 1
        if family:
            s["family"] = family
        self.total_recorded += 1
        self.save()
        logger.info(
            "Learn[%s] %s win=%s streak_L=%s wr=%.0f%% pnl=%.2f total=%s",
            self.total_recorded,
            k,
            is_win,
            s["streak_loss"],
            self.win_rate(symbol, contract_type) * 100,
            s["pnl"],
            self.global_wins + self.global_losses,
        )

    def samples(self, symbol: str, contract_type: str) -> int:
        s = self.stats.get(_key(symbol, contract_type)) or {}
        return int(s.get("wins", 0)) + int(s.get("losses", 0))

    def win_rate(self, symbol: str, contract_type: str) -> float:
        s = self.stats.get(_key(symbol, contract_type)) or {}
        w, l = int(s.get("wins", 0)), int(s.get("losses", 0))
        if w + l == 0:
            return 0.5  # neutral prior
        return w / (w + l)

    def should_skip(self, symbol: str, contract_type: str) -> Tuple[bool, str]:
        s = self.stats.get(_key(symbol, contract_type)) or {}
        streak = int(s.get("streak_loss", 0))
        # Soft-ban after 2 consecutive losses (high impact)
        if streak >= self.cold_streak_skip:
            return True, f"soft_ban_streak_{streak}"
        n = self.samples(symbol, contract_type)
        wr = self.win_rate(symbol, contract_type)
        # Ban chronically bad setups earlier
        if n >= max(self.min_samples, 5) and wr < 0.40:
            return True, f"low_winrate_{wr:.0%}_n={n}"
        if n >= 3 and wr <= 0.0:
            return True, f"all_losses_n={n}"
        # 2 losses with 0 wins → skip until a cooldown cycle
        if n >= 2 and int(s.get("wins", 0)) == 0:
            return True, f"all_losses_n={n}"
        return False, ""

    def confidence_multiplier(self, symbol: str, contract_type: str) -> float:
        """
        Multiplier applied to raw confidence.
        Harder bias toward winners / against losers than before.
        """
        n = self.samples(symbol, contract_type)
        wr = self.win_rate(symbol, contract_type)
        k = _key(symbol, contract_type)
        if n == 0:
            # Soft prior for known preferred setups
            if k in PREFERRED_SETUPS:
                return 1.0 + min(0.04, PREFERRED_SETUPS[k] * 0.5)
            return 1.0

        # Trust grows with samples
        trust = min(1.0, n / 10.0) if n >= self.min_samples else min(0.40, n / 6.0)
        # wr 0.5 → 1.0, wr 0.7 → ~1.22, wr 0.3 → ~0.78 (stronger than before)
        raw = 0.65 + wr * 0.70
        mult = 1.0 + (raw - 1.0) * trust

        s = self.stats.get(k) or {}
        if int(s.get("streak_win", 0)) >= 2:
            mult *= 1.08
        if int(s.get("streak_win", 0)) >= 4:
            mult *= 1.04
        if int(s.get("streak_loss", 0)) >= 1:
            mult *= 0.92
        if int(s.get("streak_loss", 0)) >= 2:
            mult *= 0.85  # soft-ban path usually already skips

        # PnL prior: positive cumulative pnl boosts slightly
        pnl = float(s.get("pnl") or 0)
        if n >= 4 and pnl > 0:
            mult *= 1.05
        elif n >= 4 and pnl < 0:
            mult *= 0.92

        return float(max(0.55, min(1.35, mult)))

    def adjust_confidence(
        self, symbol: str, contract_type: str, confidence: float
    ) -> float:
        skip, reason = self.should_skip(symbol, contract_type)
        if skip:
            logger.info("Learner skip %s|%s: %s", symbol, contract_type, reason)
            return 0.0
        mult = self.confidence_multiplier(symbol, contract_type)
        adj = float(confidence) * mult
        return float(max(0.0, min(0.99, adj)))

    def selection_bonus(self, symbol: str, contract_type: str) -> float:
        """
        Additive score for TradeSelector (0 .. max_selection_bonus).
        Prefer winners hard: WR, pnl, preferred setups, win streaks.
        """
        k = _key(symbol, contract_type)
        n = self.samples(symbol, contract_type)
        bonus = float(PREFERRED_SETUPS.get(k) or 0.0)

        if n >= self.min_samples:
            wr = self.win_rate(symbol, contract_type)
            # wr 0.6 → +0.05, wr 0.8 → +0.12 (before cap)
            bonus += max(0.0, (wr - 0.48) * 0.40)
            s = self.stats.get(k) or {}
            pnl = float(s.get("pnl") or 0)
            if pnl > 0:
                bonus += min(0.05, pnl * 0.01)
            if int(s.get("streak_win", 0)) >= 2:
                bonus += 0.03
            if int(s.get("streak_loss", 0)) >= 1:
                bonus -= 0.04
            if wr < 0.45 and n >= 4:
                bonus -= 0.06

        return max(0.0, min(self.max_selection_bonus, bonus))

    def effective_min_confidence(
        self,
        base: float = 0.80,
        *,
        family: str = "",
        contract_type: str = "",
    ) -> float:
        """
        Single min_confidence for ALL families (digits + rise/fall + minute).

        Source of truth: strategy.xml / orchestrator.min_confidence (default 0.80).
        Learning phase tightens *analytics gates*, not a different conf floor.
        """
        # family/contract_type kept for API compatibility — not used to diverge
        _ = (family, contract_type, self.global_samples())
        return float(base)

    def cold_start_phase(self) -> str:
        n = self.global_samples()
        if n < 50:
            return "cold"
        if n < 100:
            return "warming"
        return "mature"

    def snapshot(self) -> Dict[str, Any]:
        return {
            "always_on": self.always_on,
            "path": str(self.path),
            "global_wins": self.global_wins,
            "global_losses": self.global_losses,
            "total_recorded": self.total_recorded,
            "global_samples": self.global_samples(),
            "phase": self.cold_start_phase(),
            "keys": len(self.stats),
            "preferred": list(PREFERRED_SETUPS.keys()),
            "top": sorted(
                (
                    {
                        "key": k,
                        "wins": v.get("wins"),
                        "losses": v.get("losses"),
                        "pnl": round(float(v.get("pnl") or 0), 2),
                        "streak_loss": v.get("streak_loss"),
                        "family": v.get("family"),
                    }
                    for k, v in self.stats.items()
                ),
                key=lambda x: -(float(x.get("pnl") or 0)),
            )[:12],
        }
