"""
Anti-loss-spiral controls: setup/symbol cooldowns and soft-landing.

Prevents the bot from repeatedly entering the same losing (symbol, type)
combination that digs the account into a hole.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AntiSpiral:
    def __init__(
        self,
        *,
        setup_loss_limit: int = 2,
        setup_cooldown_sec: int = 20 * 60,
        symbol_loss_limit: int = 3,
        symbol_cooldown_sec: int = 15 * 60,
        soft_landing_losses: int = 3,
        soft_landing_sec: int = 20 * 60,
        soft_landing_min_conf: float = 0.88,
        max_same_setup_in_row: int = 2,
    ):
        self.setup_loss_limit = setup_loss_limit
        self.setup_cooldown_sec = setup_cooldown_sec
        self.symbol_loss_limit = symbol_loss_limit
        self.symbol_cooldown_sec = symbol_cooldown_sec
        self.soft_landing_losses = soft_landing_losses
        self.soft_landing_sec = soft_landing_sec
        self.soft_landing_min_conf = soft_landing_min_conf
        self.max_same_setup_in_row = max_same_setup_in_row

        # key -> {losses, until}
        self._setup_cd: Dict[str, Dict[str, Any]] = {}
        self._symbol_cd: Dict[str, Dict[str, Any]] = {}
        self._setup_streak: Dict[str, int] = {}
        self._symbol_streak: Dict[str, int] = {}
        self._global_loss_streak = 0
        self._soft_until = 0.0
        self._last_setup: Optional[str] = None
        self._same_setup_run = 0

    @staticmethod
    def _key(symbol: str, contract_type: str) -> str:
        return f"{symbol}|{str(contract_type).upper()}"

    def record(self, symbol: str, contract_type: str, is_win: bool) -> None:
        now = time.time()
        k = self._key(symbol, contract_type)
        if is_win:
            self._setup_streak[k] = 0
            self._symbol_streak[symbol] = 0
            self._global_loss_streak = 0
            self._same_setup_run = 0
            self._last_setup = None
            return

        self._setup_streak[k] = self._setup_streak.get(k, 0) + 1
        self._symbol_streak[symbol] = self._symbol_streak.get(symbol, 0) + 1
        self._global_loss_streak += 1

        if self._setup_streak[k] >= self.setup_loss_limit:
            until = now + self.setup_cooldown_sec
            self._setup_cd[k] = {"until": until, "losses": self._setup_streak[k]}
            logger.warning(
                "AntiSpiral: ban setup %s for %.0fm after %s losses",
                k,
                self.setup_cooldown_sec / 60,
                self._setup_streak[k],
            )

        if self._symbol_streak[symbol] >= self.symbol_loss_limit:
            until = now + self.symbol_cooldown_sec
            self._symbol_cd[symbol] = {
                "until": until,
                "losses": self._symbol_streak[symbol],
            }
            logger.warning(
                "AntiSpiral: ban symbol %s for %.0fm after %s losses",
                symbol,
                self.symbol_cooldown_sec / 60,
                self._symbol_streak[symbol],
            )

        if self._global_loss_streak >= self.soft_landing_losses:
            self._soft_until = now + self.soft_landing_sec
            logger.warning(
                "AntiSpiral: soft-landing ON for %.0fm (need conf≥%.2f)",
                self.soft_landing_sec / 60,
                self.soft_landing_min_conf,
            )

    def allow(
        self, symbol: str, contract_type: str, confidence: float
    ) -> Tuple[bool, str]:
        now = time.time()
        k = self._key(symbol, contract_type)

        # Expire cooldowns
        sc = self._setup_cd.get(k)
        if sc and now < float(sc["until"]):
            left = (float(sc["until"]) - now) / 60
            return False, f"setup_cooldown_{left:.0f}m"
        if sc and now >= float(sc["until"]):
            self._setup_cd.pop(k, None)

        sy = self._symbol_cd.get(symbol)
        if sy and now < float(sy["until"]):
            left = (float(sy["until"]) - now) / 60
            return False, f"symbol_cooldown_{left:.0f}m"
        if sy and now >= float(sy["until"]):
            self._symbol_cd.pop(symbol, None)

        if now < self._soft_until and confidence < self.soft_landing_min_conf:
            left = (self._soft_until - now) / 60
            return False, f"soft_landing_need_{self.soft_landing_min_conf:.0%}_{left:.0f}m"

        # Avoid hammering same setup in a row
        if self._last_setup == k and self._same_setup_run >= self.max_same_setup_in_row:
            return False, "same_setup_row_limit"

        return True, "ok"

    def note_selected(self, symbol: str, contract_type: str) -> None:
        k = self._key(symbol, contract_type)
        if self._last_setup == k:
            self._same_setup_run += 1
        else:
            self._last_setup = k
            self._same_setup_run = 1

    def clear_cooldowns(self) -> None:
        """Operator resume: clear timed bans but keep learner stats elsewhere."""
        self._setup_cd.clear()
        self._symbol_cd.clear()
        self._soft_until = 0.0
        self._global_loss_streak = 0
        self._setup_streak.clear()
        self._symbol_streak.clear()
        self._same_setup_run = 0
        self._last_setup = None
        logger.info("AntiSpiral: all cooldowns cleared")

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "global_loss_streak": self._global_loss_streak,
            "soft_landing": now < self._soft_until,
            "soft_until_min": max(0, (self._soft_until - now) / 60),
            "setup_bans": {
                k: round((float(v["until"]) - now) / 60, 1)
                for k, v in self._setup_cd.items()
                if float(v["until"]) > now
            },
            "symbol_bans": {
                k: round((float(v["until"]) - now) / 60, 1)
                for k, v in self._symbol_cd.items()
                if float(v["until"]) > now
            },
        }
