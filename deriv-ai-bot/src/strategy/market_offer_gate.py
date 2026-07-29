"""
Broker offer / session gate.

When Deriv rejects a proposal with errors like:
  "Trading is not offered for this duration"
  "Market is closed"
…cool down that symbol/duration so the bot stops retrying dead setups
and keeps scanning markets that are actually open.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REASON_DURATION = "duration_not_offered"
REASON_MARKET_CLOSED = "market_closed"
REASON_UNAVAILABLE = "unavailable"
REASON_OTHER = "other"

_DURATION_HINTS = (
    "not offered for this duration",
    "invalid duration",
    "duration is not",
    "unsupported duration",
    "duration not available",
    "please choose another duration",
)
_MARKET_CLOSED_HINTS = (
    "market is closed",
    "market is not open",
    "is currently closed",
    "markets are closed",
    "trading hours",
    "outside trading hours",
    "market closed",
)
_UNAVAILABLE_HINTS = (
    "trading is not available",
    "not available for this market",
    "symbol is not available",
    "underlying is suspended",
    "contract type is not offered",
    "is not offered for this",
    "not offered for this symbol",
)


def classify_offer_error(message: Optional[str]) -> str:
    text = str(message or "").strip().lower()
    if not text:
        return REASON_OTHER
    for h in _DURATION_HINTS:
        if h in text:
            return REASON_DURATION
    for h in _MARKET_CLOSED_HINTS:
        if h in text:
            return REASON_MARKET_CLOSED
    for h in _UNAVAILABLE_HINTS:
        if h in text:
            return REASON_UNAVAILABLE
    if "not offered" in text and "duration" in text:
        return REASON_DURATION
    if "not offered" in text:
        return REASON_UNAVAILABLE
    return REASON_OTHER


def duration_fallbacks(duration: int, unit: str, *, symbol: str = "") -> List[Tuple[int, str]]:
    """
    Alternate durations to try when the primary is rejected.
    Returns list of (duration, unit) excluding the original.
    """
    from src.strategy.session_hours import is_fx_symbol, is_spike_synthetic

    u = (unit or "t").lower()
    d = int(duration or 5)
    out: List[Tuple[int, str]] = []
    seen = {(d, u)}

    def add(nd: int, nu: str) -> None:
        key = (int(nd), nu)
        if key not in seen and nd > 0:
            seen.add(key)
            out.append(key)

    if is_fx_symbol(symbol):
        # FX: stay on minutes; try 30/15/45/60/10 then ticks as last resort
        for nd in (30, 15, 45, 60, 10, 5, 20, 40):
            add(nd, "m")
        for nd in (5, 10):
            add(nd, "t")
        return out

    if is_spike_synthetic(symbol) or u == "m":
        # Boom/Crash etc: minutes often rejected → ticks first
        for nd in (5, 3, 1, 10, 15):
            add(nd, "t")
        for nd in (1, 2, 3, 5):
            add(nd, "m")
        return out

    if u == "t":
        for nd in (5, 3, 1, 10, 15, 2, 4, 8):
            add(nd, "t")
        for nd in (1, 2, 5):
            add(nd, "m")
    else:
        for nd in (5, 1, 10):
            add(nd, u)
    return out


class MarketOfferGate:
    """Temporary skip list for symbol/duration/contract combos the broker rejects."""

    def __init__(
        self,
        *,
        duration_cooldown_sec: int = 20 * 60,
        market_closed_cooldown_sec: int = 30 * 60,
        unavailable_cooldown_sec: int = 30 * 60,
        max_market_closed_cooldown_sec: int = 2 * 60 * 60,
        max_entries: int = 500,
    ):
        self.duration_cooldown_sec = int(duration_cooldown_sec)
        self.market_closed_cooldown_sec = int(market_closed_cooldown_sec)
        self.unavailable_cooldown_sec = int(unavailable_cooldown_sec)
        self.max_market_closed_cooldown_sec = int(max_market_closed_cooldown_sec)
        self.max_entries = max_entries
        self._blocks: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _key(
        symbol: str,
        *,
        contract_type: Optional[str] = None,
        duration: Optional[int] = None,
        duration_unit: Optional[str] = None,
        whole_symbol: bool = False,
    ) -> str:
        sym = str(symbol or "").strip()
        if whole_symbol:
            return f"{sym}|*"
        ct = str(contract_type or "*").upper()
        if duration is None or duration_unit is None:
            return f"{sym}|{ct}|*"
        return f"{sym}|{ct}|{int(duration)}{str(duration_unit).lower()}"

    def _purge(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        dead = [k for k, v in self._blocks.items() if float(v.get("until") or 0) <= now]
        for k in dead:
            self._blocks.pop(k, None)
        if len(self._blocks) > self.max_entries:
            ordered = sorted(
                self._blocks.items(), key=lambda kv: float(kv[1].get("until") or 0)
            )
            for k, _ in ordered[: len(self._blocks) - self.max_entries]:
                self._blocks.pop(k, None)

    def block(
        self,
        symbol: str,
        *,
        reason: str,
        error: Optional[str] = None,
        contract_type: Optional[str] = None,
        duration: Optional[int] = None,
        duration_unit: Optional[str] = None,
        cooldown_sec: Optional[int] = None,
    ) -> str:
        now = time.time()
        self._purge(now)
        if reason == REASON_MARKET_CLOSED:
            key = self._key(symbol, whole_symbol=True)
        elif reason == REASON_DURATION:
            key = self._key(
                symbol,
                contract_type=contract_type,
                duration=duration,
                duration_unit=duration_unit,
            )
        elif reason == REASON_UNAVAILABLE and contract_type:
            key = self._key(symbol, contract_type=contract_type)
        else:
            key = self._key(symbol, whole_symbol=True)

        prev = self._blocks.get(key) or {}
        hits = int(prev.get("hits") or 0) + 1

        if cooldown_sec is None:
            if reason == REASON_MARKET_CLOSED:
                base = self.market_closed_cooldown_sec
                cooldown_sec = min(
                    self.max_market_closed_cooldown_sec,
                    int(base * (1.0 + 0.5 * min(hits - 1, 4))),
                )
            elif reason == REASON_DURATION:
                cooldown_sec = self.duration_cooldown_sec
            elif reason == REASON_UNAVAILABLE:
                cooldown_sec = self.unavailable_cooldown_sec
            else:
                cooldown_sec = self.duration_cooldown_sec

        until = now + float(cooldown_sec)
        self._blocks[key] = {
            "until": until,
            "reason": reason,
            "error": (error or "")[:240],
            "hits": hits,
            "symbol": symbol,
            "contract_type": contract_type,
            "duration": duration,
            "duration_unit": duration_unit,
            "reprobe_at": until,
        }
        logger.warning(
            "MarketOfferGate: block %s for %.0fm (reason=%s hits=%s err=%s)",
            key,
            cooldown_sec / 60.0,
            reason,
            hits,
            (error or "")[:120],
        )
        return key

    def clear_symbol(self, symbol: str, *, reason: str = "success") -> int:
        sym = str(symbol or "").strip()
        if not sym:
            return 0
        drop = [k for k in self._blocks if k == f"{sym}|*" or k.startswith(f"{sym}|")]
        for k in drop:
            self._blocks.pop(k, None)
        if drop:
            logger.info(
                "MarketOfferGate: cleared %d block(s) for %s (%s)",
                len(drop),
                sym,
                reason,
            )
        return len(drop)

    def note_success(self, symbol: str, **_kwargs: Any) -> None:
        self.clear_symbol(symbol, reason="offer_success")

    def note_error(
        self,
        symbol: str,
        error: Optional[str],
        *,
        contract_type: Optional[str] = None,
        duration: Optional[int] = None,
        duration_unit: Optional[str] = None,
    ) -> Optional[str]:
        reason = classify_offer_error(error)
        if reason == REASON_OTHER:
            return None
        self.block(
            symbol,
            reason=reason,
            error=error,
            contract_type=contract_type,
            duration=duration,
            duration_unit=duration_unit,
        )
        return reason

    def is_symbol_blocked(self, symbol: str) -> bool:
        self._purge()
        now = time.time()
        entry = self._blocks.get(self._key(symbol, whole_symbol=True))
        return bool(entry and float(entry.get("until") or 0) > now)

    def is_blocked(
        self,
        symbol: str,
        *,
        contract_type: Optional[str] = None,
        duration: Optional[int] = None,
        duration_unit: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        self._purge()
        now = time.time()
        sym = str(symbol or "").strip()
        ct = str(contract_type or "").upper() or None
        unit = str(duration_unit or "").lower() or None

        keys = [self._key(sym, whole_symbol=True)]
        if ct:
            keys.append(self._key(sym, contract_type=ct))
        if ct and duration is not None and unit:
            keys.append(
                self._key(sym, contract_type=ct, duration=duration, duration_unit=unit)
            )

        for k in keys:
            v = self._blocks.get(k)
            if v and float(v.get("until") or 0) > now:
                return True, str(v.get("reason") or REASON_OTHER)

        # type-level blocks
        for k, v in self._blocks.items():
            if float(v.get("until") or 0) <= now:
                continue
            if not k.startswith(f"{sym}|"):
                continue
            parts = k.split("|")
            if ct and len(parts) >= 2 and parts[1] == ct:
                if len(parts) == 3 and parts[2] == "*":
                    return True, str(v.get("reason") or REASON_OTHER)
                if (
                    duration is not None
                    and unit
                    and len(parts) >= 3
                    and parts[2] == f"{int(duration)}{unit}"
                ):
                    return True, str(v.get("reason") or REASON_OTHER)
        return False, None

    def snapshot(self) -> Dict[str, Any]:
        self._purge()
        now = time.time()
        active = []
        for k, v in sorted(
            self._blocks.items(), key=lambda kv: -float(kv[1].get("until") or 0)
        ):
            until = float(v.get("until") or 0)
            if until <= now:
                continue
            active.append(
                {
                    "key": k,
                    "reason": v.get("reason"),
                    "remaining_min": round((until - now) / 60.0, 1),
                    "hits": v.get("hits"),
                    "error": v.get("error"),
                    "symbol": v.get("symbol"),
                }
            )
        return {
            "active_blocks": active[:40],
            "count": len(active),
            "policy": "temporary_cooldown_then_reprobe",
        }
