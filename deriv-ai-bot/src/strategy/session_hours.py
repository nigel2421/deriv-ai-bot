"""
Session / market-open heuristics for multi-asset scanning.

Synthetics (R_*, 1HZ*, Boom/Crash, Jump, …) trade 24/7 on Deriv.
Forex majors follow the usual Sun–Fri OTC window (UTC approx).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple


def _weekday_utc(now: Optional[datetime] = None) -> Tuple[int, int, int]:
    """Return (weekday Mon=0..Sun=6, hour, minute) in UTC."""
    n = now or datetime.now(timezone.utc)
    if n.tzinfo is None:
        n = n.replace(tzinfo=timezone.utc)
    else:
        n = n.astimezone(timezone.utc)
    return n.weekday(), n.hour, n.minute


def is_fx_symbol(symbol: str) -> bool:
    s = str(symbol or "").strip()
    return s.lower().startswith("frx") or s.upper().startswith("FRX")


def is_boom_crash(symbol: str) -> bool:
    s = str(symbol or "").upper()
    return s.startswith("BOOM") or s.startswith("CRASH")


def is_boom_symbol(symbol: str) -> bool:
    return str(symbol or "").upper().startswith("BOOM")


def is_crash_symbol(symbol: str) -> bool:
    return str(symbol or "").upper().startswith("CRASH")


def sanitize_contracts_for_symbol(symbol: str, contract_types: list[str]) -> list[str]:
    """
    Enforce Deriv market restrictions:
    - BOOM indices only offer CALL for Rise/Fall
    - CRASH indices only offer PUT for Rise/Fall
    """
    if is_boom_symbol(symbol):
        return [t for t in contract_types if t.upper() == "CALL"]
    if is_crash_symbol(symbol):
        return [t for t in contract_types if t.upper() == "PUT"]
    return contract_types



def is_spike_synthetic(symbol: str) -> bool:
    """Markets that typically reject multi-minute rise/fall durations."""
    s = str(symbol or "").upper()
    return (
        s.startswith("BOOM")
        or s.startswith("CRASH")
        or s.startswith("JD")
        or s.startswith("STEP")
        or s in {"RDBULL", "RDBEAR"}
    )


def forex_session_open(now: Optional[datetime] = None) -> bool:
    """
    Approximate major FX session (OTC):
    Open: Sunday 22:00 UTC → Friday 22:00 UTC
    Closed: Fri 22:00 → Sun 22:00 (weekend).
    """
    wd, hour, minute = _weekday_utc(now)
    mins = hour * 60 + minute
    if wd == 5:  # Saturday
        return False
    if wd == 6:  # Sunday open from 22:00 UTC
        return mins >= 22 * 60
    if wd == 4:  # Friday closed after 22:00 UTC
        return mins < 22 * 60
    return True


def is_likely_session_open(
    symbol: str, now: Optional[datetime] = None
) -> Tuple[bool, str]:
    """Soft session check. Returns (open, reason)."""
    if is_fx_symbol(symbol):
        if forex_session_open(now):
            return True, "forex_session_open"
        return False, "forex_weekend_closed"
    # Synthetics and crypto-style underlyings: always on
    return True, "always_on_synthetic"


def preferred_minute_duration(symbol: str, default_minutes: int = 2) -> int:
    """
    Horizon by asset class.

    FX majors: 30–40 minute rise/fall (user target for trend holds).
    Synthetics: short minute engine (default 2–5m) or ticks.
    """
    if is_fx_symbol(symbol):
        # Prefer 30m; env can override via FX_MINUTE_DURATION
        import os

        try:
            return max(15, int(os.getenv("FX_MINUTE_DURATION", "30")))
        except ValueError:
            return 30
    if is_spike_synthetic(symbol):
        # Prefer not using minutes at all (caller should skip)
        return max(1, int(default_minutes))
    return max(1, int(default_minutes))
