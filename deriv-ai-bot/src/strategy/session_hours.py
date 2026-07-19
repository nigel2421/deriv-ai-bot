"""
Session / market-open heuristics for multi-asset scanning.

Synthetics (R_*, 1HZ*, Boom/Crash, Jump, …) trade 24/7 on Deriv.
Forex majors follow the usual Sun–Fri OTC window (UTC approx).

This is a *soft* gate: when we think a market is closed we skip proposals
(no wasted API calls). When we think it is open, the offer_gate still
learns from real broker rejections and re-probes after cooldown.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

from src.strategy.market_categories import (
    FOREX,
    DERIVED_FX,
    CRYPTO,
    STOCKS,
    INDICES,
    COMMODITIES,
    classify_market,
)

# Always-on categories on Deriv synthetics stack
_ALWAYS_OPEN = frozenset(
    {
        "synthetic_vol",
        "boom",
        "crash",
        "step",
        "dsi",
        "vol_switch",
        "jump",
        "dex",
        "trek",
        "skew_step",
        "daily_reset",
        "unknown",
    }
)


def _weekday_utc(now: Optional[datetime] = None) -> Tuple[int, int, int]:
    """Return (weekday Mon=0..Sun=6, hour, minute) in UTC."""
    n = now or datetime.now(timezone.utc)
    if n.tzinfo is None:
        n = n.replace(tzinfo=timezone.utc)
    else:
        n = n.astimezone(timezone.utc)
    return n.weekday(), n.hour, n.minute


def forex_session_open(now: Optional[datetime] = None) -> bool:
    """
    Approximate major FX session (OTC):
    Open: Sunday 22:00 UTC → Friday 22:00 UTC
    Closed: Fri 22:00 → Sun 22:00 (weekend).
    """
    wd, hour, minute = _weekday_utc(now)
    mins = hour * 60 + minute
    # Saturday = fully closed
    if wd == 5:
        return False
    # Sunday: open from 22:00 UTC
    if wd == 6:
        return mins >= 22 * 60
    # Friday: closed after 22:00 UTC
    if wd == 4:
        return mins < 22 * 60
    # Mon–Thu: open all day
    return True


def is_likely_session_open(
    symbol: str, now: Optional[datetime] = None
) -> Tuple[bool, str]:
    """
    Soft session check for a symbol.

    Returns (open, reason). reason is short for logs / dashboard.
    """
    cat = classify_market(symbol)
    if cat in _ALWAYS_OPEN:
        return True, "always_on_synthetic"

    if cat in {FOREX, DERIVED_FX}:
        if forex_session_open(now):
            return True, "forex_session_open"
        return False, "forex_weekend_closed"

    # Crypto roughly 24/7 on Deriv OTC — allow
    if cat == CRYPTO:
        return True, "crypto_24x7"

    # Stocks / indices / commodities: weekdays only (rough; broker is source of truth)
    if cat in {STOCKS, INDICES, COMMODITIES}:
        wd, hour, _ = _weekday_utc(now)
        if wd >= 5:
            return False, f"{cat}_weekend_closed"
        # Avoid thin overnight: allow 07:00–21:00 UTC as soft window
        if 7 <= hour < 21:
            return True, f"{cat}_session_soft_open"
        return False, f"{cat}_outside_soft_hours"

    return True, "default_open"
