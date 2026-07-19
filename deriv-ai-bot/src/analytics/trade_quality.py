"""
Trade Quality Score 0–100 from pattern / digit momentum / volatility / historical edge.
Gate: only auto-trade when total >= 80 (configurable).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from src.analytics.digit_analysis import digit_snapshot
from src.analytics.tick_patterns import detect_patterns
from src.strategy.chart_tools import atr_proxy, quotes_from_ticks


def trade_quality_score(
    ticks: Sequence[Dict[str, Any]],
    *,
    symbol: str = "",
    contract_type: str = "",
    historical_edge: float = 50.0,
    pattern_strength_val: float = 50.0,
    live_edge: float = 50.0,
    max_components: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Components (default weights sum to 100 potential):
      Pattern Strength  30
      Digit Momentum    20
      Volatility Score  15
      Historical Edge   25
      Live Edge residual 10 (optional blend)
    """
    weights = max_components or {
        "pattern": 30.0,
        "digit_momentum": 20.0,
        "volatility": 15.0,
        "historical": 25.0,
        "live": 10.0,
    }

    pats = detect_patterns(ticks)
    snap = digit_snapshot(ticks)
    prices = quotes_from_ticks(ticks, n=80)
    atr = atr_proxy(prices, 14) if prices else None

    # Pattern strength from alerts + external
    alert_s = float(pats.get("pattern_alert_strength") or 0)
    pattern_pts = min(
        weights["pattern"],
        (0.55 * float(pattern_strength_val) + 0.45 * alert_s)
        / 100.0
        * weights["pattern"],
    )

    # Digit momentum: deviation from fair + streak clarity
    heat = (snap.get("heatmap") or {}).get("windows", {}).get("100") or {}
    pct = heat.get("pct") or {}
    if pct:
        max_dev = max(abs(float(pct.get(d, 10)) - 10.0) for d in range(10))
    else:
        max_dev = 0.0
    streak = int((snap.get("streaks") or {}).get("current_streak") or 0)
    mom_raw = min(100.0, max_dev * 8.0 + streak * 12.0)
    digit_pts = (mom_raw / 100.0) * weights["digit_momentum"]

    # Volatility: moderate preferred for digits; extremes lower
    vol_raw = 50.0
    if atr and prices and prices[-1]:
        rel = atr / abs(prices[-1]) * 10000.0  # scale
        # Sweet spot mid volatility
        if 0.5 <= rel <= 5.0:
            vol_raw = 80.0 + min(20.0, rel * 2)
        elif rel < 0.5:
            vol_raw = 40.0
        else:
            vol_raw = max(20.0, 90.0 - rel * 5)
    vol_pts = (vol_raw / 100.0) * weights["volatility"]

    hist_pts = (max(0.0, min(100.0, historical_edge)) / 100.0) * weights["historical"]
    live_pts = (max(0.0, min(100.0, live_edge)) / 100.0) * weights["live"]

    total = pattern_pts + digit_pts + vol_pts + hist_pts + live_pts
    return {
        "quality_score": round(total, 1),
        "components": {
            "pattern_strength": round(pattern_pts, 1),
            "digit_momentum": round(digit_pts, 1),
            "volatility": round(vol_pts, 1),
            "historical_edge": round(hist_pts, 1),
            "live_edge": round(live_pts, 1),
        },
        "auto_ok": total >= 80.0,
        "gate": 80.0,
        "symbol": symbol,
        "contract_type": contract_type,
    }
