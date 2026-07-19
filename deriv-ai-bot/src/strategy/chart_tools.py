"""
Chart / technical tools for tick series (Rise-Fall decision support).

Pure-Python indicators — no external TA package required at runtime.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def quotes_from_ticks(ticks: Sequence[Dict[str, Any]], n: int = 120) -> List[float]:
    out: List[float] = []
    for t in list(ticks)[-n:]:
        try:
            q = t.get("quote") if isinstance(t, dict) else t
            if q is None:
                continue
            out.append(float(q))
        except (TypeError, ValueError):
            continue
    return out


def sma(values: Sequence[float], period: int) -> Optional[float]:
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def ema_series(values: Sequence[float], period: int) -> List[float]:
    if not values or period <= 0:
        return []
    k = 2.0 / (period + 1)
    out: List[float] = []
    ema = float(values[0])
    out.append(ema)
    for v in values[1:]:
        ema = float(v) * k + ema * (1 - k)
        out.append(ema)
    return out


def rsi(values: Sequence[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr_proxy(values: Sequence[float], period: int = 14) -> Optional[float]:
    """Mean absolute tick-to-tick change (volatility proxy)."""
    if len(values) < period + 1:
        return None
    diffs = [abs(values[i] - values[i - 1]) for i in range(-period, 0)]
    return sum(diffs) / len(diffs)


def macd(
    values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, Optional[float]]:
    if len(values) < slow + signal:
        return {"macd": None, "signal": None, "hist": None}
    ema_fast = ema_series(values, fast)
    ema_slow = ema_series(values, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    sig = ema_series(macd_line, signal)
    hist = macd_line[-1] - sig[-1] if sig else None
    return {
        "macd": macd_line[-1] if macd_line else None,
        "signal": sig[-1] if sig else None,
        "hist": hist,
    }


def higher_highs_lows(values: Sequence[float], look: int = 10) -> Dict[str, Any]:
    if len(values) < look + 2:
        return {"hh": False, "hl": False, "lh": False, "ll": False}
    a = values[-(look * 2) : -look] if len(values) >= look * 2 else values[:-look]
    b = values[-look:]
    if not a or not b:
        return {"hh": False, "hl": False, "lh": False, "ll": False}
    return {
        "hh": max(b) > max(a),
        "hl": min(b) > min(a),
        "lh": max(b) < max(a),
        "ll": min(b) < min(a),
    }


def support_resistance(values: Sequence[float], look: int = 40) -> Dict[str, Optional[float]]:
    window = list(values[-look:]) if values else []
    if len(window) < 5:
        return {"support": None, "resistance": None, "mid": None}
    return {
        "support": min(window),
        "resistance": max(window),
        "mid": sum(window) / len(window),
    }


def chart_snapshot(ticks: Sequence[Dict[str, Any]], n: int = 100) -> Dict[str, Any]:
    """Full indicator pack for a market."""
    prices = quotes_from_ticks(ticks, n=n)
    if len(prices) < 20:
        return {"n": len(prices), "ready": False}

    ema8 = ema_series(prices, 8)
    ema21 = ema_series(prices, 21)
    ema50 = ema_series(prices, 50) if len(prices) >= 50 else []
    ema200 = ema_series(prices, 200) if len(prices) >= 200 else (
        ema_series(prices, max(50, len(prices) // 2)) if len(prices) >= 50 else []
    )
    rsi14 = rsi(prices, 14)
    atr = atr_proxy(prices, 14)
    m = macd(prices)
    struct = higher_highs_lows(prices, 10)
    sr = support_resistance(prices, 40)
    last = prices[-1]
    ema8_v = ema8[-1] if ema8 else None
    ema21_v = ema21[-1] if ema21 else None
    ema50_v = ema50[-1] if ema50 else None
    ema200_v = ema200[-1] if ema200 else None

    # Distance from mid-band as % of ATR
    band_pos = None
    if atr and atr > 0 and sr.get("mid") is not None:
        band_pos = (last - float(sr["mid"])) / atr

    # Classic short stack + professional 50/200 stack
    short_bull = ema8_v is not None and ema21_v is not None and ema8_v > ema21_v
    short_bear = ema8_v is not None and ema21_v is not None and ema8_v < ema21_v
    stack_bull = (
        ema50_v is not None
        and ema200_v is not None
        and last > ema50_v
        and last > ema200_v
        and ema50_v > ema200_v
    )
    stack_bear = (
        ema50_v is not None
        and ema200_v is not None
        and last < ema50_v
        and last < ema200_v
        and ema50_v < ema200_v
    )

    return {
        "n": len(prices),
        "ready": True,
        "last": last,
        "ema8": ema8_v,
        "ema21": ema21_v,
        "ema50": ema50_v,
        "ema200": ema200_v,
        "ema_bull": short_bull or stack_bull,
        "ema_bear": short_bear or stack_bear,
        "ema_stack_bull": stack_bull,
        "ema_stack_bear": stack_bear,
        "rsi14": rsi14,
        "atr": atr,
        "macd": m.get("macd"),
        "macd_signal": m.get("signal"),
        "macd_hist": m.get("hist"),
        "structure": struct,
        "support": sr.get("support"),
        "resistance": sr.get("resistance"),
        "mid": sr.get("mid"),
        "band_pos": band_pos,
    }


def rise_fall_vote(chart: Dict[str, Any]) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Multi-tool vote for CALL/PUT.
    Returns (contract_type|None, confidence 0..1, detail).
    """
    if not chart.get("ready"):
        return None, 0.0, {"reason": "not_ready"}

    call_pts = 0.0
    put_pts = 0.0
    notes: List[str] = []

    # EMA stack (short + 50/200 professional)
    if chart.get("ema_stack_bull"):
        call_pts += 1.4
        notes.append("ema50_200_bull")
    elif chart.get("ema_bull"):
        call_pts += 1.0
        notes.append("ema_bull")
    if chart.get("ema_stack_bear"):
        put_pts += 1.4
        notes.append("ema50_200_bear")
    elif chart.get("ema_bear"):
        put_pts += 1.0
        notes.append("ema_bear")

    # RSI — trend-following gate (>50 long bias, <50 short)
    rsi_v = chart.get("rsi14")
    if rsi_v is not None:
        if rsi_v >= 50:
            call_pts += 0.7 + min(0.6, (rsi_v - 50) / 40)
            notes.append(f"rsi_up={rsi_v:.1f}")
        elif rsi_v < 50:
            put_pts += 0.7 + min(0.6, (50 - rsi_v) / 40)
            notes.append(f"rsi_dn={rsi_v:.1f}")
        # Extreme mean-reversion caution: reduce overextension
        if rsi_v >= 75:
            call_pts *= 0.75
            notes.append("rsi_overbought_dampen")
        if rsi_v <= 25:
            put_pts *= 0.75
            notes.append("rsi_oversold_dampen")

    # MACD histogram
    hist = chart.get("macd_hist")
    if hist is not None:
        if hist > 0:
            call_pts += 1.0
            notes.append("macd_pos")
        elif hist < 0:
            put_pts += 1.0
            notes.append("macd_neg")

    # Market structure HH/HL vs LH/LL
    st = chart.get("structure") or {}
    if st.get("hh") and st.get("hl"):
        call_pts += 1.0
        notes.append("hh_hl")
    if st.get("lh") and st.get("ll"):
        put_pts += 1.0
        notes.append("lh_ll")

    # Position vs mid / ATR band
    band = chart.get("band_pos")
    if band is not None:
        if band > 0.4:
            call_pts += 0.5
            notes.append("above_mid")
        elif band < -0.4:
            put_pts += 0.5
            notes.append("below_mid")

    total = call_pts + put_pts
    if total < 1.5:
        return None, min(0.65, 0.4 + total * 0.1), {"notes": notes, "call": call_pts, "put": put_pts}

    if call_pts > put_pts * 1.15:
        # Confidence: share of points + margin
        margin = (call_pts - put_pts) / total
        conf = 0.58 + 0.35 * (call_pts / max(total, 1)) + 0.12 * margin
        conf = min(0.96, conf)
        return "CALL", conf, {"notes": notes, "call": call_pts, "put": put_pts}

    if put_pts > call_pts * 1.15:
        margin = (put_pts - call_pts) / total
        conf = 0.58 + 0.35 * (put_pts / max(total, 1)) + 0.12 * margin
        conf = min(0.96, conf)
        return "PUT", conf, {"notes": notes, "call": call_pts, "put": put_pts}

    return None, 0.55, {"notes": notes, "call": call_pts, "put": put_pts, "reason": "mixed"}
