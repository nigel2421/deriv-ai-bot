"""
Professional trend-following + Boom/Crash helpers for Volatility indices.

Implements the operator playbook:
  - 50/200 EMA stack + RSI gate + pullback entries
  - Market structure (HH/HL, LH/LL) and break-of-structure
  - Break & retest, RSI divergence, candle confirmation
  - Spike fade for Boom/Crash-style moves (wait for spike end + confirm)
  - Multi-timeframe bias (longer candle = trend, shorter = entry)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategy.candles import build_candles, candle_closes, is_bearish, is_bullish
from src.strategy.chart_tools import (
    atr_proxy,
    ema_series,
    quotes_from_ticks,
    rsi,
    support_resistance,
)

logger = logging.getLogger(__name__)

# Symbols where multi-TF + strict trend rules matter most
VOL_STRICT = frozenset(
    {
        "R_50",
        "R_75",
        "R_100",
        "1HZ50V",
        "1HZ75V",
        "1HZ100V",
        "BOOM500",
        "BOOM1000",
        "CRASH500",
        "CRASH1000",
    }
)


def _ema_last(values: Sequence[float], period: int) -> Optional[float]:
    series = ema_series(list(values), period)
    return series[-1] if series else None


def market_structure(closes: Sequence[float], look: int = 8) -> Dict[str, Any]:
    """HH/HL vs LH/LL + simple break-of-structure flags."""
    if len(closes) < look * 2 + 2:
        return {
            "uptrend": False,
            "downtrend": False,
            "hh": False,
            "hl": False,
            "lh": False,
            "ll": False,
            "bullish_bos": False,
            "bearish_bos": False,
        }
    a = list(closes[-(look * 2) : -look])
    b = list(closes[-look:])
    hh = max(b) > max(a)
    hl = min(b) > min(a)
    lh = max(b) < max(a)
    ll = min(b) < min(a)
    uptrend = hh and hl
    downtrend = lh and ll
    # Break of structure: last close breaks prior swing high/low
    prior_high = max(a)
    prior_low = min(a)
    last = closes[-1]
    bullish_bos = downtrend and last > prior_high
    bearish_bos = uptrend and last < prior_low
    return {
        "uptrend": uptrend,
        "downtrend": downtrend,
        "hh": hh,
        "hl": hl,
        "lh": lh,
        "ll": ll,
        "bullish_bos": bullish_bos,
        "bearish_bos": bearish_bos,
        "prior_high": prior_high,
        "prior_low": prior_low,
    }


def rsi_divergence(closes: Sequence[float], period: int = 14) -> Dict[str, bool]:
    """Detect simple price/RSI divergence on last two swing windows."""
    if len(closes) < period + 20:
        return {"bullish": False, "bearish": False}
    # RSI series (approx at each point via rolling)
    rsi_vals: List[float] = []
    for i in range(period + 1, len(closes) + 1):
        v = rsi(closes[:i], period)
        if v is not None:
            rsi_vals.append(v)
    if len(rsi_vals) < 16:
        return {"bullish": False, "bearish": False}
    # Compare early half vs late half of last 16 bars
    price_window = list(closes[-16:])
    rsi_window = rsi_vals[-16:]
    mid = 8
    p_lo_a, p_lo_b = min(price_window[:mid]), min(price_window[mid:])
    p_hi_a, p_hi_b = max(price_window[:mid]), max(price_window[mid:])
    # RSI at those relative segments
    r_lo_a = min(rsi_window[:mid])
    r_lo_b = min(rsi_window[mid:])
    r_hi_a = max(rsi_window[:mid])
    r_hi_b = max(rsi_window[mid:])
    bullish = p_lo_b < p_lo_a and r_lo_b > r_lo_a  # price LL, RSI HL
    bearish = p_hi_b > p_hi_a and r_hi_b < r_hi_a  # price HH, RSI LH
    return {"bullish": bullish, "bearish": bearish}


def break_and_retest(
    closes: Sequence[float], look: int = 20
) -> Dict[str, Any]:
    """
    Detect break of range then retest of broken level.
    Bullish: break above resistance, pullback toward it, hold above.
    Bearish: break below support, retest from below, hold under.
    """
    empty = {"bullish": False, "bearish": False, "level": None}
    if len(closes) < look + 6:
        return empty
    base = list(closes[-(look + 6) : -6])
    recent = list(closes[-6:])
    if not base or not recent:
        return empty
    res = max(base)
    sup = min(base)
    last = recent[-1]
    # Bullish break + retest: at least one bar above res, then touch near res, close above
    broke_up = max(recent) > res
    retest_up = any(abs(c - res) / (abs(res) or 1.0) < 0.0015 for c in recent[-4:])
    hold_up = last > res * 0.999
    # Bearish
    broke_dn = min(recent) < sup
    retest_dn = any(abs(c - sup) / (abs(sup) or 1.0) < 0.0015 for c in recent[-4:])
    hold_dn = last < sup * 1.001
    return {
        "bullish": bool(broke_up and retest_up and hold_up),
        "bearish": bool(broke_dn and retest_dn and hold_dn),
        "level": res if broke_up else (sup if broke_dn else None),
        "resistance": res,
        "support": sup,
    }


def candle_reversal_hints(candles: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    """Hammer / engulfing style hints from last 1–2 OHLC candles."""
    out = {"bullish": False, "bearish": False, "pattern": None}
    if len(candles) < 2:
        return out
    c0, c1 = candles[-2], candles[-1]
    o, h, l, cl = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
    body = abs(cl - o)
    full = max(h - l, 1e-12)
    upper = h - max(o, cl)
    lower = min(o, cl) - l
    # Hammer: long lower wick, small body near top
    if lower >= body * 2 and upper <= body * 0.6 and cl >= o:
        out["bullish"] = True
        out["pattern"] = "hammer"
    # Shooting star
    if upper >= body * 2 and lower <= body * 0.6 and cl <= o:
        out["bearish"] = True
        out["pattern"] = "shooting_star"
    # Engulfing
    o0, cl0 = float(c0["open"]), float(c0["close"])
    if is_bearish(c0) and is_bullish(c1) and cl >= o0 and o <= cl0:
        out["bullish"] = True
        out["pattern"] = "bullish_engulfing"
    if is_bullish(c0) and is_bearish(c1) and cl <= o0 and o >= cl0:
        out["bearish"] = True
        out["pattern"] = "bearish_engulfing"
    return out


def detect_spike(
    prices: Sequence[float], atr: Optional[float], mult: float = 3.5
) -> Dict[str, Any]:
    """
    Large single-move candle/tick run → wait for confirmation (Boom/Crash style).
    """
    if len(prices) < 5:
        return {"spike_up": False, "spike_down": False, "size": 0.0}
    last_move = prices[-1] - prices[-2]
    atr_v = atr if atr and atr > 0 else (
        sum(abs(prices[i] - prices[i - 1]) for i in range(-5, 0)) / 5.0
    )
    thr = atr_v * mult
    size = abs(last_move)
    return {
        "spike_up": last_move > thr,
        "spike_down": last_move < -thr,
        "size": size,
        "threshold": thr,
    }


def pullback_to_ema(
    prices: Sequence[float],
    ema_value: Optional[float],
    atr: Optional[float],
    *,
    band_atr: float = 0.8,
) -> bool:
    """True if last price is near EMA (pullback zone), not extended far away."""
    if ema_value is None or not prices:
        return False
    last = prices[-1]
    dist = abs(last - ema_value)
    if atr and atr > 0:
        return dist <= atr * band_atr
    # Relative tolerance
    return dist / (abs(ema_value) or 1.0) < 0.001


def analyze_pro_trend(
    ticks: Sequence[Dict[str, Any]],
    *,
    symbol: str = "",
    min_confidence: float = 0.78,
) -> Dict[str, Any]:
    """
    Full pro-trend vote for CALL/PUT.

    Returns dict with contract_type, confidence, notes, components.
    """
    empty: Dict[str, Any] = {
        "contract_type": None,
        "confidence": 0.0,
        "direction": "flat",
        "notes": [],
        "ready": False,
    }
    prices = quotes_from_ticks(ticks, n=220)
    if len(prices) < 60:
        return empty

    # Prefer minute candles when enough ticks; else use tick closes
    candles_5 = build_candles(ticks, period_sec=60, max_candles=80)
    candles_15 = build_candles(ticks, period_sec=180, max_candles=60)  # ~3m proxy if short history
    # If operator intended 5m/15m: use 300/900 when enough data
    if len(ticks) >= 400:
        candles_5 = build_candles(ticks, period_sec=300, max_candles=80)
        candles_15 = build_candles(ticks, period_sec=900, max_candles=40)

    entry_closes = candle_closes(candles_5) if len(candles_5) >= 30 else prices
    trend_closes = candle_closes(candles_15) if len(candles_15) >= 25 else entry_closes

    ema50 = _ema_last(entry_closes, 50)
    ema200 = _ema_last(entry_closes, min(200, max(50, len(entry_closes) - 1)))
    # If not enough bars for true 200, use longest available as slow EMA
    if ema200 is None and len(entry_closes) >= 50:
        ema200 = _ema_last(entry_closes, max(50, len(entry_closes) // 2))

    rsi14 = rsi(entry_closes, 14)
    atr = atr_proxy(entry_closes, 14)
    last = entry_closes[-1]
    struct = market_structure(entry_closes, 8)
    div = rsi_divergence(entry_closes, 14)
    br = break_and_retest(entry_closes, 20)
    spike = detect_spike(entry_closes, atr)
    candles_for_pat = candles_5 if candles_5 else []
    pat = candle_reversal_hints(candles_for_pat) if candles_for_pat else {
        "bullish": False,
        "bearish": False,
        "pattern": None,
    }

    # Higher TF bias
    t_ema50 = _ema_last(trend_closes, min(50, max(10, len(trend_closes) // 2)))
    t_ema200 = _ema_last(trend_closes, min(200, max(20, len(trend_closes) - 1)))
    htf_bull = (
        t_ema50 is not None
        and t_ema200 is not None
        and t_ema50 > t_ema200
        and trend_closes[-1] > t_ema50
    )
    htf_bear = (
        t_ema50 is not None
        and t_ema200 is not None
        and t_ema50 < t_ema200
        and trend_closes[-1] < t_ema50
    )

    call_pts = 0.0
    put_pts = 0.0
    notes: List[str] = []

    # --- Core EMA 50/200 + RSI ---
    ema_bull = (
        ema50 is not None
        and ema200 is not None
        and last > ema50
        and last > ema200
        and ema50 > ema200
    )
    ema_bear = (
        ema50 is not None
        and ema200 is not None
        and last < ema50
        and last < ema200
        and ema50 < ema200
    )
    if ema_bull:
        call_pts += 1.5
        notes.append("ema50_200_bull")
    if ema_bear:
        put_pts += 1.5
        notes.append("ema50_200_bear")

    if rsi14 is not None:
        if rsi14 > 50:
            call_pts += 0.7 + min(0.4, (rsi14 - 50) / 50)
            notes.append(f"rsi_gt50={rsi14:.0f}")
        elif rsi14 < 50:
            put_pts += 0.7 + min(0.4, (50 - rsi14) / 50)
            notes.append(f"rsi_lt50={rsi14:.0f}")
        if rsi14 >= 75:
            call_pts *= 0.7
            notes.append("rsi_ob_dampen")
        if rsi14 <= 25:
            put_pts *= 0.7
            notes.append("rsi_os_dampen")

    # Pullback entry preference (reward setups near 50 EMA)
    if ema_bull and pullback_to_ema(entry_closes, ema50, atr):
        call_pts += 0.9
        notes.append("pullback_50ema")
    if ema_bear and pullback_to_ema(entry_closes, ema50, atr):
        put_pts += 0.9
        notes.append("pullback_50ema")

    # Structure
    if struct.get("uptrend"):
        call_pts += 1.0
        notes.append("struct_up")
    if struct.get("downtrend"):
        put_pts += 1.0
        notes.append("struct_down")
    if struct.get("bullish_bos"):
        call_pts += 0.8
        notes.append("bullish_bos")
    if struct.get("bearish_bos"):
        put_pts += 0.8
        notes.append("bearish_bos")

    # Break & retest
    if br.get("bullish"):
        call_pts += 1.1
        notes.append("break_retest_bull")
    if br.get("bearish"):
        put_pts += 1.1
        notes.append("break_retest_bear")

    # RSI divergence
    if div.get("bullish"):
        call_pts += 0.7
        notes.append("rsi_div_bull")
    if div.get("bearish"):
        put_pts += 0.7
        notes.append("rsi_div_bear")

    # Candles only as confirmation
    if pat.get("bullish") and (ema_bull or struct.get("uptrend") or br.get("bullish")):
        call_pts += 0.5
        notes.append(f"candle_{pat.get('pattern')}")
    if pat.get("bearish") and (ema_bear or struct.get("downtrend") or br.get("bearish")):
        put_pts += 0.5
        notes.append(f"candle_{pat.get('pattern')}")

    # HTF agreement for Vol 50/75/100
    strict = symbol in VOL_STRICT or any(
        x in str(symbol).upper() for x in ("75", "100", "BOOM", "CRASH")
    )
    if strict:
        if htf_bull:
            call_pts += 0.8
            notes.append("htf_bull")
        if htf_bear:
            put_pts += 0.8
            notes.append("htf_bear")
        # Penalize trading against HTF
        if htf_bull and put_pts > call_pts:
            put_pts *= 0.55
            notes.append("fade_vs_htf_dampen")
        if htf_bear and call_pts > put_pts:
            call_pts *= 0.55
            notes.append("fade_vs_htf_dampen")

    # Boom/Crash: do not chase spike; reward post-spike confirmation
    if spike.get("spike_up"):
        call_pts *= 0.4
        notes.append("no_chase_spike_up")
        # Crash-style short only after bearish confirm later — mild put bias if structure bearish
        if pat.get("bearish") or struct.get("bearish_bos"):
            put_pts += 0.9
            notes.append("crash_confirm")
    if spike.get("spike_down"):
        put_pts *= 0.4
        notes.append("no_chase_spike_down")
        if pat.get("bullish") or struct.get("bullish_bos"):
            call_pts += 0.9
            notes.append("boom_confirm")

    # Extended move without pullback → cut points (avoid chase)
    if ema50 is not None and atr and atr > 0:
        ext = abs(last - ema50) / atr
        if ext > 2.5:
            if last > ema50:
                call_pts *= 0.65
            else:
                put_pts *= 0.65
            notes.append(f"extended_{ext:.1f}atr")

    total = call_pts + put_pts
    result: Dict[str, Any] = {
        "ready": True,
        "contract_type": None,
        "confidence": 0.0,
        "direction": "flat",
        "notes": notes,
        "call_pts": round(call_pts, 3),
        "put_pts": round(put_pts, 3),
        "ema50": ema50,
        "ema200": ema200,
        "rsi14": rsi14,
        "structure": struct,
        "divergence": div,
        "break_retest": br,
        "spike": spike,
        "htf_bull": htf_bull,
        "htf_bear": htf_bear,
        "strict_symbol": strict,
        "pattern": pat.get("pattern"),
    }

    if total < 2.2:
        result["confidence"] = min(0.7, 0.35 + total * 0.1)
        return result

    if call_pts > put_pts * 1.12:
        margin = (call_pts - put_pts) / total
        conf = 0.58 + 0.32 * (call_pts / total) + 0.14 * margin
        # Require RSI>50 and (EMA bull or structure) for high conf on strict symbols
        if strict and not (ema_bull or struct.get("uptrend") or htf_bull):
            conf *= 0.85
        conf = min(0.97, conf)
        if conf >= min_confidence * 0.9:
            result["contract_type"] = "CALL"
            result["direction"] = "up"
        result["confidence"] = float(round(conf, 4))
        return result

    if put_pts > call_pts * 1.12:
        margin = (put_pts - call_pts) / total
        conf = 0.58 + 0.32 * (put_pts / total) + 0.14 * margin
        if strict and not (ema_bear or struct.get("downtrend") or htf_bear):
            conf *= 0.85
        conf = min(0.97, conf)
        if conf >= min_confidence * 0.9:
            result["contract_type"] = "PUT"
            result["direction"] = "down"
        result["confidence"] = float(round(conf, 4))
        return result

    result["confidence"] = 0.55
    result["notes"] = notes + ["mixed"]
    return result


def pick_pro_rise_fall(
    ticks: Sequence[Dict[str, Any]],
    *,
    symbol: str = "",
    min_confidence: float = 0.80,
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    pro = analyze_pro_trend(ticks, symbol=symbol, min_confidence=min_confidence)
    ct = pro.get("contract_type")
    conf = float(pro.get("confidence") or 0.0)
    if not ct or conf < min_confidence:
        return None, conf, pro
    return str(ct), conf, pro
