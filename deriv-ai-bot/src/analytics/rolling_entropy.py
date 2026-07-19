"""
Rolling Entropy Engine — real-time digit-stream randomness analysis.

Architecture (every tick):
  Incoming Tick → Update Window Buffers → Compute Entropy →
  Entropy Momentum / Velocity → Clarity / Pattern Strength →
  Regime → Edge / Signal gates

Windows: 25, 50, 100, 200, 500 ticks.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

# Hmax for 10 digits
HMAX_DIGITS = math.log2(10.0)  # ≈ 3.3219
HMAX_BINARY = 1.0

DEFAULT_WINDOWS = (25, 50, 100, 200, 500)


def _entropy_from_counts(counts: Sequence[int]) -> float:
    total = sum(int(c) for c in counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def _compression(h: float, hmax: float = HMAX_DIGITS) -> Dict[str, float]:
    hmax = float(hmax) if hmax > 0 else 1.0
    h_eff = min(max(0.0, float(h)), hmax)
    comp = max(0.0, min(1.0, 1.0 - h_eff / hmax))
    return {
        "h": round(h, 4),
        "hmax": round(hmax, 4),
        "compression": round(comp, 4),
        "compression_pct": round(comp * 100.0, 2),
        "entropy_loss": round(hmax - h_eff, 4),
    }


def compression_bias_label(compression_pct: float) -> str:
    """
    0–5% Random · 5–10% Mild · 10–20% Significant · 20%+ Strong anomaly
    """
    c = float(compression_pct)
    if c >= 20:
        return "Strong anomaly"
    if c >= 10:
        return "Significant"
    if c >= 5:
        return "Mild bias"
    return "Random"


def regime_from_normalized_entropy(h_ratio: float) -> str:
    """
    Classify by H/Hmax residual randomness (as % of max entropy remaining).

    Entropy remaining > 95% → RANDOM
    80–95% → NORMAL
    60–80% → BIASED
    40–60% → STRONG PATTERN
    < 40% → EXTREME ANOMALY

    h_ratio = H / Hmax  (1.0 = fully random)
    """
    # residual randomness percent
    remaining = max(0.0, min(1.0, float(h_ratio))) * 100.0
    if remaining > 95:
        return "RANDOM"
    if remaining > 80:
        return "NORMAL"
    if remaining > 60:
        return "BIASED"
    if remaining > 40:
        return "STRONG PATTERN"
    return "EXTREME ANOMALY"


class RollingEntropyEngine:
    """
    Tick-updated multi-window entropy engine for Deriv last digits.

    Call add_digit(d) or add_tick(tick_dict) every tick; read snapshot().
    """

    def __init__(self, windows: Sequence[int] = DEFAULT_WINDOWS):
        self.windows: Tuple[int, ...] = tuple(sorted({int(w) for w in windows if int(w) > 0}))
        self.buffers: Dict[int, Deque[int]] = {
            w: deque(maxlen=w) for w in self.windows
        }
        # price stream for up/down (last quotes)
        self._quotes: Deque[float] = deque(maxlen=max(self.windows) if self.windows else 500)
        # previous entropy per window for velocity
        self._prev_h: Dict[int, float] = {}
        self._tick_count = 0
        self._last_snapshot: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        for w in self.windows:
            self.buffers[w].clear()
        self._quotes.clear()
        self._prev_h.clear()
        self._tick_count = 0
        self._last_snapshot = None

    def add_digit(self, digit: int, quote: Optional[float] = None) -> Dict[str, Any]:
        """Push one last-digit (0–9); optionally quote for up/down entropy."""
        try:
            d = int(digit)
        except (TypeError, ValueError):
            return self.snapshot()
        if d < 0 or d > 9:
            # try last digit of absolute value
            d = abs(d) % 10
        for w in self.windows:
            self.buffers[w].append(d)
        if quote is not None:
            try:
                self._quotes.append(float(quote))
            except (TypeError, ValueError):
                pass
        self._tick_count += 1
        return self.snapshot(recompute=True)

    def add_tick(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """Extract last digit + quote from a Deriv-style tick dict."""
        from src.strategy.digit_contracts import extract_last_digit

        q = tick.get("quote") if isinstance(tick, dict) else None
        d = extract_last_digit(q) if q is not None else None
        if d is None and isinstance(tick, dict):
            d = tick.get("digit")
        if d is None:
            return self.snapshot()
        return self.add_digit(int(d), quote=float(q) if q is not None else None)

    def bootstrap_from_ticks(self, ticks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Warm windows from a historical tick list (oldest → newest)."""
        for t in ticks:
            if isinstance(t, dict):
                self.add_tick(t)
            else:
                try:
                    self.add_digit(int(t))
                except (TypeError, ValueError):
                    continue
        return self.snapshot()

    def _digit_entropy(self, digits: Sequence[int]) -> Dict[str, Any]:
        if len(digits) < 5:
            c = _compression(HMAX_DIGITS, HMAX_DIGITS)
            return {**c, "n": len(digits), "counts": [0] * 10, "ready": False}
        counts = [0] * 10
        for d in digits:
            counts[int(d)] += 1
        h = _entropy_from_counts(counts)
        c = _compression(h, HMAX_DIGITS)
        n = len(digits)
        expected = n / 10.0
        deviations = []
        max_dev = 0.0
        coldest = hottest = 0
        for i, cnt in enumerate(counts):
            actual_pct = 100.0 * cnt / n
            exp_pct = 10.0
            dev = abs(actual_pct - exp_pct)
            deviations.append(round(dev, 2))
            if dev > max_dev:
                max_dev = dev
            if cnt == min(counts):
                coldest = i
            if cnt == max(counts):
                hottest = i
        # max deviation 0 → score 0, 7% → ~70, 15%+ → 100
        dev_score = min(100.0, max_dev / 10.0 * 100.0)
        return {
            **c,
            "n": n,
            "counts": counts,
            "ready": True,
            "max_deviation_pct": round(max_dev, 2),
            "deviation_score": round(dev_score, 1),
            "deviations_pct": deviations,
            "coldest": coldest,
            "hottest": hottest,
            "bias_label": compression_bias_label(c["compression_pct"]),
            "h_ratio": round(min(h, HMAX_DIGITS) / HMAX_DIGITS, 4),
        }

    def _binary_entropy(self, bits: Sequence[int]) -> Dict[str, Any]:
        if len(bits) < 5:
            return {**_compression(HMAX_BINARY, HMAX_BINARY), "ready": False, "n": len(bits)}
        n0 = sum(1 for b in bits if int(b) == 0)
        n1 = len(bits) - n0
        h = _entropy_from_counts([n0, n1])
        c = _compression(h, HMAX_BINARY)
        return {**c, "ready": True, "n": len(bits), "n0": n0, "n1": n1}

    def _streak_score(self, digits: Sequence[int]) -> Dict[str, Any]:
        """Streak analysis: current repeat length + streak entropy of continue/change."""
        if len(digits) < 3:
            return {"streak_score": 30.0, "current_streak": 0, "ready": False}
        last = int(digits[-1])
        streak = 1
        for d in reversed(list(digits)[:-1]):
            if int(d) == last:
                streak += 1
            else:
                break
        bits = []
        for i in range(1, len(digits)):
            bits.append(1 if int(digits[i]) == int(digits[i - 1]) else 0)
        be = self._binary_entropy(bits)
        # Longer streaks + lower streak entropy → higher score
        streak_pts = min(100.0, 20.0 + streak * 18.0)
        ent_pts = float(be.get("compression_pct") or 0) * 2.0  # compression% * 2 capped
        ent_pts = min(100.0, ent_pts)
        score = 0.55 * streak_pts + 0.45 * ent_pts
        return {
            "streak_score": round(score, 1),
            "current_streak": streak,
            "current_digit": last,
            "streak_entropy": be,
            "ready": True,
        }

    def snapshot(self, *, recompute: bool = True) -> Dict[str, Any]:
        if not recompute and self._last_snapshot is not None:
            return self._last_snapshot

        window_stats: Dict[str, Any] = {}
        velocities: Dict[str, float] = {}

        for w in self.windows:
            digits = list(self.buffers[w])
            st = self._digit_entropy(digits)
            prev = self._prev_h.get(w)
            vel = 0.0 if prev is None else float(st["h"]) - float(prev)
            if st.get("ready"):
                self._prev_h[w] = float(st["h"])
            velocities[str(w)] = round(vel, 4)
            st["velocity"] = round(vel, 4)
            # residual randomness % for regime
            h_ratio = float(st.get("h_ratio") or 1.0)
            st["regime"] = regime_from_normalized_entropy(h_ratio)
            window_stats[str(w)] = st

        # Primary short / long
        h50 = window_stats.get("50") or window_stats.get("25") or {}
        h100 = window_stats.get("100") or h50
        h500 = window_stats.get("500") or window_stats.get("200") or h100

        # Momentum = long H − short H
        h_short = float(h50.get("h") or HMAX_DIGITS)
        h_long = float(h500.get("h") or HMAX_DIGITS)
        momentum_bits = h_long - h_short
        # Scale momentum to 0–100 (0.55 bits strong)
        momentum_score = max(0.0, min(100.0, 50.0 + (momentum_bits / HMAX_DIGITS) * 120.0))

        compression_pct = float(h50.get("compression_pct") or h100.get("compression_pct") or 0)
        compression_score = min(100.0, compression_pct * 5.0)  # 20% → 100
        # softer: use raw clarity 100*(1-H/Hmax) which IS compression_pct
        compression_score = min(100.0, compression_pct * (100.0 / 20.0))  # 20% → 100
        compression_score = min(100.0, max(0.0, compression_pct * 5.0))

        deviation_score = float(
            h100.get("deviation_score") or h50.get("deviation_score") or 0
        )

        # Multi-dimensional on best-filled buffer
        primary = list(self.buffers.get(100) or self.buffers.get(max(self.windows)))
        oe_bits = [1 if d % 2 else 0 for d in primary]
        ou_bits = [1 if d > 5 else 0 for d in primary]
        quotes = list(self._quotes)
        ud_bits = []
        for i in range(1, len(quotes)):
            ud_bits.append(1 if quotes[i] > quotes[i - 1] else 0)

        oe = self._binary_entropy(oe_bits)
        ou = self._binary_entropy(ou_bits)
        ud = self._binary_entropy(ud_bits)
        streak = self._streak_score(primary)

        # Level-1 scores (0–100) from compression / streak analysis
        dig_c = float(h100.get("compression_pct") or compression_pct)

        def _c_score(pct: float) -> float:
            # map compression% → 0–100 (25% compression → 100)
            return min(100.0, max(0.0, float(pct) * (100.0 / 25.0)))

        dig_s = _c_score(dig_c)
        oe_s = _c_score(float(oe.get("compression_pct") or 0))
        ou_s = _c_score(float(ou.get("compression_pct") or 0))
        ud_s = _c_score(float(ud.get("compression_pct") or 0))
        st_s = float(streak.get("streak_score") or 0)

        # Hierarchical default reliability weights (not equal average):
        # Digit 40% · Streak 25% · OddEven 15% · OverUnder 10% · UpDown 10%
        composite = (
            0.40 * dig_s
            + 0.25 * st_s
            + 0.15 * oe_s
            + 0.10 * ou_s
            + 0.10 * ud_s
        )
        composite_weights = {
            "digit": 0.40,
            "streak": 0.25,
            "odd_even": 0.15,
            "over_under": 0.10,
            "up_down": 0.10,
        }
        composite_contrib = {
            "digit": round(0.40 * dig_s, 1),
            "streak": round(0.25 * st_s, 1),
            "odd_even": round(0.15 * oe_s, 1),
            "over_under": round(0.10 * ou_s, 1),
            "up_down": round(0.10 * ud_s, 1),
        }

        # Track last 20 composite readings for stability
        if not hasattr(self, "_composite_hist"):
            self._composite_hist: Deque[float] = deque(maxlen=20)
        self._composite_hist.append(float(composite))
        hist_vals = list(self._composite_hist)
        if len(hist_vals) >= 3:
            mean = sum(hist_vals) / len(hist_vals)
            var = sum((x - mean) ** 2 for x in hist_vals) / len(hist_vals)
            std = math.sqrt(var)
            if std <= 2:
                stability_score = 100.0 - std * 5.0
            elif std <= 8:
                stability_score = 90.0 - (std - 2) * 3.0
            elif std <= 20:
                stability_score = 72.0 - (std - 8) * 2.5
            else:
                stability_score = max(15.0, 42.0 - (std - 20) * 1.5)
            stability_score = min(100.0, max(0.0, stability_score))
        else:
            std = 0.0
            stability_score = 55.0

        # Entropy Clarity Engine: 60% composite + 25% momentum + 15% stability
        entropy_clarity = (
            0.60 * composite + 0.25 * momentum_score + 0.15 * stability_score
        )
        entropy_clarity = max(0.0, min(100.0, entropy_clarity))

        # Real-time pattern strength (compression-led)
        rt_strength = (
            0.40 * compression_score
            + 0.25 * momentum_score
            + 0.20 * deviation_score
            + 0.15 * st_s
        )
        rt_strength = max(0.0, min(100.0, rt_strength))

        velocity = float(velocities.get("50") or velocities.get("100") or 0)
        velocity_score = max(0.0, min(100.0, 50.0 - velocity * 200.0))

        regime = h50.get("regime") or h100.get("regime") or "RANDOM"
        confidence = min(
            100.0,
            0.45 * entropy_clarity
            + 0.35 * rt_strength
            + 0.20 * (100.0 if self._tick_count >= 100 else self._tick_count),
        )
        conf_label = (
            "HIGH" if confidence >= 80 else ("MEDIUM" if confidence >= 60 else "LOW")
        )

        triggers = self._trade_triggers(
            compression_pct=compression_pct,
            rt_strength=rt_strength,
            oe=oe,
            streak=streak,
            h100=h100,
        )

        level1 = {
            "digit": round(dig_s, 1),
            "streak": round(st_s, 1),
            "odd_even": round(oe_s, 1),
            "over_under": round(ou_s, 1),
            "up_down": round(ud_s, 1),
        }

        snap = {
            "tick_count": self._tick_count,
            "ready": self._tick_count >= 25,
            "windows": window_stats,
            "velocities": velocities,
            "primary": {
                "window": 50 if "50" in window_stats else 100,
                "entropy": h50.get("h") or h100.get("h"),
                "compression_pct": compression_pct,
                "bias_label": compression_bias_label(compression_pct),
                "regime": regime,
                "momentum_bits": round(momentum_bits, 4),
                "momentum_score": round(momentum_score, 1),
                "velocity": velocity,
                "velocity_score": round(velocity_score, 1),
            },
            "compression_score": round(compression_score, 1),
            "momentum_score": round(momentum_score, 1),
            "deviation_score": round(deviation_score, 1),
            "stability_score": round(stability_score, 1),
            "stability_stdev": round(std, 3),
            "streak": streak,
            # Level 1
            "level1": level1,
            "multi": level1,  # backward compatible
            # Level 2
            "composite_entropy": round(composite, 1),
            "composite_weights": composite_weights,
            "composite_contributions": composite_contrib,
            "entropy_clarity": round(entropy_clarity, 1),
            "entropy_clarity_blend": {
                "composite_60": round(0.60 * composite, 1),
                "momentum_25": round(0.25 * momentum_score, 1),
                "stability_15": round(0.15 * stability_score, 1),
            },
            "realtime_pattern_strength": round(rt_strength, 1),
            "regime": regime,
            "confidence": round(confidence, 1),
            "confidence_label": conf_label,
            "display": {
                "market_regime": regime,
                "entropy": h50.get("h") or h100.get("h"),
                "compression": f"{compression_pct:.1f}%",
                "entropy_clarity": round(entropy_clarity, 1),
                "composite_entropy": round(composite, 1),
                "confidence": conf_label,
            },
            "triggers": triggers,
            "odd_even_detail": oe,
            "over_under_detail": ou,
            "up_down_detail": ud,
            "contributors": [
                f"{'✓' if dig_s >= 65 else '✗'} Digit Bias Strength {dig_s:.0f}",
                f"{'✓' if st_s >= 65 else '✗'} Streak Compression {st_s:.0f}",
                f"{'✓' if oe_s >= 55 else '✗'} Odd/Even Entropy {oe_s:.0f}",
                f"{'✓' if ou_s >= 55 else '✗'} Over/Under Entropy {ou_s:.0f}",
                f"{'✓' if ud_s >= 50 else '✗'} Up/Down Entropy {ud_s:.0f}",
                f"{'✓' if momentum_score >= 65 else '✗'} Entropy Momentum {momentum_score:.0f}",
                f"{'✓' if stability_score >= 60 else '✗'} Entropy Stability {stability_score:.0f}",
            ],
        }
        self._last_snapshot = snap
        return snap

    @staticmethod
    def _trade_triggers(
        *,
        compression_pct: float,
        rt_strength: float,
        oe: Dict[str, Any],
        streak: Dict[str, Any],
        h100: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Entropy-based allow flags for digit contract families.
        """
        oe_comp = float(oe.get("compression_pct") or 0)
        streak_n = int(streak.get("current_streak") or 0)
        hot = h100.get("hottest")
        cold = h100.get("coldest")
        max_dev = float(h100.get("max_deviation_pct") or 0)

        differ_ok = compression_pct > 15 and rt_strength > 75
        match_ok = compression_pct > 20 and (
            streak_n >= 2 or max_dev >= 6
        )
        even_odd_ok = oe_comp >= 8  # significantly decreased odd/even entropy

        return {
            "DIGITDIFF": {
                "allow": differ_ok,
                "reason": (
                    f"compression {compression_pct:.1f}% "
                    f"{'>' if compression_pct > 15 else '≤'} 15% · "
                    f"strength {rt_strength:.0f} "
                    f"{'>' if rt_strength > 75 else '≤'} 75"
                ),
            },
            "DIGITMATCH": {
                "allow": match_ok,
                "reason": (
                    f"compression {compression_pct:.1f}% · "
                    f"streak {streak_n} · max_dev {max_dev:.1f}% · "
                    f"hot={hot} cold={cold}"
                ),
            },
            "DIGITEVEN": {
                "allow": even_odd_ok,
                "reason": f"odd/even compression {oe_comp:.1f}%",
            },
            "DIGITODD": {
                "allow": even_odd_ok,
                "reason": f"odd/even compression {oe_comp:.1f}%",
            },
            "DIGITOVER": {
                "allow": compression_pct > 12 and rt_strength >= 70,
                "reason": f"comp {compression_pct:.1f}% strength {rt_strength:.0f}",
            },
            "DIGITUNDER": {
                "allow": compression_pct > 12 and rt_strength >= 70,
                "reason": f"comp {compression_pct:.1f}% strength {rt_strength:.0f}",
            },
        }


# Process-wide engines keyed by symbol (for multi-market scan)
_engines: Dict[str, RollingEntropyEngine] = {}


def get_engine(symbol: str = "_default") -> RollingEntropyEngine:
    if symbol not in _engines:
        _engines[symbol] = RollingEntropyEngine()
    return _engines[symbol]


def feed_ticks(symbol: str, ticks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sync engine to latest ticks (efficient: only append new if buffer shorter,
    else full rebuild when history jumps).
    """
    eng = get_engine(symbol)
    if not ticks:
        return eng.snapshot()
    # If we have fewer ticks than buffer capacity, rebuild from all
    if eng._tick_count < 20 or eng._tick_count + 5 < len(ticks):
        eng.reset()
        return eng.bootstrap_from_ticks(ticks)
    # Append only the newest tick if one new arrived
    last = ticks[-1]
    return eng.add_tick(last if isinstance(last, dict) else {"digit": last})
