"""
Online learning from closed trade outcomes.

Tracks win-rate and streak per (symbol, contract_type) and produces:
  - confidence multipliers (boost winners, cut losers)
  - market preference weights for multi-symbol selection
  - temporary skip flags after cold streaks

State is persisted under data/learning_state.json (ephemeral on Cloud Run
unless volume attached — still useful within a session / warm instance).

Enhancements (Recs #1, #2, #6):
  - confidence_level(): LOW / MEDIUM / HIGH based on trade count
  - historical_support(): total settled trades for a setup
  - record_pattern_strength(): tracks signal quality trend over time (per scan)
  - pattern_decay(): current - historical average strength
  - decay_status(): tiered status (Healthy / Watch / Warning / Block)
  - should_block_for_decay(): hard block when decay < -20 AND clarity < 0.75
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/learning_state.json")

# Pattern decay tiered thresholds (Rec #6)
DECAY_WATCH = -10.0
DECAY_WARNING = -15.0
DECAY_BLOCK = -20.0

# Confidence level thresholds (Rec #1)
CONFIDENCE_LOW_MAX = 30      # < 30 trades -> LOW
CONFIDENCE_MEDIUM_MAX = 100  # 30..99 trades -> MEDIUM, >=100 -> HIGH


def _key(symbol: str, contract_type: str) -> str:
    return f"{symbol}|{str(contract_type).upper()}"


class AdaptiveLearner:
    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        min_samples: int = 2,
        cold_streak_skip: int = 4,
        decay: float = 0.995,
        always_on: bool = True,
    ):
        self.path = Path(path) if path else DEFAULT_PATH
        self.min_samples = min_samples
        self.cold_streak_skip = cold_streak_skip
        self.decay = decay
        self.always_on = always_on
        # stats[key] = {wins, losses, streak_loss, streak_win, pnl, last_ts,
        #               conf_sum, conf_n, family, strength_history}
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
            self.total_recorded = int(data.get("total_recorded") or 0)
            logger.info(
                "AdaptiveLearner loaded %d keys (W=%s L=%s) from %s",
                len(self.stats),
                self.global_wins,
                self.global_losses,
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
        except Exception as e:
            logger.debug("AdaptiveLearner save failed: %s", e)

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
                "strength_history": [],
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
        if streak >= self.cold_streak_skip:
            return True, f"cold_streak_{streak}"
        # Skip chronically bad combos with enough samples
        n = self.samples(symbol, contract_type)
        wr = self.win_rate(symbol, contract_type)
        # Ban chronically bad setups earlier (stop re-entering loss pits)
        if n >= max(self.min_samples, 5) and wr < 0.40:
            return True, f"low_winrate_{wr:.0%}_n={n}"
        if n >= 3 and wr <= 0.0:
            return True, f"all_losses_n={n}"
        return False, ""

    def confidence_multiplier(self, symbol: str, contract_type: str) -> float:
        """
        Multiplier applied to raw confidence.
        Always-on: even 1 sample nudges slightly; more samples -> stronger effect.
        """
        n = self.samples(symbol, contract_type)
        wr = self.win_rate(symbol, contract_type)
        if n == 0:
            return 1.0

        # Trust grows with samples (always-on learning)
        trust = min(1.0, n / 12.0) if n >= self.min_samples else min(0.35, n / 8.0)
        # wr 0.5 -> 1.0, wr 0.7 -> ~1.15, wr 0.3 -> ~0.82
        raw = 0.72 + wr * 0.55
        mult = 1.0 + (raw - 1.0) * trust

        s = self.stats.get(_key(symbol, contract_type)) or {}
        if int(s.get("streak_win", 0)) >= 3:
            mult *= 1.06
        if int(s.get("streak_loss", 0)) >= 2:
            mult *= 0.90

        return float(max(0.65, min(1.28, mult)))

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
        """Small additive score for trade selector (0..0.08)."""
        n = self.samples(symbol, contract_type)
        if n < self.min_samples:
            return 0.0
        wr = self.win_rate(symbol, contract_type)
        return max(0.0, min(0.08, (wr - 0.5) * 0.2))

    # -------------------------------------------------------------------------
    # Rec #1 — Probability Confidence Level
    # -------------------------------------------------------------------------

    def confidence_level(self, symbol: str, contract_type: str) -> str:
        """
        Returns LOW / MEDIUM / HIGH based on settled trade count for this setup.

        LOW:    < 30 trades  (predictions unreliable — cold start)
        MEDIUM: 30–99 trades (learning in progress)
        HIGH:   >= 100 trades (statistically meaningful)
        """
        n = self.samples(symbol, contract_type)
        if n < CONFIDENCE_LOW_MAX:
            return "LOW"
        if n < CONFIDENCE_MEDIUM_MAX:
            return "MEDIUM"
        return "HIGH"

    def historical_support(self, symbol: str, contract_type: str) -> int:
        """Total settled trades for this symbol|contract_type setup."""
        return self.samples(symbol, contract_type)

    # -------------------------------------------------------------------------
    # Rec #6 — Pattern Decay Tracking
    # -------------------------------------------------------------------------

    def record_pattern_strength(
        self, symbol: str, contract_type: str, strength: float
    ) -> None:
        """
        Called at scan time (not trade time) with the current signal strength.
        Maintains a rolling window of the last 100 strength readings per setup.

        'strength' should be the normalized signal confidence or strength score
        scaled to [0, 100] for intuitive decay display.
        """
        k = _key(symbol, contract_type)
        s = self.stats.setdefault(
            k,
            {
                "wins": 0, "losses": 0, "streak_loss": 0, "streak_win": 0,
                "pnl": 0.0, "last_ts": 0.0, "conf_sum": 0.0, "conf_n": 0,
                "family": "", "strength_history": [],
            },
        )
        hist = s.setdefault("strength_history", [])
        hist.append(round(float(strength) * 100, 2))  # store as 0-100
        if len(hist) > 100:
            s["strength_history"] = hist[-100:]
        # Do NOT save here — called every 60s scan cycle, save on record() only

    def pattern_decay(self, symbol: str, contract_type: str) -> float:
        """
        Decay = current_avg (last 10) - historical_avg (all prior readings).

        Negative = strength is declining (decaying edge).
        Positive = strength is improving.
        Returns 0.0 if insufficient data (< 20 readings).

        Values are in the same scale as strength_history (0-100).
        """
        k = _key(symbol, contract_type)
        hist = self.stats.get(k, {}).get("strength_history", [])
        if len(hist) < 20:
            return 0.0
        baseline = sum(hist[:-10]) / len(hist[:-10])
        current = sum(hist[-10:]) / 10
        return round(current - baseline, 2)

    def decay_status(self, symbol: str, contract_type: str) -> Tuple[str, str]:
        """
        Returns (status_code, display_label) based on tiered thresholds.

        Status codes: "healthy" | "watch" | "warning" | "block" | "unknown"
        """
        hist = self.stats.get(_key(symbol, contract_type), {}).get("strength_history", [])
        if len(hist) < 20:
            return "unknown", "Unknown (insufficient data)"

        # Tiered thresholds (approved):
        #   > -10 healthy | -10..-15 watch | -15..-20 warning | < -20 block
        decay = self.pattern_decay(symbol, contract_type)
        if decay > DECAY_WATCH:
            return "healthy", f"Healthy ({decay:+.1f})"
        if decay > DECAY_WARNING:
            return "watch", f"Watch ({decay:+.1f})"
        if decay >= DECAY_BLOCK:
            return "warning", f"Warning ({decay:+.1f})"
        return "block", f"Block ({decay:+.1f})"

    def should_block_for_decay(
        self,
        symbol: str,
        contract_type: str,
        current_strength: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Hard block rule (Rec #6 production rule):
            Decay < -20 AND current pattern clarity < 0.75

        'current_strength' is the latest raw signal confidence [0,1].
        If None, uses the last recorded strength_history value as proxy.

        Returns (should_block, reason_string).
        """
        hist = self.stats.get(_key(symbol, contract_type), {}).get("strength_history", [])
        if len(hist) < 20:
            return False, ""

        decay = self.pattern_decay(symbol, contract_type)
        if decay >= DECAY_BLOCK:
            return False, ""

        # Decay threshold crossed — now check clarity
        if current_strength is not None:
            clarity = float(current_strength)
        elif hist:
            clarity = hist[-1] / 100.0  # last recorded strength as proxy
        else:
            clarity = 1.0

        if clarity < 0.75:
            reason = (
                f"Pattern decay {decay:+.1f} (< {DECAY_BLOCK}) "
                f"AND clarity {clarity:.2f} (< 0.75) — edge dying"
            )
            logger.info("PatternDecay BLOCK %s|%s: %s", symbol, contract_type, reason)
            return True, reason

        # Decay is severe but clarity is still OK — warn but allow
        return False, f"decay_warning_{decay:+.1f}_clarity_ok_{clarity:.2f}"

    def snapshot(self) -> Dict[str, Any]:
        top_entries = []
        for k, v in self.stats.items():
            n = int(v.get("wins", 0)) + int(v.get("losses", 0))
            try:
                sym, ct = k.split("|", 1)
            except ValueError:
                sym, ct = k, ""
            top_entries.append({
                "key": k,
                "wins": v.get("wins"),
                "losses": v.get("losses"),
                "pnl": round(float(v.get("pnl") or 0), 2),
                "streak_loss": v.get("streak_loss"),
                "family": v.get("family"),
                "confidence_level": self.confidence_level(sym, ct),
                "historical_support": n,
                "decay_status": self.decay_status(sym, ct)[1],
            })

        return {
            "always_on": self.always_on,
            "path": str(self.path),
            "global_wins": self.global_wins,
            "global_losses": self.global_losses,
            "total_recorded": self.total_recorded,
            "keys": len(self.stats),
            "top": sorted(
                top_entries,
                key=lambda x: -(float(x.get("pnl") or 0)),
            )[:12],
        }
