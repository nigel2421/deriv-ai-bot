"""
Convert AI (or heuristic) predictions into Deriv digit trade signals.

Supports: DIGITOVER, DIGITUNDER, DIGITEVEN, DIGITODD, DIGITMATCH, DIGITDIFF
with barriers that match Deriv win conditions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategy.digit_contracts import (
    ALL_DIGIT_TYPES,
    barrier_for_predicted_digit,
    clamp_digit,
    extract_last_digit,
    last_digits_from_ticks,
    normalize_barrier,
    normalize_contract_type,
    validate_digit_contract,
    would_win,
)

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Converts predictions to (contract_type, barrier, confidence)."""

    def __init__(self, prefer_parity: bool = False):
        """
        prefer_parity: if True, bias EVEN/ODD when parity confidence is strong.
        """
        self.prefer_parity = prefer_parity

    def generate_signal(
        self,
        prediction: Dict[str, Any],
        confidence: float,
        min_confidence: float = 0.75,
        allowed_types: Optional[Sequence[str]] = None,
    ) -> Tuple[Optional[str], Optional[int], float]:
        """
        Build a trade signal from a prediction dict.

        prediction keys (optional):
          digit: int 0-9 predicted next last digit
          parity: bool True=even
          preferred_type: force a contract family
          recent_ticks: optional ticks for digit stats fallback
        """
        if confidence is None or confidence < min_confidence:
            return None, None, 0.0

        allowed = self._normalize_allowed(allowed_types)
        predicted_digit = self._resolve_digit(prediction)
        parity_even = self._resolve_parity(prediction, predicted_digit)

        preferred = normalize_contract_type(prediction.get("preferred_type"))
        contract_type = self._select_contract_type(
            predicted_digit=predicted_digit,
            parity_even=parity_even,
            confidence=float(confidence),
            allowed=allowed,
            preferred=preferred,
        )
        if not contract_type:
            logger.debug("No valid contract type for allowed=%s", allowed)
            return None, None, 0.0

        barrier = self._resolve_barrier(
            contract_type, predicted_digit, prediction.get("barrier")
        )

        ok, reason, barrier = validate_digit_contract(contract_type, barrier)
        if not ok:
            logger.warning("Invalid signal %s barrier=%s: %s", contract_type, barrier, reason)
            return None, None, 0.0

        logger.info(
            "Signal: %s barrier=%s digit=%s parity=%s conf=%.1f%%",
            contract_type,
            barrier,
            predicted_digit,
            "even" if parity_even else "odd",
            float(confidence) * 100,
        )
        return contract_type, barrier, float(confidence)

    def parity_confidence(
        self, ticks: Sequence[Dict[str, Any]], lookback: int = 30
    ) -> Tuple[Dict[str, Any], float]:
        """
        Even/Odd streak analysis from recent last digits.
        Returns ({even: bool, even_rate, streak, n}, confidence 0..1).
        """
        digits = last_digits_from_ticks(ticks, n=lookback)
        if len(digits) < 8:
            return {"even": True, "even_rate": 0.5, "streak": 0, "n": len(digits)}, 0.0

        even_n = sum(1 for d in digits if d % 2 == 0)
        even_rate = even_n / len(digits)
        # Current streak of same parity at the end
        last_even = digits[-1] % 2 == 0
        streak = 0
        for d in reversed(digits):
            if (d % 2 == 0) == last_even:
                streak += 1
            else:
                break

        # Bias toward dominant parity; boost on streak 3+ of same parity.
        # CAP conf — live audit showed DIGITEVEN/ODD at 0.93–0.99 with ~20% WR.
        # True edge on parity is modest; never claim near-certainty.
        dominant_even = even_rate >= 0.5
        dom_rate = even_rate if dominant_even else (1.0 - even_rate)
        conf = 0.50 + (dom_rate - 0.5) * 0.9  # 0.5→0.5, 0.7→0.68
        if streak >= 3 and ((last_even and dominant_even) or (not last_even and not dominant_even)):
            conf += min(0.08, 0.02 * streak)
        # Fade if alternating (low streak despite bias)
        if streak <= 1 and 0.45 < even_rate < 0.55:
            conf *= 0.75
        # Hard ceiling: parity is not a 95%+ edge
        conf = min(0.82, conf)
        conf = max(0.0, min(0.95, conf))
        return {
            "even": dominant_even,
            "even_rate": even_rate,
            "streak": streak,
            "n": len(digits),
            "last_even": last_even,
        }, float(conf)

    def generate_from_ticks(
        self,
        ticks: Sequence[Dict[str, Any]],
        min_confidence: float = 0.75,
        allowed_types: Optional[Sequence[str]] = None,
        lookback: int = 25,
    ) -> Tuple[Optional[str], Optional[int], float, Dict[str, Any]]:
        """
        Heuristic signal from recent last-digit frequencies (no ML required).
        Returns (type, barrier, confidence, stats).
        """
        digits = last_digits_from_ticks(ticks, n=lookback)
        stats = self.digit_stats(digits)
        if not digits:
            return None, None, 0.0, stats

        # Mode digit as prediction
        pred_digit = max(range(10), key=lambda d: stats["counts"][d])
        even_share = stats["even_rate"]
        conf_digit = stats["counts"][pred_digit] / max(1, len(digits))
        conf_parity = max(even_share, 1.0 - even_share)

        # Use stronger of mode-digit vs parity signal
        if conf_parity >= conf_digit and conf_parity >= min_confidence:
            prediction = {
                "digit": pred_digit,
                "parity": even_share >= 0.5,
                "preferred_type": "DIGITEVEN" if even_share >= 0.5 else "DIGITODD",
            }
            conf = conf_parity
        else:
            prediction = {"digit": pred_digit, "parity": pred_digit % 2 == 0}
            conf = conf_digit

        ct, barrier, conf_out = self.generate_signal(
            prediction, conf, min_confidence, allowed_types
        )
        return ct, barrier, conf_out, stats

    @staticmethod
    def digit_stats(digits: Sequence[int]) -> Dict[str, Any]:
        counts = [0] * 10
        for d in digits:
            counts[clamp_digit(d)] += 1
        n = len(digits) or 1
        even = sum(counts[i] for i in range(0, 10, 2))
        return {
            "counts": counts,
            "n": len(digits),
            "even_rate": even / n,
            "odd_rate": 1.0 - (even / n),
            "mode": max(range(10), key=lambda i: counts[i]) if digits else None,
        }

    def _normalize_allowed(
        self, allowed_types: Optional[Sequence[str]]
    ) -> List[str]:
        if not allowed_types:
            return sorted(ALL_DIGIT_TYPES)
        out = []
        for t in allowed_types:
            ct = normalize_contract_type(t)
            if ct:
                out.append(ct)
        return out or sorted(ALL_DIGIT_TYPES)

    def _resolve_digit(self, prediction: Dict[str, Any]) -> int:
        if "digit" in prediction and prediction["digit"] is not None:
            return clamp_digit(prediction["digit"])
        # Fallback: last tick quote
        ticks = prediction.get("recent_ticks") or []
        if ticks:
            d = extract_last_digit(ticks[-1].get("quote"))
            if d is not None:
                return d
        return 5

    def _resolve_parity(
        self, prediction: Dict[str, Any], predicted_digit: int
    ) -> bool:
        if "parity" in prediction and prediction["parity"] is not None:
            p = prediction["parity"]
            if isinstance(p, str):
                return p.lower() in {"even", "e", "true", "1"}
            return bool(p)
        return predicted_digit % 2 == 0

    def _select_contract_type(
        self,
        predicted_digit: int,
        parity_even: bool,
        confidence: float,
        allowed: List[str],
        preferred: Optional[str],
    ) -> Optional[str]:
        if preferred and preferred in allowed:
            return preferred

        candidates: List[Tuple[float, str]] = []

        # Parity contracts (Even/Odd) — strong when parity signal aligns
        if "DIGITEVEN" in allowed:
            score = confidence * (1.05 if parity_even else 0.28)
            if self.prefer_parity and parity_even:
                score += 0.08
            candidates.append((score, "DIGITEVEN"))
        if "DIGITODD" in allowed:
            score = confidence * (1.05 if not parity_even else 0.28)
            if self.prefer_parity and not parity_even:
                score += 0.08
            candidates.append((score, "DIGITODD"))

        # Over/Under: high digits favor OVER@6 (7-9), low favor UNDER@4 (0-3)
        if "DIGITOVER" in allowed:
            # Stronger when predicted digit is in the OVER win set (7-9)
            over_bias = 0.85 if predicted_digit >= 7 else (
                0.55 if predicted_digit >= 5 else 0.25
            )
            candidates.append((confidence * over_bias, "DIGITOVER"))
        if "DIGITUNDER" in allowed:
            under_bias = 0.85 if predicted_digit <= 3 else (
                0.55 if predicted_digit <= 5 else 0.25
            )
            candidates.append((confidence * under_bias, "DIGITUNDER"))

        # Match is high-payout / low-prob — only if strong confidence
        if "DIGITMATCH" in allowed:
            candidates.append((confidence * 0.45, "DIGITMATCH"))
        if "DIGITDIFF" in allowed:
            candidates.append((confidence * 0.55, "DIGITDIFF"))

        if not candidates:
            # Fallbacks in priority order
            for fallback in (
                "DIGITEVEN",
                "DIGITODD",
                "DIGITOVER",
                "DIGITUNDER",
                "DIGITDIFF",
                "DIGITMATCH",
            ):
                if fallback in allowed:
                    return fallback
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _resolve_barrier(
        self,
        contract_type: str,
        predicted_digit: int,
        explicit_barrier: Any,
    ) -> Optional[int]:
        if explicit_barrier is not None:
            return normalize_barrier(contract_type, clamp_digit(explicit_barrier))
        return barrier_for_predicted_digit(contract_type, predicted_digit)

    def explain(
        self,
        contract_type: str,
        barrier: Optional[int],
        sample_digit: int,
    ) -> str:
        win = would_win(contract_type, barrier, sample_digit)
        return (
            f"{contract_type} barrier={barrier}: digit {sample_digit} → "
            f"{'WIN' if win else 'LOSS'}"
        )
