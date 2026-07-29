"""
Calibration Tracker — Rec #8

Tracks predicted confidence vs actual win rate across 5 buckets.
Phase 1: Display + alert only. No auto-deflation.
Phase 2: Auto-deflation enabled after:
    - cumulative_trades >= 1000
    - calibration_error > 15% for 3 consecutive audits

Persistence: data/calibration_state.json

Schema:
{
  "buckets": {
    "50-60": {"predicted_sum": 85.3, "n": 14, "wins": 8,
              "consecutive_overconfident": 0},
    "60-70": ...,
    "70-80": ...,
    "80-90": ...,
    "90+":   {"predicted_sum": 194.2, "n": 21, "wins": 12,
              "consecutive_overconfident": 3}
  },
  "cumulative_trades": 234,
  "auto_deflation_enabled": false,
  "calibration_factors": {"90+": 0.78},
  "updated_at": 1750000000.0
}
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/calibration_state.json")

BUCKETS = ["50-60", "60-70", "70-80", "80-90", "90+"]

# Phase 2 thresholds
PHASE2_MIN_TRADES = 1000
PHASE2_ERROR_THRESHOLD = 0.15
PHASE2_CONSECUTIVE_AUDITS = 3


def _bucket_for(confidence: float) -> str:
    pct = confidence * 100
    if pct < 60:
        return "50-60"
    if pct < 70:
        return "60-70"
    if pct < 80:
        return "70-80"
    if pct < 90:
        return "80-90"
    return "90+"


def _empty_bucket() -> Dict[str, Any]:
    return {
        "predicted_sum": 0.0,
        "n": 0,
        "wins": 0,
        "consecutive_overconfident": 0,
    }


class CalibrationTracker:
    """
    Tracks whether the bot's predicted confidence matches actual win rates.

    Phase 1: Alert only — no automatic confidence adjustment.
    Phase 2: After sufficient data, applies a calibration factor to
             raw confidence when overconfidence is persistent.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.buckets: Dict[str, Dict[str, Any]] = {b: _empty_bucket() for b in BUCKETS}
        self.cumulative_trades: int = 0
        self.auto_deflation_enabled: bool = False
        self.calibration_factors: Dict[str, float] = {}
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.buckets = data.get("buckets") or {b: _empty_bucket() for b in BUCKETS}
            self.cumulative_trades = int(data.get("cumulative_trades") or 0)
            self.auto_deflation_enabled = bool(data.get("auto_deflation_enabled", False))
            self.calibration_factors = data.get("calibration_factors") or {}
            logger.info(
                "CalibrationTracker loaded %d trades from %s", self.cumulative_trades, self.path
            )
        except Exception as e:
            logger.warning("CalibrationTracker load failed: %s", e)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "buckets": self.buckets,
                "cumulative_trades": self.cumulative_trades,
                "auto_deflation_enabled": self.auto_deflation_enabled,
                "calibration_factors": self.calibration_factors,
                "updated_at": time.time(),
            }
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("CalibrationTracker save failed: %s", e)

    def record(self, confidence: float, is_win: bool) -> None:
        """Record a settled trade for calibration tracking."""
        bucket = _bucket_for(confidence)
        b = self.buckets.setdefault(bucket, _empty_bucket())
        b["predicted_sum"] = float(b.get("predicted_sum", 0.0)) + confidence
        b["n"] = int(b.get("n", 0)) + 1
        if is_win:
            b["wins"] = int(b.get("wins", 0)) + 1
        self.cumulative_trades += 1
        self.save()

    def actual_win_rate(self, bucket: str) -> Optional[float]:
        b = self.buckets.get(bucket, {})
        n = int(b.get("n", 0))
        if n == 0:
            return None
        return round(int(b.get("wins", 0)) / n, 4)

    def avg_predicted(self, bucket: str) -> Optional[float]:
        b = self.buckets.get(bucket, {})
        n = int(b.get("n", 0))
        if n == 0:
            return None
        return round(float(b.get("predicted_sum", 0.0)) / n, 4)

    def calibration_error(self, bucket: str) -> Optional[float]:
        """
        |avg_predicted - actual_win_rate|.
        Positive error = overconfident (predicted > actual).
        Returns None if insufficient data.
        """
        pred = self.avg_predicted(bucket)
        actual = self.actual_win_rate(bucket)
        if pred is None or actual is None:
            return None
        return round(pred - actual, 4)  # positive = overconfident

    def overall_error(self) -> Optional[float]:
        """Weighted mean absolute calibration error across all buckets with data."""
        total_n = 0
        weighted_err = 0.0
        for bucket in BUCKETS:
            err = self.calibration_error(bucket)
            n = int(self.buckets.get(bucket, {}).get("n", 0))
            if err is not None and n > 0:
                weighted_err += abs(err) * n
                total_n += n
        if total_n == 0:
            return None
        return round(weighted_err / total_n, 4)

    def is_healthy(self, bucket: str) -> bool:
        """
        Phase 8/10/16: Returns False if calibration error >= 15% (SEVERELY OVERCONFIDENT).
        """
        err = self.calibration_error(bucket)
        if err is None:
            return True # allow if insufficient data
        return err < PHASE2_ERROR_THRESHOLD

    def status_for(self, bucket: str) -> Tuple[str, str]:
        """
        Phase 1 display labels (no probability mutation).

        status_code:
          "good" | "watch" | "overconfident" | "severely_overconfident"
          | "underconfident" | "insufficient"
        """
        err = self.calibration_error(bucket)
        n = int(self.buckets.get(bucket, {}).get("n", 0))
        if n < 10:
            return "insufficient", "Insufficient data"
        if err is None:
            return "insufficient", "Insufficient data"
        abs_err = abs(err)
        if abs_err < 0.05:
            return "good", "Good"
        if abs_err < 0.10:
            return "watch", "Watch"
        if err > 0:
            # Display-only severity: >15% matches Phase 2 error threshold
            if err > PHASE2_ERROR_THRESHOLD:
                return "severely_overconfident", "SEVERELY OVERCONFIDENT"
            return "overconfident", "OVERCONFIDENT"
        return "underconfident", "Underconfident"

    def apply_calibration(self, confidence: float) -> float:
        """
        Phase 2 only: apply calibration factor to raw confidence.
        Phase 1: returns raw confidence unchanged.

        Safety rule: only deflate if auto_deflation_enabled=True,
        which requires cumulative_trades > 1000 AND error persisted.
        """
        if not self.auto_deflation_enabled:
            return confidence
        bucket = _bucket_for(confidence)
        factor = self.calibration_factors.get(bucket)
        if factor is None:
            return confidence
        adjusted = confidence * factor
        logger.debug(
            "Calibration deflation %s: %.3f * %.3f = %.3f",
            bucket, confidence, factor, adjusted,
        )
        return round(max(0.0, min(1.0, adjusted)), 4)

    def audit_and_maybe_enable_deflation(self) -> None:
        """
        Called after each AI audit (every 100 closed trades).

        Phase 2 auto-deflation gates (approved):
            cumulative_trades > 1000
            AND calibration_error > 15%
            AND that condition holds for 3 consecutive audits
        """
        if self.cumulative_trades <= PHASE2_MIN_TRADES:
            return

        updated = False
        for bucket in BUCKETS:
            err = self.calibration_error(bucket)
            n = int(self.buckets.get(bucket, {}).get("n", 0))
            if err is None or n < 50:
                continue
            b = self.buckets[bucket]
            if err > PHASE2_ERROR_THRESHOLD:
                # Overconfident beyond 15% — increment consecutive audit counter
                b["consecutive_overconfident"] = int(b.get("consecutive_overconfident", 0)) + 1
            else:
                b["consecutive_overconfident"] = 0

            consec = int(b.get("consecutive_overconfident", 0))
            if consec >= PHASE2_CONSECUTIVE_AUDITS:
                # Activate deflation for this bucket:
                # Adjusted = Raw * (actual / predicted)
                actual = self.actual_win_rate(bucket)
                pred = self.avg_predicted(bucket)
                if pred and pred > 0 and actual is not None:
                    factor = round(actual / pred, 4)
                    # Never inflate confidence; only deflate
                    factor = min(1.0, max(0.1, factor))
                    self.calibration_factors[bucket] = factor
                    self.auto_deflation_enabled = True
                    updated = True
                    logger.warning(
                        "Calibration Phase 2 ACTIVATED for bucket %s: "
                        "factor=%.4f (predicted=%.3f actual=%.3f, consec=%d)",
                        bucket, factor, pred, actual, consec,
                    )

        if updated:
            self.save()
        else:
            # Persist consecutive counters even when not yet activated
            self.save()

    def snapshot(self) -> Dict[str, Any]:
        rows = []
        for bucket in BUCKETS:
            pred = self.avg_predicted(bucket)
            actual = self.actual_win_rate(bucket)
            err = self.calibration_error(bucket)
            n = int(self.buckets.get(bucket, {}).get("n", 0))
            status_code, status_label = self.status_for(bucket)
            rows.append({
                "bucket": bucket,
                "predicted_avg": round(pred * 100, 1) if pred else None,
                "actual_wr": round(actual * 100, 1) if actual else None,
                "error": round(err * 100, 1) if err is not None else None,
                "n": n,
                "status": status_label,
                "status_code": status_code,
                "factor": self.calibration_factors.get(bucket),
            })
        return {
            "rows": rows,
            "overall_error": self.overall_error(),
            "cumulative_trades": self.cumulative_trades,
            "auto_deflation_enabled": self.auto_deflation_enabled,
        }
