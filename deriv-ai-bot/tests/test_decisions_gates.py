"""
Unit tests for the three approved production decisions:

1. Pattern decay tiered thresholds + hard block (decay < -20 AND clarity < 0.75)
2. AI Auditor frequency: persistent cumulative closes (100 minor / 1000 major)
3. Calibration: Phase 1 display-only; Phase 2 auto-deflation gates
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.strategy.adaptive_learner import (
    DECAY_BLOCK,
    DECAY_WARNING,
    DECAY_WATCH,
    AdaptiveLearner,
)
from src.strategy.ai_auditor import (
    MAJOR_INTERVAL,
    MINOR_INTERVAL,
    AIAuditor,
)
from src.strategy.calibration_tracker import (
    PHASE2_CONSECUTIVE_AUDITS,
    PHASE2_ERROR_THRESHOLD,
    PHASE2_MIN_TRADES,
    CalibrationTracker,
)


# ---------------------------------------------------------------------------
# 1. Pattern Decay
# ---------------------------------------------------------------------------

class TestPatternDecayThresholds:
    def test_constants_match_approved_decisions(self):
        assert DECAY_WATCH == -10.0
        assert DECAY_WARNING == -15.0
        assert DECAY_BLOCK == -20.0

    def _learner_with_history(self, hist, tmp_path: Path) -> AdaptiveLearner:
        learner = AdaptiveLearner(path=tmp_path / "learn.json")
        learner.stats["R_100|DIGITOVER"] = {
            "wins": 0,
            "losses": 0,
            "streak_loss": 0,
            "streak_win": 0,
            "pnl": 0.0,
            "last_ts": 0.0,
            "conf_sum": 0.0,
            "conf_n": 0,
            "family": "digits",
            "strength_history": list(hist),
        }
        return learner

    def test_decay_formula_current_minus_historical(self, tmp_path: Path):
        # 20 baseline @ 88, 10 current @ 66 → decay = -22
        hist = [88.0] * 20 + [66.0] * 10
        learner = self._learner_with_history(hist, tmp_path)
        decay = learner.pattern_decay("R_100", "DIGITOVER")
        assert decay == pytest.approx(-22.0, abs=0.01)

    def test_tiered_status_labels(self, tmp_path: Path):
        # Build histories that yield known decays via formula
        # healthy: current slightly above baseline
        healthy = [50.0] * 20 + [55.0] * 10  # decay = +5
        # watch: decay ~ -12
        watch = [80.0] * 20 + [68.0] * 10  # -12
        # warning: decay ~ -17
        warning = [80.0] * 20 + [63.0] * 10  # -17
        # block status (display): decay ~ -22
        block = [88.0] * 20 + [66.0] * 10  # -22

        cases = [
            (healthy, "healthy"),
            (watch, "watch"),
            (warning, "warning"),
            (block, "block"),
        ]
        for hist, expected in cases:
            learner = self._learner_with_history(hist, tmp_path)
            code, _label = learner.decay_status("R_100", "DIGITOVER")
            assert code == expected, f"hist decay status expected {expected}, got {code}"

    def test_block_requires_decay_and_clarity(self, tmp_path: Path):
        hist = [88.0] * 20 + [66.0] * 10  # decay -22
        learner = self._learner_with_history(hist, tmp_path)

        # Decay severe + clarity low → BLOCK
        blocked, reason = learner.should_block_for_decay(
            "R_100", "DIGITOVER", current_strength=0.66
        )
        assert blocked is True
        assert "clarity" in reason.lower() or "0.66" in reason

        # Decay severe but clarity still high → allow (warn only)
        blocked2, reason2 = learner.should_block_for_decay(
            "R_100", "DIGITOVER", current_strength=0.90
        )
        assert blocked2 is False
        assert "clarity_ok" in reason2

    def test_exactly_minus_20_is_not_hard_block(self, tmp_path: Path):
        # Force decay == -20 by construction: baseline 80, current 60
        hist = [80.0] * 20 + [60.0] * 10
        learner = self._learner_with_history(hist, tmp_path)
        decay = learner.pattern_decay("R_100", "DIGITOVER")
        assert decay == pytest.approx(-20.0, abs=0.01)
        # Production rule is Decay < -20, so -20 alone must not hard-block
        blocked, _ = learner.should_block_for_decay(
            "R_100", "DIGITOVER", current_strength=0.50
        )
        assert blocked is False
        code, _ = learner.decay_status("R_100", "DIGITOVER")
        assert code == "warning"


# ---------------------------------------------------------------------------
# 2. AI Auditor frequency
# ---------------------------------------------------------------------------

class TestAIAuditorFrequency:
    def test_intervals_match_approved_decisions(self):
        assert MINOR_INTERVAL == 100
        assert MAJOR_INTERVAL == 1000

    def test_triggers_on_persistent_cumulative_count(self, tmp_path: Path, monkeypatch):
        history = tmp_path / "trade_history.jsonl"
        report = tmp_path / "auditor_report.json"
        # Seed enough fake trades so audits can load data
        rows = [
            {
                "confidence": 0.8 + (i % 10) * 0.01,
                "is_win": i % 2 == 0,
                "mor_score": 70 + (i % 20),
                "contract_type": "DIGITOVER",
                "symbol": "R_100",
                "profit": 1.0 if i % 2 == 0 else -1.0,
            }
            for i in range(1000)
        ]
        history.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

        auditor = AIAuditor(history_path=history, report_path=report)
        monkeypatch.setattr(
            "src.strategy.ai_auditor.HISTORY_LOG_PATH",
            tmp_path / "auditor_history.jsonl",
        )
        auditor.history_log_path = tmp_path / "auditor_history.jsonl"

        assert auditor.check_and_run(0) is None
        assert auditor.check_and_run(50) is None
        assert auditor.check_and_run(99) is None

        minor = auditor.check_and_run(100)
        assert minor is not None
        assert minor.get("type") == "minor"

        assert auditor.check_and_run(200) is not None

        major = auditor.check_and_run(1000)
        assert major is not None
        assert major.get("type") == "major"
        assert "mor_validation" in major
        assert "best_setups" in major


# ---------------------------------------------------------------------------
# 3. Calibration Phase 1 / Phase 2
# ---------------------------------------------------------------------------

class TestCalibrationGates:
    def test_phase2_constants(self):
        assert PHASE2_MIN_TRADES == 1000
        assert PHASE2_ERROR_THRESHOLD == 0.15
        assert PHASE2_CONSECUTIVE_AUDITS == 3

    def test_phase1_does_not_mutate_confidence(self, tmp_path: Path):
        cal = CalibrationTracker(path=tmp_path / "cal.json")
        # Simulate severe overconfidence with small sample (Phase 1)
        for _ in range(20):
            cal.record(0.97, is_win=False)  # predicted 97%, always lose
        assert cal.auto_deflation_enabled is False
        assert cal.apply_calibration(0.97) == 0.97

        code, label = cal.status_for("90+")
        assert code == "severely_overconfident"
        assert "SEVERELY OVERCONFIDENT" in label

    def test_phase2_requires_sample_error_and_three_audits(self, tmp_path: Path):
        cal = CalibrationTracker(path=tmp_path / "cal.json")

        # Seed 1001 overconfident trades in 90+ bucket
        for i in range(1001):
            # ~56% actual win rate at ~97% predicted → error ~0.41
            cal.record(0.97, is_win=(i % 100) < 56)

        assert cal.cumulative_trades > PHASE2_MIN_TRADES
        err = cal.calibration_error("90+")
        assert err is not None and err > PHASE2_ERROR_THRESHOLD

        # Still Phase 1 until 3 consecutive audits
        cal.audit_and_maybe_enable_deflation()
        assert cal.auto_deflation_enabled is False
        cal.audit_and_maybe_enable_deflation()
        assert cal.auto_deflation_enabled is False
        cal.audit_and_maybe_enable_deflation()
        assert cal.auto_deflation_enabled is True

        # Factor ≈ actual/predicted < 1 → deflates
        adjusted = cal.apply_calibration(0.97)
        assert adjusted < 0.97
        assert adjusted == pytest.approx(0.97 * cal.calibration_factors["90+"], abs=1e-3)

    def test_error_reset_breaks_consecutive_streak(self, tmp_path: Path):
        cal = CalibrationTracker(path=tmp_path / "cal.json")
        for i in range(1001):
            cal.record(0.97, is_win=(i % 100) < 56)

        cal.audit_and_maybe_enable_deflation()
        cal.audit_and_maybe_enable_deflation()
        # Inject a "good" audit by zeroing the error path: fill a well-calibrated bucket
        # Easier: manually reset consecutive via a temporary well-calibrated state
        # Force error below threshold by rewriting bucket to match predicted ≈ actual
        b = cal.buckets["90+"]
        n = int(b["n"])
        b["wins"] = int(round(0.97 * n))  # actual ≈ predicted
        b["predicted_sum"] = 0.97 * n
        cal.audit_and_maybe_enable_deflation()  # should reset streak
        assert int(cal.buckets["90+"]["consecutive_overconfident"]) == 0
        assert cal.auto_deflation_enabled is False
