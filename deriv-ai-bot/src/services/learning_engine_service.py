"""
Learning Engine Microservice daemon.
Runs continuous pattern detection, win-rate tracking, pattern decay evaluation,
confidence calibration audits, and stores outcomes in PostgreSQL & Redis.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.db import db_manager
from src.strategy.adaptive_learner import AdaptiveLearner
from src.strategy.ai_auditor import AIAuditor
from src.strategy.calibration_tracker import CalibrationTracker
from src.strategy.deepseek_advisor import DeepSeekAdvisor
from src.utils.logger import setup_logger

logger = setup_logger()


class LearningEngineService:
    """Decoupled background worker for AI learning and optimization."""

    def __init__(self):
        self.interval = int(os.getenv("LEARNING_CYCLE_SECONDS", "300"))  # Default 5 min
        self.learner = AdaptiveLearner(path=Path("data/learning.json"), always_on=True)
        self.calibration = CalibrationTracker(path=Path("data/calibration_state.json"))
        self.auditor = AIAuditor(
            history_path=Path("data/trade_history.jsonl"),
            report_path=Path("data/auditor_report.json"),
        )
        self.deepseek = DeepSeekAdvisor(
            history_path=Path("data/trade_history.jsonl"),
            report_path=Path("data/deepseek_report.json"),
            state_path=Path("data/deepseek_state.json"),
        )

    async def run_cycle(self):
        logger.info("Executing Learning Engine cycle...")
        recent_trades = db_manager.fetch_recent_trades(limit=500)
        trades_count = len(recent_trades)

        if trades_count == 0:
            logger.info("No trades found in DB. Skipping learning cycle.")
            return

        wins = sum(1 for t in recent_trades if t.get("status") == "win")
        win_rate = round((wins / trades_count) * 100, 2) if trades_count > 0 else 0.0

        # Calibration error calculation
        overall_error = float(self.calibration.overall_error or 0.0)

        cycle_summary: Dict[str, Any] = {
            "trades_analyzed": trades_count,
            "overall_win_rate": win_rate,
            "overall_calibration_error": overall_error,
            "top_patterns": self.learner.top_performers(limit=10),
            "banned_setups": list(self.learner.setup_bans.keys()),
            "boosted_setups": [],
            "confidence_adjustments": self.learner.snapshot(),
            "audit_report": self.auditor.snapshot(),
            "deepseek_summary": "DeepSeek Advisor active",
        }

        # Record cycle to PostgreSQL
        db_manager.record_learning_cycle(cycle_summary)

        # Cache top patterns and updated confidence adjustments in Redis
        db_manager.cache_set("learning:confidence_adjustments", cycle_summary["confidence_adjustments"], ttl_seconds=600)
        db_manager.cache_set("learning:win_rate", win_rate, ttl_seconds=600)
        db_manager.cache_set("learning:banned_setups", cycle_summary["banned_setups"], ttl_seconds=600)

        logger.info(
            "Learning Cycle completed: %d trades analyzed, win_rate=%.2f%%, error=%.4f",
            trades_count,
            win_rate,
            overall_error,
        )

    async def start(self):
        logger.info("Learning Engine Service started. Running every %d seconds.", self.interval)
        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.exception("Error in Learning Engine cycle: %s", e)
            await asyncio.sleep(self.interval)


if __name__ == "__main__":
    service = LearningEngineService()
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logger.info("Learning Engine Service shutdown.")
