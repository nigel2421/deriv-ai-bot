"""
Experiment Engine Framework for CONTROL vs CHALLENGER Shadow Testing.

Allows testing alternative production parameters (e.g. EV threshold 0.05 vs 0.08,
Quorum 2 vs 3 agents, Expiry 5 vs 8 ticks) strictly in SHADOW MODE without look-ahead bias
or real money risk.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    experiment_name: str
    control_config: Dict[str, Any]
    challenger_config: Dict[str, Any]
    status: str = "active"  # active | paused | completed
    mode: str = "shadow"  # ALWAYS shadow mode (never live execution)
    control_results: List[Dict[str, Any]] = field(default_factory=list)
    challenger_results: List[Dict[str, Any]] = field(default_factory=list)


class ExperimentEngine:
    """Manages CONTROL vs CHALLENGER experiment setups strictly in SHADOW MODE."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/experiments_state.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentConfig] = {}
        self._seed_default_experiments()

    def _seed_default_experiments(self) -> None:
        """Seed core A/B experiment comparisons."""
        defaults = [
            ExperimentConfig(
                experiment_name="EV_Threshold_Exp",
                control_config={"ev_threshold": 0.08},
                challenger_config={"ev_threshold": 0.05},
            ),
            ExperimentConfig(
                experiment_name="Quorum_Exp",
                control_config={"min_quorum": 2},
                challenger_config={"min_quorum": 3},
            ),
            ExperimentConfig(
                experiment_name="Vol_Expiry_Exp",
                control_config={"base_duration": 5},
                challenger_config={"high_vol_duration": 8},
            ),
        ]
        for exp in defaults:
            self.experiments[exp.experiment_name] = exp
        self._load_state()

    def _load_state(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for exp_data in data.get("experiments", []):
                        exp = ExperimentConfig(**exp_data)
                        self.experiments[exp.experiment_name] = exp
            except Exception as e:
                logger.warning("ExperimentEngine state load warning: %s", e)

    def save_state(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"experiments": [asdict(e) for e in self.experiments.values()]},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error("Failed to save ExperimentEngine state: %s", e)

    def evaluate_opportunity(
        self,
        symbol: str,
        contract_type: str,
        control_passed: bool,
        control_ev: float,
        control_quorum: int,
        shadow_trader: Any,
        entry_price: float,
        duration: int = 5,
        duration_unit: str = "t",
    ) -> Dict[str, Any]:
        """
        Evaluate opportunity across active challenger configurations in shadow mode.
        If challenger would accept a trade that CONTROL rejected, spawn shadow trade.
        SAFETY: Challenger configurations NEVER execute real trades.
        """
        evaluations = {}

        for name, exp in self.experiments.items():
            if exp.status != "active":
                continue

            challenger_accepted = False
            c_cfg = exp.challenger_config

            if "ev_threshold" in c_cfg:
                threshold = float(c_cfg["ev_threshold"])
                challenger_accepted = control_ev >= threshold
            elif "min_quorum" in c_cfg:
                min_q = int(c_cfg["min_quorum"])
                challenger_accepted = control_quorum >= min_q
            elif "high_vol_duration" in c_cfg:
                challenger_accepted = control_passed

            evaluations[name] = {
                "control_passed": control_passed,
                "challenger_accepted": challenger_accepted,
            }

            # If challenger accepts an opportunity that CONTROL rejected, track in shadow mode
            if challenger_accepted and not control_passed and shadow_trader:
                st = shadow_trader.spawn_shadow_trade(
                    symbol=symbol,
                    contract_type=contract_type,
                    rejection_gate=f"EXP_{name}_CHALLENGER",
                    entry_price=entry_price,
                    duration=c_cfg.get("high_vol_duration", duration),
                    duration_unit=duration_unit,
                )
                shadow_id = st if isinstance(st, str) else getattr(st, "shadow_id", str(st))
                exp.challenger_results.append({
                    "shadow_id": shadow_id,
                    "symbol": symbol,
                    "contract_type": contract_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                self.save_state()

        return evaluations

    def get_summary(self) -> List[Dict[str, Any]]:
        """Return experiment status summary for Web Dashboard."""
        res = []
        for exp in self.experiments.values():
            res.append({
                "experiment_name": exp.experiment_name,
                "status": exp.status,
                "mode": exp.mode,
                "control": exp.control_config,
                "challenger": exp.challenger_config,
                "challenger_shadow_samples": len(exp.challenger_results),
            })
        return res

    def summary(self) -> Dict[str, Any]:
        """Return structured summary for Dashboard & unit tests."""
        active = []
        for exp in self.experiments.values():
            active.append({
                "name": exp.experiment_name,
                "status": exp.status,
                "mode": exp.mode,
                "control": {
                    "config": exp.control_config,
                    "samples": len(exp.control_results),
                    "win_rate": 0.0,
                },
                "challenger": {
                    "config": exp.challenger_config,
                    "samples": len(exp.challenger_results),
                    "win_rate": 0.0,
                },
            })
        return {"active_experiments": active}
