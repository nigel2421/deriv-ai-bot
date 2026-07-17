"""Train the hybrid LSTM + XGBoost model.

Run from project root (with venv active):
    python scripts/train_model.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.ai.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA = ROOT / "data" / "historical" / "ticks.csv"
ALT_DATA = ROOT / "data" / "training" / "features.csv"


def ensure_sample_ticks(path: Path, n: int = 2000) -> None:
    """Create synthetic tick data so first-time training can run offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    # Rough random-walk around a typical synthetic index price
    returns = rng.normal(0, 0.0008, n)
    quotes = 1000 + np.cumsum(returns)
    epochs = np.arange(1_700_000_000, 1_700_000_000 + n)
    pd.DataFrame({"epoch": epochs, "quote": quotes}).to_csv(path, index=False)
    logger.info("Created sample training ticks at %s (%d rows)", path, n)


def resolve_data_path() -> Path:
    if DEFAULT_DATA.exists():
        return DEFAULT_DATA
    if ALT_DATA.exists():
        return ALT_DATA
    ensure_sample_ticks(DEFAULT_DATA)
    return DEFAULT_DATA


if __name__ == "__main__":
    data_path = resolve_data_path()
    # Paths relative to project root for modules that use string paths
    rel = data_path.relative_to(ROOT).as_posix()
    logger.info("Training with data: %s", rel)

    trainer = ModelTrainer()
    ok = trainer.train(data_path=rel)
    if ok:
        print("Model training completed!")
    else:
        print("Model training failed — see logs above.")
        sys.exit(1)
