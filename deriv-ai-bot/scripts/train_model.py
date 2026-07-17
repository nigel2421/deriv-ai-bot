"""Train the hybrid LSTM + XGBoost model.

Run from project root (with venv active):
    python scripts/train_model.py
    python scripts/train_model.py --lstm
    python scripts/train_model.py --min-accuracy 0.15 --force
    python scripts/train_model.py --retrain
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.ai.paths import DEFAULT_MODEL_DIR
from src.ai.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA = ROOT / "data" / "historical" / "ticks.csv"
ALT_DATA = ROOT / "data" / "training" / "features.csv"


def ensure_sample_ticks(path: Path, n: int = 3000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.0008, n)
    quotes = np.round(1000 + np.cumsum(returns), 2)
    epochs = np.arange(1_700_000_000, 1_700_000_000 + n)
    pd.DataFrame({"epoch": epochs, "quote": quotes}).to_csv(path, index=False)
    logger.info("Created sample training ticks at %s (%d rows)", path, n)


def resolve_data_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if DEFAULT_DATA.exists() and DEFAULT_DATA.stat().st_size > 50:
        return DEFAULT_DATA
    # Prefer symbol CSVs from tick bootstrap if present
    hist = ROOT / "data" / "historical"
    for name in ("R_100_ticks.csv", "R_75_ticks.csv"):
        p = hist / name
        if p.exists() and p.stat().st_size > 50:
            return p
    if ALT_DATA.exists() and ALT_DATA.stat().st_size > 50:
        return ALT_DATA
    ensure_sample_ticks(DEFAULT_DATA)
    return DEFAULT_DATA


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Deriv AI digit models")
    parser.add_argument("--lstm", action="store_true", help="Also train LSTM")
    parser.add_argument("--epochs", type=int, default=None, help="LSTM epochs")
    parser.add_argument("--data", type=str, default=None, help="Ticks CSV path")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--retrain", action="store_true", help="Mark as retrain run")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Min holdout accuracy to save (default env MIN_MODEL_ACCURACY or 0.12)",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=None,
        help="Min accuracy - baseline lift required (default 0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Save models even if accuracy gate fails",
    )
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)
    if not data_path.is_file():
        ensure_sample_ticks(data_path)

    logger.info("Training with data: %s", data_path)
    logger.info(
        "Model dir: %s | lstm=%s | retrain=%s | force=%s",
        args.model_dir,
        args.lstm,
        args.retrain,
        args.force,
    )

    trainer = ModelTrainer(model_dir=Path(args.model_dir))
    ok = trainer.train(
        data_path=str(data_path),
        retrain=args.retrain,
        train_lstm=args.lstm,
        train_xgb=True,
        epochs=args.epochs,
        min_accuracy=args.min_accuracy,
        min_lift=args.min_lift,
        force_save=args.force,
    )
    if ok:
        print("Model training completed!")
        print(f"Artifacts: {args.model_dir}")
        if trainer.last_metrics:
            print(f"Metrics: {trainer.last_metrics.summary_line()}")
            print(f"Schema: v{trainer.last_metrics.schema_version} hash={trainer.last_metrics.schema_hash}")
        return 0

    print("Model training failed or accuracy gate blocked save — see logs.")
    if trainer.last_metrics:
        print(f"Last metrics: {trainer.last_metrics.summary_line()}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
