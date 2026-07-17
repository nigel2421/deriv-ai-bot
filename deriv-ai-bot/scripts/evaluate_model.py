"""Evaluate saved models on a ticks CSV without retraining.

  python scripts/evaluate_model.py
  python scripts/evaluate_model.py --data data/historical/R_100_ticks.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.paths import DEFAULT_MODEL_DIR, META_FILENAME
from src.ai.schema import FeatureSchema
from src.ai.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate digit models")
    p.add_argument("--data", default=str(ROOT / "data" / "historical" / "ticks.csv"))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    schema = FeatureSchema.load(model_dir)
    if schema:
        logger.info(
            "Schema v%s hash=%s n=%s",
            schema.version,
            schema.columns_hash,
            schema.n_features,
        )
    meta_path = model_dir / META_FILENAME
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info(
            "Saved meta accuracy=%s gate=%s",
            meta.get("ensemble_test_accuracy") or meta.get("xgb_test_accuracy"),
            meta.get("passed_gate"),
        )

    trainer = ModelTrainer(model_dir=model_dir)
    report = trainer.evaluate_only(args.data)
    if not report:
        print("Evaluation failed.")
        return 1
    print(report.summary_line())
    print(json.dumps({k: v for k, v in report.to_dict().items() if k != "extras"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
