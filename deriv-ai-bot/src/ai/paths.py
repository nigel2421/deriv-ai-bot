"""Canonical paths for trained model artifacts."""
from __future__ import annotations

from pathlib import Path

# Project root: src/ai/paths.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "src" / "models"

LSTM_FILENAME = "lstm_model.keras"  # modern Keras format (fallback .h5)
LSTM_FILENAME_H5 = "lstm_model.h5"
XGB_FILENAME = "xgboost_model.pkl"
SCALER_FILENAME = "scaler.pkl"
FEATURE_COLUMNS_FILENAME = "feature_columns.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
META_FILENAME = "model_meta.json"
METRICS_HISTORY_FILENAME = "metrics_history.jsonl"
