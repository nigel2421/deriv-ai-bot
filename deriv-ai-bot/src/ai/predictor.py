from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.ai.feature_engineering import FeatureEngineer
from src.ai.models import HybridModel
from src.ai.paths import DEFAULT_MODEL_DIR
from src.ai.schema import FeatureSchema
from src.strategy.digit_contracts import extract_last_digit, last_digits_from_ticks

logger = logging.getLogger(__name__)


class Predictor:
    """
    Real-time digit predictions.

    Priority:
      1. Loaded XGBoost / LSTM ensemble (if artifacts present)
      2. Last-digit frequency heuristic fallback
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        *,
        auto_load: bool = True,
        min_ticks: int = 50,
    ):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.min_ticks = min_ticks
        self.engineer = FeatureEngineer()
        self.model = HybridModel(self.model_dir)
        self.models_loaded = False
        self.schema: Optional[FeatureSchema] = None
        self.schema_status: Dict[str, Any] = {}

        if auto_load:
            self.load()

    def load(self) -> bool:
        self.models_loaded = self.model.load_models(self.model_dir)
        self.schema = getattr(self.model, "schema", None) or FeatureSchema.load(
            self.model_dir
        )
        if self.models_loaded:
            logger.info(
                "Predictor ready with models from %s (xgb=%s lstm=%s cols=%s schema=%s)",
                self.model_dir,
                self.model.xgb_model is not None,
                self.model.lstm_model is not None,
                len(self.model.feature_columns or []),
                self.schema.columns_hash if self.schema else None,
            )
        else:
            logger.info(
                "Predictor using heuristic fallback (no models in %s). "
                "Run: python scripts/train_model.py",
                self.model_dir,
            )
        return self.models_loaded

    def predict(self, recent_ticks: List[Dict[str, Any]]) -> dict:
        """Predict next last digit + confidence from recent ticks."""
        if not recent_ticks:
            return self._empty_prediction(source="no_ticks")

        # Try model path first
        if self.model.is_ready:
            model_out = self._predict_with_models(recent_ticks)
            if model_out is not None:
                return model_out

        return self._predict_heuristic(recent_ticks)

    def _predict_with_models(
        self, recent_ticks: List[Dict[str, Any]]
    ) -> Optional[dict]:
        try:
            df = pd.DataFrame(recent_ticks)
            if "quote" not in df.columns or len(df) < self.min_ticks:
                return None

            features = self.engineer.create_features(df, quiet=True)
            if features.empty:
                return None

            X = self.engineer.model_matrix(features, self.model.feature_columns)
            if X.empty:
                return None

            # Validate live matrix against training schema (non-strict: missing=bad)
            if self.schema:
                self.schema_status = self.schema.validate_frame(X, strict=False)
                if not self.schema_status.get("ok"):
                    logger.warning(
                        "Live features missing schema columns: %s",
                        self.schema_status.get("missing"),
                    )

            result = self.model.predict_digit(X)
            if result.get("confidence", 0) <= 0 and result.get("source") == "no_model":
                return None

            result["source"] = f"model:{result.get('source', 'ensemble')}"
            # Blend a bit of recent frequency for stability when model conf is flat
            heur = self._predict_heuristic(recent_ticks)
            if result["confidence"] < 0.2:
                # very flat → lean heuristic
                return {
                    **heur,
                    "source": "model_low_conf+heuristic",
                    "model": result,
                }

            result["parity"] = bool(result["digit"] % 2 == 0)
            result["heuristic_digit"] = heur.get("digit")
            result["heuristic_confidence"] = heur.get("confidence")
            logger.debug(
                "Model predict digit=%s conf=%.3f src=%s",
                result["digit"],
                result["confidence"],
                result["source"],
            )
            return result
        except Exception as e:
            logger.warning("Model inference failed (%s); using heuristic.", e)
            return None

    def _predict_heuristic(self, recent_ticks: List[Dict[str, Any]]) -> dict:
        digits = last_digits_from_ticks(recent_ticks, n=50)
        if len(digits) < 10:
            last = extract_last_digit(
                recent_ticks[-1].get("quote") if recent_ticks else None
            )
            d = last if last is not None else 5
            return {
                "digit": d,
                "confidence": 0.5,
                "parity": d % 2 == 0,
                "source": "sparse",
            }

        counts = Counter(digits)
        mode_digit, mode_n = counts.most_common(1)[0]
        conf = min(0.95, 0.45 + (mode_n / len(digits)) * 0.5)
        even_n = sum(1 for x in digits if x % 2 == 0)
        return {
            "digit": int(mode_digit),
            "confidence": float(conf),
            "parity": even_n >= (len(digits) / 2),
            "source": "digit_freq",
            "digit_counts": {int(k): int(v) for k, v in counts.items()},
        }

    @staticmethod
    def _empty_prediction(source: str = "empty") -> dict:
        return {
            "digit": 5,
            "confidence": 0.0,
            "parity": False,
            "source": source,
        }
