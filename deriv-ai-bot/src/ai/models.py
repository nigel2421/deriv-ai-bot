from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.ai.paths import (
    DEFAULT_MODEL_DIR,
    FEATURE_COLUMNS_FILENAME,
    LSTM_FILENAME,
    LSTM_FILENAME_H5,
    META_FILENAME,
    SCALER_FILENAME,
    XGB_FILENAME,
)
from src.ai.schema import FeatureSchema

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class HybridModel:
    """
    LSTM (sequence) + XGBoost (tabular) digit classifier ensemble.

    TensorFlow is imported lazily so the bot can start without loading TF
    until training or LSTM inference needs it.
    """

    def __init__(self, model_dir: PathLike = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = None  # optional StandardScaler for LSTM inputs
        self.feature_columns: Optional[List[str]] = None
        self.schema: Optional[FeatureSchema] = None
        self.meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------ TF lazy
    def _tf(self):
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential

        return tf, Sequential, LSTM, Dense, Dropout

    # ------------------------------------------------------------------ train
    def build_lstm(self, n_features: int):
        _, Sequential, LSTM, Dense, Dropout = self._tf()
        self.lstm_model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=(n_features, 1)),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(10, activation="softmax"),
            ]
        )
        self.lstm_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("LSTM model built for %d features.", n_features)

    def _to_lstm_array(self, X: pd.DataFrame) -> np.ndarray:
        arr = X.values.astype(np.float32)
        # Optional scale
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        return arr.reshape((arr.shape[0], arr.shape[1], 1))

    def train_lstm(self, X_train: pd.DataFrame, y_train, epochs: int = 15, verbose: int = 1):
        from sklearn.preprocessing import StandardScaler

        self.feature_columns = list(X_train.columns)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train.values.astype(np.float32))

        if self.lstm_model is None:
            self.build_lstm(X_train.shape[1])

        X_lstm = self._to_lstm_array(X_train)
        y = np.asarray(y_train).astype(int)
        self.lstm_model.fit(
            X_lstm,
            y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.15,
            verbose=verbose,
        )
        logger.info("LSTM training finished (%d epochs).", epochs)

    def train_xgb(self, X_train: pd.DataFrame, y_train, **kwargs):
        import xgboost as xgb

        self.feature_columns = list(X_train.columns)
        params = {
            "n_estimators": kwargs.get("n_estimators", 120),
            "learning_rate": kwargs.get("learning_rate", 0.08),
            "max_depth": kwargs.get("max_depth", 5),
            "subsample": kwargs.get("subsample", 0.9),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.9),
            "objective": "multi:softprob",
            "num_class": 10,
            "eval_metric": "mlogloss",
            "n_jobs": -1,
            "random_state": 42,
        }
        self.xgb_model = xgb.XGBClassifier(**params)
        X_fit, y_fit = self._ensure_digit_classes(X_train, y_train)
        self.xgb_model.fit(X_fit, y_fit)
        logger.info(
            "XGBoost trained on %d samples × %d features (classes padded to 0-9).",
            len(X_train),
            X_train.shape[1],
        )

    @staticmethod
    def _ensure_digit_classes(
        X: pd.DataFrame, y
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        XGBoost sklearn API requires contiguous class labels starting at 0.
        Inject one synthetic row per missing digit (0-9) so num_class=10 works.
        """
        y_arr = np.asarray(y).astype(int)
        present = set(int(v) for v in y_arr.tolist())
        missing = [d for d in range(10) if d not in present]
        if not missing:
            return X, y_arr

        # Duplicate first row for each missing class (minimal synthetic support)
        base = X.iloc[[0]]
        extras_X = pd.concat([base] * len(missing), ignore_index=True)
        extras_y = np.array(missing, dtype=int)
        X_out = pd.concat([X.reset_index(drop=True), extras_X], ignore_index=True)
        y_out = np.concatenate([y_arr, extras_y])
        logger.debug("Padded missing digit classes for XGB: %s", missing)
        return X_out, y_out

    # ------------------------------------------------------------------ predict
    def align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if X is None or X.empty:
            return pd.DataFrame()
        Xn = X.select_dtypes(include=["number"]).astype(float)
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in Xn.columns:
                    Xn[col] = 0.0
            Xn = Xn.reindex(columns=self.feature_columns, fill_value=0.0)
        return Xn.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def predict_xgb_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.xgb_model is None:
            return None
        Xa = self.align_features(X)
        if Xa.empty:
            return None
        proba = self.xgb_model.predict_proba(Xa)
        # Ensure shape (n, 10) — pad missing classes
        full = np.zeros((proba.shape[0], 10), dtype=float)
        classes = list(getattr(self.xgb_model, "classes_", range(proba.shape[1])))
        for i, cls in enumerate(classes):
            if 0 <= int(cls) < 10:
                full[:, int(cls)] = proba[:, i]
        # Renormalize rows (padding can leave mass ≠ 1 if classes remapped)
        row_sums = full.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        full = full / row_sums
        return full

    def predict_lstm_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.lstm_model is None:
            return None
        Xa = self.align_features(X)
        if Xa.empty:
            return None
        X_lstm = self._to_lstm_array(Xa)
        proba = self.lstm_model.predict(X_lstm, verbose=0)
        return np.asarray(proba, dtype=float)

    def predict_digit(
        self,
        X: pd.DataFrame,
        *,
        xgb_weight: float = 0.6,
        lstm_weight: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Ensemble digit prediction for the last row of X (or all rows → last).

        Returns digit, confidence, parity, per-model probs, source.
        """
        Xa = self.align_features(X)
        if Xa.empty:
            return {
                "digit": 5,
                "confidence": 0.0,
                "parity": False,
                "source": "empty",
            }

        # Use last row only for online inference
        row = Xa.iloc[[-1]]
        xgb_p = self.predict_xgb_proba(row)
        lstm_p = self.predict_lstm_proba(row)

        parts = []
        weights = []
        sources = []
        if xgb_p is not None:
            parts.append(xgb_p[-1])
            weights.append(xgb_weight)
            sources.append("xgb")
        if lstm_p is not None:
            parts.append(lstm_p[-1])
            weights.append(lstm_weight)
            sources.append("lstm")

        if not parts:
            return {
                "digit": 5,
                "confidence": 0.0,
                "parity": False,
                "source": "no_model",
            }

        w = np.array(weights, dtype=float)
        w = w / w.sum()
        ensemble = sum(wi * pi for wi, pi in zip(w, parts))
        ensemble = ensemble / max(ensemble.sum(), 1e-12)
        digit = int(np.argmax(ensemble))
        confidence = float(ensemble[digit])

        detail = {
            "digit": digit,
            "confidence": confidence,
            "parity": digit % 2 == 0,
            "source": "+".join(sources),
            "proba": ensemble.tolist(),
        }
        if xgb_p is not None:
            detail["xgb_digit"] = int(np.argmax(xgb_p[-1]))
            detail["xgb_confidence"] = float(np.max(xgb_p[-1]))
        if lstm_p is not None:
            detail["lstm_digit"] = int(np.argmax(lstm_p[-1]))
            detail["lstm_confidence"] = float(np.max(lstm_p[-1]))
        return detail

    # ------------------------------------------------------------------ persist
    def save_models(self, path: Optional[PathLike] = None) -> Path:
        out = Path(path) if path is not None else self.model_dir
        out.mkdir(parents=True, exist_ok=True)

        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, out / XGB_FILENAME)
            logger.info("Saved XGBoost → %s", out / XGB_FILENAME)

        if self.lstm_model is not None:
            # Prefer modern Keras format; also try h5 for compatibility
            try:
                self.lstm_model.save(out / LSTM_FILENAME)
                logger.info("Saved LSTM → %s", out / LSTM_FILENAME)
            except Exception as e:
                logger.warning("Keras save failed (%s); trying .h5", e)
                self.lstm_model.save(str(out / LSTM_FILENAME_H5))
                logger.info("Saved LSTM → %s", out / LSTM_FILENAME_H5)

        if self.scaler is not None:
            joblib.dump(self.scaler, out / SCALER_FILENAME)

        if self.feature_columns:
            # Prefer versioned schema writer (also writes feature_columns.json)
            try:
                schema = FeatureSchema.from_columns(
                    self.feature_columns,
                    extras={"saved_with_meta_keys": list(self.meta.keys())},
                )
                if self.meta.get("schema_hash"):
                    schema.columns_hash = str(self.meta["schema_hash"])
                if self.meta.get("schema_version"):
                    schema.version = str(self.meta["schema_version"])
                schema.save(out)
            except Exception:
                (out / FEATURE_COLUMNS_FILENAME).write_text(
                    json.dumps(self.feature_columns, indent=2), encoding="utf-8"
                )

        meta = {
            **self.meta,
            "feature_count": len(self.feature_columns or []),
            "has_xgb": self.xgb_model is not None,
            "has_lstm": self.lstm_model is not None,
            "has_scaler": self.scaler is not None,
        }
        (out / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Model artifacts saved under %s", out)
        return out

    def load_models(self, path: Optional[PathLike] = None) -> bool:
        """Load saved models if present. Returns True if anything loaded."""
        root = Path(path) if path is not None else self.model_dir
        if not root.is_dir():
            logger.warning("Model dir missing: %s", root)
            return False

        loaded = False
        self.schema: Optional[FeatureSchema] = FeatureSchema.load(root)

        if self.schema and self.schema.columns:
            self.feature_columns = list(self.schema.columns)
            logger.info(
                "Loaded schema v%s (%d cols, hash=%s)",
                self.schema.version,
                self.schema.n_features,
                self.schema.columns_hash,
            )
        else:
            cols_path = root / FEATURE_COLUMNS_FILENAME
            if cols_path.is_file():
                try:
                    self.feature_columns = json.loads(
                        cols_path.read_text(encoding="utf-8")
                    )
                    logger.info(
                        "Loaded %d feature columns (legacy)",
                        len(self.feature_columns),
                    )
                except Exception as e:
                    logger.warning("Failed to load feature columns: %s", e)

        scaler_path = root / SCALER_FILENAME
        if scaler_path.is_file():
            try:
                self.scaler = joblib.load(scaler_path)
                loaded = True
            except Exception as e:
                logger.warning("Failed to load scaler: %s", e)

        xgb_path = root / XGB_FILENAME
        if xgb_path.is_file():
            try:
                self.xgb_model = joblib.load(xgb_path)
                loaded = True
                logger.info("Loaded XGBoost from %s", xgb_path)
            except Exception as e:
                logger.warning("Failed to load XGBoost: %s", e)

        lstm_path = root / LSTM_FILENAME
        lstm_h5 = root / LSTM_FILENAME_H5
        target = lstm_path if lstm_path.is_file() else (
            lstm_h5 if lstm_h5.is_file() else None
        )
        if target is not None:
            try:
                from tensorflow.keras.models import load_model

                self.lstm_model = load_model(str(target))
                loaded = True
                logger.info("Loaded LSTM from %s", target)
            except Exception as e:
                logger.warning("Failed to load LSTM: %s", e)

        meta_path = root / META_FILENAME
        if meta_path.is_file():
            try:
                self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        if not loaded:
            logger.info("No model artifacts found in %s", root)
        return loaded

    @property
    def is_ready(self) -> bool:
        return self.xgb_model is not None or self.lstm_model is not None
