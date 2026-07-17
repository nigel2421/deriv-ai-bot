from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from src.ai.data_pipeline import DataPipeline
from src.ai.metrics import evaluate_predictions, passes_accuracy_gate
from src.ai.models import HybridModel
from src.ai.paths import DEFAULT_MODEL_DIR, META_FILENAME, METRICS_HISTORY_FILENAME
from src.ai.schema import SCHEMA_VERSION, FeatureSchema, MetricsReport

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class ModelTrainer:
    """Training pipeline with schema, metrics, accuracy gates, and history."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.pipeline = DataPipeline()
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.model = HybridModel(self.model_dir)
        self.last_metrics: Optional[MetricsReport] = None
        self.last_schema: Optional[FeatureSchema] = None

    def train(
        self,
        data_path: str = "data/historical/ticks.csv",
        retrain: bool = False,
        *,
        train_lstm: Optional[bool] = None,
        train_xgb: bool = True,
        epochs: Optional[int] = None,
        min_accuracy: Optional[float] = None,
        min_lift: Optional[float] = None,
        force_save: bool = False,
    ) -> bool:
        """
        Train models, evaluate holdout metrics, enforce accuracy gate, save schema.

        Env:
          TRAIN_LSTM=true|false
          MIN_MODEL_ACCURACY=0.12   (digit random baseline ~0.10)
          MIN_MODEL_LIFT=0.0       (must beat majority baseline by this)
          FORCE_SAVE_MODEL=true    (save even if gate fails)
        """
        if train_lstm is None:
            train_lstm = _env_bool("TRAIN_LSTM", False)
        if min_accuracy is None:
            min_accuracy = _env_float("MIN_MODEL_ACCURACY", 0.12)
        if min_lift is None:
            min_lift = _env_float("MIN_MODEL_LIFT", 0.0)
        force_save = force_save or _env_bool("FORCE_SAVE_MODEL", False)

        prev_meta = self._load_prev_meta()

        df = self.pipeline.load_historical(data_path)
        if df.empty:
            logger.error("No training data available.")
            return False

        X, y_digit, _ = self.pipeline.preprocess_for_training(df)
        if X.empty or len(y_digit) == 0:
            logger.error("No usable features after preprocessing.")
            return False

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_digit, test_size=0.2, random_state=42, stratify=y_digit
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_digit, test_size=0.2, random_state=42
            )

        schema = FeatureSchema.from_columns(
            list(X_train.columns),
            version=SCHEMA_VERSION,
            notes="next_digit classification features",
            extras={"data_path": str(data_path), "retrain": retrain},
        )
        self.last_schema = schema
        self.model.feature_columns = schema.columns

        # Schema compatibility with previous deployment
        old_schema = FeatureSchema.load(self.model_dir)
        if old_schema and not old_schema.is_compatible(schema):
            logger.warning(
                "Feature schema changed vs deployed model "
                "(old_hash=%s new_hash=%s). Full retrain artifacts will replace it.",
                old_schema.columns_hash,
                schema.columns_hash,
            )

        metrics_extra: Dict[str, Any] = {
            "train_lstm": bool(train_lstm),
            "train_xgb": bool(train_xgb),
            "retrain": retrain,
        }

        y_test_arr = np.asarray(y_test).astype(int)
        best_proba: Optional[np.ndarray] = None
        best_pred: Optional[np.ndarray] = None
        primary_acc = 0.0

        if train_xgb:
            self.model.train_xgb(X_train, y_train)
            xgb_pred = self.model.xgb_model.predict(X_test)
            xgb_proba = self.model.predict_xgb_proba(X_test)
            ev = evaluate_predictions(y_test_arr, xgb_pred, xgb_proba)
            metrics_extra["xgb_test_accuracy"] = ev["accuracy"]
            metrics_extra["xgb_eval"] = {
                k: v for k, v in ev.items() if k != "classification_report"
            }
            primary_acc = ev["accuracy"]
            best_pred = np.asarray(xgb_pred)
            best_proba = xgb_proba
            logger.info("XGBoost holdout accuracy: %.3f", ev["accuracy"])
            if ev.get("classification_report"):
                logger.info("XGBoost report:\n%s", ev["classification_report"])

        if train_lstm:
            ep = epochs if epochs is not None else (8 if retrain else 12)
            self.model.train_lstm(X_train, y_train, epochs=ep, verbose=1)
            try:
                lstm_proba = self.model.predict_lstm_proba(X_test)
                if lstm_proba is not None:
                    lstm_pred = np.argmax(lstm_proba, axis=1)
                    ev_l = evaluate_predictions(y_test_arr, lstm_pred, lstm_proba)
                    metrics_extra["lstm_test_accuracy"] = ev_l["accuracy"]
                    metrics_extra["lstm_eval"] = {
                        k: v for k, v in ev_l.items() if k != "classification_report"
                    }
                    logger.info("LSTM holdout accuracy: %.3f", ev_l["accuracy"])
            except Exception as e:
                logger.warning("LSTM eval failed: %s", e)

        # Ensemble
        if self.model.xgb_model is not None or self.model.lstm_model is not None:
            ens_preds: List[int] = []
            ens_proba_rows: List[List[float]] = []
            for i in range(len(X_test)):
                out = self.model.predict_digit(X_test.iloc[[i]])
                ens_preds.append(int(out["digit"]))
                if out.get("proba"):
                    ens_proba_rows.append(out["proba"])
            ens_proba = (
                np.asarray(ens_proba_rows, dtype=float) if ens_proba_rows else None
            )
            ev_e = evaluate_predictions(y_test_arr, ens_preds, ens_proba)
            metrics_extra["ensemble_test_accuracy"] = ev_e["accuracy"]
            metrics_extra["ensemble_eval"] = {
                k: v for k, v in ev_e.items() if k != "classification_report"
            }
            primary_acc = ev_e["accuracy"]
            best_pred = np.asarray(ens_preds)
            best_proba = ens_proba
            logger.info("Ensemble holdout accuracy: %.3f", ev_e["accuracy"])

        # Aggregate report
        if best_pred is None:
            logger.error("No model was trained.")
            return False

        final_ev = evaluate_predictions(y_test_arr, best_pred, best_proba)
        baseline = float(final_ev["baseline_accuracy"])
        gate_ok = passes_accuracy_gate(
            primary_acc, baseline, min_accuracy=min_accuracy, min_lift=min_lift
        )

        report = MetricsReport(
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            n_features=int(X_train.shape[1]),
            xgb_test_accuracy=metrics_extra.get("xgb_test_accuracy"),
            lstm_test_accuracy=metrics_extra.get("lstm_test_accuracy"),
            ensemble_test_accuracy=metrics_extra.get("ensemble_test_accuracy"),
            top3_accuracy=final_ev.get("top3_accuracy"),
            log_loss=final_ev.get("log_loss"),
            baseline_accuracy=baseline,
            lift_vs_baseline=float(final_ev["lift_vs_baseline"]),
            per_class_f1=final_ev.get("per_class_f1") or {},
            confusion_diag=final_ev.get("confusion_diag") or {},
            passed_gate=gate_ok or force_save,
            min_accuracy_required=float(min_accuracy),
            data_path=str(data_path),
            schema_version=schema.version,
            schema_hash=schema.columns_hash,
            extras={
                **metrics_extra,
                "min_lift_required": min_lift,
                "gate_passed_raw": gate_ok,
                "force_save": force_save,
                "prev_accuracy": (
                    prev_meta.get("ensemble_test_accuracy")
                    or prev_meta.get("xgb_test_accuracy")
                    if prev_meta
                    else None
                ),
            },
        )
        self.last_metrics = report
        self.model.meta = report.to_dict()

        logger.info("Metrics: %s", report.summary_line())
        if prev_meta:
            prev_acc = prev_meta.get("ensemble_test_accuracy") or prev_meta.get(
                "xgb_test_accuracy"
            )
            if prev_acc is not None:
                delta = primary_acc - float(prev_acc)
                logger.info(
                    "vs previous model: %.3f → %.3f (Δ %+.3f)",
                    float(prev_acc),
                    primary_acc,
                    delta,
                )

        if not gate_ok and not force_save:
            logger.error(
                "Accuracy gate FAILED (acc=%.3f baseline=%.3f min=%.3f lift_min=%.3f). "
                "Artifacts NOT saved. Use --force or FORCE_SAVE_MODEL=true to override.",
                primary_acc,
                baseline,
                min_accuracy,
                min_lift,
            )
            self._append_history(report, saved=False)
            return False

        if not gate_ok and force_save:
            logger.warning("Accuracy gate failed but force_save=True — saving anyway.")

        # Persist schema + models
        schema.save(self.model_dir)
        self.model.save_models(self.model_dir)
        self._append_history(report, saved=True)

        logger.info(
            "%s complete | %s | dir=%s",
            "Retrain" if retrain else "Train",
            report.summary_line(),
            self.model_dir,
        )
        return True

    def evaluate_only(self, data_path: str) -> Optional[MetricsReport]:
        """Load existing model and score on data (no retrain)."""
        if not self.model.load_models(self.model_dir):
            logger.error("No models to evaluate in %s", self.model_dir)
            return None

        df = self.pipeline.load_historical(data_path)
        if df.empty:
            return None
        X, y, _ = self.pipeline.preprocess_for_training(df)
        if X.empty:
            return None

        schema = FeatureSchema.load(self.model_dir)
        if schema:
            report = schema.validate_frame(X, strict=False)
            logger.info("Schema validation: %s", report)

        preds = []
        probas = []
        for i in range(len(X)):
            out = self.model.predict_digit(X.iloc[[i]])
            preds.append(out["digit"])
            if out.get("proba"):
                probas.append(out["proba"])
        proba = np.asarray(probas) if probas else None
        ev = evaluate_predictions(y, preds, proba)
        m = MetricsReport(
            n_train=0,
            n_test=len(X),
            n_features=X.shape[1],
            ensemble_test_accuracy=ev["accuracy"],
            top3_accuracy=ev.get("top3_accuracy"),
            log_loss=ev.get("log_loss"),
            baseline_accuracy=ev["baseline_accuracy"],
            lift_vs_baseline=ev["lift_vs_baseline"],
            per_class_f1=ev.get("per_class_f1") or {},
            confusion_diag=ev.get("confusion_diag") or {},
            passed_gate=True,
            data_path=str(data_path),
            schema_version=schema.version if schema else SCHEMA_VERSION,
            schema_hash=schema.columns_hash if schema else "",
            extras={"mode": "evaluate_only"},
        )
        self.last_metrics = m
        logger.info("Evaluate-only: %s", m.summary_line())
        if ev.get("classification_report"):
            logger.info("\n%s", ev["classification_report"])
        return m

    def _load_prev_meta(self) -> Dict[str, Any]:
        path = self.model_dir / META_FILENAME
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _append_history(self, report: MetricsReport, *, saved: bool) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / METRICS_HISTORY_FILENAME
        row = report.to_dict()
        row["saved"] = saved
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            logger.warning("Could not append metrics history: %s", e)
