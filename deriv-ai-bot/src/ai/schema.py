"""
Feature schema contract between training and inference.

Artifacts:
  feature_columns.json  — ordered column list (legacy + still written)
  feature_schema.json   — versioned schema with hashes and metadata
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src.ai.paths import (
    DEFAULT_MODEL_DIR,
    FEATURE_COLUMNS_FILENAME,
    FEATURE_SCHEMA_FILENAME,
)

logger = logging.getLogger(__name__)

# Bump when feature engineering changes break compatibility
SCHEMA_VERSION = "1.1.0"


@dataclass
class FeatureSchema:
    """Versioned description of the model input matrix."""

    version: str
    columns: List[str]
    n_features: int
    columns_hash: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    target: str = "next_digit"
    target_classes: List[int] = field(default_factory=lambda: list(range(10)))
    notes: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def hash_columns(columns: Sequence[str]) -> str:
        payload = json.dumps(list(columns), separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @classmethod
    def from_columns(
        cls,
        columns: Sequence[str],
        *,
        version: str = SCHEMA_VERSION,
        notes: str = "",
        extras: Optional[Dict[str, Any]] = None,
    ) -> "FeatureSchema":
        cols = list(columns)
        return cls(
            version=version,
            columns=cols,
            n_features=len(cols),
            columns_hash=cls.hash_columns(cols),
            notes=notes,
            extras=extras or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSchema":
        return cls(
            version=str(data.get("version") or SCHEMA_VERSION),
            columns=list(data.get("columns") or []),
            n_features=int(data.get("n_features") or len(data.get("columns") or [])),
            columns_hash=str(data.get("columns_hash") or ""),
            created_at=str(data.get("created_at") or ""),
            target=str(data.get("target") or "next_digit"),
            target_classes=list(data.get("target_classes") or list(range(10))),
            notes=str(data.get("notes") or ""),
            extras=dict(data.get("extras") or {}),
        )

    def save(self, model_dir: Union[str, Path] = DEFAULT_MODEL_DIR) -> Path:
        root = Path(model_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / FEATURE_SCHEMA_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        # Keep legacy flat list in sync
        (root / FEATURE_COLUMNS_FILENAME).write_text(
            json.dumps(self.columns, indent=2), encoding="utf-8"
        )
        logger.info(
            "Saved feature schema v%s (%d cols, hash=%s) → %s",
            self.version,
            self.n_features,
            self.columns_hash,
            path,
        )
        return path

    @classmethod
    def load(
        cls, model_dir: Union[str, Path] = DEFAULT_MODEL_DIR
    ) -> Optional["FeatureSchema"]:
        root = Path(model_dir)
        schema_path = root / FEATURE_SCHEMA_FILENAME
        if schema_path.is_file():
            try:
                data = json.loads(schema_path.read_text(encoding="utf-8"))
                schema = cls.from_dict(data)
                # Repair hash if missing
                if not schema.columns_hash and schema.columns:
                    schema.columns_hash = cls.hash_columns(schema.columns)
                return schema
            except Exception as e:
                logger.warning("Failed to load feature_schema.json: %s", e)

        # Fallback: feature_columns.json only
        cols_path = root / FEATURE_COLUMNS_FILENAME
        if cols_path.is_file():
            try:
                cols = json.loads(cols_path.read_text(encoding="utf-8"))
                return cls.from_columns(cols, notes="migrated from feature_columns.json")
            except Exception as e:
                logger.warning("Failed to load feature_columns.json: %s", e)
        return None

    def validate_columns(
        self, columns: Sequence[str], *, strict: bool = False
    ) -> Dict[str, Any]:
        """
        Compare live feature columns to this schema.

        Returns report with ok, missing, extra, order_match, hash_match.
        """
        live = list(columns)
        expected = list(self.columns)
        missing = [c for c in expected if c not in live]
        extra = [c for c in live if c not in expected]
        order_match = live == expected if not missing and not extra else False
        live_hash = self.hash_columns(live) if live else ""
        hash_match = live_hash == self.columns_hash if self.columns_hash else order_match

        ok = not missing
        if strict:
            ok = ok and not extra and order_match

        report = {
            "ok": ok,
            "strict": strict,
            "missing": missing,
            "extra": extra,
            "order_match": order_match,
            "hash_match": hash_match,
            "expected_n": len(expected),
            "live_n": len(live),
            "schema_version": self.version,
            "schema_hash": self.columns_hash,
            "live_hash": live_hash,
        }
        if not ok:
            logger.warning(
                "Feature schema mismatch: missing=%s extra=%s",
                missing,
                extra[:10] if len(extra) > 10 else extra,
            )
        return report

    def validate_frame(
        self, X: pd.DataFrame, *, strict: bool = False
    ) -> Dict[str, Any]:
        cols = list(X.columns) if X is not None and not X.empty else []
        return self.validate_columns(cols, strict=strict)

    def is_compatible(self, other: "FeatureSchema") -> bool:
        return self.columns_hash == other.columns_hash and self.columns == other.columns


@dataclass
class MetricsReport:
    """Holdout evaluation bundle stored in model_meta.json."""

    n_train: int
    n_test: int
    n_features: int
    xgb_test_accuracy: Optional[float] = None
    lstm_test_accuracy: Optional[float] = None
    ensemble_test_accuracy: Optional[float] = None
    top3_accuracy: Optional[float] = None
    log_loss: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    lift_vs_baseline: Optional[float] = None
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    confusion_diag: Dict[str, int] = field(default_factory=dict)
    passed_gate: bool = True
    min_accuracy_required: float = 0.0
    trained_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    data_path: str = ""
    schema_version: str = SCHEMA_VERSION
    schema_hash: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsReport":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        kwargs = {k: v for k, v in data.items() if k in known}
        extras = {k: v for k, v in data.items() if k not in known}
        if extras:
            kwargs["extras"] = {**kwargs.get("extras", {}), **extras}
        return cls(**kwargs)  # type: ignore[arg-type]

    def primary_accuracy(self) -> float:
        for key in (
            self.ensemble_test_accuracy,
            self.xgb_test_accuracy,
            self.lstm_test_accuracy,
        ):
            if key is not None:
                return float(key)
        return 0.0

    def summary_line(self) -> str:
        return (
            f"acc={self.primary_accuracy():.3f} top3={self.top3_accuracy} "
            f"baseline={self.baseline_accuracy} lift={self.lift_vs_baseline} "
            f"gate={'PASS' if self.passed_gate else 'FAIL'} "
            f"(min={self.min_accuracy_required})"
        )
