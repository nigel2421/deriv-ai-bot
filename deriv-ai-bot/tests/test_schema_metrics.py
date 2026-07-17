import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai.metrics import (
    evaluate_predictions,
    majority_baseline,
    passes_accuracy_gate,
    top_k_accuracy,
)
from src.ai.schema import FeatureSchema, MetricsReport, SCHEMA_VERSION


def test_feature_schema_roundtrip(tmp_path: Path):
    cols = ["a", "b", "last_digit"]
    schema = FeatureSchema.from_columns(cols, notes="unit")
    schema.save(tmp_path)
    assert (tmp_path / "feature_schema.json").is_file()
    assert (tmp_path / "feature_columns.json").is_file()

    loaded = FeatureSchema.load(tmp_path)
    assert loaded is not None
    assert loaded.is_compatible(schema)
    assert loaded.version == SCHEMA_VERSION
    assert loaded.columns_hash == schema.columns_hash


def test_schema_validate_missing_extra():
    schema = FeatureSchema.from_columns(["a", "b", "c"])
    r = schema.validate_columns(["a", "b"], strict=False)
    assert r["ok"] is False
    assert r["missing"] == ["c"]

    r2 = schema.validate_columns(["a", "b", "c", "d"], strict=False)
    assert r2["ok"] is True  # missing none
    assert r2["extra"] == ["d"]
    r3 = schema.validate_columns(["a", "b", "c", "d"], strict=True)
    assert r3["ok"] is False


def test_majority_baseline_and_gate():
    y = [1, 1, 1, 2, 3]
    base = majority_baseline(y)
    assert abs(base - 0.6) < 1e-9
    assert passes_accuracy_gate(0.7, base, min_accuracy=0.12, min_lift=0.0)
    assert not passes_accuracy_gate(0.1, base, min_accuracy=0.12, min_lift=0.0)
    assert not passes_accuracy_gate(0.61, base, min_accuracy=0.12, min_lift=0.05)


def test_top_k_and_evaluate():
    y = [0, 1, 2]
    proba = np.array(
        [
            [0.5, 0.3, 0.2] + [0] * 7,
            [0.1, 0.6, 0.3] + [0] * 7,
            [0.2, 0.2, 0.6] + [0] * 7,
        ]
    )
    assert top_k_accuracy(y, proba, k=1) == 1.0
    pred = [0, 1, 1]
    ev = evaluate_predictions(y, pred, proba)
    assert "accuracy" in ev
    assert "baseline_accuracy" in ev
    assert "per_class_f1" in ev


def test_metrics_report_dict():
    m = MetricsReport(
        n_train=10,
        n_test=5,
        n_features=3,
        xgb_test_accuracy=0.4,
        baseline_accuracy=0.2,
        lift_vs_baseline=0.2,
        passed_gate=True,
        min_accuracy_required=0.12,
    )
    d = m.to_dict()
    m2 = MetricsReport.from_dict(d)
    assert m2.primary_accuracy() == 0.4
    assert "acc=" in m2.summary_line()
