"""Evaluation helpers for digit classifiers."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

logger = logging.getLogger(__name__)


def majority_baseline(y_true: Sequence[int]) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=10)
    return float(counts.max() / len(y))


def top_k_accuracy(
    y_true: Sequence[int], proba: np.ndarray, k: int = 3
) -> float:
    """Fraction of rows where true label is in top-k predicted classes."""
    y = np.asarray(y_true, dtype=int)
    if proba is None or len(y) == 0:
        return 0.0
    k = min(k, proba.shape[1])
    topk = np.argsort(-proba, axis=1)[:, :k]
    hits = [y[i] in topk[i] for i in range(len(y))]
    return float(np.mean(hits))


def safe_log_loss(y_true: Sequence[int], proba: np.ndarray) -> Optional[float]:
    try:
        y = np.asarray(y_true, dtype=int)
        # Pad labels 0-9 for sklearn
        return float(log_loss(y, proba, labels=list(range(10))))
    except Exception as e:
        logger.debug("log_loss failed: %s", e)
        return None


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    acc = float(accuracy_score(y, p)) if len(y) else 0.0
    baseline = majority_baseline(y)
    lift = acc - baseline

    per_class: Dict[str, float] = {}
    try:
        f1s = f1_score(y, p, labels=list(range(10)), average=None, zero_division=0)
        per_class = {str(i): float(f1s[i]) for i in range(10)}
    except Exception:
        pass

    diag: Dict[str, int] = {}
    try:
        cm = confusion_matrix(y, p, labels=list(range(10)))
        for i in range(10):
            diag[str(i)] = int(cm[i, i])
    except Exception:
        pass

    report_txt = ""
    try:
        report_txt = classification_report(y, p, digits=3, zero_division=0)
    except Exception:
        pass

    out: Dict[str, Any] = {
        "accuracy": acc,
        "baseline_accuracy": baseline,
        "lift_vs_baseline": lift,
        "per_class_f1": per_class,
        "confusion_diag": diag,
        "classification_report": report_txt,
        "n": int(len(y)),
    }
    if proba is not None:
        out["top3_accuracy"] = top_k_accuracy(y, proba, k=3)
        out["log_loss"] = safe_log_loss(y, proba)
    return out


def passes_accuracy_gate(
    accuracy: float,
    baseline: float,
    *,
    min_accuracy: float,
    min_lift: float = 0.0,
) -> bool:
    """
    Gate: accuracy >= min_accuracy AND accuracy >= baseline + min_lift.
    """
    if accuracy < min_accuracy:
        return False
    if accuracy + 1e-12 < baseline + min_lift:
        return False
    return True
