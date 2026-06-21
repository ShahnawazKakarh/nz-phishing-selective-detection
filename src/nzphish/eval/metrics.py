"""Evaluation metrics for binary phishing classification.

Includes classification metrics (Acc / P / R / F1 / ROC-AUC) plus selective
metrics (coverage, selective accuracy, selective F1) used by the deferral layer.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_phish": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_phish": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_phish": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "n": int(len(y_true)),
        "n_phish": int(np.sum(y_true == 1)),
    }


def selective_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, defer_mask: np.ndarray, threshold: float = 0.5
) -> dict:
    """Metrics on the *kept* (non-deferred) subset.

    Args:
        defer_mask: boolean array, True where the sample is deferred to a human.
    """
    keep = ~defer_mask
    if keep.sum() == 0:
        return {"coverage": 0.0, "n_kept": 0}
    y_pred = (y_prob[keep] >= threshold).astype(int)
    return {
        "coverage": float(keep.mean()),
        "n_kept": int(keep.sum()),
        "n_deferred": int(defer_mask.sum()),
        "selective_accuracy": float(accuracy_score(y_true[keep], y_pred)),
        "selective_precision_phish": float(precision_score(y_true[keep], y_pred, zero_division=0)),
        "selective_recall_phish": float(recall_score(y_true[keep], y_pred, zero_division=0)),
        "selective_f1_phish": float(f1_score(y_true[keep], y_pred, zero_division=0)),
    }
