"""Anomaly-detection metrics for SentinelNet."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_anomaly_scores(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate anomaly scores against a binary ground truth."""
    y_true = np.asarray(y_true, dtype=np.int32)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64)
    y_pred = (anomaly_scores >= threshold).astype(np.int32)

    if np.max(anomaly_scores) > np.min(anomaly_scores):
        normalized_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    else:
        normalized_scores = np.zeros_like(anomaly_scores, dtype=np.float64)

    roc_auc = None
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, normalized_scores))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "score_min": float(np.min(anomaly_scores)),
        "score_max": float(np.max(anomaly_scores)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
    }
