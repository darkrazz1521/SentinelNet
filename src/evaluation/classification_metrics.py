"""Classification metrics for SentinelNet model evaluation."""

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


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: list[int],
    task_name: str,
) -> dict[str, Any]:
    """Compute classification metrics for binary or multiclass tasks."""
    average = "binary" if task_name == "binary" else "weighted"
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    if task_name == "binary":
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    else:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_proba,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )

    return metrics

