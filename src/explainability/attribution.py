"""Attribution and explainability helpers for SentinelNet Phase 10."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd


@dataclass(slots=True)
class Phase6Selection:
    """Best Phase 6 model for a task under a configured metric."""

    task_name: str
    model_name: str
    metric_name: str
    metric_value: float
    artifact_path: Path


@dataclass(slots=True)
class ShapExplanationResult:
    """Local and global SHAP-style attribution output."""

    contributions: np.ndarray
    base_values: np.ndarray
    explained_values: np.ndarray
    target_labels: np.ndarray
    target_label_names: list[str]
    backend: str


def _as_feature_frame(X_raw: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Wrap raw feature arrays in a named DataFrame for sklearn-compatible inference."""
    return pd.DataFrame(X_raw, columns=feature_names, copy=False)


def _format_features_for_artifact(
    artifact: Any,
    X_raw: np.ndarray,
    feature_names: list[str],
) -> np.ndarray | pd.DataFrame:
    """Match inference inputs to the representation used when an artifact was fit."""
    if artifact is not None and hasattr(artifact, "feature_names_in_"):
        return _as_feature_frame(X_raw, feature_names)
    return X_raw


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase6_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load the Phase 6 metrics summary."""
    frame = pd.read_csv(metrics_path)
    if frame.empty:
        raise ValueError(f"No Phase 6 metrics found in {metrics_path}")
    return frame


def select_best_phase6_model(metrics_frame: pd.DataFrame, task_name: str, metric_name: str) -> Phase6Selection:
    """Select the highest-scoring Phase 6 model for the requested task."""
    task_frame = metrics_frame.loc[metrics_frame["task"] == task_name].copy()
    if task_frame.empty:
        raise ValueError(f"No Phase 6 metrics found for task '{task_name}'.")
    if metric_name not in task_frame.columns:
        raise ValueError(f"Metric '{metric_name}' not present in Phase 6 metrics.")
    task_frame = task_frame.loc[task_frame["status"] == "trained"].copy()
    if task_frame.empty:
        raise ValueError(f"No trained Phase 6 models found for task '{task_name}'.")
    best_row = task_frame.sort_values(metric_name, ascending=False).iloc[0]
    return Phase6Selection(
        task_name=task_name,
        model_name=str(best_row["model"]),
        metric_name=metric_name,
        metric_value=float(best_row[metric_name]),
        artifact_path=Path(str(best_row["artifact_path"])).resolve(),
    )


def load_phase6_payload(artifact_path: Path) -> dict[str, Any]:
    """Load a persisted Phase 6 model payload."""
    payload = joblib.load(artifact_path)
    required = {"estimator", "feature_names", "labels", "task", "model_name"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Phase 6 artifact {artifact_path} is missing keys: {sorted(missing)}")
    return payload


def predict_phase6_probabilities(
    payload: dict[str, Any],
    X_raw: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Run probability inference for a Phase 6 payload on raw selected features."""
    estimator = payload["estimator"]
    scaler = payload.get("scaler")
    X_scaler = _format_features_for_artifact(scaler, X_raw, feature_names) if scaler is not None else X_raw
    X_model = scaler.transform(X_scaler) if scaler is not None else X_scaler
    X_estimator = _format_features_for_artifact(estimator, X_model, feature_names)
    return np.asarray(estimator.predict_proba(X_estimator), dtype=np.float64)


def native_feature_importance(
    payload: dict[str, Any],
    inverse_multiclass_mapping: dict[int, str],
) -> pd.DataFrame:
    """Extract model-native feature importance or coefficient magnitudes."""
    estimator = payload["estimator"]
    feature_names = [str(feature) for feature in payload["feature_names"]]
    task_name = str(payload["task"])
    model_name = str(payload["model_name"])
    labels = [int(label) for label in payload["labels"]]

    rows: list[dict[str, Any]] = []
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=np.float64)
        for feature_name, importance in zip(feature_names, importances, strict=True):
            rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "importance_scope": "aggregated",
                    "class_label": None,
                    "class_name": None,
                    "feature_name": feature_name,
                    "importance": float(importance),
                    "importance_type": "native_feature_importance",
                }
            )
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=np.float64)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)

        aggregated = np.mean(np.abs(coefficients), axis=0)
        for feature_name, importance in zip(feature_names, aggregated, strict=True):
            rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "importance_scope": "aggregated",
                    "class_label": None,
                    "class_name": None,
                    "feature_name": feature_name,
                    "importance": float(importance),
                    "importance_type": "absolute_coefficient",
                }
            )

        for row_index, coefficients_row in enumerate(coefficients):
            class_label = labels[min(row_index, len(labels) - 1)] if task_name == "multiclass" else 1
            class_name = "ATTACK" if task_name == "binary" else inverse_multiclass_mapping.get(class_label, str(class_label))
            for feature_name, coefficient in zip(feature_names, np.abs(coefficients_row), strict=True):
                rows.append(
                    {
                        "task": task_name,
                        "model": model_name,
                        "importance_scope": "class_specific",
                        "class_label": class_label,
                        "class_name": class_name,
                        "feature_name": feature_name,
                        "importance": float(coefficient),
                        "importance_type": "absolute_coefficient",
                    }
                )
    else:
        raise ValueError(
            f"Estimator {type(estimator).__name__} in model '{model_name}' does not expose "
            "feature_importances_ or coef_."
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["importance_scope", "importance", "feature_name"], ascending=[True, False, True]).reset_index(drop=True)


def build_phase6_predictor(
    payload: dict[str, Any],
    feature_names: list[str],
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a vectorized probability predictor for a Phase 6 payload."""

    def predictor(X_values: np.ndarray) -> np.ndarray:
        return predict_phase6_probabilities(payload, X_values, feature_names)

    return predictor


def monte_carlo_shap_for_target_probabilities(
    predictor: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    X_explain: np.ndarray,
    target_columns: np.ndarray,
    target_labels: np.ndarray,
    target_label_names: list[str],
    n_permutations: int,
    random_state: int,
) -> ShapExplanationResult:
    """Approximate SHAP values by sampling random feature permutations."""
    if len(X_background) == 0:
        raise ValueError("Background set for Monte Carlo SHAP cannot be empty.")
    if len(X_explain) == 0:
        raise ValueError("Explain set for Monte Carlo SHAP cannot be empty.")
    if len(X_explain) != len(target_columns) or len(target_columns) != len(target_labels) or len(target_labels) != len(target_label_names):
        raise ValueError("Explain rows, target columns, target labels, and target label names must have matching lengths.")

    background_probabilities = predictor(X_background)
    explain_probabilities = predictor(X_explain)
    n_samples, n_features = X_explain.shape
    contributions = np.zeros((n_samples, n_features), dtype=np.float64)
    base_values = np.zeros(n_samples, dtype=np.float64)
    explained_values = np.zeros(n_samples, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    for sample_index, sample in enumerate(X_explain):
        target_column = int(target_columns[sample_index])
        target_label = int(target_labels[sample_index])
        base_value = float(np.mean(background_probabilities[:, target_column]))
        explained_value = float(explain_probabilities[sample_index, target_column])
        base_values[sample_index] = base_value
        explained_values[sample_index] = explained_value

        for _ in range(max(n_permutations, 1)):
            permutation = rng.permutation(n_features)
            coalition = X_background.copy()
            previous_value = base_value
            for feature_index in permutation:
                coalition[:, feature_index] = sample[feature_index]
                current_value = float(np.mean(predictor(coalition)[:, target_column]))
                contributions[sample_index, feature_index] += current_value - previous_value
                previous_value = current_value

        contributions[sample_index] /= max(n_permutations, 1)

    return ShapExplanationResult(
        contributions=contributions,
        base_values=base_values,
        explained_values=explained_values,
        target_labels=target_labels.astype(np.int32, copy=False),
        target_label_names=target_label_names,
        backend="internal_monte_carlo_shap",
    )


def build_global_contribution_summary(
    contributions: np.ndarray,
    feature_names: list[str],
    task_name: str,
    explainer_name: str,
    contribution_scope: str,
) -> pd.DataFrame:
    """Aggregate local contribution matrices into a global importance summary."""
    mean_abs = np.mean(np.abs(contributions), axis=0)
    mean_value = np.mean(contributions, axis=0)
    frame = pd.DataFrame(
        {
            "task": task_name,
            "explainer_name": explainer_name,
            "contribution_scope": contribution_scope,
            "feature_name": feature_names,
            "mean_abs_contribution": mean_abs.astype(np.float64, copy=False),
            "mean_contribution": mean_value.astype(np.float64, copy=False),
        }
    )
    return frame.sort_values(["mean_abs_contribution", "feature_name"], ascending=[False, True]).reset_index(drop=True)


def build_top_contribution_rows(
    contributions: np.ndarray,
    feature_names: list[str],
    sample_indices: np.ndarray,
    true_labels: np.ndarray,
    true_label_names: list[str],
    target_labels: np.ndarray,
    target_label_names: list[str],
    predicted_labels: np.ndarray,
    predicted_label_names: list[str],
    explained_values: np.ndarray,
    base_values: np.ndarray,
    explainer_name: str,
    task_name: str,
    output_space: str,
    top_k: int,
) -> pd.DataFrame:
    """Flatten the top local contributions per sample into a readable long-form table."""
    rows: list[dict[str, Any]] = []
    n_features = contributions.shape[1]
    effective_top_k = min(max(top_k, 1), n_features)

    for sample_position, sample_index in enumerate(sample_indices):
        contribution_row = contributions[sample_position]
        ranking = np.argsort(np.abs(contribution_row))[::-1][:effective_top_k]
        for rank, feature_index in enumerate(ranking, start=1):
            rows.append(
                {
                    "task": task_name,
                    "explainer_name": explainer_name,
                    "sample_index": int(sample_index),
                    "true_label": int(true_labels[sample_position]),
                    "true_label_name": true_label_names[sample_position],
                    "target_label": int(target_labels[sample_position]),
                    "target_label_name": target_label_names[sample_position],
                    "predicted_label": int(predicted_labels[sample_position]),
                    "predicted_label_name": predicted_label_names[sample_position],
                    "rank": rank,
                    "feature_name": feature_names[feature_index],
                    "contribution": float(contribution_row[feature_index]),
                    "abs_contribution": float(abs(contribution_row[feature_index])),
                    "base_value": float(base_values[sample_position]),
                    "explained_value": float(explained_values[sample_position]),
                    "output_space": output_space,
                }
            )

    return pd.DataFrame(rows)


def label_names_from_ids(labels: np.ndarray, inverse_multiclass_mapping: dict[int, str], task_name: str) -> list[str]:
    """Convert label ids into stable readable names."""
    if task_name == "binary":
        return ["ATTACK" if int(label) == 1 else "BENIGN" for label in labels]
    return [inverse_multiclass_mapping.get(int(label), str(int(label))) for label in labels]
