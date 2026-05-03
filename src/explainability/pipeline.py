"""SentinelNet Phase 10 explainability pipeline."""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_pipeline.logging_utils import configure_logging
from src.models.ensemble.config import EnsembleConfig
from src.models.ensemble.training import (
    _build_binary_probability_maps,
    _build_multiclass_probability_maps,
    _extract_binary_weight_map,
    _extract_multiclass_weight_map,
    _weighted_average,
)

from .attribution import (
    build_global_contribution_summary,
    build_phase6_predictor,
    build_top_contribution_rows,
    label_names_from_ids,
    load_json,
    load_phase6_metrics,
    load_phase6_payload,
    monte_carlo_shap_for_target_probabilities,
    native_feature_importance,
    select_best_phase6_model,
)
from .config import ExplainabilityConfig
from .data import ExplainabilityDataset, load_explainability_dataset, sample_subset

LOGGER = logging.getLogger("sentinelnet.phase10")


@dataclass(slots=True)
class ExplainabilityArtifact:
    """Single generated Phase 10 artifact."""

    scope: str
    task_name: str
    explainer_name: str
    artifact_type: str
    artifact_path: str
    row_count: int
    backend: str | None
    output_space: str | None
    status: str
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the artifact entry into a JSON-serializable dictionary."""
        return {
            "scope": self.scope,
            "task": self.task_name,
            "explainer_name": self.explainer_name,
            "artifact_type": self.artifact_type,
            "artifact_path": self.artifact_path,
            "row_count": self.row_count,
            "backend": self.backend,
            "output_space": self.output_space,
            "status": self.status,
            "message": self.message,
        }


@dataclass(slots=True)
class ExplainabilityReport:
    """Serializable report for SentinelNet Phase 10."""

    created_at_utc: str
    input_path: str
    output_dir: str
    report_path: str
    feature_count: int
    train_rows: int
    test_rows: int
    selected_phase6_models: dict[str, Any]
    artifacts: list[dict[str, Any]]
    validation_passed: bool
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "report_path": self.report_path,
            "feature_count": self.feature_count,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "selected_phase6_models": self.selected_phase6_models,
            "artifacts": self.artifacts,
            "validation_passed": self.validation_passed,
            "config": self.config,
        }


def build_phase10_logger(config: ExplainabilityConfig) -> logging.Logger:
    """Create the dedicated Phase 10 logger."""
    return configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase10")


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Write a DataFrame to CSV and ensure its parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _register_artifact(
    artifacts: list[ExplainabilityArtifact],
    *,
    scope: str,
    task_name: str,
    explainer_name: str,
    artifact_type: str,
    artifact_path: Path,
    row_count: int,
    backend: str | None,
    output_space: str | None,
    status: str = "generated",
    message: str | None = None,
) -> None:
    """Append a generated artifact to the running manifest."""
    artifacts.append(
        ExplainabilityArtifact(
            scope=scope,
            task_name=task_name,
            explainer_name=explainer_name,
            artifact_type=artifact_type,
            artifact_path=str(artifact_path),
            row_count=int(row_count),
            backend=backend,
            output_space=output_space,
            status=status,
            message=message,
        )
    )


def _build_report(
    config: ExplainabilityConfig,
    dataset: ExplainabilityDataset,
    selected_phase6_models: dict[str, Any],
    artifacts: list[ExplainabilityArtifact],
) -> ExplainabilityReport:
    """Build the current Phase 10 report snapshot."""
    return ExplainabilityReport(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        input_path=str(config.input_data_path),
        output_dir=str(config.output_dir),
        report_path=str(config.report_path),
        feature_count=len(dataset.feature_names),
        train_rows=len(dataset.X_train),
        test_rows=len(dataset.X_test),
        selected_phase6_models=selected_phase6_models,
        artifacts=[artifact.to_dict() for artifact in artifacts],
        validation_passed=all(artifact.status == "generated" for artifact in artifacts) and len(artifacts) > 0,
        config=config.to_dict(),
    )


def _to_ensemble_config(config: ExplainabilityConfig) -> EnsembleConfig:
    """Build a lightweight Phase 9 config wrapper for shared inference helpers."""
    return EnsembleConfig(
        project_root=config.project_root,
        input_data_path=config.input_data_path,
        feature_manifest_path=config.feature_manifest_path,
        train_indices_path=config.train_indices_path,
        test_indices_path=config.test_indices_path,
        phase6_output_dir=config.phase6_output_dir,
        phase7_output_dir=config.phase7_output_dir,
        phase8_output_dir=config.phase8_output_dir,
        output_dir=config.phase9_output_dir,
        logs_dir=config.logs_dir,
    )


def _phase6_native_importance(
    config: ExplainabilityConfig,
    dataset: ExplainabilityDataset,
    artifacts: list[ExplainabilityArtifact],
) -> None:
    """Extract native feature importance tables for every trained Phase 6 model."""
    metrics_frame = load_phase6_metrics(config.phase6_output_dir / "metrics_summary.csv")
    trained_rows = metrics_frame.loc[metrics_frame["status"] == "trained"].copy()

    for row in trained_rows.itertuples(index=False):
        payload = load_phase6_payload(Path(str(row.artifact_path)))
        importance_frame = native_feature_importance(payload, dataset.inverse_multiclass_mapping)
        output_path = config.phase6_native_dir / f"{row.task}_{row.model}.csv"
        _write_dataframe(output_path, importance_frame)
        _register_artifact(
            artifacts,
            scope="phase6",
            task_name=str(row.task),
            explainer_name=str(row.model),
            artifact_type="native_importance",
            artifact_path=output_path,
            row_count=len(importance_frame),
            backend="native_model_attributes",
            output_space=None,
        )


def _run_phase6_shap_task(
    config: ExplainabilityConfig,
    dataset: ExplainabilityDataset,
    artifacts: list[ExplainabilityArtifact],
    task_name: str,
    selection_metric: str,
) -> dict[str, Any]:
    """Generate raw-feature SHAP-style explanations for the best Phase 6 model in a task."""
    metrics_frame = load_phase6_metrics(config.phase6_output_dir / "metrics_summary.csv")
    selection = select_best_phase6_model(metrics_frame, task_name, selection_metric)
    payload = load_phase6_payload(selection.artifact_path)

    if task_name == "binary":
        X_background, _, _, _ = sample_subset(
            dataset.X_train,
            dataset.y_binary_train,
            dataset.source_file_train,
            dataset.train_indices,
            config.background_sample_cap,
            config.random_state,
        )
        X_explain, y_true, _, sample_indices = sample_subset(
            dataset.X_test,
            dataset.y_binary_test,
            dataset.source_file_test,
            dataset.test_indices,
            config.binary_explain_cap,
            config.random_state,
        )
        predictor = build_phase6_predictor(payload, dataset.feature_names)
        probabilities = predictor(X_explain)
        label_ids = [int(label) for label in payload["labels"]]
        predicted_labels = np.asarray(label_ids, dtype=np.int32)[np.argmax(probabilities, axis=1)]
        attack_column = label_ids.index(1) if 1 in label_ids else len(label_ids) - 1
        target_columns = np.full(len(X_explain), attack_column, dtype=np.int32)
        target_labels = np.full(len(X_explain), 1, dtype=np.int32)
        target_label_names = ["ATTACK"] * len(X_explain)
        output_space = "attack_probability"
    else:
        X_background, _, _, _ = sample_subset(
            dataset.X_train,
            dataset.y_multiclass_train,
            dataset.source_file_train,
            dataset.train_indices,
            config.background_sample_cap,
            config.random_state,
        )
        X_explain, y_true, _, sample_indices = sample_subset(
            dataset.X_test,
            dataset.y_multiclass_test,
            dataset.source_file_test,
            dataset.test_indices,
            config.multiclass_explain_cap,
            config.random_state,
        )
        predictor = build_phase6_predictor(payload, dataset.feature_names)
        probabilities = predictor(X_explain)
        label_ids = [int(label) for label in payload["labels"]]
        predicted_columns = np.argmax(probabilities, axis=1).astype(np.int32, copy=False)
        predicted_labels = np.asarray([label_ids[index] for index in predicted_columns], dtype=np.int32)
        target_columns = predicted_columns
        target_labels = predicted_labels
        target_label_names = label_names_from_ids(target_labels, dataset.inverse_multiclass_mapping, task_name)
        output_space = "predicted_class_probability"

    shap_result = monte_carlo_shap_for_target_probabilities(
        predictor=predictor,
        X_background=X_background,
        X_explain=X_explain,
        target_columns=target_columns,
        target_labels=target_labels,
        target_label_names=target_label_names,
        n_permutations=config.shap_permutations,
        random_state=config.random_state,
    )

    true_label_names = label_names_from_ids(y_true, dataset.inverse_multiclass_mapping, task_name)
    predicted_label_names = label_names_from_ids(predicted_labels, dataset.inverse_multiclass_mapping, task_name)
    summary_frame = build_global_contribution_summary(
        shap_result.contributions,
        dataset.feature_names,
        task_name=task_name,
        explainer_name=selection.model_name,
        contribution_scope="raw_feature_shap",
    )
    local_frame = build_top_contribution_rows(
        shap_result.contributions,
        dataset.feature_names,
        sample_indices,
        y_true,
        true_label_names,
        target_labels,
        target_label_names,
        predicted_labels,
        predicted_label_names,
        shap_result.explained_values,
        shap_result.base_values,
        explainer_name=selection.model_name,
        task_name=task_name,
        output_space=output_space,
        top_k=config.top_features_per_explanation,
    )

    summary_path = config.phase6_shap_dir / f"{task_name}_{selection.model_name}_summary.csv"
    local_path = config.phase6_shap_dir / f"{task_name}_{selection.model_name}_local.csv"
    _write_dataframe(summary_path, summary_frame)
    _write_dataframe(local_path, local_frame)
    _register_artifact(
        artifacts,
        scope="phase6",
        task_name=task_name,
        explainer_name=selection.model_name,
        artifact_type="shap_summary",
        artifact_path=summary_path,
        row_count=len(summary_frame),
        backend=shap_result.backend,
        output_space=output_space,
    )
    _register_artifact(
        artifacts,
        scope="phase6",
        task_name=task_name,
        explainer_name=selection.model_name,
        artifact_type="local_explanations",
        artifact_path=local_path,
        row_count=len(local_frame),
        backend=shap_result.backend,
        output_space=output_space,
    )

    return {
        "task": task_name,
        "model_name": selection.model_name,
        "metric_name": selection.metric_name,
        "metric_value": selection.metric_value,
        "artifact_path": str(selection.artifact_path),
        "backend": shap_result.backend,
    }


def _soft_or_weighted_binary_explanations(
    *,
    config: ExplainabilityConfig,
    artifacts: list[ExplainabilityArtifact],
    variant_name: str,
    weight_map: dict[str, float],
    probability_map_background: dict[str, np.ndarray],
    probability_map_explain: dict[str, np.ndarray],
    y_true: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    """Explain a binary Phase 9 voting/scoring ensemble in probability space."""
    component_names = list(probability_map_explain)
    raw_weights = np.asarray([float(weight_map[name]) for name in component_names], dtype=np.float64)
    raw_weights = np.clip(raw_weights, a_min=0.0, a_max=None)
    raw_weights /= raw_weights.sum() if raw_weights.sum() > 0 else 1.0
    baseline_vector = np.asarray(
        [np.mean(probability_map_background[name][:, 1]) for name in component_names],
        dtype=np.float64,
    )
    explain_matrix = np.column_stack([probability_map_explain[name][:, 1] for name in component_names]).astype(np.float64, copy=False)
    weight_vector = raw_weights
    contributions = (explain_matrix - baseline_vector.reshape(1, -1)) * weight_vector.reshape(1, -1)
    base_values = np.full(len(explain_matrix), float(np.dot(weight_vector, baseline_vector)), dtype=np.float64)
    explained_values = base_values + np.sum(contributions, axis=1)
    predicted_labels = (explained_values >= 0.5).astype(np.int32)

    summary_frame = build_global_contribution_summary(
        contributions,
        component_names,
        task_name="binary",
        explainer_name=variant_name,
        contribution_scope="ensemble_component_probability",
    )
    local_frame = build_top_contribution_rows(
        contributions,
        component_names,
        sample_indices,
        y_true,
        label_names_from_ids(y_true, {}, "binary"),
        np.full(len(y_true), 1, dtype=np.int32),
        ["ATTACK"] * len(y_true),
        predicted_labels,
        label_names_from_ids(predicted_labels, {}, "binary"),
        explained_values,
        base_values,
        explainer_name=variant_name,
        task_name="binary",
        output_space="attack_probability",
        top_k=config.top_features_per_explanation,
    )

    summary_path = config.phase9_dir / f"binary_{variant_name}_summary.csv"
    local_path = config.phase9_dir / f"binary_{variant_name}_local.csv"
    _write_dataframe(summary_path, summary_frame)
    _write_dataframe(local_path, local_frame)
    _register_artifact(
        artifacts,
        scope="phase9",
        task_name="binary",
        explainer_name=variant_name,
        artifact_type="component_summary",
        artifact_path=summary_path,
        row_count=len(summary_frame),
        backend="weighted_probability_decomposition",
        output_space="attack_probability",
    )
    _register_artifact(
        artifacts,
        scope="phase9",
        task_name="binary",
        explainer_name=variant_name,
        artifact_type="local_explanations",
        artifact_path=local_path,
        row_count=len(local_frame),
        backend="weighted_probability_decomposition",
        output_space="attack_probability",
    )


def _stacking_binary_explanations(
    *,
    config: ExplainabilityConfig,
    artifacts: list[ExplainabilityArtifact],
    probability_map_background: dict[str, np.ndarray],
    probability_map_explain: dict[str, np.ndarray],
    y_true: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    """Explain the binary Phase 9 stacking model in logit space."""
    payload = joblib.load(config.phase9_output_dir / "binary" / "stacking.joblib")
    model = payload["model"]
    component_names = [str(name) for name in payload["feature_order"]]
    X_background = np.column_stack([probability_map_background[name][:, 1] for name in component_names]).astype(np.float64, copy=False)
    X_explain = np.column_stack([probability_map_explain[name][:, 1] for name in component_names]).astype(np.float64, copy=False)
    baseline = np.mean(X_background, axis=0)
    coefficients = np.asarray(model.coef_[0], dtype=np.float64)
    contributions = (X_explain - baseline.reshape(1, -1)) * coefficients.reshape(1, -1)
    base_value = float(model.intercept_[0] + np.dot(baseline, coefficients))
    base_values = np.full(len(X_explain), base_value, dtype=np.float64)
    explained_values = np.asarray(model.decision_function(X_explain), dtype=np.float64).reshape(-1)
    predicted_labels = np.asarray(model.predict(X_explain), dtype=np.int32)

    summary_frame = build_global_contribution_summary(
        contributions,
        component_names,
        task_name="binary",
        explainer_name="stacking",
        contribution_scope="ensemble_component_logit",
    )
    local_frame = build_top_contribution_rows(
        contributions,
        component_names,
        sample_indices,
        y_true,
        label_names_from_ids(y_true, {}, "binary"),
        np.full(len(y_true), 1, dtype=np.int32),
        ["ATTACK"] * len(y_true),
        predicted_labels,
        label_names_from_ids(predicted_labels, {}, "binary"),
        explained_values,
        base_values,
        explainer_name="stacking",
        task_name="binary",
        output_space="attack_logit",
        top_k=config.top_features_per_explanation,
    )

    summary_path = config.phase9_dir / "binary_stacking_summary.csv"
    local_path = config.phase9_dir / "binary_stacking_local.csv"
    _write_dataframe(summary_path, summary_frame)
    _write_dataframe(local_path, local_frame)
    _register_artifact(
        artifacts,
        scope="phase9",
        task_name="binary",
        explainer_name="stacking",
        artifact_type="component_summary",
        artifact_path=summary_path,
        row_count=len(summary_frame),
        backend="exact_linear_decomposition",
        output_space="attack_logit",
    )
    _register_artifact(
        artifacts,
        scope="phase9",
        task_name="binary",
        explainer_name="stacking",
        artifact_type="local_explanations",
        artifact_path=local_path,
        row_count=len(local_frame),
        backend="exact_linear_decomposition",
        output_space="attack_logit",
    )


def _soft_or_weighted_multiclass_explanations(
    *,
    config: ExplainabilityConfig,
    dataset: ExplainabilityDataset,
    artifacts: list[ExplainabilityArtifact],
    variant_name: str,
    weight_map: dict[str, float],
    multiclass_labels: list[int],
    probability_map_background: dict[str, np.ndarray],
    probability_map_explain: dict[str, np.ndarray],
    y_true: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    """Explain a multiclass Phase 9 voting/scoring ensemble in probability space."""
    component_names = list(probability_map_explain)
    raw_weights = np.asarray([float(weight_map[name]) for name in component_names], dtype=np.float64)
    raw_weights = np.clip(raw_weights, a_min=0.0, a_max=None)
    raw_weights /= raw_weights.sum() if raw_weights.sum() > 0 else 1.0
    weights = {name: float(raw_weights[index]) for index, name in enumerate(component_names)}
    final_probabilities = _weighted_average(probability_map_explain, weights)
    predicted_columns = np.argmax(final_probabilities, axis=1).astype(np.int32, copy=False)
    predicted_labels = np.asarray([multiclass_labels[index] for index in predicted_columns], dtype=np.int32)

    baseline_probabilities = {
        name: np.mean(probability_map_background[name], axis=0)
        for name in component_names
    }
    contributions = np.zeros((len(sample_indices), len(component_names)), dtype=np.float64)
    base_values = np.zeros(len(sample_indices), dtype=np.float64)
    explained_values = np.zeros(len(sample_indices), dtype=np.float64)

    for sample_index, predicted_column in enumerate(predicted_columns):
        contribution_row = []
        base_value = 0.0
        explained_value = float(final_probabilities[sample_index, predicted_column])
        for component_name in component_names:
            weight = weights[component_name]
            baseline_probability = float(baseline_probabilities[component_name][predicted_column])
            sample_probability = float(probability_map_explain[component_name][sample_index, predicted_column])
            base_value += weight * baseline_probability
            contribution_row.append(weight * (sample_probability - baseline_probability))
        contributions[sample_index] = np.asarray(contribution_row, dtype=np.float64)
        base_values[sample_index] = base_value
        explained_values[sample_index] = explained_value

    summary_frame = build_global_contribution_summary(
        contributions,
        component_names,
        task_name="multiclass",
        explainer_name=variant_name,
        contribution_scope="ensemble_component_probability",
    )
    local_frame = build_top_contribution_rows(
        contributions,
        component_names,
        sample_indices,
        y_true,
        label_names_from_ids(y_true, dataset.inverse_multiclass_mapping, "multiclass"),
        predicted_labels,
        label_names_from_ids(predicted_labels, dataset.inverse_multiclass_mapping, "multiclass"),
        predicted_labels,
        label_names_from_ids(predicted_labels, dataset.inverse_multiclass_mapping, "multiclass"),
        explained_values,
        base_values,
        explainer_name=variant_name,
        task_name="multiclass",
        output_space="predicted_class_probability",
        top_k=config.top_features_per_explanation,
    )

    summary_path = config.phase9_dir / f"multiclass_{variant_name}_summary.csv"
    local_path = config.phase9_dir / f"multiclass_{variant_name}_local.csv"
    _write_dataframe(summary_path, summary_frame)
    _write_dataframe(local_path, local_frame)
    _register_artifact(
        artifacts,
        scope="phase9",
        task_name="multiclass",
        explainer_name=variant_name,
        artifact_type="component_summary",
        artifact_path=summary_path,
        row_count=len(summary_frame),
        backend="weighted_probability_decomposition",
        output_space="predicted_class_probability",
    )
    _register_artifact(
        artifacts,
        scope="phase9",
        task_name="multiclass",
        explainer_name=variant_name,
        artifact_type="local_explanations",
        artifact_path=local_path,
        row_count=len(local_frame),
        backend="weighted_probability_decomposition",
        output_space="predicted_class_probability",
    )


def _stacking_multiclass_explanations(
    *,
    config: ExplainabilityConfig,
    dataset: ExplainabilityDataset,
    artifacts: list[ExplainabilityArtifact],
    multiclass_labels: list[int],
    probability_map_background: dict[str, np.ndarray],
    probability_map_explain: dict[str, np.ndarray],
    anomaly_map_background: dict[str, np.ndarray],
    anomaly_map_explain: dict[str, np.ndarray],
    y_true: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    """Explain the multiclass Phase 9 stacking model in linear score space."""
    payload = joblib.load(config.phase9_output_dir / "multiclass" / "stacking.joblib")
    model = payload["model"]
    multiclass_component_names = [str(name) for name in payload["feature_order"]]
    anomaly_component_names = [str(name) for name in payload["anomaly_feature_order"]]
    component_names = [*multiclass_component_names, *anomaly_component_names]
    n_classes = len(multiclass_labels)

    X_background = np.column_stack(
        [probability_map_background[name] for name in multiclass_component_names]
        + [anomaly_map_background[name][:, 1].reshape(-1, 1) for name in anomaly_component_names]
    ).astype(np.float64, copy=False)
    X_explain = np.column_stack(
        [probability_map_explain[name] for name in multiclass_component_names]
        + [anomaly_map_explain[name][:, 1].reshape(-1, 1) for name in anomaly_component_names]
    ).astype(np.float64, copy=False)
    background_means = np.mean(X_background, axis=0)
    decision_scores = np.asarray(model.decision_function(X_explain), dtype=np.float64)
    predicted_labels = np.asarray(model.predict(X_explain), dtype=np.int32)
    class_to_row = {int(label): index for index, label in enumerate(model.classes_.astype(int).tolist())}

    contributions = np.zeros((len(sample_indices), len(component_names)), dtype=np.float64)
    base_values = np.zeros(len(sample_indices), dtype=np.float64)
    explained_values = np.zeros(len(sample_indices), dtype=np.float64)

    for sample_position, predicted_label in enumerate(predicted_labels):
        class_row = class_to_row[int(predicted_label)]
        coefficients = np.asarray(model.coef_[class_row], dtype=np.float64)
        offset = 0
        base_value = float(model.intercept_[class_row])
        component_contributions: list[float] = []

        for component_name in multiclass_component_names:
            baseline_vector = background_means[offset : offset + n_classes]
            sample_vector = X_explain[sample_position, offset : offset + n_classes]
            coefficient_vector = coefficients[offset : offset + n_classes]
            base_value += float(np.dot(baseline_vector, coefficient_vector))
            component_contributions.append(float(np.dot(sample_vector - baseline_vector, coefficient_vector)))
            offset += n_classes

        for _component_name in anomaly_component_names:
            baseline_value = float(background_means[offset])
            sample_value = float(X_explain[sample_position, offset])
            coefficient_value = float(coefficients[offset])
            base_value += baseline_value * coefficient_value
            component_contributions.append((sample_value - baseline_value) * coefficient_value)
            offset += 1

        contributions[sample_position] = np.asarray(component_contributions, dtype=np.float64)
        base_values[sample_position] = base_value
        explained_values[sample_position] = float(decision_scores[sample_position, class_row])

    summary_frame = build_global_contribution_summary(
        contributions,
        component_names,
        task_name="multiclass",
        explainer_name="stacking",
        contribution_scope="ensemble_component_score",
    )
    local_frame = build_top_contribution_rows(
        contributions,
        component_names,
        sample_indices,
        y_true,
        label_names_from_ids(y_true, dataset.inverse_multiclass_mapping, "multiclass"),
        predicted_labels,
        label_names_from_ids(predicted_labels, dataset.inverse_multiclass_mapping, "multiclass"),
        predicted_labels,
        label_names_from_ids(predicted_labels, dataset.inverse_multiclass_mapping, "multiclass"),
        explained_values,
        base_values,
        explainer_name="stacking",
        task_name="multiclass",
        output_space="predicted_class_score",
        top_k=config.top_features_per_explanation,
    )

    summary_path = config.phase9_dir / "multiclass_stacking_summary.csv"
    local_path = config.phase9_dir / "multiclass_stacking_local.csv"
    _write_dataframe(summary_path, summary_frame)
    _write_dataframe(local_path, local_frame)
    _register_artifact(
        artifacts,
        scope="phase9",
        task_name="multiclass",
        explainer_name="stacking",
        artifact_type="component_summary",
        artifact_path=summary_path,
        row_count=len(summary_frame),
        backend="exact_linear_decomposition",
        output_space="predicted_class_score",
    )
    _register_artifact(
        artifacts,
        scope="phase9",
        task_name="multiclass",
        explainer_name="stacking",
        artifact_type="local_explanations",
        artifact_path=local_path,
        row_count=len(local_frame),
        backend="exact_linear_decomposition",
        output_space="predicted_class_score",
    )


def _phase9_explanations(
    config: ExplainabilityConfig,
    dataset: ExplainabilityDataset,
    artifacts: list[ExplainabilityArtifact],
) -> None:
    """Generate ensemble-component explanations for all Phase 9 variants."""
    ensemble_config = _to_ensemble_config(config)
    phase6_report = load_json(config.phase6_output_dir / "ml_training_report.json")
    phase7_report = load_json(config.phase7_output_dir / "deep_learning_report.json")
    phase8_report = load_json(config.phase8_output_dir / "anomaly_detection_report.json")

    binary_background_X, _, binary_background_sources, binary_background_indices = sample_subset(
        dataset.X_train,
        dataset.y_binary_train,
        dataset.source_file_train,
        dataset.train_indices,
        config.background_sample_cap,
        config.random_state,
    )
    binary_explain_X, binary_y_true, binary_explain_sources, binary_explain_indices = sample_subset(
        dataset.X_test,
        dataset.y_binary_test,
        dataset.source_file_test,
        dataset.test_indices,
        config.ensemble_binary_explain_cap,
        config.random_state,
    )
    binary_priors = np.bincount(dataset.y_binary_train, minlength=2).astype(np.float64)
    binary_default = (binary_priors / binary_priors.sum()).reshape(1, -1)

    binary_background_map = _build_binary_probability_maps(
        ensemble_config,
        dataset.feature_names,
        binary_background_X,
        binary_background_sources,
        binary_background_indices,
        phase6_report,
        phase7_report,
        phase8_report,
        binary_default,
    )
    binary_explain_map = _build_binary_probability_maps(
        ensemble_config,
        dataset.feature_names,
        binary_explain_X,
        binary_explain_sources,
        binary_explain_indices,
        phase6_report,
        phase7_report,
        phase8_report,
        binary_default,
    )
    binary_uniform_weights = {name: 1.0 for name in binary_explain_map}
    binary_weighted_weights = _extract_binary_weight_map(phase6_report, phase7_report, phase8_report)
    _soft_or_weighted_binary_explanations(
        config=config,
        artifacts=artifacts,
        variant_name="soft_voting",
        weight_map=binary_uniform_weights,
        probability_map_background=binary_background_map,
        probability_map_explain=binary_explain_map,
        y_true=binary_y_true,
        sample_indices=binary_explain_indices,
    )
    _soft_or_weighted_binary_explanations(
        config=config,
        artifacts=artifacts,
        variant_name="weighted_scoring",
        weight_map=binary_weighted_weights,
        probability_map_background=binary_background_map,
        probability_map_explain=binary_explain_map,
        y_true=binary_y_true,
        sample_indices=binary_explain_indices,
    )
    _stacking_binary_explanations(
        config=config,
        artifacts=artifacts,
        probability_map_background=binary_background_map,
        probability_map_explain=binary_explain_map,
        y_true=binary_y_true,
        sample_indices=binary_explain_indices,
    )

    multiclass_background_X, _, multiclass_background_sources, multiclass_background_indices = sample_subset(
        dataset.X_train,
        dataset.y_multiclass_train,
        dataset.source_file_train,
        dataset.train_indices,
        config.background_sample_cap,
        config.random_state,
    )
    multiclass_explain_X, multiclass_y_true, multiclass_explain_sources, multiclass_explain_indices = sample_subset(
        dataset.X_test,
        dataset.y_multiclass_test,
        dataset.source_file_test,
        dataset.test_indices,
        config.ensemble_multiclass_explain_cap,
        config.random_state,
    )
    multiclass_labels = sorted(np.unique(dataset.y_multiclass_train).astype(int).tolist())
    multiclass_priors = np.bincount(dataset.y_multiclass_train, minlength=max(multiclass_labels) + 1).astype(np.float64)
    multiclass_default = (multiclass_priors[multiclass_labels] / multiclass_priors[multiclass_labels].sum()).reshape(1, -1)

    multiclass_background_map = _build_multiclass_probability_maps(
        ensemble_config,
        dataset.feature_names,
        multiclass_background_X,
        multiclass_background_sources,
        multiclass_background_indices,
        phase6_report,
        phase7_report,
        multiclass_labels,
        multiclass_default,
    )
    multiclass_explain_map = _build_multiclass_probability_maps(
        ensemble_config,
        dataset.feature_names,
        multiclass_explain_X,
        multiclass_explain_sources,
        multiclass_explain_indices,
        phase6_report,
        phase7_report,
        multiclass_labels,
        multiclass_default,
    )
    anomaly_background_map = _build_binary_probability_maps(
        ensemble_config,
        dataset.feature_names,
        multiclass_background_X,
        multiclass_background_sources,
        multiclass_background_indices,
        phase6_report,
        phase7_report,
        phase8_report,
        binary_default,
    )
    anomaly_explain_map = _build_binary_probability_maps(
        ensemble_config,
        dataset.feature_names,
        multiclass_explain_X,
        multiclass_explain_sources,
        multiclass_explain_indices,
        phase6_report,
        phase7_report,
        phase8_report,
        binary_default,
    )
    anomaly_background_map = {name: values for name, values in anomaly_background_map.items() if name.startswith("phase8_")}
    anomaly_explain_map = {name: values for name, values in anomaly_explain_map.items() if name.startswith("phase8_")}

    multiclass_uniform_weights = {name: 1.0 for name in multiclass_explain_map}
    multiclass_weighted_weights = _extract_multiclass_weight_map(phase6_report, phase7_report)
    _soft_or_weighted_multiclass_explanations(
        config=config,
        dataset=dataset,
        artifacts=artifacts,
        variant_name="soft_voting",
        weight_map=multiclass_uniform_weights,
        multiclass_labels=multiclass_labels,
        probability_map_background=multiclass_background_map,
        probability_map_explain=multiclass_explain_map,
        y_true=multiclass_y_true,
        sample_indices=multiclass_explain_indices,
    )
    _soft_or_weighted_multiclass_explanations(
        config=config,
        dataset=dataset,
        artifacts=artifacts,
        variant_name="weighted_scoring",
        weight_map=multiclass_weighted_weights,
        multiclass_labels=multiclass_labels,
        probability_map_background=multiclass_background_map,
        probability_map_explain=multiclass_explain_map,
        y_true=multiclass_y_true,
        sample_indices=multiclass_explain_indices,
    )
    _stacking_multiclass_explanations(
        config=config,
        dataset=dataset,
        artifacts=artifacts,
        multiclass_labels=multiclass_labels,
        probability_map_background=multiclass_background_map,
        probability_map_explain=multiclass_explain_map,
        anomaly_map_background=anomaly_background_map,
        anomaly_map_explain=anomaly_explain_map,
        y_true=multiclass_y_true,
        sample_indices=multiclass_explain_indices,
    )


def run_explainability_pipeline(
    config: ExplainabilityConfig,
    logger: logging.Logger | None = None,
) -> ExplainabilityReport:
    """Execute the complete SentinelNet Phase 10 explainability workflow."""
    active_logger = logger or LOGGER
    config.ensure_directories()

    required_paths = (
        config.input_data_path,
        config.feature_manifest_path,
        config.label_mapping_path,
        config.train_indices_path,
        config.test_indices_path,
        config.phase6_output_dir / "metrics_summary.csv",
        config.phase6_output_dir / "ml_training_report.json",
        config.phase7_output_dir / "deep_learning_report.json",
        config.phase8_output_dir / "anomaly_detection_report.json",
        config.phase9_output_dir / "metrics_summary.csv",
    )
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Phase 10 dependency not found at {required_path}")

    dataset = load_explainability_dataset(config)
    active_logger.info(
        "Loaded Phase 10 dataset | train_rows=%d | test_rows=%d | features=%d",
        len(dataset.X_train),
        len(dataset.X_test),
        len(dataset.feature_names),
    )

    artifacts: list[ExplainabilityArtifact] = []
    active_logger.info("Generating native feature importance tables for all Phase 6 models")
    _phase6_native_importance(config, dataset, artifacts)

    active_logger.info("Generating SHAP-style raw-feature explanations for the best binary and multiclass Phase 6 models")
    selected_binary = _run_phase6_shap_task(config, dataset, artifacts, "binary", config.binary_selection_metric)
    selected_multiclass = _run_phase6_shap_task(config, dataset, artifacts, "multiclass", config.multiclass_selection_metric)

    active_logger.info("Generating component-level explanations for all Phase 9 ensemble variants")
    _phase9_explanations(config, dataset, artifacts)

    summary_frame = pd.DataFrame([artifact.to_dict() for artifact in artifacts])
    _write_dataframe(config.summary_path, summary_frame)
    report = _build_report(
        config,
        dataset,
        selected_phase6_models={"binary": selected_binary, "multiclass": selected_multiclass},
        artifacts=artifacts,
    )
    config.report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if not report.validation_passed:
        raise ValueError("Phase 10 validation failed because one or more explainability artifacts were not generated successfully.")

    active_logger.info(
        "Completed Phase 10 explainability | artifacts=%d | validation_passed=%s",
        len(artifacts),
        report.validation_passed,
    )
    gc.collect()
    return report
