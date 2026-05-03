"""Persistent streaming predictor for SentinelNet Phase 11."""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.ensemble.training import _extract_binary_weight_map, _extract_multiclass_weight_map

from .config import StreamingConfig
from .data import StreamBatch, StreamingMetadata


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


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


def _align_probabilities(probabilities: np.ndarray, current_labels: list[int], target_labels: list[int]) -> np.ndarray:
    """Align probability columns to a shared label ordering."""
    aligned = np.zeros((len(probabilities), len(target_labels)), dtype=np.float64)
    label_to_column = {int(label): index for index, label in enumerate(current_labels)}
    for target_index, label in enumerate(target_labels):
        if label in label_to_column:
            aligned[:, target_index] = probabilities[:, label_to_column[label]]
    return aligned


def _binary_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Convert a single attack probability into a two-column matrix."""
    flattened = probabilities.reshape(-1).astype(np.float64, copy=False)
    return np.column_stack([1.0 - flattened, flattened])


def _logistic_transform(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert anomaly scores into probability-like attack scores."""
    scores = np.asarray(scores, dtype=np.float64)
    scale = max(float(np.std(scores)), 1e-6)
    centered = np.clip((scores - threshold) / scale, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-centered))


def _score_as_anomaly(estimator: Any, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convert detector output into scores where higher means more anomalous."""
    if hasattr(estimator, "score_samples"):
        return -np.asarray(estimator.score_samples(X), dtype=np.float64)
    if hasattr(estimator, "decision_function"):
        return -np.asarray(estimator.decision_function(X), dtype=np.float64)
    raise AttributeError(f"Estimator {type(estimator).__name__} does not expose a supported scoring method.")


def _normalize_weights(weight_map: dict[str, float]) -> dict[str, float]:
    """Normalize weights onto a simplex."""
    positive = {name: max(0.0, float(weight)) for name, weight in weight_map.items()}
    total = sum(positive.values())
    if total <= 0:
        uniform = 1.0 / max(len(weight_map), 1)
        return {name: uniform for name in weight_map}
    return {name: value / total for name, value in positive.items()}


@dataclass(slots=True)
class LoadedPhase6Model:
    """Persisted Phase 6 model with its scaler and labels."""

    name: str
    estimator: Any
    scaler: Any | None
    labels: list[int]
    feature_names: list[str]

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        """Predict probabilities on raw selected-feature arrays."""
        X_scaler = _format_features_for_artifact(self.scaler, X_raw, self.feature_names) if self.scaler is not None else X_raw
        X_model = self.scaler.transform(X_scaler) if self.scaler is not None else X_scaler
        X_estimator = _format_features_for_artifact(self.estimator, X_model, self.feature_names)
        return np.asarray(self.estimator.predict_proba(X_estimator), dtype=np.float64)


@dataclass(slots=True)
class LoadedAnomalyModel:
    """Persisted Phase 8 anomaly detector."""

    name: str
    estimator: Any
    threshold: float


@dataclass(slots=True)
class SelectedEnsembleVariants:
    """Resolved Phase 9 variants used during streaming inference."""

    binary_variant: str
    multiclass_variant: str
    binary_metric_name: str
    binary_metric_value: float
    multiclass_metric_name: str
    multiclass_metric_value: float


@dataclass(slots=True)
class BatchPredictionResult:
    """Final prediction outputs for a streamed batch."""

    binary_variant: str
    multiclass_variant: str
    binary_probabilities: np.ndarray
    binary_predicted_labels: np.ndarray
    multiclass_probabilities: np.ndarray
    multiclass_predicted_labels: np.ndarray


class StreamingEnsemblePredictor:
    """Persistent predictor that reuses Phase 6 to 9 artifacts across streaming batches."""

    def __init__(self, config: StreamingConfig, metadata: StreamingMetadata) -> None:
        self.config = config
        self.metadata = metadata
        self.feature_names = metadata.feature_names
        self.inverse_multiclass_mapping = metadata.inverse_multiclass_mapping
        self.binary_labels = [0, 1]
        self.multiclass_labels = sorted(self.inverse_multiclass_mapping)

        self.phase6_report = _read_json(config.phase6_output_dir / "ml_training_report.json")
        self.phase7_report = _read_json(config.phase7_output_dir / "deep_learning_report.json")
        self.phase8_report = _read_json(config.phase8_output_dir / "anomaly_detection_report.json")

        self.binary_default_probabilities, self.multiclass_default_probabilities = self._load_default_probabilities()
        self.phase6_binary_models = self._load_phase6_models(task_name="binary")
        self.phase6_multiclass_models = self._load_phase6_models(task_name="multiclass")
        self._load_phase7_models()
        self._load_phase8_models()
        self.selected_variants = self._select_variants()
        self._load_phase9_artifacts()

    def _load_default_probabilities(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute training priors used for early LSTM rows."""
        label_frame = pd.read_csv(
            self.config.input_data_path,
            usecols=[self.config.binary_target_column, self.config.multiclass_target_column],
            low_memory=False,
        )
        train_indices = np.load(self.config.train_indices_path).astype(np.int64, copy=False)
        binary_train = label_frame.iloc[train_indices][self.config.binary_target_column].to_numpy(dtype=np.int32, copy=False)
        multiclass_train = label_frame.iloc[train_indices][self.config.multiclass_target_column].to_numpy(dtype=np.int32, copy=False)

        binary_counts = np.bincount(binary_train, minlength=2).astype(np.float64)
        binary_default = (binary_counts / binary_counts.sum()).reshape(1, -1)

        max_label = max(self.multiclass_labels)
        multiclass_counts = np.bincount(multiclass_train, minlength=max_label + 1).astype(np.float64)
        multiclass_default = (multiclass_counts[self.multiclass_labels] / multiclass_counts[self.multiclass_labels].sum()).reshape(1, -1)
        return binary_default, multiclass_default

    def _load_phase6_models(self, task_name: str) -> dict[str, LoadedPhase6Model]:
        """Load all Phase 6 models for a task."""
        directory = self.config.phase6_output_dir / task_name
        loaded: dict[str, LoadedPhase6Model] = {}
        for model_name in ("logistic_regression", "random_forest", "xgboost", "lightgbm"):
            payload = joblib.load(directory / f"{model_name}.joblib")
            loaded[model_name] = LoadedPhase6Model(
                name=model_name,
                estimator=payload["estimator"],
                scaler=payload.get("scaler"),
                labels=[int(label) for label in payload["labels"]],
                feature_names=[str(feature) for feature in payload["feature_names"]],
            )
        return loaded

    def _load_phase7_models(self) -> None:
        """Load shared Phase 7 artifacts and initialize LSTM history."""
        self.phase7_scaler = joblib.load(self.config.phase7_output_dir / "common" / "feature_scaler.joblib")
        self.phase7_binary_dnn = tf.keras.models.load_model(self.config.phase7_output_dir / "dnn" / "binary.keras")
        self.phase7_multiclass_dnn = tf.keras.models.load_model(self.config.phase7_output_dir / "dnn" / "multiclass.keras")
        self.phase7_binary_lstm = tf.keras.models.load_model(self.config.phase7_output_dir / "lstm" / "binary.keras")
        self.phase7_multiclass_lstm = tf.keras.models.load_model(self.config.phase7_output_dir / "lstm" / "multiclass.keras")

        binary_metadata = _read_json(self.config.phase7_output_dir / "dnn" / "binary_metadata.json")
        multiclass_metadata = _read_json(self.config.phase7_output_dir / "dnn" / "multiclass_metadata.json")
        self.phase7_binary_labels = [int(label) for label in binary_metadata["labels"]]
        self.phase7_multiclass_labels = [int(label) for label in multiclass_metadata["labels"]]
        self.lstm_sequence_length = int(self.phase7_report["config"]["lstm_sequence_length"])
        self.sequence_history: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.lstm_sequence_length))

    def _load_phase8_models(self) -> None:
        """Load Phase 8 anomaly models and metadata."""
        self.phase8_scaler = joblib.load(self.config.phase8_output_dir / "common" / "feature_scaler.joblib")
        isolation_payload = joblib.load(self.config.phase8_output_dir / "isolation_forest" / "isolation_forest.joblib")
        one_class_payload = joblib.load(self.config.phase8_output_dir / "one_class_svm" / "one_class_svm.joblib")
        lof_payload = joblib.load(self.config.phase8_output_dir / "lof" / "lof.joblib")
        self.phase8_models = {
            "isolation_forest": LoadedAnomalyModel(
                name="isolation_forest",
                estimator=isolation_payload["estimator"],
                threshold=float(isolation_payload["threshold"]),
            ),
            "one_class_svm": LoadedAnomalyModel(
                name="one_class_svm",
                estimator=one_class_payload["estimator"],
                threshold=float(one_class_payload["threshold"]),
            ),
            "lof": LoadedAnomalyModel(
                name="lof",
                estimator=lof_payload["estimator"],
                threshold=float(lof_payload["threshold"]),
            ),
        }
        self.phase8_autoencoder = tf.keras.models.load_model(self.config.phase8_output_dir / "autoencoder" / "autoencoder.keras")
        self.phase8_autoencoder_threshold = float(
            _read_json(self.config.phase8_output_dir / "autoencoder" / "autoencoder_metadata.json")["threshold"]
        )

    def _select_variants(self) -> SelectedEnsembleVariants:
        """Resolve the best Phase 9 binary and multiclass variants for deployment."""
        metrics = pd.read_csv(self.config.phase9_output_dir / "metrics_summary.csv")
        binary_frame = metrics.loc[(metrics["task"] == "binary") & (metrics["status"] == "trained")].copy()
        multiclass_frame = metrics.loc[(metrics["task"] == "multiclass") & (metrics["status"] == "trained")].copy()

        binary_row = self._select_variant_row(
            frame=binary_frame,
            metric_name=self.config.binary_selection_metric,
            override=self.config.binary_variant_override,
        )
        multiclass_row = self._select_variant_row(
            frame=multiclass_frame,
            metric_name=self.config.multiclass_selection_metric,
            override=self.config.multiclass_variant_override,
        )
        return SelectedEnsembleVariants(
            binary_variant=str(binary_row["ensemble"]),
            multiclass_variant=str(multiclass_row["ensemble"]),
            binary_metric_name=self.config.binary_selection_metric,
            binary_metric_value=float(binary_row[self.config.binary_selection_metric]),
            multiclass_metric_name=self.config.multiclass_selection_metric,
            multiclass_metric_value=float(multiclass_row[self.config.multiclass_selection_metric]),
        )

    @staticmethod
    def _select_variant_row(frame: pd.DataFrame, metric_name: str, override: str | None) -> pd.Series:
        """Select a trained ensemble variant by override or metric."""
        if frame.empty:
            raise ValueError("No trained Phase 9 ensemble variants available for selection.")
        if override is not None:
            matched = frame.loc[frame["ensemble"] == override].copy()
            if matched.empty:
                raise ValueError(f"Requested ensemble variant '{override}' was not found.")
            return matched.iloc[0]
        if metric_name not in frame.columns:
            raise ValueError(f"Phase 9 metrics do not include selection metric '{metric_name}'.")
        return frame.sort_values(metric_name, ascending=False).iloc[0]

    def _load_phase9_artifacts(self) -> None:
        """Load Phase 9 stacking artifacts and cached weight maps."""
        self.binary_weights = _normalize_weights(_extract_binary_weight_map(self.phase6_report, self.phase7_report, self.phase8_report))
        self.multiclass_weights = _normalize_weights(_extract_multiclass_weight_map(self.phase6_report, self.phase7_report))

        binary_payload = joblib.load(self.config.phase9_output_dir / "binary" / "stacking.joblib")
        self.binary_stacking_model = binary_payload["model"]
        self.binary_stacking_feature_order = [str(name) for name in binary_payload["feature_order"]]

        multiclass_payload = joblib.load(self.config.phase9_output_dir / "multiclass" / "stacking.joblib")
        self.multiclass_stacking_model = multiclass_payload["model"]
        self.multiclass_stacking_feature_order = [str(name) for name in multiclass_payload["feature_order"]]
        self.multiclass_stacking_anomaly_order = [str(name) for name in multiclass_payload["anomaly_feature_order"]]

    def _scale_with_phase7(self, X_raw: np.ndarray) -> np.ndarray:
        """Scale raw features for Phase 7 models."""
        X_scaler = _format_features_for_artifact(self.phase7_scaler, X_raw, self.feature_names)
        return self.phase7_scaler.transform(X_scaler).astype(np.float32, copy=False)

    def _scale_with_phase8(self, X_raw: np.ndarray) -> np.ndarray:
        """Scale raw features for Phase 8 models."""
        X_scaler = _format_features_for_artifact(self.phase8_scaler, X_raw, self.feature_names)
        return self.phase8_scaler.transform(X_scaler).astype(np.float32, copy=False)

    def _build_lstm_sequences(
        self,
        X_phase7_scaled: np.ndarray,
        source_files: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update source-specific history and return ready LSTM sequences for this batch."""
        positions: list[int] = []
        sequences: list[np.ndarray] = []
        for batch_index, (row, source_file) in enumerate(zip(X_phase7_scaled, source_files, strict=True)):
            history = self.sequence_history[str(source_file)]
            history.append(np.asarray(row, dtype=np.float32))
            if len(history) == self.lstm_sequence_length:
                sequences.append(np.stack(history).astype(np.float32, copy=False))
                positions.append(batch_index)

        if not sequences:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, self.lstm_sequence_length, len(self.feature_names)), dtype=np.float32),
            )
        return np.asarray(positions, dtype=np.int64), np.stack(sequences).astype(np.float32, copy=False)

    def _predict_phase6_maps(self, X_raw: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Predict all Phase 6 base model probability maps."""
        binary_map: dict[str, np.ndarray] = {}
        multiclass_map: dict[str, np.ndarray] = {}
        for model_name, model in self.phase6_binary_models.items():
            binary_map[f"phase6_{model_name}"] = _align_probabilities(model.predict_proba(X_raw), model.labels, self.binary_labels)
        for model_name, model in self.phase6_multiclass_models.items():
            multiclass_map[f"phase6_{model_name}"] = _align_probabilities(model.predict_proba(X_raw), model.labels, self.multiclass_labels)
        return binary_map, multiclass_map

    def _predict_phase7_maps(
        self,
        X_phase7_scaled: np.ndarray,
        source_files: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Predict all Phase 7 base model probability maps, preserving LSTM stream state."""
        binary_dnn_raw = np.asarray(self.phase7_binary_dnn.predict(X_phase7_scaled, batch_size=256, verbose=0), dtype=np.float64)
        multiclass_dnn_raw = np.asarray(self.phase7_multiclass_dnn.predict(X_phase7_scaled, batch_size=256, verbose=0), dtype=np.float64)
        binary_map = {
            "phase7_dnn": _align_probabilities(_binary_probability_matrix(binary_dnn_raw), self.phase7_binary_labels, self.binary_labels),
        }
        multiclass_map = {
            "phase7_dnn": _align_probabilities(multiclass_dnn_raw, self.phase7_multiclass_labels, self.multiclass_labels),
        }

        positions, sequences = self._build_lstm_sequences(X_phase7_scaled, source_files)
        binary_lstm = np.tile(self.binary_default_probabilities, (len(X_phase7_scaled), 1))
        multiclass_lstm = np.tile(self.multiclass_default_probabilities, (len(X_phase7_scaled), 1))
        if len(positions) > 0:
            binary_lstm_raw = np.asarray(self.phase7_binary_lstm.predict(sequences, batch_size=256, verbose=0), dtype=np.float64)
            multiclass_lstm_raw = np.asarray(self.phase7_multiclass_lstm.predict(sequences, batch_size=256, verbose=0), dtype=np.float64)
            binary_lstm[positions] = _align_probabilities(_binary_probability_matrix(binary_lstm_raw), self.phase7_binary_labels, self.binary_labels)
            multiclass_lstm[positions] = _align_probabilities(multiclass_lstm_raw, self.phase7_multiclass_labels, self.multiclass_labels)

        binary_map["phase7_lstm"] = binary_lstm.astype(np.float64, copy=False)
        multiclass_map["phase7_lstm"] = multiclass_lstm.astype(np.float64, copy=False)
        return binary_map, multiclass_map

    def _predict_phase8_map(self, X_raw: np.ndarray, X_phase8_scaled: np.ndarray) -> dict[str, np.ndarray]:
        """Predict all Phase 8 anomaly probability maps."""
        anomaly_map: dict[str, np.ndarray] = {}
        for model_name, model in self.phase8_models.items():
            X_estimator = _format_features_for_artifact(model.estimator, X_phase8_scaled, self.feature_names)
            scores = _score_as_anomaly(model.estimator, X_estimator)
            anomaly_probability = _logistic_transform(scores, model.threshold)
            anomaly_map[f"phase8_{model_name}"] = np.column_stack([1.0 - anomaly_probability, anomaly_probability])

        reconstructed = self.phase8_autoencoder.predict(X_phase8_scaled, batch_size=256, verbose=0)
        reconstruction_error = np.mean(np.square(X_phase8_scaled - reconstructed), axis=1)
        autoencoder_probability = _logistic_transform(reconstruction_error, self.phase8_autoencoder_threshold)
        anomaly_map["phase8_autoencoder"] = np.column_stack([1.0 - autoencoder_probability, autoencoder_probability])
        return anomaly_map

    def _build_probability_maps(self, batch: StreamBatch) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Build all component probability maps for a streaming batch."""
        X_raw = batch.feature_frame.to_numpy(dtype=np.float32, copy=False)
        binary_phase6, multiclass_phase6 = self._predict_phase6_maps(X_raw)
        X_phase7_scaled = self._scale_with_phase7(X_raw)
        binary_phase7, multiclass_phase7 = self._predict_phase7_maps(X_phase7_scaled, batch.source_files)
        X_phase8_scaled = self._scale_with_phase8(X_raw)
        anomaly_map = self._predict_phase8_map(X_raw, X_phase8_scaled)

        binary_map = {**binary_phase6, **binary_phase7, **anomaly_map}
        multiclass_map = {**multiclass_phase6, **multiclass_phase7}
        return binary_map, multiclass_map, anomaly_map

    def _predict_binary_final(self, binary_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Predict the final binary ensemble output for the configured variant."""
        variant = self.selected_variants.binary_variant
        if variant == "soft_voting":
            weight_map = _normalize_weights({name: 1.0 for name in binary_map})
            probability = self._weighted_average_map(binary_map, weight_map)
        elif variant == "weighted_scoring":
            probability = self._weighted_average_map(binary_map, self.binary_weights)
        elif variant == "stacking":
            X_meta = np.column_stack([binary_map[name][:, 1] for name in self.binary_stacking_feature_order]).astype(np.float64, copy=False)
            probability = np.asarray(self.binary_stacking_model.predict_proba(X_meta), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported binary ensemble variant '{variant}'.")
        predictions = np.asarray(self.binary_labels, dtype=np.int32)[np.argmax(probability, axis=1)]
        return probability, predictions

    def _predict_multiclass_final(
        self,
        multiclass_map: dict[str, np.ndarray],
        anomaly_map: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the final multiclass ensemble output for the configured variant."""
        variant = self.selected_variants.multiclass_variant
        if variant == "soft_voting":
            weight_map = _normalize_weights({name: 1.0 for name in multiclass_map})
            probability = self._weighted_average_map(multiclass_map, weight_map)
        elif variant == "weighted_scoring":
            probability = self._weighted_average_map(multiclass_map, self.multiclass_weights)
        elif variant == "stacking":
            X_meta = np.column_stack(
                [multiclass_map[name] for name in self.multiclass_stacking_feature_order]
                + [anomaly_map[name][:, 1].reshape(-1, 1) for name in self.multiclass_stacking_anomaly_order]
            ).astype(np.float64, copy=False)
            raw_probability = np.asarray(self.multiclass_stacking_model.predict_proba(X_meta), dtype=np.float64)
            probability = _align_probabilities(
                raw_probability,
                self.multiclass_stacking_model.classes_.astype(int).tolist(),
                self.multiclass_labels,
            )
        else:
            raise ValueError(f"Unsupported multiclass ensemble variant '{variant}'.")
        predictions = np.asarray([self.multiclass_labels[index] for index in np.argmax(probability, axis=1)], dtype=np.int32)
        return probability, predictions

    @staticmethod
    def _weighted_average_map(probability_map: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
        """Apply a normalized weighted average across probability matrices."""
        normalized = _normalize_weights(weights)
        first = next(iter(probability_map.values()))
        result = np.zeros_like(first, dtype=np.float64)
        for name, probabilities in probability_map.items():
            result += probabilities * normalized[name]
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return result / row_sums

    def predict_batch(self, batch: StreamBatch) -> BatchPredictionResult:
        """Predict the selected binary and multiclass ensemble outputs for one streamed batch."""
        binary_map, multiclass_map, anomaly_map = self._build_probability_maps(batch)
        binary_probability, binary_predictions = self._predict_binary_final(binary_map)
        multiclass_probability, multiclass_predictions = self._predict_multiclass_final(multiclass_map, anomaly_map)
        return BatchPredictionResult(
            binary_variant=self.selected_variants.binary_variant,
            multiclass_variant=self.selected_variants.multiclass_variant,
            binary_probabilities=binary_probability,
            binary_predicted_labels=binary_predictions,
            multiclass_probabilities=multiclass_probability,
            multiclass_predicted_labels=multiclass_predictions,
        )

    def reset_state(self) -> None:
        """Reset mutable streaming state so future predictions start from a clean history."""
        self.sequence_history.clear()

    def close(self) -> None:
        """Release loaded TensorFlow state."""
        tf.keras.backend.clear_session()
