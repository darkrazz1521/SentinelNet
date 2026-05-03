"""Phase 8 anomaly-detection training pipeline implementation."""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_pipeline.logging_utils import configure_logging
from src.evaluation.anomaly_metrics import evaluate_anomaly_scores
from src.models.deep_learning.architectures import build_autoencoder

from .config import AnomalyDetectionConfig
from .data import DetectorTaskData, build_detector_task, load_anomaly_dataset
from .registry import DetectorSpec, get_detector_specs

LOGGER = logging.getLogger("sentinelnet.phase8")


@dataclass(slots=True)
class AnomalyRunResult:
    """Stored result of a single Phase 8 detector run."""

    model_name: str
    status: str
    metrics: dict[str, Any] | None
    artifact_path: str | None
    metadata_path: str | None
    history_path: str | None
    train_rows: int
    test_rows: int
    message: str | None = None


@dataclass(slots=True)
class AnomalyDetectionReport:
    """Serializable report for Phase 8 anomaly detection."""

    created_at_utc: str
    input_path: str
    output_dir: str
    report_path: str
    feature_count: int
    train_rows: int
    test_rows: int
    models: list[dict[str, Any]]
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
            "models": self.models,
            "validation_passed": self.validation_passed,
            "config": self.config,
        }


def build_phase8_logger(config: AnomalyDetectionConfig) -> logging.Logger:
    """Create the dedicated Phase 8 logger."""
    return configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase8")


def configure_tensorflow_runtime(config: AnomalyDetectionConfig) -> None:
    """Set TensorFlow seeds and threading controls."""
    tf.keras.utils.set_random_seed(config.random_state)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(config.tf_intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(config.tf_inter_op_threads)
    except RuntimeError:
        pass


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload to disk."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _serialize_history(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    """Convert a Keras history object into JSON-serializable primitives."""
    return {
        str(key): [float(value) for value in values]
        for key, values in history.history.items()
    }


def _make_callbacks(model_path: Path, patience: int) -> list[tf.keras.callbacks.Callback]:
    """Create the common training callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
    ]


def _score_as_anomaly(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Convert detector output into scores where higher means more anomalous."""
    if hasattr(estimator, "score_samples"):
        return -np.asarray(estimator.score_samples(X), dtype=np.float64)
    if hasattr(estimator, "decision_function"):
        return -np.asarray(estimator.decision_function(X), dtype=np.float64)
    raise AttributeError(f"Estimator {type(estimator).__name__} does not expose a supported scoring method.")


def _adjust_estimator_for_task(estimator: Any, task: DetectorTaskData) -> Any:
    """Adjust estimator hyperparameters that depend on the sampled dataset size."""
    if hasattr(estimator, "n_neighbors"):
        max_neighbors = max(2, len(task.X_train) - 1)
        estimator.set_params(n_neighbors=min(int(estimator.n_neighbors), max_neighbors))
    if hasattr(estimator, "max_samples") and isinstance(getattr(estimator, "max_samples"), int):
        estimator.set_params(max_samples=min(int(estimator.max_samples), len(task.X_train)))
    return estimator


def _train_sklearn_detector(
    config: AnomalyDetectionConfig,
    spec: DetectorSpec,
    task: DetectorTaskData,
    feature_names: list[str],
    artifact_dir: Path,
    logger: logging.Logger,
) -> AnomalyRunResult:
    """Train and evaluate a classical anomaly detector."""
    estimator = _adjust_estimator_for_task(spec.builder(config), task)
    artifact_path = artifact_dir / f"{spec.name}.joblib"
    metadata_path = artifact_dir / f"{spec.name}_metadata.json"

    logger.info(
        "Training %s | train_rows=%d | validation_rows=%d | test_rows=%d",
        spec.name,
        len(task.X_train),
        len(task.X_validation),
        len(task.X_test),
    )
    estimator.fit(task.X_train)
    validation_scores = _score_as_anomaly(estimator, task.X_validation)
    threshold = float(np.quantile(validation_scores, config.threshold_quantile))
    test_scores = _score_as_anomaly(estimator, task.X_test)
    metrics = evaluate_anomaly_scores(task.y_test_binary, test_scores, threshold)

    joblib.dump(
        {
            "model_name": spec.name,
            "feature_names": feature_names,
            "threshold": threshold,
            "estimator": estimator,
        },
        artifact_path,
    )
    _write_json(
        metadata_path,
        {
            "model_name": spec.name,
            "train_rows": int(len(task.X_train)),
            "validation_rows": int(len(task.X_validation)),
            "test_rows": int(len(task.X_test)),
            "threshold": threshold,
            "artifact_path": str(artifact_path),
            "feature_count": len(feature_names),
        },
    )
    return AnomalyRunResult(
        model_name=spec.name,
        status="trained",
        metrics=metrics,
        artifact_path=str(artifact_path),
        metadata_path=str(metadata_path),
        history_path=None,
        train_rows=int(len(task.X_train)),
        test_rows=int(len(task.X_test)),
        message=None,
    )


def _reconstruction_errors(model: tf.keras.Model, X: np.ndarray, batch_size: int) -> np.ndarray:
    """Compute per-row mean squared reconstruction errors."""
    reconstructed = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=1)


def _train_autoencoder_detector(
    config: AnomalyDetectionConfig,
    task: DetectorTaskData,
    feature_names: list[str],
    artifact_dir: Path,
    logger: logging.Logger,
) -> AnomalyRunResult:
    """Train and evaluate the autoencoder anomaly detector."""
    tf.keras.backend.clear_session()
    artifact_path = artifact_dir / "autoencoder.keras"
    history_path = artifact_dir / "autoencoder_history.json"
    metadata_path = artifact_dir / "autoencoder_metadata.json"

    model = build_autoencoder(config, task.X_train.shape[1])
    logger.info(
        "Training autoencoder | train_rows=%d | validation_rows=%d | test_rows=%d",
        len(task.X_train),
        len(task.X_validation),
        len(task.X_test),
    )
    history = model.fit(
        task.X_train,
        task.X_train,
        validation_data=(task.X_validation, task.X_validation),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=0,
        callbacks=_make_callbacks(artifact_path, config.patience),
        shuffle=True,
    )

    best_model = tf.keras.models.load_model(artifact_path)
    validation_scores = _reconstruction_errors(best_model, task.X_validation, config.batch_size)
    threshold = float(np.quantile(validation_scores, config.threshold_quantile))
    test_scores = _reconstruction_errors(best_model, task.X_test, config.batch_size)
    metrics = evaluate_anomaly_scores(task.y_test_binary, test_scores, threshold)

    _write_json(history_path, _serialize_history(history))
    _write_json(
        metadata_path,
        {
            "model_name": "autoencoder",
            "train_rows": int(len(task.X_train)),
            "validation_rows": int(len(task.X_validation)),
            "test_rows": int(len(task.X_test)),
            "threshold": threshold,
            "artifact_path": str(artifact_path),
            "history_path": str(history_path),
            "feature_count": len(feature_names),
        },
    )
    return AnomalyRunResult(
        model_name="autoencoder",
        status="trained",
        metrics=metrics,
        artifact_path=str(artifact_path),
        metadata_path=str(metadata_path),
        history_path=str(history_path),
        train_rows=int(len(task.X_train)),
        test_rows=int(len(task.X_test)),
        message=None,
    )


def _serialize_results(results: list[AnomalyRunResult]) -> list[dict[str, Any]]:
    """Convert model results into report-friendly dictionaries."""
    return [
        {
            "model": result.model_name,
            "status": result.status,
            "artifact_path": result.artifact_path,
            "metadata_path": result.metadata_path,
            "history_path": result.history_path,
            "train_rows": result.train_rows,
            "test_rows": result.test_rows,
            "message": result.message,
            "metrics": result.metrics,
        }
        for result in results
    ]


def _write_metrics_summary(results: list[AnomalyRunResult], metrics_path: Path) -> None:
    """Write a flat metrics CSV across all Phase 8 detectors."""
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "model": result.model_name,
            "status": result.status,
            "artifact_path": result.artifact_path,
            "metadata_path": result.metadata_path,
            "history_path": result.history_path,
            "train_rows": result.train_rows,
            "test_rows": result.test_rows,
            "message": result.message,
        }
        if result.metrics:
            row.update(
                {
                    "accuracy": result.metrics.get("accuracy"),
                    "precision": result.metrics.get("precision"),
                    "recall": result.metrics.get("recall"),
                    "f1_score": result.metrics.get("f1_score"),
                    "roc_auc": result.metrics.get("roc_auc"),
                    "threshold": result.metrics.get("threshold"),
                }
            )
        rows.append(row)

    pd.DataFrame(rows).to_csv(metrics_path, index=False)


def _build_report(
    config: AnomalyDetectionConfig,
    feature_count: int,
    train_rows: int,
    test_rows: int,
    results: list[AnomalyRunResult],
) -> AnomalyDetectionReport:
    """Build the current report snapshot."""
    return AnomalyDetectionReport(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        input_path=str(config.input_data_path),
        output_dir=str(config.output_dir),
        report_path=str(config.report_path),
        feature_count=feature_count,
        train_rows=train_rows,
        test_rows=test_rows,
        models=_serialize_results(results),
        validation_passed=all(result.status == "trained" for result in results) and len(results) == 4,
        config=config.to_dict(),
    )


def run_anomaly_detection_pipeline(
    config: AnomalyDetectionConfig,
    logger: logging.Logger | None = None,
) -> AnomalyDetectionReport:
    """Execute the complete Phase 8 anomaly-detection workflow."""
    active_logger = logger or LOGGER
    config.ensure_directories()
    configure_tensorflow_runtime(config)

    if not config.input_data_path.exists():
        raise FileNotFoundError(f"Phase 8 input dataset not found at {config.input_data_path}")
    if not config.feature_manifest_path.exists():
        raise FileNotFoundError(f"Phase 8 feature manifest not found at {config.feature_manifest_path}")
    if not config.label_mapping_path.exists():
        raise FileNotFoundError(f"Phase 8 label mapping artifact not found at {config.label_mapping_path}")

    dataset = load_anomaly_dataset(config)
    joblib.dump(dataset.scaler, config.scaler_path)
    active_logger.info(
        "Loaded Phase 8 dataset | train_rows=%d | test_rows=%d | features=%d",
        len(dataset.X_train),
        len(dataset.X_test),
        len(dataset.feature_names),
    )

    results: list[AnomalyRunResult] = []

    task_builders = {
        "isolation_forest": build_detector_task(
            dataset=dataset,
            train_cap=config.isolation_forest_train_cap,
            test_cap=config.evaluation_test_cap,
            validation_size=config.validation_size,
            random_state=config.random_state,
        ),
        "one_class_svm": build_detector_task(
            dataset=dataset,
            train_cap=config.one_class_svm_train_cap,
            test_cap=config.evaluation_test_cap,
            validation_size=config.validation_size,
            random_state=config.random_state,
        ),
        "lof": build_detector_task(
            dataset=dataset,
            train_cap=config.lof_train_cap,
            test_cap=config.evaluation_test_cap,
            validation_size=config.validation_size,
            random_state=config.random_state,
        ),
    }

    for spec in get_detector_specs():
        artifact_dir = getattr(config, f"{spec.name}_dir")
        task = task_builders[spec.name]
        results.append(
            _train_sklearn_detector(
                config=config,
                spec=spec,
                task=task,
                feature_names=dataset.feature_names,
                artifact_dir=artifact_dir,
                logger=active_logger,
            )
        )

    autoencoder_task = build_detector_task(
        dataset=dataset,
        train_cap=config.autoencoder_train_cap,
        test_cap=config.autoencoder_test_cap,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )
    results.append(
        _train_autoencoder_detector(
            config=config,
            task=autoencoder_task,
            feature_names=dataset.feature_names,
            artifact_dir=config.autoencoder_dir,
            logger=active_logger,
        )
    )

    _write_metrics_summary(results, config.metrics_path)
    report = _build_report(
        config=config,
        feature_count=len(dataset.feature_names),
        train_rows=len(dataset.X_train),
        test_rows=len(dataset.X_test),
        results=results,
    )
    config.report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if not report.validation_passed:
        raise ValueError("Phase 8 validation failed because one or more anomaly detectors did not train successfully.")

    active_logger.info(
        "Completed Phase 8 anomaly detection | detectors=%d | validation_passed=%s",
        len(results),
        report.validation_passed,
    )
    tf.keras.backend.clear_session()
    gc.collect()
    return report
