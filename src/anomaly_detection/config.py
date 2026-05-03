"""Configuration for SentinelNet Phase 8 anomaly detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data_pipeline.config import default_project_root


def _resolve_path(project_root: Path, value: str | Path | None, fallback: str) -> Path:
    """Resolve a possibly relative path against the project root."""
    candidate = Path(value) if value is not None else Path(fallback)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _optional_int(value: int | None) -> int | None:
    """Normalize an optional integer value."""
    if value is None:
        return None
    return int(value)


@dataclass(slots=True)
class AnomalyDetectionConfig:
    """Runtime configuration for Phase 8 anomaly detection."""

    project_root: Path = field(default_factory=default_project_root)
    input_data_path: Path | None = None
    feature_manifest_path: Path | None = None
    label_mapping_path: Path | None = None
    train_indices_path: Path | None = None
    test_indices_path: Path | None = None
    output_dir: Path | None = None
    logs_dir: Path | None = None
    report_filename: str = "anomaly_detection_report.json"
    metrics_filename: str = "metrics_summary.csv"
    scaler_filename: str = "feature_scaler.joblib"
    log_filename: str = "phase8_anomaly_detection.log"
    source_file_column: str = "source_file"
    binary_target_column: str = "label_binary"
    multiclass_target_column: str = "label_multiclass"
    random_state: int = 42
    validation_size: float = 0.1
    threshold_quantile: float = 0.995
    evaluation_test_cap: int | None = 250_000
    isolation_forest_train_cap: int | None = 400_000
    isolation_forest_n_estimators: int = 200
    isolation_forest_contamination: float = 0.01
    isolation_forest_max_samples: int = 256
    one_class_svm_train_cap: int | None = 40_000
    one_class_svm_nu: float = 0.01
    one_class_svm_kernel: str = "rbf"
    one_class_svm_gamma: str = "scale"
    lof_train_cap: int | None = 60_000
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.01
    batch_size: int = 256
    epochs: int = 20
    patience: int = 4
    learning_rate: float = 0.001
    tf_intra_op_threads: int = 1
    tf_inter_op_threads: int = 1
    autoencoder_hidden_units: tuple[int, ...] = (128, 64)
    autoencoder_latent_dim: int = 32
    autoencoder_train_cap: int | None = 300_000
    autoencoder_test_cap: int | None = 250_000
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.input_data_path = _resolve_path(
            self.project_root,
            self.input_data_path,
            "data/processed/feature_engineered/selected_dataset.csv",
        )
        self.feature_manifest_path = _resolve_path(
            self.project_root,
            self.feature_manifest_path,
            "data/processed/feature_engineered/selected_feature_manifest.json",
        )
        self.label_mapping_path = _resolve_path(
            self.project_root,
            self.label_mapping_path,
            "data/processed/label_mappings.json",
        )
        self.train_indices_path = _resolve_path(
            self.project_root,
            self.train_indices_path,
            "data/processed/feature_engineered/train_indices.npy",
        )
        self.test_indices_path = _resolve_path(
            self.project_root,
            self.test_indices_path,
            "data/processed/feature_engineered/test_indices.npy",
        )
        self.output_dir = _resolve_path(self.project_root, self.output_dir, "models/saved_models/phase8_anomaly_detection")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.random_state = int(self.random_state)
        self.validation_size = float(self.validation_size)
        self.threshold_quantile = float(self.threshold_quantile)
        self.evaluation_test_cap = _optional_int(self.evaluation_test_cap)
        self.isolation_forest_train_cap = _optional_int(self.isolation_forest_train_cap)
        self.isolation_forest_n_estimators = int(self.isolation_forest_n_estimators)
        self.isolation_forest_contamination = float(self.isolation_forest_contamination)
        self.isolation_forest_max_samples = int(self.isolation_forest_max_samples)
        self.one_class_svm_train_cap = _optional_int(self.one_class_svm_train_cap)
        self.one_class_svm_nu = float(self.one_class_svm_nu)
        self.lof_train_cap = _optional_int(self.lof_train_cap)
        self.lof_n_neighbors = int(self.lof_n_neighbors)
        self.lof_contamination = float(self.lof_contamination)
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.patience = int(self.patience)
        self.learning_rate = float(self.learning_rate)
        self.tf_intra_op_threads = int(self.tf_intra_op_threads)
        self.tf_inter_op_threads = int(self.tf_inter_op_threads)
        self.autoencoder_hidden_units = tuple(int(unit) for unit in self.autoencoder_hidden_units)
        self.autoencoder_latent_dim = int(self.autoencoder_latent_dim)
        self.autoencoder_train_cap = _optional_int(self.autoencoder_train_cap)
        self.autoencoder_test_cap = _optional_int(self.autoencoder_test_cap)
        self.log_level = self.log_level.upper()

    @property
    def common_dir(self) -> Path:
        """Return the directory for shared artifacts."""
        return self.output_dir / "common"

    @property
    def isolation_forest_dir(self) -> Path:
        """Return the isolation-forest artifact directory."""
        return self.output_dir / "isolation_forest"

    @property
    def one_class_svm_dir(self) -> Path:
        """Return the one-class SVM artifact directory."""
        return self.output_dir / "one_class_svm"

    @property
    def lof_dir(self) -> Path:
        """Return the local outlier factor artifact directory."""
        return self.output_dir / "lof"

    @property
    def autoencoder_dir(self) -> Path:
        """Return the autoencoder artifact directory."""
        return self.output_dir / "autoencoder"

    @property
    def scaler_path(self) -> Path:
        """Return the saved feature-scaler path."""
        return self.common_dir / self.scaler_filename

    @property
    def report_path(self) -> Path:
        """Return the Phase 8 report destination."""
        return self.output_dir / self.report_filename

    @property
    def metrics_path(self) -> Path:
        """Return the Phase 8 metrics summary path."""
        return self.output_dir / self.metrics_filename

    @property
    def log_path(self) -> Path:
        """Return the Phase 8 log destination."""
        return self.logs_dir / self.log_filename

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        for directory in (
            self.output_dir,
            self.common_dir,
            self.isolation_forest_dir,
            self.one_class_svm_dir,
            self.lof_dir,
            self.autoencoder_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(
        cls,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "AnomalyDetectionConfig":
        """Create a Phase 8 configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "anomaly_detection_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            input_data_path=payload.get("input_data_path"),
            feature_manifest_path=payload.get("feature_manifest_path"),
            label_mapping_path=payload.get("label_mapping_path"),
            train_indices_path=payload.get("train_indices_path"),
            test_indices_path=payload.get("test_indices_path"),
            output_dir=payload.get("output_dir"),
            logs_dir=payload.get("logs_dir"),
            report_filename=payload.get("report_filename", "anomaly_detection_report.json"),
            metrics_filename=payload.get("metrics_filename", "metrics_summary.csv"),
            scaler_filename=payload.get("scaler_filename", "feature_scaler.joblib"),
            log_filename=payload.get("log_filename", "phase8_anomaly_detection.log"),
            source_file_column=payload.get("source_file_column", "source_file"),
            binary_target_column=payload.get("binary_target_column", "label_binary"),
            multiclass_target_column=payload.get("multiclass_target_column", "label_multiclass"),
            random_state=payload.get("random_state", 42),
            validation_size=payload.get("validation_size", 0.1),
            threshold_quantile=payload.get("threshold_quantile", 0.995),
            evaluation_test_cap=payload.get("evaluation_test_cap", 250_000),
            isolation_forest_train_cap=payload.get("isolation_forest_train_cap", 400_000),
            isolation_forest_n_estimators=payload.get("isolation_forest_n_estimators", 200),
            isolation_forest_contamination=payload.get("isolation_forest_contamination", 0.01),
            isolation_forest_max_samples=payload.get("isolation_forest_max_samples", 256),
            one_class_svm_train_cap=payload.get("one_class_svm_train_cap", 40_000),
            one_class_svm_nu=payload.get("one_class_svm_nu", 0.01),
            one_class_svm_kernel=payload.get("one_class_svm_kernel", "rbf"),
            one_class_svm_gamma=payload.get("one_class_svm_gamma", "scale"),
            lof_train_cap=payload.get("lof_train_cap", 60_000),
            lof_n_neighbors=payload.get("lof_n_neighbors", 20),
            lof_contamination=payload.get("lof_contamination", 0.01),
            batch_size=payload.get("batch_size", 256),
            epochs=payload.get("epochs", 20),
            patience=payload.get("patience", 4),
            learning_rate=payload.get("learning_rate", 0.001),
            tf_intra_op_threads=payload.get("tf_intra_op_threads", 1),
            tf_inter_op_threads=payload.get("tf_inter_op_threads", 1),
            autoencoder_hidden_units=tuple(payload.get("autoencoder_hidden_units", (128, 64))),
            autoencoder_latent_dim=payload.get("autoencoder_latent_dim", 32),
            autoencoder_train_cap=payload.get("autoencoder_train_cap", 300_000),
            autoencoder_test_cap=payload.get("autoencoder_test_cap", 250_000),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for reports."""
        return {
            "project_root": str(self.project_root),
            "input_data_path": str(self.input_data_path),
            "feature_manifest_path": str(self.feature_manifest_path),
            "label_mapping_path": str(self.label_mapping_path),
            "train_indices_path": str(self.train_indices_path),
            "test_indices_path": str(self.test_indices_path),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "report_filename": self.report_filename,
            "metrics_filename": self.metrics_filename,
            "scaler_filename": self.scaler_filename,
            "log_filename": self.log_filename,
            "source_file_column": self.source_file_column,
            "binary_target_column": self.binary_target_column,
            "multiclass_target_column": self.multiclass_target_column,
            "random_state": self.random_state,
            "validation_size": self.validation_size,
            "threshold_quantile": self.threshold_quantile,
            "evaluation_test_cap": self.evaluation_test_cap,
            "isolation_forest_train_cap": self.isolation_forest_train_cap,
            "isolation_forest_n_estimators": self.isolation_forest_n_estimators,
            "isolation_forest_contamination": self.isolation_forest_contamination,
            "isolation_forest_max_samples": self.isolation_forest_max_samples,
            "one_class_svm_train_cap": self.one_class_svm_train_cap,
            "one_class_svm_nu": self.one_class_svm_nu,
            "one_class_svm_kernel": self.one_class_svm_kernel,
            "one_class_svm_gamma": self.one_class_svm_gamma,
            "lof_train_cap": self.lof_train_cap,
            "lof_n_neighbors": self.lof_n_neighbors,
            "lof_contamination": self.lof_contamination,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "learning_rate": self.learning_rate,
            "tf_intra_op_threads": self.tf_intra_op_threads,
            "tf_inter_op_threads": self.tf_inter_op_threads,
            "autoencoder_hidden_units": list(self.autoencoder_hidden_units),
            "autoencoder_latent_dim": self.autoencoder_latent_dim,
            "autoencoder_train_cap": self.autoencoder_train_cap,
            "autoencoder_test_cap": self.autoencoder_test_cap,
            "log_level": self.log_level,
        }
