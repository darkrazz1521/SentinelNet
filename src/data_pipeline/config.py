"""Configuration objects for SentinelNet data pipeline stages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def default_project_root() -> Path:
    """Resolve the repository root from the package location."""
    return Path(__file__).resolve().parents[2]


def _resolve_path(project_root: Path, value: str | Path | None, fallback: str) -> Path:
    """Resolve a possibly relative path against the project root."""
    candidate = Path(value) if value is not None else Path(fallback)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


@dataclass(slots=True)
class IngestionConfig:
    """Runtime configuration for Phase 1 multi-file ingestion."""

    project_root: Path = field(default_factory=default_project_root)
    raw_data_dir: Path | None = None
    interim_data_dir: Path | None = None
    logs_dir: Path | None = None
    output_filename: str = "combined.csv"
    report_filename: str = "combined_report.json"
    log_filename: str = "phase1_ingestion.log"
    chunk_size: int = 100_000
    encoding_candidates: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    delimiter_candidates: tuple[str, ...] = (",", ";", "\t", "|")
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.raw_data_dir = _resolve_path(self.project_root, self.raw_data_dir, "data/raw")
        self.interim_data_dir = _resolve_path(self.project_root, self.interim_data_dir, "data/interim")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.chunk_size = int(self.chunk_size)
        self.log_level = self.log_level.upper()
        self.encoding_candidates = tuple(self.encoding_candidates)
        self.delimiter_candidates = tuple(self.delimiter_candidates)

    @property
    def output_path(self) -> Path:
        """Return the combined dataset destination."""
        return self.interim_data_dir / self.output_filename

    @property
    def report_path(self) -> Path:
        """Return the ingestion report destination."""
        return self.interim_data_dir / self.report_filename

    @property
    def log_path(self) -> Path:
        """Return the pipeline log destination."""
        return self.logs_dir / self.log_filename

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        for directory in (self.raw_data_dir, self.interim_data_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(
        cls,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "IngestionConfig":
        """Create a configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "ingestion_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            raw_data_dir=payload.get("raw_data_dir"),
            interim_data_dir=payload.get("interim_data_dir"),
            logs_dir=payload.get("logs_dir"),
            output_filename=payload.get("output_filename", "combined.csv"),
            report_filename=payload.get("report_filename", "combined_report.json"),
            log_filename=payload.get("log_filename", "phase1_ingestion.log"),
            chunk_size=payload.get("chunk_size", 100_000),
            encoding_candidates=tuple(payload.get("encoding_candidates", ("utf-8-sig", "utf-8", "cp1252", "latin-1"))),
            delimiter_candidates=tuple(payload.get("delimiter_candidates", (",", ";", "\t", "|"))),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for logs and reports."""
        return {
            "project_root": str(self.project_root),
            "raw_data_dir": str(self.raw_data_dir),
            "interim_data_dir": str(self.interim_data_dir),
            "logs_dir": str(self.logs_dir),
            "output_filename": self.output_filename,
            "report_filename": self.report_filename,
            "log_filename": self.log_filename,
            "chunk_size": self.chunk_size,
            "encoding_candidates": list(self.encoding_candidates),
            "delimiter_candidates": list(self.delimiter_candidates),
            "log_level": self.log_level,
        }


def default_label_aliases() -> dict[str, str]:
    """Return canonical label aliases for CICIDS2017 attack categories."""
    return {
        "benign": "BENIGN",
        "bot": "Bot",
        "ddos": "DDoS",
        "dos goldeneye": "DoS GoldenEye",
        "dos hulk": "DoS Hulk",
        "dos slowhttptest": "DoS Slowhttptest",
        "dos slowloris": "DoS slowloris",
        "ftp patator": "FTP-Patator",
        "ftp-patator": "FTP-Patator",
        "heartbleed": "Heartbleed",
        "infiltration": "Infiltration",
        "portscan": "PortScan",
        "ssh patator": "SSH-Patator",
        "ssh-patator": "SSH-Patator",
        "web attack brute force": "Web Attack - Brute Force",
        "web attack sql injection": "Web Attack - Sql Injection",
        "web attack xss": "Web Attack - XSS",
    }


def default_allowed_labels() -> tuple[str, ...]:
    """Return the ordered, supported label vocabulary for Phase 3."""
    ordered_labels = list(dict.fromkeys(default_label_aliases().values()))
    if "BENIGN" in ordered_labels:
        ordered_labels = ["BENIGN", *[label for label in ordered_labels if label != "BENIGN"]]
    else:
        ordered_labels.insert(0, "BENIGN")
    return tuple(ordered_labels)


@dataclass(slots=True)
class CleaningConfig:
    """Runtime configuration for Phase 2 data cleaning."""

    project_root: Path = field(default_factory=default_project_root)
    input_data_path: Path | None = None
    interim_data_dir: Path | None = None
    logs_dir: Path | None = None
    output_filename: str = "cleaned.csv"
    report_filename: str = "cleaned_report.json"
    log_filename: str = "phase2_cleaning.log"
    chunk_size: int = 100_000
    label_column: str = "label"
    source_file_column: str = "source_file"
    critical_columns: tuple[str, ...] = (
        "label",
        "source_file",
        "destination_port",
        "flow_duration",
        "total_fwd_packets",
        "total_backward_packets",
    )
    deduplication_excluded_columns: tuple[str, ...] = ("source_file",)
    zero_fill_suffixes: tuple[str, ...] = ("_per_s", "_ratio", "_rate")
    label_aliases: dict[str, str] = field(default_factory=default_label_aliases)
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.input_data_path = _resolve_path(self.project_root, self.input_data_path, "data/interim/combined.csv")
        self.interim_data_dir = _resolve_path(self.project_root, self.interim_data_dir, "data/interim")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.chunk_size = int(self.chunk_size)
        self.critical_columns = tuple(self.critical_columns)
        self.deduplication_excluded_columns = tuple(self.deduplication_excluded_columns)
        self.zero_fill_suffixes = tuple(self.zero_fill_suffixes)
        self.log_level = self.log_level.upper()
        self.label_aliases = dict(self.label_aliases)

    @property
    def output_path(self) -> Path:
        """Return the cleaned dataset destination."""
        return self.interim_data_dir / self.output_filename

    @property
    def report_path(self) -> Path:
        """Return the cleaning report destination."""
        return self.interim_data_dir / self.report_filename

    @property
    def log_path(self) -> Path:
        """Return the Phase 2 log destination."""
        return self.logs_dir / self.log_filename

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        for directory in (self.interim_data_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(
        cls,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "CleaningConfig":
        """Create a cleaning configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "cleaning_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            input_data_path=payload.get("input_data_path"),
            interim_data_dir=payload.get("interim_data_dir"),
            logs_dir=payload.get("logs_dir"),
            output_filename=payload.get("output_filename", "cleaned.csv"),
            report_filename=payload.get("report_filename", "cleaned_report.json"),
            log_filename=payload.get("log_filename", "phase2_cleaning.log"),
            chunk_size=payload.get("chunk_size", 100_000),
            label_column=payload.get("label_column", "label"),
            source_file_column=payload.get("source_file_column", "source_file"),
            critical_columns=tuple(
                payload.get(
                    "critical_columns",
                    ("label", "source_file", "destination_port", "flow_duration", "total_fwd_packets", "total_backward_packets"),
                )
            ),
            deduplication_excluded_columns=tuple(payload.get("deduplication_excluded_columns", ("source_file",))),
            zero_fill_suffixes=tuple(payload.get("zero_fill_suffixes", ("_per_s", "_ratio", "_rate"))),
            label_aliases=dict(payload.get("label_aliases", default_label_aliases())),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for logs and reports."""
        return {
            "project_root": str(self.project_root),
            "input_data_path": str(self.input_data_path),
            "interim_data_dir": str(self.interim_data_dir),
            "logs_dir": str(self.logs_dir),
            "output_filename": self.output_filename,
            "report_filename": self.report_filename,
            "log_filename": self.log_filename,
            "chunk_size": self.chunk_size,
            "label_column": self.label_column,
            "source_file_column": self.source_file_column,
            "critical_columns": list(self.critical_columns),
            "deduplication_excluded_columns": list(self.deduplication_excluded_columns),
            "zero_fill_suffixes": list(self.zero_fill_suffixes),
            "label_aliases": dict(self.label_aliases),
            "log_level": self.log_level,
        }


@dataclass(slots=True)
class LabelHandlingConfig:
    """Runtime configuration for Phase 3 label engineering."""

    project_root: Path = field(default_factory=default_project_root)
    input_data_path: Path | None = None
    processed_data_dir: Path | None = None
    logs_dir: Path | None = None
    output_filename: str = "labeled_dataset.csv"
    mapping_filename: str = "label_mappings.json"
    report_filename: str = "label_report.json"
    log_filename: str = "phase3_label_handling.log"
    chunk_size: int = 100_000
    label_column: str = "label"
    binary_target_column: str = "label_binary"
    multiclass_target_column: str = "label_multiclass"
    benign_label: str = "BENIGN"
    allowed_labels: tuple[str, ...] = field(default_factory=default_allowed_labels)
    label_aliases: dict[str, str] = field(default_factory=default_label_aliases)
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.input_data_path = _resolve_path(self.project_root, self.input_data_path, "data/interim/cleaned.csv")
        self.processed_data_dir = _resolve_path(self.project_root, self.processed_data_dir, "data/processed")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.chunk_size = int(self.chunk_size)
        self.allowed_labels = tuple(self.allowed_labels)
        self.label_aliases = dict(self.label_aliases)
        self.log_level = self.log_level.upper()

        if self.benign_label not in self.allowed_labels:
            raise ValueError(f"Benign label {self.benign_label!r} must exist in allowed_labels.")
        if self.allowed_labels[0] != self.benign_label:
            self.allowed_labels = (self.benign_label, *[label for label in self.allowed_labels if label != self.benign_label])

    @property
    def output_path(self) -> Path:
        """Return the label-engineered dataset destination."""
        return self.processed_data_dir / self.output_filename

    @property
    def mapping_path(self) -> Path:
        """Return the label mapping artifact destination."""
        return self.processed_data_dir / self.mapping_filename

    @property
    def report_path(self) -> Path:
        """Return the Phase 3 report destination."""
        return self.processed_data_dir / self.report_filename

    @property
    def log_path(self) -> Path:
        """Return the Phase 3 log destination."""
        return self.logs_dir / self.log_filename

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        for directory in (self.processed_data_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(
        cls,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "LabelHandlingConfig":
        """Create a Phase 3 configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "label_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            input_data_path=payload.get("input_data_path"),
            processed_data_dir=payload.get("processed_data_dir"),
            logs_dir=payload.get("logs_dir"),
            output_filename=payload.get("output_filename", "labeled_dataset.csv"),
            mapping_filename=payload.get("mapping_filename", "label_mappings.json"),
            report_filename=payload.get("report_filename", "label_report.json"),
            log_filename=payload.get("log_filename", "phase3_label_handling.log"),
            chunk_size=payload.get("chunk_size", 100_000),
            label_column=payload.get("label_column", "label"),
            binary_target_column=payload.get("binary_target_column", "label_binary"),
            multiclass_target_column=payload.get("multiclass_target_column", "label_multiclass"),
            benign_label=payload.get("benign_label", "BENIGN"),
            allowed_labels=tuple(payload.get("allowed_labels", default_allowed_labels())),
            label_aliases=dict(payload.get("label_aliases", default_label_aliases())),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for logs and reports."""
        return {
            "project_root": str(self.project_root),
            "input_data_path": str(self.input_data_path),
            "processed_data_dir": str(self.processed_data_dir),
            "logs_dir": str(self.logs_dir),
            "output_filename": self.output_filename,
            "mapping_filename": self.mapping_filename,
            "report_filename": self.report_filename,
            "log_filename": self.log_filename,
            "chunk_size": self.chunk_size,
            "label_column": self.label_column,
            "binary_target_column": self.binary_target_column,
            "multiclass_target_column": self.multiclass_target_column,
            "benign_label": self.benign_label,
            "allowed_labels": list(self.allowed_labels),
            "label_aliases": dict(self.label_aliases),
            "log_level": self.log_level,
        }


@dataclass(slots=True)
class PreprocessingConfig:
    """Runtime configuration for Phase 4 preprocessing and dataset splitting."""

    project_root: Path = field(default_factory=default_project_root)
    input_data_path: Path | None = None
    output_dir: Path | None = None
    logs_dir: Path | None = None
    preprocessor_filename: str = "preprocessor.joblib"
    feature_manifest_filename: str = "feature_manifest.json"
    report_filename: str = "preprocessing_report.json"
    log_filename: str = "phase4_preprocessing.log"
    chunk_size: int = 100_000
    test_size: float = 0.2
    random_state: int = 42
    label_column: str = "label"
    source_file_column: str = "source_file"
    binary_target_column: str = "label_binary"
    multiclass_target_column: str = "label_multiclass"
    include_source_file_feature: bool = False
    excluded_feature_columns: tuple[str, ...] = ("label", "label_binary", "label_multiclass")
    metadata_columns: tuple[str, ...] = ("label", "source_file", "label_binary", "label_multiclass")
    binary_resampling_method: str = "adasyn"
    binary_majority_cap: int = 500_000
    binary_target_ratio: float = 1.0
    multiclass_resampling_method: str = "smote"
    multiclass_benign_cap: int = 500_000
    multiclass_attack_cap: int = 100_000
    multiclass_min_target_count: int = 20_000
    knn_neighbors: int = 5
    output_dtype: str = "float32"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.input_data_path = _resolve_path(self.project_root, self.input_data_path, "data/processed/labeled_dataset.csv")
        self.output_dir = _resolve_path(self.project_root, self.output_dir, "data/processed/preprocessed")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.chunk_size = int(self.chunk_size)
        self.test_size = float(self.test_size)
        self.random_state = int(self.random_state)
        self.binary_majority_cap = int(self.binary_majority_cap)
        self.multiclass_benign_cap = int(self.multiclass_benign_cap)
        self.multiclass_attack_cap = int(self.multiclass_attack_cap)
        self.multiclass_min_target_count = int(self.multiclass_min_target_count)
        self.knn_neighbors = int(self.knn_neighbors)
        self.excluded_feature_columns = tuple(self.excluded_feature_columns)
        self.metadata_columns = tuple(self.metadata_columns)
        self.binary_resampling_method = self.binary_resampling_method.lower()
        self.multiclass_resampling_method = self.multiclass_resampling_method.lower()
        self.output_dtype = self.output_dtype.lower()
        self.log_level = self.log_level.upper()

    @property
    def common_dir(self) -> Path:
        """Return the directory for split-independent preprocessing artifacts."""
        return self.output_dir / "common"

    @property
    def binary_dir(self) -> Path:
        """Return the binary-classification artifact directory."""
        return self.output_dir / "binary"

    @property
    def multiclass_dir(self) -> Path:
        """Return the multiclass-classification artifact directory."""
        return self.output_dir / "multiclass"

    @property
    def preprocessor_path(self) -> Path:
        """Return the fitted preprocessor artifact path."""
        return self.common_dir / self.preprocessor_filename

    @property
    def feature_manifest_path(self) -> Path:
        """Return the feature manifest artifact path."""
        return self.common_dir / self.feature_manifest_filename

    @property
    def report_path(self) -> Path:
        """Return the Phase 4 report destination."""
        return self.output_dir / self.report_filename

    @property
    def log_path(self) -> Path:
        """Return the Phase 4 log destination."""
        return self.logs_dir / self.log_filename

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        for directory in (self.output_dir, self.common_dir, self.binary_dir, self.multiclass_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(
        cls,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "PreprocessingConfig":
        """Create a Phase 4 configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "preprocessing_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            input_data_path=payload.get("input_data_path"),
            output_dir=payload.get("output_dir"),
            logs_dir=payload.get("logs_dir"),
            preprocessor_filename=payload.get("preprocessor_filename", "preprocessor.joblib"),
            feature_manifest_filename=payload.get("feature_manifest_filename", "feature_manifest.json"),
            report_filename=payload.get("report_filename", "preprocessing_report.json"),
            log_filename=payload.get("log_filename", "phase4_preprocessing.log"),
            chunk_size=payload.get("chunk_size", 100_000),
            test_size=payload.get("test_size", 0.2),
            random_state=payload.get("random_state", 42),
            label_column=payload.get("label_column", "label"),
            source_file_column=payload.get("source_file_column", "source_file"),
            binary_target_column=payload.get("binary_target_column", "label_binary"),
            multiclass_target_column=payload.get("multiclass_target_column", "label_multiclass"),
            include_source_file_feature=payload.get("include_source_file_feature", False),
            excluded_feature_columns=tuple(payload.get("excluded_feature_columns", ("label", "label_binary", "label_multiclass"))),
            metadata_columns=tuple(payload.get("metadata_columns", ("label", "source_file", "label_binary", "label_multiclass"))),
            binary_resampling_method=payload.get("binary_resampling_method", "adasyn"),
            binary_majority_cap=payload.get("binary_majority_cap", 500_000),
            binary_target_ratio=payload.get("binary_target_ratio", 1.0),
            multiclass_resampling_method=payload.get("multiclass_resampling_method", "smote"),
            multiclass_benign_cap=payload.get("multiclass_benign_cap", 500_000),
            multiclass_attack_cap=payload.get("multiclass_attack_cap", 100_000),
            multiclass_min_target_count=payload.get("multiclass_min_target_count", 20_000),
            knn_neighbors=payload.get("knn_neighbors", 5),
            output_dtype=payload.get("output_dtype", "float32"),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for logs and reports."""
        return {
            "project_root": str(self.project_root),
            "input_data_path": str(self.input_data_path),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "preprocessor_filename": self.preprocessor_filename,
            "feature_manifest_filename": self.feature_manifest_filename,
            "report_filename": self.report_filename,
            "log_filename": self.log_filename,
            "chunk_size": self.chunk_size,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "label_column": self.label_column,
            "source_file_column": self.source_file_column,
            "binary_target_column": self.binary_target_column,
            "multiclass_target_column": self.multiclass_target_column,
            "include_source_file_feature": self.include_source_file_feature,
            "excluded_feature_columns": list(self.excluded_feature_columns),
            "metadata_columns": list(self.metadata_columns),
            "binary_resampling_method": self.binary_resampling_method,
            "binary_majority_cap": self.binary_majority_cap,
            "binary_target_ratio": self.binary_target_ratio,
            "multiclass_resampling_method": self.multiclass_resampling_method,
            "multiclass_benign_cap": self.multiclass_benign_cap,
            "multiclass_attack_cap": self.multiclass_attack_cap,
            "multiclass_min_target_count": self.multiclass_min_target_count,
            "knn_neighbors": self.knn_neighbors,
            "output_dtype": self.output_dtype,
            "log_level": self.log_level,
        }
