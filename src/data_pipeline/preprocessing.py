"""Phase 4 preprocessing pipeline for SentinelNet v2."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cleaning import normalize_string_series
from .config import PreprocessingConfig
from .resampling import (
    adasyn_resample_binary,
    class_distribution,
    downsample_by_class,
    smote_resample_multiclass,
)
from .schema import normalize_columns

LOGGER = logging.getLogger("sentinelnet.phase4")


@dataclass(slots=True)
class FeatureSchema:
    """Describes the feature layout used during preprocessing."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    metadata_columns: list[str]
    feature_columns: list[str]


@dataclass(slots=True)
class FittedPreprocessor:
    """Container for fitted preprocessing components."""

    schema: FeatureSchema
    scaler: StandardScaler | None
    encoder: OneHotEncoder | None
    categories: dict[str, list[str]]
    feature_names: list[str]


@dataclass(slots=True)
class PreprocessingReport:
    """Serializable report returned after Phase 4 preprocessing completes."""

    created_at_utc: str
    input_path: str
    output_dir: str
    report_path: str
    rows_read: int
    train_rows: int
    test_rows: int
    transformed_feature_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_names: list[str]
    binary_train_distribution_before: dict[int, int]
    binary_train_distribution_after: dict[int, int]
    binary_test_distribution: dict[int, int]
    multiclass_train_distribution_before: dict[int, int]
    multiclass_train_distribution_after: dict[int, int]
    multiclass_test_distribution: dict[int, int]
    artifact_paths: dict[str, str]
    validation_passed: bool
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "report_path": self.report_path,
            "rows_read": self.rows_read,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "transformed_feature_count": self.transformed_feature_count,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "feature_names": self.feature_names,
            "binary_train_distribution_before": self.binary_train_distribution_before,
            "binary_train_distribution_after": self.binary_train_distribution_after,
            "binary_test_distribution": self.binary_test_distribution,
            "multiclass_train_distribution_before": self.multiclass_train_distribution_before,
            "multiclass_train_distribution_after": self.multiclass_train_distribution_after,
            "multiclass_test_distribution": self.multiclass_test_distribution,
            "artifact_paths": self.artifact_paths,
            "validation_passed": self.validation_passed,
            "config": self.config,
        }


def _infer_feature_schema(sample_frame: pd.DataFrame, config: PreprocessingConfig) -> FeatureSchema:
    """Infer numeric and categorical feature columns from a representative sample."""
    sample_frame.columns = normalize_columns(list(sample_frame.columns))

    metadata_columns = list(dict.fromkeys(config.metadata_columns))
    excluded = set(config.excluded_feature_columns)
    if not config.include_source_file_feature:
        excluded.add(config.source_file_column)

    feature_columns = [
        column
        for column in sample_frame.columns
        if column not in excluded
    ]

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in feature_columns:
        dtype = sample_frame[column].dtype
        if is_numeric_dtype(dtype) and not is_bool_dtype(dtype):
            numeric_columns.append(column)
        elif is_object_dtype(dtype) or is_string_dtype(dtype) or is_bool_dtype(dtype):
            categorical_columns.append(column)
        else:
            numeric_columns.append(column)

    return FeatureSchema(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        metadata_columns=metadata_columns,
        feature_columns=feature_columns,
    )


def _read_target_arrays(config: PreprocessingConfig, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray, int]:
    """Read target vectors incrementally to build a reproducible stratified split."""
    binary_targets: list[np.ndarray] = []
    multiclass_targets: list[np.ndarray] = []
    total_rows = 0

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            config.input_data_path,
            usecols=[config.binary_target_column, config.multiclass_target_column],
            chunksize=config.chunk_size,
            low_memory=False,
        ),
        start=1,
    ):
        chunk.columns = normalize_columns(list(chunk.columns))
        binary_targets.append(chunk[config.binary_target_column].to_numpy(dtype=np.int8, copy=False))
        multiclass_targets.append(chunk[config.multiclass_target_column].to_numpy(dtype=np.int16, copy=False))
        total_rows += len(chunk)
        logger.info("Indexed split targets for chunk %d | rows=%d | cumulative_rows=%d", chunk_index, len(chunk), total_rows)

    return (
        np.concatenate(binary_targets),
        np.concatenate(multiclass_targets),
        total_rows,
    )


def _build_split_mask(config: PreprocessingConfig, multiclass_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a single stratified train/test split based on multiclass targets."""
    counts = Counter(multiclass_targets.astype(int).tolist())
    smallest_class = min(counts.values())
    if smallest_class < 2:
        raise ValueError("Phase 4 requires at least two samples in every class for stratified splitting.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
    indices = np.arange(len(multiclass_targets))
    train_indices, test_indices = next(splitter.split(indices, multiclass_targets))
    train_mask = np.zeros(len(multiclass_targets), dtype=bool)
    train_mask[train_indices] = True
    return train_mask, train_indices, test_indices


def _fit_preprocessor(
    config: PreprocessingConfig,
    schema: FeatureSchema,
    train_mask: np.ndarray,
    logger: logging.Logger,
) -> FittedPreprocessor:
    """Fit the scaler and categorical encoder using only training rows."""
    scaler = StandardScaler() if schema.numeric_columns else None
    category_sets = {column: set() for column in schema.categorical_columns}
    row_cursor = 0

    for chunk_index, chunk in enumerate(
        pd.read_csv(config.input_data_path, chunksize=config.chunk_size, low_memory=False),
        start=1,
    ):
        chunk.columns = normalize_columns(list(chunk.columns))
        mask = train_mask[row_cursor : row_cursor + len(chunk)]
        train_chunk = chunk.loc[mask].copy()
        row_cursor += len(chunk)

        if train_chunk.empty:
            continue

        if schema.numeric_columns:
            scaler.partial_fit(train_chunk.loc[:, schema.numeric_columns].to_numpy(dtype=np.float64, copy=False))

        for column in schema.categorical_columns:
            normalized_values = normalize_string_series(train_chunk[column]).fillna("__missing__")
            category_sets[column].update(normalized_values.astype(str).tolist())

        logger.info("Fitted preprocessing stats from chunk %d | train_rows=%d", chunk_index, len(train_chunk))

    encoder = None
    categories: dict[str, list[str]] = {}
    categorical_feature_names: list[str] = []

    if schema.categorical_columns:
        categories = {
            column: sorted(values) if values else ["__missing__"]
            for column, values in category_sets.items()
        }
        encoder = OneHotEncoder(
            categories=[categories[column] for column in schema.categorical_columns],
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        dummy_frame = pd.DataFrame(
            {
                column: [categories[column][0]]
                for column in schema.categorical_columns
            }
        )
        encoder.fit(dummy_frame)
        categorical_feature_names = encoder.get_feature_names_out(schema.categorical_columns).tolist()

    feature_names = [*schema.numeric_columns, *categorical_feature_names]
    return FittedPreprocessor(
        schema=schema,
        scaler=scaler,
        encoder=encoder,
        categories=categories,
        feature_names=feature_names,
    )


def _transform_features(
    frame: pd.DataFrame,
    fitted_preprocessor: FittedPreprocessor,
    output_dtype: str,
) -> np.ndarray:
    """Transform a frame into the model-ready feature matrix."""
    parts: list[np.ndarray] = []
    dtype = np.dtype(output_dtype)

    if fitted_preprocessor.schema.numeric_columns:
        numeric_values = frame.loc[:, fitted_preprocessor.schema.numeric_columns].to_numpy(dtype=np.float64, copy=False)
        if fitted_preprocessor.scaler is not None:
            numeric_values = fitted_preprocessor.scaler.transform(numeric_values)
        parts.append(numeric_values.astype(dtype, copy=False))

    if fitted_preprocessor.schema.categorical_columns:
        categorical_frame = frame.loc[:, fitted_preprocessor.schema.categorical_columns].copy()
        for column in fitted_preprocessor.schema.categorical_columns:
            categorical_frame[column] = normalize_string_series(categorical_frame[column]).fillna("__missing__")
        encoded = fitted_preprocessor.encoder.transform(categorical_frame)
        parts.append(encoded.astype(dtype, copy=False))

    if not parts:
        return np.empty((len(frame), 0), dtype=dtype)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1).astype(dtype, copy=False)


def _write_array(path: Path, array: np.ndarray) -> None:
    """Write a NumPy array to disk."""
    np.save(path, array)


def _append_metadata(path: Path, frame: pd.DataFrame, header_written: bool) -> bool:
    """Append metadata rows to a CSV artifact."""
    frame.to_csv(path, mode="a" if header_written else "w", header=not header_written, index=False)
    return True


def _prepare_binary_training_set(
    config: PreprocessingConfig,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap the majority class and apply the configured binary resampler."""
    minority_count = int(np.sum(train_targets == 1))
    effective_majority_cap = max(config.binary_majority_cap, minority_count)
    downsampled_features, downsampled_targets = downsample_by_class(
        train_features,
        train_targets,
        class_caps={0: effective_majority_cap},
        random_state=config.random_state,
    )
    class_counts = class_distribution(downsampled_targets)
    if class_counts.get(1, 0) > class_counts.get(0, 0):
        downsampled_features, downsampled_targets = downsample_by_class(
            downsampled_features,
            downsampled_targets,
            class_caps={1: class_counts.get(0, 0)},
            random_state=config.random_state,
        )

    if config.binary_resampling_method == "adasyn":
        return adasyn_resample_binary(
            downsampled_features,
            downsampled_targets,
            minority_label=1,
            majority_label=0,
            target_ratio=config.binary_target_ratio,
            k_neighbors=config.knn_neighbors,
            random_state=config.random_state,
        )

    raise ValueError(f"Unsupported binary resampling method: {config.binary_resampling_method}")


def _prepare_multiclass_training_set(
    config: PreprocessingConfig,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap dominant classes and apply multiclass SMOTE-style balancing."""
    class_caps = {0: config.multiclass_benign_cap}
    for class_value in np.unique(train_targets).tolist():
        if int(class_value) != 0:
            class_caps[int(class_value)] = config.multiclass_attack_cap

    downsampled_features, downsampled_targets = downsample_by_class(
        train_features,
        train_targets,
        class_caps=class_caps,
        random_state=config.random_state,
    )

    attack_counts = [
        count
        for class_value, count in class_distribution(downsampled_targets).items()
        if class_value != 0
    ]
    target_count = max(config.multiclass_min_target_count, int(np.median(attack_counts))) if attack_counts else config.multiclass_min_target_count

    if config.multiclass_resampling_method == "smote":
        return smote_resample_multiclass(
            downsampled_features,
            downsampled_targets,
            benign_label=0,
            target_count=target_count,
            attack_cap=config.multiclass_attack_cap,
            k_neighbors=config.knn_neighbors,
            random_state=config.random_state,
        )

    raise ValueError(f"Unsupported multiclass resampling method: {config.multiclass_resampling_method}")


def run_preprocessing_pipeline(
    config: PreprocessingConfig,
    logger: logging.Logger | None = None,
) -> PreprocessingReport:
    """Execute the complete Phase 4 preprocessing workflow."""
    active_logger = logger or LOGGER
    config.ensure_directories()

    if not config.input_data_path.exists():
        raise FileNotFoundError(f"Phase 3 labeled dataset not found at {config.input_data_path}")

    sample_frame = pd.read_csv(config.input_data_path, nrows=min(1000, config.chunk_size))
    schema = _infer_feature_schema(sample_frame, config)
    binary_targets, multiclass_targets, total_rows = _read_target_arrays(config, active_logger)
    train_mask, train_indices, test_indices = _build_split_mask(config, multiclass_targets)
    fitted_preprocessor = _fit_preprocessor(config, schema, train_mask, active_logger)

    train_rows = len(train_indices)
    test_rows = len(test_indices)
    transformed_feature_count = len(fitted_preprocessor.feature_names)

    train_feature_path = config.common_dir / "X_train.npy"
    test_feature_path = config.common_dir / "X_test.npy"
    y_binary_train_path = config.common_dir / "y_binary_train.npy"
    y_binary_test_path = config.common_dir / "y_binary_test.npy"
    y_multiclass_train_path = config.common_dir / "y_multiclass_train.npy"
    y_multiclass_test_path = config.common_dir / "y_multiclass_test.npy"
    train_metadata_path = config.common_dir / "train_metadata.csv"
    test_metadata_path = config.common_dir / "test_metadata.csv"

    train_feature_map = np.lib.format.open_memmap(
        train_feature_path,
        mode="w+",
        dtype=config.output_dtype,
        shape=(train_rows, transformed_feature_count),
    )
    test_feature_map = np.lib.format.open_memmap(
        test_feature_path,
        mode="w+",
        dtype=config.output_dtype,
        shape=(test_rows, transformed_feature_count),
    )

    train_cursor = 0
    test_cursor = 0
    row_cursor = 0
    train_metadata_written = False
    test_metadata_written = False

    for chunk_index, chunk in enumerate(
        pd.read_csv(config.input_data_path, chunksize=config.chunk_size, low_memory=False),
        start=1,
    ):
        chunk.columns = normalize_columns(list(chunk.columns))
        mask = train_mask[row_cursor : row_cursor + len(chunk)]
        row_cursor += len(chunk)

        features_chunk = chunk.loc[:, schema.feature_columns].copy()
        transformed_chunk = _transform_features(features_chunk, fitted_preprocessor, config.output_dtype)

        train_chunk = transformed_chunk[mask]
        test_chunk = transformed_chunk[~mask]
        train_chunk_rows = len(train_chunk)
        test_chunk_rows = len(test_chunk)

        train_feature_map[train_cursor : train_cursor + train_chunk_rows] = train_chunk
        test_feature_map[test_cursor : test_cursor + test_chunk_rows] = test_chunk

        if config.metadata_columns:
            train_metadata_written = _append_metadata(
                train_metadata_path,
                chunk.loc[mask, list(config.metadata_columns)].copy(),
                train_metadata_written,
            )
            test_metadata_written = _append_metadata(
                test_metadata_path,
                chunk.loc[~mask, list(config.metadata_columns)].copy(),
                test_metadata_written,
            )

        train_cursor += train_chunk_rows
        test_cursor += test_chunk_rows

        active_logger.info(
            "Transformed chunk %d | train_rows=%d | test_rows=%d | cumulative_train=%d | cumulative_test=%d",
            chunk_index,
            train_chunk_rows,
            test_chunk_rows,
            train_cursor,
            test_cursor,
        )

    del train_feature_map
    del test_feature_map

    y_binary_train = binary_targets[train_indices].astype(np.int8, copy=False)
    y_binary_test = binary_targets[test_indices].astype(np.int8, copy=False)
    y_multiclass_train = multiclass_targets[train_indices].astype(np.int16, copy=False)
    y_multiclass_test = multiclass_targets[test_indices].astype(np.int16, copy=False)

    _write_array(y_binary_train_path, y_binary_train)
    _write_array(y_binary_test_path, y_binary_test)
    _write_array(y_multiclass_train_path, y_multiclass_train)
    _write_array(y_multiclass_test_path, y_multiclass_test)

    train_features = np.load(train_feature_path, mmap_mode="r")
    binary_train_features, binary_train_targets = _prepare_binary_training_set(
        config,
        np.asarray(train_features, dtype=np.float32),
        y_binary_train,
    )
    multiclass_train_features, multiclass_train_targets = _prepare_multiclass_training_set(
        config,
        np.asarray(train_features, dtype=np.float32),
        y_multiclass_train,
    )

    binary_train_feature_path = config.binary_dir / "X_train_resampled.npy"
    binary_train_target_path = config.binary_dir / "y_train_resampled.npy"
    binary_test_feature_path = config.binary_dir / "X_test.npy"
    binary_test_target_path = config.binary_dir / "y_test.npy"
    multiclass_train_feature_path = config.multiclass_dir / "X_train_resampled.npy"
    multiclass_train_target_path = config.multiclass_dir / "y_train_resampled.npy"
    multiclass_test_feature_path = config.multiclass_dir / "X_test.npy"
    multiclass_test_target_path = config.multiclass_dir / "y_test.npy"

    _write_array(binary_train_feature_path, binary_train_features.astype(config.output_dtype, copy=False))
    _write_array(binary_train_target_path, binary_train_targets)
    _write_array(binary_test_feature_path, np.load(test_feature_path))
    _write_array(binary_test_target_path, y_binary_test)

    _write_array(multiclass_train_feature_path, multiclass_train_features.astype(config.output_dtype, copy=False))
    _write_array(multiclass_train_target_path, multiclass_train_targets)
    _write_array(multiclass_test_feature_path, np.load(test_feature_path))
    _write_array(multiclass_test_target_path, y_multiclass_test)

    artifact_paths = {
        "preprocessor": str(config.preprocessor_path),
        "feature_manifest": str(config.feature_manifest_path),
        "common_X_train": str(train_feature_path),
        "common_X_test": str(test_feature_path),
        "common_y_binary_train": str(y_binary_train_path),
        "common_y_binary_test": str(y_binary_test_path),
        "common_y_multiclass_train": str(y_multiclass_train_path),
        "common_y_multiclass_test": str(y_multiclass_test_path),
        "train_metadata": str(train_metadata_path),
        "test_metadata": str(test_metadata_path),
        "binary_X_train_resampled": str(binary_train_feature_path),
        "binary_y_train_resampled": str(binary_train_target_path),
        "binary_X_test": str(binary_test_feature_path),
        "binary_y_test": str(binary_test_target_path),
        "multiclass_X_train_resampled": str(multiclass_train_feature_path),
        "multiclass_y_train_resampled": str(multiclass_train_target_path),
        "multiclass_X_test": str(multiclass_test_feature_path),
        "multiclass_y_test": str(multiclass_test_target_path),
    }

    feature_manifest = {
        "numeric_columns": schema.numeric_columns,
        "categorical_columns": schema.categorical_columns,
        "feature_columns": schema.feature_columns,
        "metadata_columns": schema.metadata_columns,
        "feature_names": fitted_preprocessor.feature_names,
        "categories": fitted_preprocessor.categories,
    }
    config.feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "schema": feature_manifest,
            "scaler": fitted_preprocessor.scaler,
            "encoder": fitted_preprocessor.encoder,
            "output_dtype": config.output_dtype,
        },
        config.preprocessor_path,
    )

    report = PreprocessingReport(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        input_path=str(config.input_data_path),
        output_dir=str(config.output_dir),
        report_path=str(config.report_path),
        rows_read=total_rows,
        train_rows=train_rows,
        test_rows=test_rows,
        transformed_feature_count=transformed_feature_count,
        numeric_columns=schema.numeric_columns,
        categorical_columns=schema.categorical_columns,
        feature_names=fitted_preprocessor.feature_names,
        binary_train_distribution_before=class_distribution(y_binary_train),
        binary_train_distribution_after=class_distribution(binary_train_targets),
        binary_test_distribution=class_distribution(y_binary_test),
        multiclass_train_distribution_before=class_distribution(y_multiclass_train),
        multiclass_train_distribution_after=class_distribution(multiclass_train_targets),
        multiclass_test_distribution=class_distribution(y_multiclass_test),
        artifact_paths=artifact_paths,
        validation_passed=(
            total_rows == train_rows + test_rows
            and train_cursor == train_rows
            and test_cursor == test_rows
            and not np.isnan(binary_train_features).any()
            and not np.isnan(multiclass_train_features).any()
        ),
        config=config.to_dict(),
    )
    config.report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if not report.validation_passed:
        raise ValueError("Phase 4 validation failed while writing preprocessing artifacts.")

    active_logger.info(
        "Completed Phase 4 preprocessing | train_rows=%d | test_rows=%d | transformed_features=%d",
        train_rows,
        test_rows,
        transformed_feature_count,
    )
    return report
