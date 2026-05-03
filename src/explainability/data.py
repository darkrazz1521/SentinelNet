"""Data-loading helpers for SentinelNet Phase 10 explainability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.deep_learning.data import load_feature_names, load_label_metadata, stratified_subsample_indices

from .config import ExplainabilityConfig


@dataclass(slots=True)
class ExplainabilityDataset:
    """Shared raw-feature dataset for Phase 10 explainability."""

    feature_names: list[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_binary_train: np.ndarray
    y_binary_test: np.ndarray
    y_multiclass_train: np.ndarray
    y_multiclass_test: np.ndarray
    source_file_train: np.ndarray
    source_file_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    inverse_multiclass_mapping: dict[int, str]


def load_explainability_dataset(config: ExplainabilityConfig) -> ExplainabilityDataset:
    """Load the selected-feature dataset into raw train/test arrays."""
    feature_names = load_feature_names(config.feature_manifest_path)
    label_metadata = load_label_metadata(config.label_mapping_path)
    usecols = [
        *feature_names,
        config.source_file_column,
        config.binary_target_column,
        config.multiclass_target_column,
    ]
    frame = pd.read_csv(config.input_data_path, usecols=usecols, low_memory=False)

    train_indices = np.load(config.train_indices_path).astype(np.int64, copy=False)
    test_indices = np.load(config.test_indices_path).astype(np.int64, copy=False)
    feature_frame = frame.loc[:, feature_names].astype(np.float32)

    return ExplainabilityDataset(
        feature_names=feature_names,
        X_train=feature_frame.iloc[train_indices].to_numpy(dtype=np.float32, copy=False),
        X_test=feature_frame.iloc[test_indices].to_numpy(dtype=np.float32, copy=False),
        y_binary_train=frame.iloc[train_indices][config.binary_target_column].to_numpy(dtype=np.int32, copy=False),
        y_binary_test=frame.iloc[test_indices][config.binary_target_column].to_numpy(dtype=np.int32, copy=False),
        y_multiclass_train=frame.iloc[train_indices][config.multiclass_target_column].to_numpy(dtype=np.int32, copy=False),
        y_multiclass_test=frame.iloc[test_indices][config.multiclass_target_column].to_numpy(dtype=np.int32, copy=False),
        source_file_train=frame.iloc[train_indices][config.source_file_column].astype(str).to_numpy(copy=False),
        source_file_test=frame.iloc[test_indices][config.source_file_column].astype(str).to_numpy(copy=False),
        train_indices=train_indices,
        test_indices=test_indices,
        inverse_multiclass_mapping=label_metadata["inverse_multiclass_mapping"],
    )


def sample_subset(
    X: np.ndarray,
    y: np.ndarray,
    source_files: np.ndarray,
    original_indices: np.ndarray,
    cap: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a stratified subset with source-file and index lineage preserved."""
    selection = stratified_subsample_indices(y, cap, random_state)
    return (
        X[selection].astype(np.float32, copy=False),
        y[selection].astype(np.int32, copy=False),
        source_files[selection],
        original_indices[selection].astype(np.int64, copy=False),
    )
