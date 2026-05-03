"""Data helpers for SentinelNet Phase 8 anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from src.models.deep_learning.data import DeepLearningDataset, load_deep_learning_dataset, stratified_subsample_indices

from .config import AnomalyDetectionConfig


@dataclass(slots=True)
class DetectorTaskData:
    """Benign-only train/validation plus full evaluation data for a detector."""

    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_test_binary: np.ndarray


def load_anomaly_dataset(config: AnomalyDetectionConfig) -> DeepLearningDataset:
    """Load the shared selected-feature dataset and standardized splits."""
    return load_deep_learning_dataset(config)


def build_detector_task(
    dataset: DeepLearningDataset,
    train_cap: int | None,
    test_cap: int | None,
    validation_size: float,
    random_state: int,
) -> DetectorTaskData:
    """Create benign-only training data plus a stratified anomaly-evaluation split."""
    benign_mask = dataset.y_binary_train == 0
    benign_train = dataset.X_train[benign_mask]
    benign_selection = stratified_subsample_indices(np.zeros(len(benign_train), dtype=np.int8), train_cap, random_state)
    benign_train = benign_train[benign_selection]

    validation_size = min(max(validation_size, 0.05), 0.4)
    train_rows, validation_rows = train_test_split(
        benign_train,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )

    test_selection = stratified_subsample_indices(dataset.y_binary_test, test_cap, random_state)
    return DetectorTaskData(
        X_train=train_rows.astype(np.float32, copy=False),
        X_validation=validation_rows.astype(np.float32, copy=False),
        X_test=dataset.X_test[test_selection].astype(np.float32, copy=False),
        y_test_binary=dataset.y_binary_test[test_selection].astype(np.int32, copy=False),
    )
