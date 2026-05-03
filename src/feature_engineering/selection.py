"""Feature-selection utilities for SentinelNet Phase 5."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import FeatureEngineeringConfig


@dataclass(slots=True)
class SelectionArtifacts:
    """Outputs from the feature-selection stage."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    selection_sample_indices: np.ndarray
    rfe_sample_indices: np.ndarray
    correlation_selected_features: list[str]
    correlation_dropped_features: dict[str, list[str]]
    mutual_information_scores: pd.DataFrame
    mutual_information_top_features: list[str]
    rfe_selected_features: list[str]
    rfe_ranking: dict[str, int]
    pca_component_count: int
    pca_explained_variance_ratio: list[float]


def build_train_test_split(
    targets: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/test split."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    indices = np.arange(len(targets))
    train_indices, test_indices = next(splitter.split(indices, targets))
    return train_indices, test_indices


def stratified_sample_indices(
    indices: np.ndarray,
    targets: np.ndarray,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """Sample row indices with per-class proportional allocation."""
    if len(indices) <= sample_size:
        return np.sort(indices.astype(np.int64, copy=False))

    rng = np.random.default_rng(random_state)
    target_subset = targets[indices]
    class_counts = Counter(target_subset.astype(int).tolist())
    total_count = len(indices)

    allocations: dict[int, int] = {}
    remainders: list[tuple[float, int]] = []

    for class_value, count in sorted(class_counts.items()):
        raw = sample_size * (count / total_count)
        allocation = max(1, int(np.floor(raw)))
        allocation = min(allocation, count)
        allocations[class_value] = allocation
        remainders.append((raw - np.floor(raw), class_value))

    allocated = sum(allocations.values())
    while allocated > sample_size:
        for _, class_value in sorted(remainders):
            if allocated <= sample_size:
                break
            if allocations[class_value] > 1:
                allocations[class_value] -= 1
                allocated -= 1

    while allocated < sample_size:
        for _, class_value in sorted(remainders, reverse=True):
            if allocated >= sample_size:
                break
            if allocations[class_value] < class_counts[class_value]:
                allocations[class_value] += 1
                allocated += 1

    selected_indices: list[np.ndarray] = []
    for class_value, allocation in allocations.items():
        class_indices = indices[target_subset == class_value]
        sampled = rng.choice(class_indices, size=allocation, replace=False)
        selected_indices.append(np.sort(sampled))

    return np.sort(np.concatenate(selected_indices).astype(np.int64, copy=False))


def collect_rows_by_indices(
    csv_path: str | bytes | "os.PathLike[str]" | "os.PathLike[bytes]",
    selected_indices: np.ndarray,
    chunk_size: int,
) -> pd.DataFrame:
    """Collect a subset of rows from a large CSV using global row indices."""
    sorted_indices = np.sort(selected_indices.astype(np.int64, copy=False))
    frames: list[pd.DataFrame] = []
    row_cursor = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        start = row_cursor
        end = row_cursor + len(chunk)
        left = np.searchsorted(sorted_indices, start, side="left")
        right = np.searchsorted(sorted_indices, end, side="left")
        if right > left:
            local_positions = sorted_indices[left:right] - start
            frames.append(chunk.iloc[local_positions].copy())
        row_cursor = end

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def correlation_filter(
    frame: pd.DataFrame,
    candidate_features: list[str],
    threshold: float,
) -> tuple[list[str], dict[str, list[str]]]:
    """Drop highly correlated features while preserving the original order."""
    if not candidate_features:
        return [], {}

    correlation_matrix = frame.loc[:, candidate_features].corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    dropped_features: dict[str, list[str]] = {}

    for column in upper_triangle.columns:
        correlated_with = upper_triangle.index[upper_triangle[column] > threshold].tolist()
        if correlated_with:
            dropped_features[column] = correlated_with

    selected_features = [feature for feature in candidate_features if feature not in dropped_features]
    return selected_features, dropped_features


def compute_mutual_information(
    frame: pd.DataFrame,
    candidate_features: list[str],
    targets: np.ndarray,
    random_state: int,
) -> pd.DataFrame:
    """Rank features by multiclass mutual information."""
    if not candidate_features:
        return pd.DataFrame(columns=["feature", "score"])

    scores = mutual_info_classif(
        frame.loc[:, candidate_features].to_numpy(dtype=np.float64, copy=False),
        targets.astype(int, copy=False),
        discrete_features=False,
        random_state=random_state,
    )
    score_frame = pd.DataFrame({"feature": candidate_features, "score": scores})
    return score_frame.sort_values(by="score", ascending=False, kind="stable").reset_index(drop=True)


def run_rfe(
    frame: pd.DataFrame,
    candidate_features: list[str],
    targets: np.ndarray,
    n_features_to_select: int,
    random_state: int,
) -> tuple[list[str], dict[str, int]]:
    """Run recursive feature elimination using a linear SVM estimator."""
    if not candidate_features:
        return [], {}

    selection_count = min(n_features_to_select, len(candidate_features))
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(frame.loc[:, candidate_features].to_numpy(dtype=np.float64, copy=False))

    estimator = LinearSVC(random_state=random_state, dual=False, max_iter=20000, tol=1e-3)
    selector = RFE(estimator=estimator, n_features_to_select=selection_count, step=0.2)
    selector.fit(scaled_features, targets.astype(int, copy=False))

    selected_features = [feature for feature, keep in zip(candidate_features, selector.support_) if keep]
    ranking = {feature: int(rank) for feature, rank in zip(candidate_features, selector.ranking_)}
    return selected_features, ranking


def fit_optional_pca(
    frame: pd.DataFrame,
    selected_features: list[str],
    config: FeatureEngineeringConfig,
) -> tuple[int, list[float]]:
    """Fit and persist an optional PCA model over the selected features."""
    if not config.pca_enabled or not selected_features:
        return 0, []

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(frame.loc[:, selected_features].to_numpy(dtype=np.float64, copy=False))
    pca = PCA(n_components=config.pca_variance_threshold, svd_solver="full", random_state=config.random_state)
    pca.fit(scaled_features)

    joblib.dump(
        {
            "selected_features": selected_features,
            "scaler": scaler,
            "pca": pca,
        },
        config.pca_path,
    )
    return int(pca.n_components_), pca.explained_variance_ratio_.astype(float).tolist()
