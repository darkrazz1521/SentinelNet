"""Custom resampling utilities for SentinelNet Phase 4."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _ensure_rng(random_state: int) -> np.random.Generator:
    """Create a deterministic NumPy random generator."""
    return np.random.default_rng(random_state)


def _shuffle_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle a feature matrix and target vector in sync."""
    permutation = rng.permutation(len(targets))
    return features[permutation], targets[permutation]


def downsample_by_class(
    features: np.ndarray,
    targets: np.ndarray,
    class_caps: dict[int, int],
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly downsample classes that exceed configured caps."""
    rng = _ensure_rng(random_state)
    keep_indices: list[np.ndarray] = []

    for class_value in sorted(np.unique(targets).tolist()):
        class_indices = np.where(targets == class_value)[0]
        cap = class_caps.get(int(class_value), len(class_indices))
        if cap < len(class_indices):
            class_indices = rng.choice(class_indices, size=cap, replace=False)
        keep_indices.append(np.sort(class_indices))

    selected_indices = np.concatenate(keep_indices)
    downsampled_features = features[selected_indices]
    downsampled_targets = targets[selected_indices]
    return _shuffle_dataset(downsampled_features, downsampled_targets, rng)


def _generate_interpolated_samples(
    class_samples: np.ndarray,
    neighbors: np.ndarray,
    sample_counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic samples by interpolating between nearest neighbors."""
    total_samples = int(sample_counts.sum())
    if total_samples == 0:
        return np.empty((0, class_samples.shape[1]), dtype=class_samples.dtype)

    synthetic = np.empty((total_samples, class_samples.shape[1]), dtype=np.float32)
    cursor = 0

    for index, count in enumerate(sample_counts):
        if count <= 0:
            continue

        chosen_neighbors = rng.choice(neighbors[index], size=int(count), replace=True)
        base_samples = np.repeat(class_samples[index : index + 1], repeats=int(count), axis=0)
        neighbor_samples = class_samples[chosen_neighbors]
        gaps = rng.random(size=(int(count), 1), dtype=np.float32)
        synthetic[cursor : cursor + int(count)] = base_samples + gaps * (neighbor_samples - base_samples)
        cursor += int(count)

    return synthetic


def smote_resample_binary(
    features: np.ndarray,
    targets: np.ndarray,
    minority_label: int,
    target_count: int,
    k_neighbors: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback SMOTE oversampling for a binary minority class."""
    rng = _ensure_rng(random_state)
    minority_mask = targets == minority_label
    minority_samples = features[minority_mask]

    if len(minority_samples) >= target_count or len(minority_samples) < 2:
        return _shuffle_dataset(features.copy(), targets.copy(), rng)

    k_value = min(k_neighbors, len(minority_samples) - 1)
    if k_value < 1:
        return _shuffle_dataset(features.copy(), targets.copy(), rng)

    n_generate = target_count - len(minority_samples)
    class_neighbors = NearestNeighbors(n_neighbors=k_value + 1)
    class_neighbors.fit(minority_samples)
    neighbor_indices = class_neighbors.kneighbors(minority_samples, return_distance=False)[:, 1:]

    base_counts = np.full(len(minority_samples), n_generate // len(minority_samples), dtype=int)
    remainder = n_generate - int(base_counts.sum())
    if remainder > 0:
        chosen = rng.choice(len(minority_samples), size=remainder, replace=False)
        base_counts[chosen] += 1

    synthetic_samples = _generate_interpolated_samples(minority_samples, neighbor_indices, base_counts, rng)
    synthetic_targets = np.full(len(synthetic_samples), minority_label, dtype=targets.dtype)

    resampled_features = np.vstack([features, synthetic_samples.astype(features.dtype, copy=False)])
    resampled_targets = np.concatenate([targets, synthetic_targets])
    return _shuffle_dataset(resampled_features, resampled_targets, rng)


def adasyn_resample_binary(
    features: np.ndarray,
    targets: np.ndarray,
    minority_label: int,
    majority_label: int,
    target_ratio: float,
    k_neighbors: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ADASYN-style oversampling to a binary minority class."""
    rng = _ensure_rng(random_state)
    majority_count = int(np.sum(targets == majority_label))
    minority_count = int(np.sum(targets == minority_label))
    target_count = int(round(majority_count * target_ratio))

    if minority_count >= target_count:
        return _shuffle_dataset(features.copy(), targets.copy(), rng)

    minority_mask = targets == minority_label
    minority_samples = features[minority_mask]
    if len(minority_samples) < 2:
        return _shuffle_dataset(features.copy(), targets.copy(), rng)

    n_generate = target_count - minority_count
    overall_k = min(k_neighbors, len(features) - 1)
    minority_k = min(k_neighbors, len(minority_samples) - 1)

    if overall_k < 1 or minority_k < 1:
        return smote_resample_binary(features, targets, minority_label, target_count, k_neighbors, random_state)

    overall_neighbors = NearestNeighbors(n_neighbors=overall_k + 1)
    overall_neighbors.fit(features)
    overall_neighbor_indices = overall_neighbors.kneighbors(minority_samples, return_distance=False)[:, 1:]
    difficulty = np.mean(targets[overall_neighbor_indices] == majority_label, axis=1)

    if float(difficulty.sum()) == 0.0:
        return smote_resample_binary(features, targets, minority_label, target_count, k_neighbors, random_state)

    minority_neighbors = NearestNeighbors(n_neighbors=minority_k + 1)
    minority_neighbors.fit(minority_samples)
    minority_neighbor_indices = minority_neighbors.kneighbors(minority_samples, return_distance=False)[:, 1:]

    difficulty_weights = difficulty / difficulty.sum()
    raw_allocations = difficulty_weights * n_generate
    sample_counts = np.floor(raw_allocations).astype(int)
    remainder = n_generate - int(sample_counts.sum())
    if remainder > 0:
        remainder_indices = rng.choice(len(sample_counts), size=remainder, replace=True, p=difficulty_weights)
        for index in remainder_indices:
            sample_counts[index] += 1

    synthetic_samples = _generate_interpolated_samples(minority_samples, minority_neighbor_indices, sample_counts, rng)
    synthetic_targets = np.full(len(synthetic_samples), minority_label, dtype=targets.dtype)

    resampled_features = np.vstack([features, synthetic_samples.astype(features.dtype, copy=False)])
    resampled_targets = np.concatenate([targets, synthetic_targets])
    return _shuffle_dataset(resampled_features, resampled_targets, rng)


def smote_resample_multiclass(
    features: np.ndarray,
    targets: np.ndarray,
    benign_label: int,
    target_count: int,
    attack_cap: int,
    k_neighbors: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply class-wise SMOTE oversampling to attack classes in a multiclass dataset."""
    rng = _ensure_rng(random_state)
    unique_classes = sorted(np.unique(targets).tolist())
    augmented_features = [features]
    augmented_targets = [targets]

    for class_value in unique_classes:
        if int(class_value) == benign_label:
            continue

        class_mask = targets == class_value
        class_samples = features[class_mask]
        class_count = len(class_samples)
        desired_count = min(max(class_count, target_count), attack_cap)

        if class_count >= desired_count or class_count < 2:
            continue

        k_value = min(k_neighbors, class_count - 1)
        if k_value < 1:
            continue

        n_generate = desired_count - class_count
        neighbors = NearestNeighbors(n_neighbors=k_value + 1)
        neighbors.fit(class_samples)
        neighbor_indices = neighbors.kneighbors(class_samples, return_distance=False)[:, 1:]

        sample_counts = np.full(class_count, n_generate // class_count, dtype=int)
        remainder = n_generate - int(sample_counts.sum())
        if remainder > 0:
            chosen = rng.choice(class_count, size=remainder, replace=True)
            for index in chosen:
                sample_counts[index] += 1

        synthetic_samples = _generate_interpolated_samples(class_samples, neighbor_indices, sample_counts, rng)
        synthetic_targets = np.full(len(synthetic_samples), int(class_value), dtype=targets.dtype)
        augmented_features.append(synthetic_samples.astype(features.dtype, copy=False))
        augmented_targets.append(synthetic_targets)

    resampled_features = np.vstack(augmented_features)
    resampled_targets = np.concatenate(augmented_targets)
    return _shuffle_dataset(resampled_features, resampled_targets, rng)


def class_distribution(targets: np.ndarray) -> dict[int, int]:
    """Return a deterministic class distribution mapping."""
    counts = Counter(targets.astype(int).tolist())
    return dict(sorted(counts.items()))

