"""Model registry for SentinelNet Phase 8 anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .config import AnomalyDetectionConfig


@dataclass(slots=True)
class DetectorSpec:
    """A trainable anomaly-detector specification."""

    name: str
    builder: Callable[[AnomalyDetectionConfig], Any]


def build_isolation_forest(config: AnomalyDetectionConfig) -> IsolationForest:
    """Create an isolation-forest detector."""
    return IsolationForest(
        n_estimators=config.isolation_forest_n_estimators,
        contamination=config.isolation_forest_contamination,
        max_samples=min(config.isolation_forest_max_samples, max(2, config.isolation_forest_train_cap or config.isolation_forest_max_samples)),
        random_state=config.random_state,
        n_jobs=1,
    )


def build_one_class_svm(config: AnomalyDetectionConfig) -> OneClassSVM:
    """Create a one-class SVM detector."""
    return OneClassSVM(
        kernel=config.one_class_svm_kernel,
        gamma=config.one_class_svm_gamma,
        nu=config.one_class_svm_nu,
    )


def build_lof(config: AnomalyDetectionConfig) -> LocalOutlierFactor:
    """Create a novelty-enabled local outlier factor detector."""
    return LocalOutlierFactor(
        n_neighbors=config.lof_n_neighbors,
        contamination=config.lof_contamination,
        novelty=True,
        n_jobs=1,
    )


def get_detector_specs() -> list[DetectorSpec]:
    """Return the ordered list of Phase 8 anomaly-detector specifications."""
    return [
        DetectorSpec(name="isolation_forest", builder=build_isolation_forest),
        DetectorSpec(name="one_class_svm", builder=build_one_class_svm),
        DetectorSpec(name="lof", builder=build_lof),
    ]
