"""Tests for the SentinelNet Phase 6 classical ML training pipeline."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.ml.config import MLTrainingConfig
from src.models.ml.training import run_ml_training_pipeline


def _build_phase6_fixture(rows_per_class: int = 20) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    classes = [
        ("BENIGN", 0, 0, 0.0),
        ("Bot", 1, 1, 5.0),
        ("DDoS", 1, 2, 10.0),
        ("PortScan", 1, 10, 15.0),
    ]
    for label, binary_target, multiclass_target, offset in classes:
        for index in range(rows_per_class):
            rows.append(
                {
                    "destination_port": 20 + index + offset,
                    "flow_duration": 1000 + index * 10 + offset * 10,
                    "flow_bytes_per_s": 30 + index + offset,
                    "flow_iat_mean": 5 + index / 10.0 + offset,
                    "fwd_header_length": 40 + index + offset,
                    "packet_length_variance": 10 + index * 0.5 + offset,
                    "forward_payload_efficiency": 0.2 + index / 100.0 + offset / 50.0,
                    "burstiness_score": 0.1 + index / 100.0 + offset / 60.0,
                    "rolling_unique_destination_ports_w20": 1 + (index % 5) + offset / 10.0,
                    "label": label,
                    "source_file": f"{label.lower()}.csv",
                    "label_binary": binary_target,
                    "label_multiclass": multiclass_target,
                }
            )
    return pd.DataFrame(rows)


def _is_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def test_run_ml_training_pipeline_trains_available_models_and_reports_optional_ones() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "ml_training_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "processed" / "feature_engineered").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "saved_models").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    frame = _build_phase6_fixture()
    input_path = project_root / "data" / "processed" / "feature_engineered" / "selected_dataset.csv"
    manifest_path = project_root / "data" / "processed" / "feature_engineered" / "selected_feature_manifest.json"
    train_indices_path = project_root / "data" / "processed" / "feature_engineered" / "train_indices.npy"
    test_indices_path = project_root / "data" / "processed" / "feature_engineered" / "test_indices.npy"
    frame.to_csv(input_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "selected_features": [
                    "destination_port",
                    "flow_duration",
                    "flow_bytes_per_s",
                    "flow_iat_mean",
                    "fwd_header_length",
                    "packet_length_variance",
                    "forward_payload_efficiency",
                    "burstiness_score",
                    "rolling_unique_destination_ports_w20",
                ]
            }
        ),
        encoding="utf-8",
    )

    train_indices = np.arange(0, len(frame), 2, dtype=np.int64)
    test_indices = np.arange(1, len(frame), 2, dtype=np.int64)
    np.save(train_indices_path, train_indices)
    np.save(test_indices_path, test_indices)

    config = MLTrainingConfig(
        project_root=project_root,
        input_data_path=input_path,
        feature_manifest_path=manifest_path,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
        output_dir=project_root / "models" / "saved_models" / "phase6_ml",
        logs_dir=project_root / "logs",
        random_forest_n_estimators=20,
        logistic_max_iter=300,
    )

    try:
        report = run_ml_training_pipeline(config)
        metrics_summary = pd.read_csv(config.metrics_path)

        trained = {(item["task"], item["model"]) for item in report.models if item["status"] == "trained"}
        unavailable = {(item["task"], item["model"]) for item in report.models if item["status"] == "unavailable"}

        assert report.validation_passed is True
        assert ("binary", "logistic_regression") in trained
        assert ("binary", "random_forest") in trained
        assert ("multiclass", "logistic_regression") in trained
        assert ("multiclass", "random_forest") in trained
        assert len(metrics_summary) == len(report.models)
        assert (config.binary_dir / "logistic_regression.joblib").exists()
        assert (config.multiclass_dir / "random_forest.joblib").exists()

        expected_optional = {
            ("binary", "xgboost"): _is_available("xgboost"),
            ("multiclass", "xgboost"): _is_available("xgboost"),
            ("binary", "lightgbm"): _is_available("lightgbm"),
            ("multiclass", "lightgbm"): _is_available("lightgbm"),
        }
        for model_key, is_available in expected_optional.items():
            if is_available:
                assert model_key in trained
            else:
                assert model_key in unavailable
    finally:
        shutil.rmtree(project_root, ignore_errors=True)


def test_run_ml_training_pipeline_reuses_existing_artifacts_on_resume() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "ml_training_resume_workspace"
    shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "processed" / "feature_engineered").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "saved_models").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    frame = _build_phase6_fixture(rows_per_class=10)
    input_path = project_root / "data" / "processed" / "feature_engineered" / "selected_dataset.csv"
    manifest_path = project_root / "data" / "processed" / "feature_engineered" / "selected_feature_manifest.json"
    train_indices_path = project_root / "data" / "processed" / "feature_engineered" / "train_indices.npy"
    test_indices_path = project_root / "data" / "processed" / "feature_engineered" / "test_indices.npy"
    frame.to_csv(input_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "selected_features": [
                    "destination_port",
                    "flow_duration",
                    "flow_bytes_per_s",
                    "flow_iat_mean",
                    "fwd_header_length",
                    "packet_length_variance",
                    "forward_payload_efficiency",
                    "burstiness_score",
                    "rolling_unique_destination_ports_w20",
                ]
            }
        ),
        encoding="utf-8",
    )

    train_indices = np.arange(0, len(frame), 2, dtype=np.int64)
    test_indices = np.arange(1, len(frame), 2, dtype=np.int64)
    np.save(train_indices_path, train_indices)
    np.save(test_indices_path, test_indices)

    config = MLTrainingConfig(
        project_root=project_root,
        input_data_path=input_path,
        feature_manifest_path=manifest_path,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
        output_dir=project_root / "models" / "saved_models" / "phase6_ml",
        logs_dir=project_root / "logs",
        random_forest_n_estimators=10,
        logistic_max_iter=200,
        xgboost_n_estimators=10,
        lightgbm_n_estimators=10,
    )

    try:
        first_report = run_ml_training_pipeline(config)
        second_report = run_ml_training_pipeline(config)

        first_status = {(item["task"], item["model"]): item["status"] for item in first_report.models}
        second_status = {(item["task"], item["model"]): item["status"] for item in second_report.models}

        assert first_status[("binary", "logistic_regression")] == "trained"
        assert first_status[("multiclass", "random_forest")] == "trained"
        assert second_status[("binary", "logistic_regression")] == "reused"
        assert second_status[("multiclass", "random_forest")] == "reused"

        for model_name, module_name in (("xgboost", "xgboost"), ("lightgbm", "lightgbm")):
            for task_name in ("binary", "multiclass"):
                status = second_status[(task_name, model_name)]
                if _is_available(module_name):
                    assert status == "reused"
                else:
                    assert status == "unavailable"
    finally:
        shutil.rmtree(project_root, ignore_errors=True)


def test_run_ml_training_pipeline_continues_random_forest_from_checkpoint() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "ml_training_rf_checkpoint_workspace"
    shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "processed" / "feature_engineered").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "saved_models").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    frame = _build_phase6_fixture(rows_per_class=10)
    input_path = project_root / "data" / "processed" / "feature_engineered" / "selected_dataset.csv"
    manifest_path = project_root / "data" / "processed" / "feature_engineered" / "selected_feature_manifest.json"
    train_indices_path = project_root / "data" / "processed" / "feature_engineered" / "train_indices.npy"
    test_indices_path = project_root / "data" / "processed" / "feature_engineered" / "test_indices.npy"
    frame.to_csv(input_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "selected_features": [
                    "destination_port",
                    "flow_duration",
                    "flow_bytes_per_s",
                    "flow_iat_mean",
                    "fwd_header_length",
                    "packet_length_variance",
                    "forward_payload_efficiency",
                    "burstiness_score",
                    "rolling_unique_destination_ports_w20",
                ]
            }
        ),
        encoding="utf-8",
    )

    train_indices = np.arange(0, len(frame), 2, dtype=np.int64)
    test_indices = np.arange(1, len(frame), 2, dtype=np.int64)
    np.save(train_indices_path, train_indices)
    np.save(test_indices_path, test_indices)

    first_config = MLTrainingConfig(
        project_root=project_root,
        input_data_path=input_path,
        feature_manifest_path=manifest_path,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
        output_dir=project_root / "models" / "saved_models" / "phase6_ml",
        logs_dir=project_root / "logs",
        random_forest_n_estimators=2,
        random_forest_batch_size=1,
        logistic_max_iter=200,
        xgboost_n_estimators=5,
        lightgbm_n_estimators=5,
    )
    second_config = MLTrainingConfig(
        project_root=project_root,
        input_data_path=input_path,
        feature_manifest_path=manifest_path,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
        output_dir=project_root / "models" / "saved_models" / "phase6_ml",
        logs_dir=project_root / "logs",
        random_forest_n_estimators=4,
        random_forest_batch_size=1,
        logistic_max_iter=200,
        xgboost_n_estimators=5,
        lightgbm_n_estimators=5,
    )

    try:
        run_ml_training_pipeline(first_config)
        second_report = run_ml_training_pipeline(second_config)

        statuses = {(item["task"], item["model"]): item["status"] for item in second_report.models}
        binary_rf_artifact = joblib.load(second_config.binary_dir / "random_forest.joblib")
        multiclass_rf_artifact = joblib.load(second_config.multiclass_dir / "random_forest.joblib")

        assert statuses[("binary", "random_forest")] == "trained"
        assert statuses[("multiclass", "random_forest")] == "trained"
        assert binary_rf_artifact["trained_trees"] == 4
        assert multiclass_rf_artifact["trained_trees"] == 4
    finally:
        shutil.rmtree(project_root, ignore_errors=True)
