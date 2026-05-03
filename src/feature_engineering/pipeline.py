"""Phase 5 feature-engineering pipeline implementation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline.logging_utils import configure_logging

from .config import FeatureEngineeringConfig
from .feature_builder import (
    RollingWindowState,
    engineer_feature_chunk,
    validate_required_columns,
)
from .selection import (
    SelectionArtifacts,
    build_train_test_split,
    collect_rows_by_indices,
    compute_mutual_information,
    correlation_filter,
    fit_optional_pca,
    run_rfe,
    stratified_sample_indices,
)

LOGGER = logging.getLogger("sentinelnet.phase5")


@dataclass(slots=True)
class FeatureEngineeringReport:
    """Serializable report returned after Phase 5 completes."""

    created_at_utc: str
    input_path: str
    engineered_output_path: str
    selected_output_path: str
    report_path: str
    rows_read: int
    rows_written: int
    train_rows: int
    test_rows: int
    engineered_feature_count: int
    candidate_feature_count: int
    correlation_selected_count: int
    rfe_selected_count: int
    statistical_features: list[str]
    time_based_features: list[str]
    domain_features: list[str]
    selected_features: list[str]
    pca_component_count: int
    artifact_paths: dict[str, str]
    assumptions: list[str]
    validation_passed: bool
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "input_path": self.input_path,
            "engineered_output_path": self.engineered_output_path,
            "selected_output_path": self.selected_output_path,
            "report_path": self.report_path,
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "engineered_feature_count": self.engineered_feature_count,
            "candidate_feature_count": self.candidate_feature_count,
            "correlation_selected_count": self.correlation_selected_count,
            "rfe_selected_count": self.rfe_selected_count,
            "statistical_features": self.statistical_features,
            "time_based_features": self.time_based_features,
            "domain_features": self.domain_features,
            "selected_features": self.selected_features,
            "pca_component_count": self.pca_component_count,
            "artifact_paths": self.artifact_paths,
            "assumptions": self.assumptions,
            "validation_passed": self.validation_passed,
            "config": self.config,
        }


def _read_numeric_targets(config: FeatureEngineeringConfig, logger: logging.Logger) -> tuple[np.ndarray, int]:
    """Read the multiclass targets from the engineered dataset."""
    targets: list[np.ndarray] = []
    total_rows = 0

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            config.engineered_output_path,
            usecols=[config.multiclass_target_column],
            chunksize=config.chunk_size,
            low_memory=False,
        ),
        start=1,
    ):
        targets.append(chunk[config.multiclass_target_column].to_numpy(dtype=np.int16, copy=False))
        total_rows += len(chunk)
        logger.info("Indexed engineered target chunk %d | rows=%d | cumulative_rows=%d", chunk_index, len(chunk), total_rows)

    return np.concatenate(targets), total_rows


def _build_selection_artifacts(config: FeatureEngineeringConfig, logger: logging.Logger) -> SelectionArtifacts:
    """Fit correlation, MI, RFE, and optional PCA on a train-only sample."""
    targets, total_rows = _read_numeric_targets(config, logger)
    train_indices, test_indices = build_train_test_split(targets, config.test_size, config.random_state)
    np.save(config.train_indices_path, train_indices.astype(np.int64, copy=False))
    np.save(config.test_indices_path, test_indices.astype(np.int64, copy=False))

    selection_sample_indices = stratified_sample_indices(
        train_indices,
        targets,
        config.selection_sample_size,
        config.random_state,
    )
    selection_sample_frame = collect_rows_by_indices(
        config.engineered_output_path,
        selection_sample_indices,
        config.chunk_size,
    )
    if selection_sample_frame.empty:
        raise ValueError("Phase 5 could not collect a non-empty selection sample.")

    metadata_columns = {
        config.label_column,
        config.source_file_column,
        config.binary_target_column,
        config.multiclass_target_column,
    }
    candidate_features = [
        column
        for column in selection_sample_frame.columns
        if column not in metadata_columns
    ]

    correlation_selected_features, dropped_features = correlation_filter(
        selection_sample_frame,
        candidate_features,
        config.correlation_threshold,
    )
    mi_scores = compute_mutual_information(
        selection_sample_frame,
        correlation_selected_features,
        selection_sample_frame[config.multiclass_target_column].to_numpy(dtype=np.int16, copy=False),
        config.random_state,
    )
    mi_scores.to_csv(config.mi_scores_path, index=False)
    mi_top_features = mi_scores["feature"].head(min(config.mutual_information_top_k, len(mi_scores))).tolist()

    rfe_sample_indices = stratified_sample_indices(
        selection_sample_indices,
        targets,
        min(config.rfe_sample_size, len(selection_sample_indices)),
        config.random_state + 1,
    )
    rfe_sample_frame = collect_rows_by_indices(
        config.engineered_output_path,
        rfe_sample_indices,
        config.chunk_size,
    )
    rfe_selected_features, rfe_ranking = run_rfe(
        rfe_sample_frame,
        mi_top_features,
        rfe_sample_frame[config.multiclass_target_column].to_numpy(dtype=np.int16, copy=False),
        config.rfe_n_features_to_select,
        config.random_state,
    )
    pca_component_count, pca_explained_variance_ratio = fit_optional_pca(rfe_sample_frame, rfe_selected_features, config)

    selection_report = {
        "total_rows": total_rows,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "selection_sample_rows": int(len(selection_sample_indices)),
        "rfe_sample_rows": int(len(rfe_sample_indices)),
        "candidate_features": candidate_features,
        "correlation_selected_features": correlation_selected_features,
        "correlation_dropped_features": dropped_features,
        "mutual_information_top_features": mi_top_features,
        "rfe_selected_features": rfe_selected_features,
        "rfe_ranking": rfe_ranking,
        "pca_component_count": pca_component_count,
        "pca_explained_variance_ratio": pca_explained_variance_ratio,
    }
    config.selection_report_path.write_text(json.dumps(selection_report, indent=2), encoding="utf-8")

    return SelectionArtifacts(
        train_indices=train_indices,
        test_indices=test_indices,
        selection_sample_indices=selection_sample_indices,
        rfe_sample_indices=rfe_sample_indices,
        correlation_selected_features=correlation_selected_features,
        correlation_dropped_features=dropped_features,
        mutual_information_scores=mi_scores,
        mutual_information_top_features=mi_top_features,
        rfe_selected_features=rfe_selected_features,
        rfe_ranking=rfe_ranking,
        pca_component_count=pca_component_count,
        pca_explained_variance_ratio=pca_explained_variance_ratio,
    )


def _materialize_selected_dataset(
    config: FeatureEngineeringConfig,
    selected_features: list[str],
    logger: logging.Logger,
) -> int:
    """Write a compact dataset containing only the selected features and metadata."""
    if config.selected_output_path.exists():
        config.selected_output_path.unlink()

    columns_to_keep = [
        *selected_features,
        config.label_column,
        config.source_file_column,
        config.binary_target_column,
        config.multiclass_target_column,
    ]

    rows_written = 0
    header_written = False
    for chunk_index, chunk in enumerate(
        pd.read_csv(config.engineered_output_path, usecols=columns_to_keep, chunksize=config.chunk_size, low_memory=False),
        start=1,
    ):
        chunk.to_csv(
            config.selected_output_path,
            mode="a" if header_written else "w",
            header=not header_written,
            index=False,
        )
        header_written = True
        rows_written += len(chunk)
        logger.info("Wrote selected dataset chunk %d | rows=%d | cumulative_rows=%d", chunk_index, len(chunk), rows_written)

    return rows_written


def run_feature_engineering_pipeline(
    config: FeatureEngineeringConfig,
    logger: logging.Logger | None = None,
) -> FeatureEngineeringReport:
    """Execute the full Phase 5 feature-engineering workflow."""
    active_logger = logger or LOGGER
    config.ensure_directories()

    if not config.input_data_path.exists():
        raise FileNotFoundError(f"Phase 5 input dataset not found at {config.input_data_path}")

    if config.engineered_output_path.exists():
        config.engineered_output_path.unlink()

    state = RollingWindowState()
    rows_read = 0
    rows_written = 0
    header_written = False
    engineered_columns: list[str] | None = None

    for chunk_index, chunk in enumerate(
        pd.read_csv(config.input_data_path, chunksize=config.chunk_size, low_memory=False),
        start=1,
    ):
        validate_required_columns(list(chunk.columns))
        engineered_chunk = engineer_feature_chunk(chunk, config, state)
        engineered_columns = list(engineered_chunk.columns)
        engineered_chunk.to_csv(
            config.engineered_output_path,
            mode="a" if header_written else "w",
            header=not header_written,
            index=False,
        )
        header_written = True
        rows_read += len(chunk)
        rows_written += len(engineered_chunk)
        active_logger.info(
            "Engineered chunk %d | rows=%d | cumulative_rows=%d",
            chunk_index,
            len(engineered_chunk),
            rows_written,
        )

    if engineered_columns is None:
        raise ValueError("Phase 5 received no rows to engineer.")

    selection_artifacts = _build_selection_artifacts(config, active_logger)
    selected_rows_written = _materialize_selected_dataset(config, selection_artifacts.rfe_selected_features, active_logger)

    statistical_features = [
        "total_packets",
        "total_bytes",
        "bytes_per_packet",
        "fwd_bwd_packet_ratio",
        "fwd_bwd_byte_ratio",
        "packet_length_range",
        "flow_iat_cv",
        "packet_length_cv",
        "header_payload_ratio",
        "forward_payload_efficiency",
        "idle_active_ratio",
        "burstiness_score",
        "segment_size_asymmetry",
        "ack_push_ratio",
        "payload_density",
    ]
    time_based_features = [
        f"rolling_flow_bytes_mean_w{window}" for window in config.rolling_windows
    ] + [
        f"rolling_flow_packets_mean_w{window}" for window in config.rolling_windows
    ] + [
        f"rolling_short_flow_fraction_w{window}" for window in config.rolling_windows
    ] + [
        f"rolling_syn_sum_w{window}" for window in config.rolling_windows
    ] + [
        f"rolling_rst_sum_w{window}" for window in config.rolling_windows
    ]
    max_window = max(config.rolling_windows)
    domain_features = [
        "syn_ratio",
        "connection_failure_score",
        "connection_reset_ratio",
        f"rolling_unique_destination_ports_w{max_window}",
        f"rolling_syn_ratio_w{max_window}",
        f"rolling_connection_failure_ratio_w{max_window}",
        f"port_scan_pattern_score_w{max_window}",
    ]

    feature_manifest = {
        "engineered_columns": engineered_columns,
        "selected_features": selection_artifacts.rfe_selected_features,
        "statistical_features": statistical_features,
        "time_based_features": time_based_features,
        "domain_features": domain_features,
        "correlation_selected_features": selection_artifacts.correlation_selected_features,
        "mutual_information_top_features": selection_artifacts.mutual_information_top_features,
        "rfe_ranking": selection_artifacts.rfe_ranking,
        "pca_component_count": selection_artifacts.pca_component_count,
        "pca_explained_variance_ratio": selection_artifacts.pca_explained_variance_ratio,
    }
    config.feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    artifact_paths = {
        "engineered_dataset": str(config.engineered_output_path),
        "selected_dataset": str(config.selected_output_path),
        "feature_manifest": str(config.feature_manifest_path),
        "selection_report": str(config.selection_report_path),
        "mutual_information_scores": str(config.mi_scores_path),
        "train_indices": str(config.train_indices_path),
        "test_indices": str(config.test_indices_path),
    }
    if config.pca_enabled and config.pca_path.exists():
        artifact_paths["pca"] = str(config.pca_path)

    assumptions = [
        "Rolling time-based features are computed in the preserved row order of each source_file because the processed dataset does not include an explicit packet timestamp column.",
        f"Short-flow behavior is approximated with flow_duration <= {config.short_flow_duration_threshold}.",
        "Feature selection is fit on the train split only to avoid leaking test-set signal.",
    ]

    report = FeatureEngineeringReport(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        input_path=str(config.input_data_path),
        engineered_output_path=str(config.engineered_output_path),
        selected_output_path=str(config.selected_output_path),
        report_path=str(config.report_path),
        rows_read=rows_read,
        rows_written=rows_written,
        train_rows=int(len(selection_artifacts.train_indices)),
        test_rows=int(len(selection_artifacts.test_indices)),
        engineered_feature_count=len(engineered_columns),
        candidate_feature_count=len(engineered_columns) - 4,
        correlation_selected_count=len(selection_artifacts.correlation_selected_features),
        rfe_selected_count=len(selection_artifacts.rfe_selected_features),
        statistical_features=statistical_features,
        time_based_features=time_based_features,
        domain_features=domain_features,
        selected_features=selection_artifacts.rfe_selected_features,
        pca_component_count=selection_artifacts.pca_component_count,
        artifact_paths=artifact_paths,
        assumptions=assumptions,
        validation_passed=(
            rows_read == rows_written
            and rows_written == selected_rows_written
            and bool(selection_artifacts.rfe_selected_features)
            and config.engineered_output_path.exists()
            and config.selected_output_path.exists()
        ),
        config=config.to_dict(),
    )
    config.report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if not report.validation_passed:
        raise ValueError("Phase 5 validation failed while generating engineered artifacts.")

    active_logger.info(
        "Completed Phase 5 feature engineering | rows=%d | engineered_columns=%d | selected_features=%d",
        report.rows_written,
        report.engineered_feature_count,
        report.rfe_selected_count,
    )
    return report


def build_phase5_logger(config: FeatureEngineeringConfig) -> logging.Logger:
    """Create the dedicated Phase 5 logger."""
    return configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase5")

