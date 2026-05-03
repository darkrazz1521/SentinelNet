"""Tests for the SentinelNet Phase 5 feature-engineering pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.feature_engineering.config import FeatureEngineeringConfig
from src.feature_engineering.pipeline import run_feature_engineering_pipeline


def _build_phase5_fixture(rows_per_class: int = 24) -> pd.DataFrame:
    labels = [
        ("BENIGN", 0, 0),
        ("DDoS", 1, 2),
        ("PortScan", 1, 10),
        ("Bot", 1, 1),
    ]
    rows: list[dict[str, object]] = []
    row_id = 0

    for source_index, (label, binary_target, multiclass_target) in enumerate(labels):
        source_file = f"source_{source_index}.csv"
        for sample_index in range(rows_per_class):
            syn_count = 1 + (sample_index % 3) + (4 if label in {"PortScan", "DDoS"} else 0)
            rst_count = 1 if label == "Bot" and sample_index % 5 == 0 else 0
            fin_count = 1 if label == "Bot" and sample_index % 7 == 0 else 0
            flow_duration = 100_000 if label == "PortScan" else 5_000_000 + sample_index * 10_000
            destination_port = 20 + sample_index if label == "PortScan" else 80 + (sample_index % 3)
            flow_packets = 50 + sample_index + (25 if label == "DDoS" else 0)
            flow_bytes = flow_packets * (10 + multiclass_target)
            total_fwd_packets = 5 + (sample_index % 4) + (10 if label in {"DDoS", "PortScan"} else 0)
            total_backward_packets = 1 + (sample_index % 3)
            total_fwd_bytes = total_fwd_packets * (20 + multiclass_target)
            total_bwd_bytes = total_backward_packets * 10

            rows.append(
                {
                    "destination_port": destination_port,
                    "flow_duration": float(flow_duration),
                    "total_fwd_packets": float(total_fwd_packets),
                    "total_backward_packets": float(total_backward_packets),
                    "total_length_of_fwd_packets": float(total_fwd_bytes),
                    "total_length_of_bwd_packets": float(total_bwd_bytes),
                    "max_packet_length": float(60 + sample_index),
                    "min_packet_length": float(20 + (sample_index % 5)),
                    "packet_length_mean": float(30 + sample_index / 4.0),
                    "packet_length_std": float(5 + sample_index / 10.0),
                    "flow_iat_mean": float(1000 + sample_index * 5),
                    "flow_iat_std": float(100 + sample_index * 2),
                    "fwd_header_length": float(40 + sample_index),
                    "bwd_header_length": float(20 + sample_index / 2.0),
                    "act_data_pkt_fwd": float(max(1, total_fwd_packets - 1)),
                    "flow_bytes_per_s": float(flow_bytes),
                    "flow_packets_per_s": float(flow_packets),
                    "syn_flag_count": float(syn_count),
                    "rst_flag_count": float(rst_count),
                    "fin_flag_count": float(fin_count),
                    "ack_flag_count": float(2 + sample_index % 4),
                    "psh_flag_count": float(1 + sample_index % 2),
                    "idle_mean": float(2000 + sample_index * 3),
                    "active_mean": float(500 + sample_index * 2),
                    "avg_fwd_segment_size": float(25 + sample_index / 3.0),
                    "avg_bwd_segment_size": float(10 + sample_index / 5.0),
                    "label": label,
                    "source_file": source_file,
                    "label_binary": binary_target,
                    "label_multiclass": multiclass_target,
                    "extra_numeric_signal": float((row_id % 11) * 1.5),
                }
            )
            row_id += 1

    return pd.DataFrame(rows)


def test_run_feature_engineering_pipeline_creates_engineered_and_selected_outputs() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "feature_engineering_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    fixture = _build_phase5_fixture()
    input_path = project_root / "data" / "processed" / "labeled_dataset.csv"
    fixture.to_csv(input_path, index=False)

    config = FeatureEngineeringConfig(
        project_root=project_root,
        input_data_path=input_path,
        output_dir=project_root / "data" / "processed" / "feature_engineered",
        logs_dir=project_root / "logs",
        chunk_size=17,
        rolling_windows=(3, 5),
        selection_sample_size=64,
        rfe_sample_size=40,
        mutual_information_top_k=12,
        rfe_n_features_to_select=6,
    )

    try:
        report = run_feature_engineering_pipeline(config)
        engineered = pd.read_csv(config.engineered_output_path)
        selected = pd.read_csv(config.selected_output_path)
        manifest = json.loads(config.feature_manifest_path.read_text(encoding="utf-8"))

        assert report.validation_passed is True
        assert report.rows_written == len(fixture)
        assert "total_packets" in engineered.columns
        assert "syn_ratio" in engineered.columns
        assert "rolling_unique_destination_ports_w5" in engineered.columns
        assert "port_scan_pattern_score_w5" in engineered.columns
        assert engineered.loc[fixture["label"] == "PortScan", "rolling_unique_destination_ports_w5"].max() > 1
        assert set(report.selected_features).issubset(set(selected.columns))
        assert len(report.selected_features) == 6
        assert manifest["selected_features"] == report.selected_features
        assert config.pca_path.exists()
    finally:
        shutil.rmtree(project_root, ignore_errors=True)

