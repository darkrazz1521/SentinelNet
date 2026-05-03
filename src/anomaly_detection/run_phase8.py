"""CLI entry point for the SentinelNet Phase 8 anomaly-detection workflow."""

from __future__ import annotations

import argparse
import json

from .config import AnomalyDetectionConfig
from .training import build_phase8_logger, run_anomaly_detection_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 8 runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 8 anomaly detection.")
    parser.add_argument("--config", type=str, default="config/anomaly_detection_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = AnomalyDetectionConfig.from_json(config_path=args.config, project_root=args.project_root)

    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = build_phase8_logger(config)
    logger.info("Starting SentinelNet Phase 8 anomaly detection")
    report = run_anomaly_detection_pipeline(config, logger)
    print(
        json.dumps(
            {
                "feature_count": report.feature_count,
                "train_rows": report.train_rows,
                "test_rows": report.test_rows,
                "trained_models": sum(1 for model in report.models if model["status"] == "trained"),
                "validation_passed": report.validation_passed,
                "report_path": report.report_path,
                "output_dir": report.output_dir,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
