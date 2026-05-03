"""CLI entry point for the SentinelNet Phase 5 feature-engineering workflow."""

from __future__ import annotations

import argparse
import json

from .config import FeatureEngineeringConfig
from .pipeline import build_phase5_logger, run_feature_engineering_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 5 runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 5 feature engineering.")
    parser.add_argument("--config", type=str, default="config/feature_engineering_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional chunk size override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = FeatureEngineeringConfig.from_json(config_path=args.config, project_root=args.project_root)

    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = build_phase5_logger(config)
    logger.info("Starting SentinelNet Phase 5 feature engineering")
    report = run_feature_engineering_pipeline(config, logger)
    print(
        json.dumps(
            {
                "rows_written": report.rows_written,
                "engineered_feature_count": report.engineered_feature_count,
                "selected_feature_count": report.rfe_selected_count,
                "train_rows": report.train_rows,
                "test_rows": report.test_rows,
                "validation_passed": report.validation_passed,
                "engineered_output_path": report.engineered_output_path,
                "selected_output_path": report.selected_output_path,
                "report_path": report.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

