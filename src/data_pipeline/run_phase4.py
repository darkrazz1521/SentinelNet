"""CLI entry point for the SentinelNet Phase 4 preprocessing workflow."""

from __future__ import annotations

import argparse
import json

from .config import PreprocessingConfig
from .logging_utils import configure_logging
from .preprocessing import run_preprocessing_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 4 runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 4 preprocessing.")
    parser.add_argument("--config", type=str, default="config/preprocessing_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional chunk size override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = PreprocessingConfig.from_json(config_path=args.config, project_root=args.project_root)

    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase4")
    logger.info("Starting SentinelNet Phase 4 preprocessing")
    report = run_preprocessing_pipeline(config, logger)
    print(
        json.dumps(
            {
                "rows_read": report.rows_read,
                "train_rows": report.train_rows,
                "test_rows": report.test_rows,
                "transformed_feature_count": report.transformed_feature_count,
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

