"""CLI entry point for the SentinelNet Phase 2 cleaning workflow."""

from __future__ import annotations

import argparse
import json

from .cleaning import run_cleaning_pipeline
from .config import CleaningConfig
from .logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the cleaning runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 2 data cleaning.")
    parser.add_argument("--config", type=str, default="config/cleaning_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional chunk size override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = CleaningConfig.from_json(config_path=args.config, project_root=args.project_root)

    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase2")
    logger.info("Starting SentinelNet Phase 2 cleaning")
    report = run_cleaning_pipeline(config, logger)
    print(
        json.dumps(
            {
                "rows_read": report.rows_read,
                "rows_written": report.rows_written,
                "duplicate_rows_removed": report.duplicate_rows_removed,
                "rows_dropped_missing_critical": report.rows_dropped_missing_critical,
                "infinite_values_replaced": report.infinite_values_replaced,
                "missing_values_imputed": report.missing_values_imputed,
                "validation_passed": report.validation_passed,
                "output_path": report.output_path,
                "report_path": report.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
