"""CLI entry point for the SentinelNet Phase 3 label engineering workflow."""

from __future__ import annotations

import argparse
import json

from .config import LabelHandlingConfig
from .label_handling import run_label_handling_pipeline
from .logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 3 runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 3 label handling.")
    parser.add_argument("--config", type=str, default="config/label_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional chunk size override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = LabelHandlingConfig.from_json(config_path=args.config, project_root=args.project_root)

    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase3")
    logger.info("Starting SentinelNet Phase 3 label engineering")
    report = run_label_handling_pipeline(config, logger)
    print(
        json.dumps(
            {
                "rows_read": report.rows_read,
                "rows_written": report.rows_written,
                "missing_labels": report.missing_labels,
                "unknown_labels": report.unknown_labels,
                "validation_passed": report.validation_passed,
                "output_path": report.output_path,
                "mapping_path": report.mapping_path,
                "report_path": report.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

