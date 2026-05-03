"""CLI entry point for the SentinelNet Phase 1 ingestion workflow."""

from __future__ import annotations

import argparse
import json

from .config import IngestionConfig
from .ingestion import run_ingestion_pipeline
from .logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ingestion runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 1 multi-file ingestion.")
    parser.add_argument("--config", type=str, default="config/ingestion_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional chunk size override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = IngestionConfig.from_json(config_path=args.config, project_root=args.project_root)

    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase1")
    logger.info("Starting SentinelNet Phase 1 ingestion")
    report = run_ingestion_pipeline(config, logger)
    print(
        json.dumps(
            {
                "files_processed": report.files_processed,
                "total_rows": report.total_rows,
                "total_columns": report.total_columns,
                "output_path": report.output_path,
                "report_path": report.report_path,
                "schema_mismatch_detected": report.schema_mismatch_detected,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
