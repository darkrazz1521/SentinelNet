"""CLI entry point for the SentinelNet Phase 11 streaming workflow."""

from __future__ import annotations

import argparse
import json

from .config import StreamingConfig
from .streaming import build_phase11_logger, run_streaming_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 11 runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 11 streaming simulation.")
    parser.add_argument("--config", type=str, default="config/streaming_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = StreamingConfig.from_json(config_path=args.config, project_root=args.project_root)

    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = build_phase11_logger(config)
    report = run_streaming_pipeline(config, logger)
    print(
        json.dumps(
            {
                "stream_split": report.stream_split,
                "rows_streamed": report.rows_streamed,
                "selected_binary_variant": report.selected_binary_variant,
                "selected_multiclass_variant": report.selected_multiclass_variant,
                "average_batch_latency_ms": report.average_batch_latency_ms,
                "throughput_rows_per_second": report.throughput_rows_per_second,
                "validation_passed": report.validation_passed,
                "report_path": report.report_path,
                "predictions_path": report.predictions_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
