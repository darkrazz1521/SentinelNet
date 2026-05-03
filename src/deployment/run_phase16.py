"""CLI entry point for the SentinelNet Phase 16 advanced response workflow."""

from __future__ import annotations

import argparse
import json

from .phase16 import build_phase16_logger, run_phase16_pipeline
from .phase16_config import Phase16Config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 16 runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 16 advanced response features.")
    parser.add_argument("--config", type=str, default="config/phase16_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = Phase16Config.from_json(config_path=args.config, project_root=args.project_root)
    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = build_phase16_logger(config)
    report = run_phase16_pipeline(config, logger)
    print(
        json.dumps(
            {
                "feature_count": report.feature_count,
                "drift_reference_rows": report.drift_reference_rows,
                "total_rows_processed": report.total_rows_processed,
                "zero_day_candidate_rows": report.zero_day_candidate_rows,
                "auto_block_rows": report.auto_block_rows,
                "continuous_learning_queue_rows": report.continuous_learning_queue_rows,
                "drifted_feature_count": report.drifted_feature_count,
                "retraining_recommended": report.retraining_recommended,
                "validation_passed": report.validation_passed,
                "report_path": report.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
