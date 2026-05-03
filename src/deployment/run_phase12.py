"""CLI entry point for the SentinelNet Phase 12 alerting workflow."""

from __future__ import annotations

import argparse
import json

from .alerting import build_phase12_logger, run_alerting_pipeline
from .alerting_config import AlertingConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 12 runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 12 alerting.")
    parser.add_argument("--config", type=str, default="config/alerting_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = AlertingConfig.from_json(config_path=args.config, project_root=args.project_root)

    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = build_phase12_logger(config)
    report = run_alerting_pipeline(config, logger)
    print(
        json.dumps(
            {
                "total_rows_processed": report.total_rows_processed,
                "alert_rows_written": report.alert_rows_written,
                "level_counts": report.level_counts,
                "average_risk_score": report.average_risk_score,
                "max_risk_score": report.max_risk_score,
                "validation_passed": report.validation_passed,
                "report_path": report.report_path,
                "alerts_path": report.alerts_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
