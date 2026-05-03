"""CLI entry point for the SentinelNet Phase 10 explainability workflow."""

from __future__ import annotations

import argparse
import json

from .config import ExplainabilityConfig
from .pipeline import build_phase10_logger, run_explainability_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 10 runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 10 explainability.")
    parser.add_argument("--config", type=str, default="config/explainability_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = ExplainabilityConfig.from_json(config_path=args.config, project_root=args.project_root)

    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = build_phase10_logger(config)
    logger.info("Starting SentinelNet Phase 10 explainability")
    report = run_explainability_pipeline(config, logger)
    print(
        json.dumps(
            {
                "feature_count": report.feature_count,
                "train_rows": report.train_rows,
                "test_rows": report.test_rows,
                "generated_artifacts": len(report.artifacts),
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
