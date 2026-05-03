"""CLI entry point for the SentinelNet Phase 15 optimization workflow."""

from __future__ import annotations

import argparse
import json

from .performance import build_phase15_logger, run_performance_optimization_pipeline
from .performance_config import PerformanceConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 15 runner."""
    parser = argparse.ArgumentParser(description="Run SentinelNet v2 Phase 15 performance optimization.")
    parser.add_argument("--config", type=str, default="config/performance_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional log level override.")
    return parser.parse_args()


def main() -> int:
    """Execute the CLI workflow and print a compact JSON summary."""
    args = parse_args()
    config = PerformanceConfig.from_json(config_path=args.config, project_root=args.project_root)
    if args.log_level is not None:
        config.log_level = args.log_level.upper()

    logger = build_phase15_logger(config)
    report = run_performance_optimization_pipeline(config, logger)
    print(
        json.dumps(
            {
                "benchmark_rows": report.benchmark_rows,
                "predictor_load_seconds": report.predictor_load_seconds,
                "recommended_inference_batch_size": report.recommended_inference_batch_size,
                "recommended_api_predict_batch_size": report.recommended_api_predict_batch_size,
                "recommended_stream_page_limit": report.recommended_stream_page_limit,
                "recommended_alert_page_limit": report.recommended_alert_page_limit,
                "metrics_refresh_latency_ms": report.metrics_refresh_latency_ms,
                "metrics_cached_latency_ms": report.metrics_cached_latency_ms,
                "metrics_cache_speedup": report.metrics_cache_speedup,
                "validation_passed": report.validation_passed,
                "report_path": report.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
