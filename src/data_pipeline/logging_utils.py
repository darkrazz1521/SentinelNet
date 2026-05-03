"""Logging helpers for SentinelNet pipeline stages."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    log_path: Path,
    level: str = "INFO",
    logger_name: str = "sentinelnet.pipeline",
) -> logging.Logger:
    """Configure a dedicated pipeline logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
