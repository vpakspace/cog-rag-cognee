"""Structured logging configuration — JSON or human-readable."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )


def setup_logging(*, json_logs: bool = False, log_level: str = "INFO") -> None:
    """Configure the root logger with JSON or text formatting.

    Args:
        json_logs: If True, emit structured JSON lines (for production).
        log_level: Logging threshold (DEBUG, INFO, WARNING, ERROR).
    """
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    # Remove existing handlers to allow re-configuration
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if json_logs:
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
        )

    root.addHandler(handler)
