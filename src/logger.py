"""
Structured JSON logger for the Globus Bank AI agent.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("intent extracted", extra={"intent": intent, "confidence": confidence})

Each log line is a single JSON object with at minimum:
    timestamp, level, component, message
Plus any extra fields passed via the `extra` dict.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra fields the caller passed
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that emits JSON to stdout.
    Calling this multiple times with the same name returns the same logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)

    # Don't propagate to root — avoids duplicate output if root is also configured
    logger.propagate = False

    return logger
