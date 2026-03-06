"""Centralised logging configuration.

Call ``configure_logging()`` once at application startup (the FastAPI lifespan
does this automatically).  Everywhere else in the codebase just call
``get_logger(__name__)`` to obtain a named logger.

Two formatters are supported:
- ``text`` — human-readable, coloured-friendly, ideal for development.
- ``json`` — newline-delimited JSON, ideal for log-aggregation pipelines.

The module is intentionally free of third-party dependencies so it can be
imported before any packages are installed (e.g. during settings validation).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Guard so configure_logging() is idempotent even if called multiple times.
_LOGGING_CONFIGURED: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Configure the root logger exactly once.

    Subsequent calls are silently ignored, making the function safe to call at
    module-import time in multiple places.

    Args:
        level: One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
        fmt:   ``"text"`` for human-readable output or ``"json"`` for structured
               newline-delimited JSON.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter: logging.Formatter
    if fmt == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handler.setFormatter(formatter)
    # Replace any handlers that may have been added before our configuration.
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers at WARNING by default.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    If ``configure_logging`` has not been called yet this will trigger a
    default text-format INFO configuration so the logger is always usable.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A standard :class:`logging.Logger` instance.
    """
    configure_logging()
    return logging.getLogger(name)


def reset_logging() -> None:
    """Reset the logging configuration (intended for tests only).

    Allows ``configure_logging()`` to be called again with different settings.
    """
    global _LOGGING_CONFIGURED
    _LOGGING_CONFIGURED = False
    root = logging.getLogger()
    root.handlers.clear()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter with zero external dependencies.

    Each log record is emitted as a single line of JSON containing at least
    ``timestamp``, ``level``, ``logger``, and ``message`` keys.  Additional
    keys are added when the record carries exception information.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False)
