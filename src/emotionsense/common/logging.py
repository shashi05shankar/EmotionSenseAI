"""Structured JSON logging with correlation-id context (NFR-6).

Never use bare ``print`` in ``src/``. Bind a correlation id at the request edge with
``bind_correlation_id`` so every downstream log line is traceable end-to-end.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

import structlog

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

_configured = False


def _add_correlation_id(_: object, __: str, event_dict: dict) -> dict:
    cid = _correlation_id.get()
    if cid is not None:
        event_dict.setdefault("correlation_id", cid)
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to emit JSON to stdout. Idempotent."""
    global _configured
    if _configured:
        return
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_correlation_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def bind_correlation_id(correlation_id: str) -> None:
    """Bind a correlation id to the current context (per request)."""
    _correlation_id.set(correlation_id)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
