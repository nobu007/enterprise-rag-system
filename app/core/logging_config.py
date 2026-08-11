"""
Logging Configuration Module

This module provides structured logging configuration for the enterprise RAG system.
It replaces print() statements with proper logging that includes:
- Timestamps
- Log levels
- Module names
- Request IDs (for API requests)

Environment Variables:
    LOG_LEVEL: Control logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                Default: INFO
"""

import logging
import sys
from typing import Optional
import os
import re
import contextvars


# Per-request id for log correlation. This is the async-safe carrier of the
# request id from RequestIDMiddleware to log records: a ContextVar is copied
# per asyncio task, so concurrent requests each see their own id, and a single
# handler-level RequestIDFilter can read it for records emitted on ANY logger.
#
# Why a ContextVar and not (as before) a per-request filter added to the root
# LOGGER: CPython applies a logger's own filters only to records emitted
# directly on that logger. Records emitted on child loggers (every app.*
# module via get_logger(__name__)) propagate to the root HANDLER via
# callHandlers, which runs handler filters but NEVER the parent logger's
# filters. So the middleware's root-logger filter was invisible to app logs
# and every app log line showed request_id=N/A -- request tracing was
# silently non-functional. The handler-level RequestIDFilter below, reading
# this ContextVar, is the fix (handler filters DO run for propagated records).
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="N/A"
)


def set_request_id(request_id: str):
    """Bind ``request_id`` to the current async context (returns a reset token).

    Called by RequestIDMiddleware for the duration of a request.
    """
    return _request_id_var.set(request_id)


def reset_request_id(token) -> None:
    """Undo a prior ``set_request_id`` using the token it returned."""
    _request_id_var.reset(token)


class RequestIDFilter(logging.Filter):
    """Filter to add request ID to log records.

    Attached to the logging handler in setup_logging. Reads the per-request id
    from ``_request_id_var`` so it applies to records on every app.* child
    logger. A record that already carries a ``request_id`` (set explicitly
    before reaching the handler) is preserved unchanged.
    """

    def filter(self, record):
        # If the record already has a request_id, keep it; otherwise fall back
        # to the per-request contextvar (default 'N/A' outside any request).
        record.request_id = getattr(record, "request_id", _request_id_var.get())
        return True


def setup_logging(
    log_level: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
) -> None:
    """
    Setup application logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                   If None, reads from LOG_LEVEL environment variable (default: INFO)
        log_format: Format string for log messages

    Returns:
        None
    """
    # Determine log level
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate log level
    numeric_level = getattr(logging, log_level, logging.INFO)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create formatter
    formatter = logging.Formatter(
        fmt=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create handler (stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()  # Remove any existing handlers

    # Add our handler
    handler.addFilter(RequestIDFilter())
    root_logger.addHandler(handler)

    # Set up specific loggers for third-party libraries
    # Reduce noise from verbose libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# C0 control characters + DEL. Client-controlled values (query text, request
# IDs, ...) that reach log records can carry CR/LF and forge extra log lines
# (CWE-117 log injection). The validation middleware scans request bodies for
# XSS/SQL/path/command payloads but NOT for control characters, so a query like
# "q\n2000-01-01 INFO admin login ok" otherwise flows through to the
# rag_pipeline / streaming log calls unstripped. sanitize_for_log() neutralises
# only the log representation; the underlying value is left untouched.
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_CONTROL_CHAR_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def sanitize_for_log(value: object) -> str:
    """Escape control characters so client-controlled text is safe in a log line.

    Returns ``str(value)`` with C0 control characters and DEL rendered as
    visible escape sequences (newline -> ``\\n``, carriage return -> ``\\r``,
    tab -> ``\\t``, others -> ``\\xNN``). CR/LF — the line-forging vectors — are
    neutralised, so the result can be embedded in a log message without
    injecting a fake subsequent line. Only the returned string is affected; the
    passed value is not mutated.

    Use this whenever an attacker-controlled string (query body, header, cache
    key, ...) is interpolated into a log message.
    """
    text = str(value)
    # Fast path: the overwhelming majority of values contain no control chars.
    if not _CONTROL_CHAR_PATTERN.search(text):
        return text
    return _CONTROL_CHAR_PATTERN.sub(
        lambda match: _CONTROL_CHAR_ESCAPES.get(
            match.group(0), f"\\x{ord(match.group(0)):02x}"
        ),
        text,
    )
