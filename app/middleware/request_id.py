"""
Request ID Middleware for distributed tracing and debugging.

This module provides middleware to add unique request IDs to each HTTP request
for distributed tracing, debugging, and log correlation.
"""

import uuid
import logging
import re
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging_config import set_request_id, reset_request_id


logger = logging.getLogger(__name__)

# C0 control characters + DEL. A client-supplied X-Request-ID carrying any of
# these (notably \r / \n) is a log-injection (CWE-117) and response-splitting
# vector: the value is written to log records via [%(request_id)s] AND echoed
# in the response X-Request-ID header. Printable special chars, empty string,
# and long values are intentionally NOT rejected — only control characters,
# which no legitimate tracing ID ever contains.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


class RequestContextFilter(logging.Filter):
    """Custom logging filter to add request_id to log records."""

    def __init__(self, request_id: str):
        """
        Initialize the filter with a request ID.

        Args:
            request_id: The unique request ID to add to log records
        """
        super().__init__()
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add request_id to the log record.

        Args:
            record: The log record to filter

        Returns:
            True to allow the record to be logged
        """
        record.request_id = self.request_id
        return True


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add unique request ID to each request for tracing.

    This middleware:
    1. Generates or retrieves a request ID from the X-Request-ID header
    2. Stores it in request.state for access in endpoints
    3. Adds it to all log records via logging filter
    4. Adds it to response headers

    Usage:
        app.add_middleware(RequestIDMiddleware)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and add request ID tracking.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler in the chain

        Returns:
            The HTTP response with X-Request-ID header
        """
        # Generate or use existing request ID from header. A client-supplied
        # value is trusted ONLY if it is free of control characters; otherwise
        # a fresh UUID is substituted so CRLF cannot reach the logs or the
        # response header (CWE-117 / response splitting).
        raw_request_id = request.headers.get("X-Request-ID")
        if raw_request_id is not None and not _CONTROL_CHAR_RE.search(raw_request_id):
            request_id = raw_request_id
        else:
            request_id = str(uuid.uuid4())

        # Store in request state for access in endpoints
        request.state.request_id = request_id

        # Bind the request id to this async context so the handler-level
        # RequestIDFilter (logging_config) attaches it to every app.* log
        # record emitted while serving the request. A ContextVar is
        # async-task-scoped, so concurrent requests keep distinct ids.
        # NB: adding a per-request filter to the ROOT LOGGER (the previous
        # approach) does NOT work -- CPython never applies a logger's own
        # filters to records emitted on child loggers, so app log lines were
        # always request_id=N/A. RequestContextFilter is retained for
        # standalone/test use but is no longer the production carrier.
        token = set_request_id(request_id)
        try:
            # Process request
            response = await call_next(request)

            # Add request ID to response header
            response.headers["X-Request-ID"] = request_id

            return response
        finally:
            reset_request_id(token)


def get_request_id(request: Request) -> str | None:
    """
    Helper function to get the request ID from a request object.

    Args:
        request: The FastAPI request object

    Returns:
        The request ID if available, None otherwise
    """
    return getattr(request.state, "request_id", None)
