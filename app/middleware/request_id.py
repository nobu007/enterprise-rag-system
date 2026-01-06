"""
Request ID Middleware for distributed tracing and debugging.

This module provides middleware to add unique request IDs to each HTTP request
for distributed tracing, debugging, and log correlation.
"""

import uuid
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


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
        # Generate or use existing request ID from header
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request state for access in endpoints
        request.state.request_id = request_id

        # Create logging filter for this request
        context_filter = RequestContextFilter(request_id)

        # Add filter to root logger for this request
        root_logger = logging.getLogger()
        root_logger.addFilter(context_filter)

        try:
            # Process request
            response = await call_next(request)

            # Add request ID to response header
            response.headers["X-Request-ID"] = request_id

            return response
        finally:
            # Clean up filter after request to prevent memory leaks
            root_logger.removeFilter(context_filter)


def get_request_id(request: Request) -> str | None:
    """
    Helper function to get the request ID from a request object.

    Args:
        request: The FastAPI request object

    Returns:
        The request ID if available, None otherwise
    """
    return getattr(request.state, "request_id", None)
