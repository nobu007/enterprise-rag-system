"""
Rate limiting module for API endpoints

This module implements rate limiting using slowapi to prevent API abuse
and ensure fair resource allocation.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from typing import Callable


def get_user_id(request: Request) -> str:
    """
    Get unique identifier for rate limiting.

    Uses API key if provided (for authenticated users),
    otherwise falls back to IP address (for anonymous users).

    Args:
        request: FastAPI request object

    Returns:
        Unique identifier string (key:xxx or ip:xxx)
    """
    # Check for API key in headers
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"

    # Fallback to IP address
    return f"ip:{get_remote_address(request)}"


def get_identifier(request: Request) -> str:
    """
    Alias for get_user_id for compatibility with slowapi.

    Args:
        request: FastAPI request object

    Returns:
        Unique identifier string
    """
    return get_user_id(request)


# Create limiter instance
limiter = Limiter(key_func=get_identifier)
