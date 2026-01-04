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

    # Get real client IP (handles proxy/X-Forwarded-For headers)
    client_ip = get_client_ip(request)
    return f"ip:{client_ip}"


def get_client_ip(request: Request) -> str:
    """
    Get the real client IP address, considering proxy headers.

    This function checks multiple headers in order of reliability:
    1. X-Forwarded-For (standard proxy header)
    2. X-Real-IP (Nginx/Apache header)
    3. CF-Connecting-IP (Cloudflare)
    4. Direct connection (request.client.host)

    Args:
        request: FastAPI request object

    Returns:
        Client IP address as string
    """
    # X-Forwarded-For header (can contain multiple IPs)
    # Format: X-Forwarded-For: client, proxy1, proxy2
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        # Take the first IP (original client)
        client_ip = x_forwarded_for.split(",")[0].strip()
        return client_ip

    # X-Real-IP header (common in Nginx/Apache)
    x_real_ip = request.headers.get("X-Real-IP")
    if x_real_ip:
        return x_real_ip

    # Cloudflare connecting IP
    cf_connecting_ip = request.headers.get("CF-Connecting-IP")
    if cf_connecting_ip:
        return cf_connecting_ip

    # Fallback to direct connection IP
    return request.client.host if request.client else "unknown"


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
