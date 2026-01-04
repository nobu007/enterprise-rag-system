"""
Request/Response validation middleware for security and robustness.

This middleware provides:
- Request size limits (DoS protection)
- Security validation (XSS, SQL injection, path traversal)
- Header validation
- Security headers injection
- Suspicious request logging
"""

import json
import logging
from typing import Dict, Any

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.security import SecurityValidator


logger = logging.getLogger(__name__)


class ValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for validating requests and adding security headers.

    Features:
    - Request size limits to prevent DoS attacks
    - Security validation (XSS, SQL injection, path traversal)
    - Header validation
    - Security headers injection
    - Logging of suspicious requests
    """

    def __init__(
        self,
        app,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB default
        enable_security_validation: bool = True,
        log_suspicious: bool = True
    ):
        """
        Initialize validation middleware.

        Args:
            app: FastAPI application instance
            max_request_size: Maximum request size in bytes (default: 10MB)
            enable_security_validation: Enable security checks (default: True)
            log_suspicious: Log suspicious requests (default: True)
        """
        super().__init__(app)
        self.max_request_size = max_request_size
        self.enable_security_validation = enable_security_validation
        self.log_suspicious = log_suspicious
        self.validator = SecurityValidator()

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request through validation pipeline.

        Args:
            request: Incoming request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response with security headers

        Raises:
            HTTPException: If validation fails
        """
        # 1. Content-Length check (DoS protection)
        await self._validate_content_length(request)

        # 2. Body validation for POST/PUT/PATCH requests
        if request.method in ["POST", "PUT", "PATCH"]:
            await self._validate_request_body(request)

        # 3. Header validation
        await self._validate_headers(request)

        # 4. Process request through next middleware/handler
        response = await call_next(request)

        # 5. Add security headers to response
        await self._add_security_headers(request, response)

        return response

    async def _validate_content_length(self, request: Request):
        """
        Validate Content-Length header to prevent DoS attacks.

        Args:
            request: Incoming request

        Raises:
            HTTPException: 413 if request too large
        """
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_request_size:
                    msg = f"Request entity too large (max {self.max_request_size} bytes)"
                    logger.warning(f"Request size limit exceeded: {length} bytes")
                    raise HTTPException(413, detail=msg)
            except ValueError:
                # Invalid Content-Length header
                logger.warning("Invalid Content-Length header")
                raise HTTPException(400, detail="Invalid Content-Length header")

    async def _validate_request_body(self, request: Request):
        """
        Validate request body for security threats.

        Args:
            request: Incoming request

        Raises:
            HTTPException: 400 if malicious content detected
        """
        try:
            # Read body
            body = await request.body()

            if not body:
                return

            # Try to parse as JSON
            try:
                data = json.loads(body.decode())
                if isinstance(data, dict):
                    await self._validate_request_data(data, request)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Not JSON, skip validation for non-JSON requests
                pass

        except HTTPException:
            # Re-raise HTTPExceptions
            raise
        except Exception as e:
            logger.error(f"Error validating request body: {e}")
            # Don't block on unexpected errors, log and continue

    async def _validate_request_data(self, data: Dict[str, Any], request: Request):
        """
        Validate request data for security threats.

        Args:
            data: Parsed JSON data
            request: Incoming request

        Raises:
            HTTPException: 400 if malicious content detected
        """
        if not self.enable_security_validation:
            return

        # Recursively validate all string values
        await self._validate_dict_recursive(data, request)

    async def _validate_dict_recursive(self, data: Any, request: Request, path: str = ""):
        """
        Recursively validate dictionary/list structures.

        Args:
            data: Data to validate
            request: Incoming request
            path: Current path in data structure (for logging)

        Raises:
            HTTPException: 400 if malicious content detected
        """
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                await self._validate_dict_recursive(value, request, current_path)

        elif isinstance(data, list):
            for idx, item in enumerate(data):
                current_path = f"{path}[{idx}]" if path else f"[{idx}]"
                await self._validate_dict_recursive(item, request, current_path)

        elif isinstance(data, str):
            await self._validate_string_value(data, request, path)

    async def _validate_string_value(self, value: str, request: Request, path: str):
        """
        Validate individual string value for security threats.

        Args:
            value: String value to validate
            request: Incoming request
            path: Path in data structure

        Raises:
            HTTPException: 400 if malicious content detected
        """
        # XSS detection
        if self.validator.detect_xss(value):
            msg = f"Potentially malicious content (XSS) in field: {path}"
            if self.log_suspicious:
                logger.warning(f"{msg} - Client: {request.client.host}")
            raise HTTPException(400, detail=msg)

        # SQL injection detection
        if self.validator.detect_sql_injection(value):
            msg = f"SQL injection pattern detected in field: {path}"
            if self.log_suspicious:
                logger.warning(f"{msg} - Client: {request.client.host}")
            raise HTTPException(400, detail=msg)

        # Path traversal detection
        if self.validator.detect_path_traversal(value):
            msg = f"Path traversal pattern detected in field: {path}"
            if self.log_suspicious:
                logger.warning(f"{msg} - Client: {request.client.host}")
            raise HTTPException(400, detail=msg)

        # Command injection detection
        if self.validator.detect_command_injection(value):
            msg = f"Command injection pattern detected in field: {path}"
            if self.log_suspicious:
                logger.warning(f"{msg} - Client: {request.client.host}")
            raise HTTPException(400, detail=msg)

    async def _validate_headers(self, request: Request):
        """
        Validate request headers.

        Args:
            request: Incoming request

        Raises:
            HTTPException: 400 if header validation fails
        """
        # User-Agent length check
        user_agent = request.headers.get("user-agent", "")
        if len(user_agent) > 500:
            msg = "User-Agent header too long (max 500 chars)"
            logger.warning(f"User-Agent too long: {len(user_agent)} chars")
            raise HTTPException(400, detail=msg)

        # Check for suspicious headers
        suspicious_headers = [
            "X-Forwarded-Host",
            "X-Original-URL",
            "X-Rewrite-URL"
        ]

        for header in suspicious_headers:
            if header in request.headers:
                logger.info(f"Suspicious header detected: {header} from {request.client.host}")

    async def _add_security_headers(self, request: Request, response: Response):
        """
        Add security headers to response.

        Args:
            request: Original request
            response: Response to modify
        """
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # HSTS (HTTP Strict Transport Security)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )

        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions Policy (formerly Feature-Policy)
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=()"
        )
