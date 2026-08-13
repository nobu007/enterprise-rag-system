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
from typing import Any

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.core.security import SecurityValidator
from app.core.logging_config import sanitize_for_log


logger = logging.getLogger(__name__)


def _client_host(request: Request) -> str:
    """Best-effort client host for rejection/suspicion logging.

    ``request.client`` is an OPTIONAL ``(host, port)`` tuple per the ASGI spec
    and is ``None`` whenever the server did not supply peer info (certain
    reverse-proxy / internal-call paths). The middleware's rejection logging
    previously dereferenced ``request.client.host`` unconditionally; when
    ``client`` is ``None`` that raised ``AttributeError`` — and because
    ``_validate_request_body`` swallows any non-``HTTPException`` error
    (fail-open: "log and continue"), a malicious payload *detected* during
    validation crashed the logger BEFORE the ``HTTPException(400)`` was raised,
    so the broad ``except`` aborted validation and the payload reached the
    handler UNVALIDATED (HTTP 200). Mirrors the ``None`` guard already present
    in ``rate_limit.get_client_ip``.
    """
    return request.client.host if request.client else "unknown"


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
            HTTP response with security headers. Validation failures are
            returned as JSONResponse with the intended status code.

        Raises:
            HTTPException: If validation fails
        """
        # NOTE: BaseHTTPMiddleware executes *outside* FastAPI's
        # ExceptionMiddleware. An HTTPException raised here (before call_next)
        # would bubble up to ServerErrorMiddleware and be rendered as a 500
        # instead of the intended client-facing status. Catch and convert it
        # into a proper response so malicious/oversized requests are rejected
        # with their real status code (400/413) under modern Starlette.
        try:
            # 1. Content-Length check (DoS protection)
            await self._validate_content_length(request)

            # 2. Body validation for POST/PUT/PATCH requests
            if request.method in ["POST", "PUT", "PATCH"]:
                await self._validate_request_body(request)

            # 3. Header validation
            await self._validate_headers(request)
        except HTTPException as exc:
            if self.log_suspicious:
                logger.info(
                    f"Request rejected ({exc.status_code}): {exc.detail} "
                    f"- Client: {_client_host(request)}"
                )
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )

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
            # Read the body with an actual-byte size limit. The Content-Length
            # header checked in _validate_content_length is client-controlled
            # and ABSENT for Transfer-Encoding: chunked requests, so reading the
            # full body unconditionally (request.body()) would buffer an
            # unbounded payload into memory and bypass the DoS guard entirely.
            body = await self._read_body_limited(request)

            if not body:
                return

            # Try to parse as JSON
            try:
                data = json.loads(body.decode())
                # Validate any JSON payload that can carry a string: objects,
                # arrays, AND bare string scalars. A top-level JSON array
                # previously skipped _validate_request_data entirely (fixed),
                # but a top-level JSON *string* (e.g. body '"1 UNION SELECT ..."')
                # still failed the isinstance guard and passed UNVALIDATED — the
                # same validator-skip class (INV-VAL-001), the scalar sibling of
                # the array case. _validate_dict_recursive already routes bare
                # strings through its isinstance(data, str) branch. int/float/
                # bool/null are intentionally excluded: a JSON number or boolean
                # cannot carry an injection payload.
                if isinstance(data, (dict, list, str)):
                    await self._validate_request_data(data, request)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Not JSON, skip validation for non-JSON requests
                pass
            except RecursionError:
                # Deeply nested JSON exhausts the recursive validator's stack.
                # Fail CLOSED (INV-VAL-001): a structure too deep to validate
                # safely is rejected at the boundary rather than passed through
                # unvalidated — otherwise an attacker can wrap any payload in
                # enough nesting to abort the scan before the payload is reached.
                logger.warning("Rejected deeply nested JSON (RecursionError during validation)")
                raise HTTPException(400, detail="Request structure too deeply nested")

        except HTTPException:
            # Re-raise HTTPExceptions
            raise
        except Exception as e:
            logger.error(f"Error validating request body: {e}")
            # Don't block on unexpected errors, log and continue

    async def _read_body_limited(self, request: Request) -> bytes:
        """Read the request body, enforcing max_request_size on actual bytes.

        Starlette's ``Request.body()`` buffers the entire body before returning
        with no upper bound. A request that omits ``Content-Length`` (e.g.
        ``Transfer-Encoding: chunked``) slips past ``_validate_content_length``
        — the size guard only inspects that client-controlled header — so a call
        to ``request.body()`` would read an attacker-controlled payload fully
        into memory regardless of the configured limit, defeating the DoS
        protection this middleware exists to provide. Read incrementally via
        the request stream instead and abort with 413 the instant the
        accumulated size exceeds the limit.

        The assembled bytes are cached on ``request._body`` (mirroring what
        Starlette's own ``body()`` does) so downstream handlers and
        ``BaseHTTPMiddleware``'s ``_CachedRequest`` can re-read the body:
        consuming ``request.stream()`` without setting ``_body`` leaves the
        downstream app with an empty body.
        """
        if hasattr(request, "_body"):
            # Already buffered (e.g. by an earlier reader): still enforce the limit.
            body = request._body
            if len(body) > self.max_request_size:
                logger.warning(
                    f"Request size limit exceeded: {len(body)} bytes (no Content-Length)"
                )
                raise HTTPException(
                    413,
                    detail=f"Request entity too large (max {self.max_request_size} bytes)",
                )
            return body

        chunks = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > self.max_request_size:
                logger.warning(
                    f"Request size limit exceeded: {total} bytes (no Content-Length)"
                )
                raise HTTPException(
                    413,
                    detail=f"Request entity too large (max {self.max_request_size} bytes)",
                )
            chunks.append(chunk)
        body = b"".join(chunks)
        request._body = body
        return body

    async def _validate_request_data(self, data: Any, request: Request):
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
                current_path = f"{path}.{key}" if path else str(key)
                # JSON object keys are attacker-controlled strings too
                # (json.loads always yields string keys), and live endpoints
                # accept arbitrary-keyed dicts (e.g. QueryRequest.filters:
                # Dict[str, Any]). Scan keys as well as values — otherwise a
                # malicious key like {"<script>": "x"} or {"1 UNION SELECT ..":
                # "x"} reaches the handler unscanned, the key/value sibling of
                # the value-only validation (INV-VAL-001).
                await self._validate_string_value(str(key), request, f"{current_path}#key")
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
        # ``path`` is assembled from client-controlled JSON object keys
        # (_validate_dict_recursive: ``current_path = f"{path}.{key}"``), and live
        # endpoints accept arbitrary-keyed dicts (QueryRequest.filters:
        # Dict[str, Any]). A key that BOTH trips a detector AND carries CR/LF
        # therefore reaches this log line raw, and the embedded newline forges a
        # fake subsequent log record (CWE-117 log injection) -- the same class as
        # the body/header sites neutralised in rag_pipeline + streaming, which a
        # field-name-keyed sweep missed here. ``sanitize_for_log`` neutralises the
        # log representation only; the HTTPException detail (and thus the 400
        # response body) is left untouched, so request handling is unchanged.
        # XSS detection
        if self.validator.detect_xss(value):
            msg = f"Potentially malicious content (XSS) in field: {path}"
            if self.log_suspicious:
                logger.warning(f"{sanitize_for_log(msg)} - Client: {_client_host(request)}")
            raise HTTPException(400, detail=msg)

        # SQL injection detection
        if self.validator.detect_sql_injection(value):
            msg = f"SQL injection pattern detected in field: {path}"
            if self.log_suspicious:
                logger.warning(f"{sanitize_for_log(msg)} - Client: {_client_host(request)}")
            raise HTTPException(400, detail=msg)

        # Path traversal detection
        if self.validator.detect_path_traversal(value):
            msg = f"Path traversal pattern detected in field: {path}"
            if self.log_suspicious:
                logger.warning(f"{sanitize_for_log(msg)} - Client: {_client_host(request)}")
            raise HTTPException(400, detail=msg)

        # Command injection detection
        if self.validator.detect_command_injection(value):
            msg = f"Command injection pattern detected in field: {path}"
            if self.log_suspicious:
                logger.warning(f"{sanitize_for_log(msg)} - Client: {_client_host(request)}")
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
                logger.info(f"Suspicious header detected: {header} from {_client_host(request)}")

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
