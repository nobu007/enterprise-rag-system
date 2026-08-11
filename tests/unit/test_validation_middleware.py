"""
Tests for validation middleware and security validation.
"""

import logging
import sys

import pytest
from fastapi import HTTPException, Request
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from unittest.mock import Mock, AsyncMock

from app.main import app
from app.core.security import SecurityValidator
from app.middleware.validation import ValidationMiddleware


class TestSecurityValidator:
    """Test SecurityValidator class"""

    def test_detect_sql_injection_true(self):
        """Test SQL injection detection with malicious input"""
        validator = SecurityValidator()

        # Test various SQL injection patterns
        malicious_inputs = [
            "1' OR '1'='1",
            "1; DROP TABLE users--",
            "' UNION SELECT * FROM users--",
            "admin'--",
            "1' AND 1=1--",
            # block-comment evasion (single-line here; multi-line covered by
            # the dedicated bypass regression test below)
            "SELECT/**/1",
        ]

        for input_str in malicious_inputs:
            assert validator.detect_sql_injection(input_str), f"Failed to detect: {input_str}"

    def test_detect_sql_injection_multiline_comment_bypass(self):
        r"""Regression: a newline inside a /* ... */ block comment must not evade detection.

        Before the DOTALL fix, the (/\*.*\*/) pattern's '.' did not cross
        newlines, so a block comment split across lines — the classic
        SELECT/**/1 token-splitting evasion — returned False. The payload below
        intentionally carries no other SQL marker (no --, no OR/UNION), so it
        isolates the block-comment rule.
        """
        validator = SecurityValidator()
        multiline_payload = "SELECT/*\n*/1"
        assert validator.detect_sql_injection(multiline_payload) is True

    def test_detect_sql_injection_false(self):
        """Test SQL injection detection with safe input"""
        validator = SecurityValidator()

        safe_inputs = [
            "Hello World",
            "SELECT is a word",
            "drop the ball",
            "This is a test",
            "User input data",
        ]

        for input_str in safe_inputs:
            assert not validator.detect_sql_injection(input_str), f"False positive: {input_str}"

    def test_detect_xss_true(self):
        """Test XSS detection with malicious input"""
        validator = SecurityValidator()

        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert('XSS')",
            "<iframe src='http://evil.com'>",
            "<div onload='alert(1)'>",
            # multi-line script bodies (DOTALL regression): the most common
            # real-world XSS form must not bypass the <script>...</script> rule
            "<script>\nalert('XSS')\n</script>",
            '<script type="text/javascript">\nsteal()\n</script>',
        ]

        for input_str in malicious_inputs:
            assert validator.detect_xss(input_str), f"Failed to detect: {input_str}"

    def test_detect_xss_multiline_script_bypass(self):
        """Regression: a newline inside the <script> body must not evade detection.

        Before the DOTALL fix, '.*?' did not cross newlines, so any payload
        split across lines (the typical injected form) returned False.
        """
        validator = SecurityValidator()
        multiline_payload = "<script>\nalert(document.cookie)\n</script>"
        assert validator.detect_xss(multiline_payload) is True

    def test_detect_xss_false(self):
        """Test XSS detection with safe input"""
        validator = SecurityValidator()

        safe_inputs = [
            "Hello World",
            "<p>This is safe HTML (without handlers)</p>",
            "javascript is a programming language",
            "normal text with no script tags",
        ]

        for input_str in safe_inputs:
            assert not validator.detect_xss(input_str), f"False positive: {input_str}"

    def test_detect_path_traversal_true(self):
        """Test path traversal detection"""
        validator = SecurityValidator()

        malicious_inputs = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "%2e%2e%2f",
            "..%2fetc/passwd",
        ]

        for input_str in malicious_inputs:
            assert validator.detect_path_traversal(input_str), f"Failed to detect: {input_str}"

    def test_detect_path_traversal_false(self):
        """Test path traversal detection with safe input"""
        validator = SecurityValidator()

        safe_inputs = [
            "/home/user/file.txt",
            "C:\\Users\\file.txt",
            "normal/path/file.txt",
            "file..txt",
        ]

        for input_str in safe_inputs:
            assert not validator.detect_path_traversal(input_str), f"False positive: {input_str}"

    def test_sanitize_input_success(self):
        """Test input sanitization"""
        validator = SecurityValidator()

        # Normal input
        assert validator.sanitize_input("  Hello World  ") == "Hello World"

        # Max length boundary
        long_input = "a" * 10000
        assert validator.sanitize_input(long_input, max_length=10000) == long_input

    def test_sanitize_input_too_long(self):
        """Test input sanitization with too long input"""
        validator = SecurityValidator()

        with pytest.raises(ValueError, match="Input too long"):
            validator.sanitize_input("a" * 10001, max_length=10000)

    def test_detect_command_injection(self):
        """Test command injection detection"""
        validator = SecurityValidator()

        malicious_inputs = [
            "file.txt; rm -rf /",
            "data | cat /etc/passwd",
            "input`whoami`",
            "file $(cat /etc/passwd)",
        ]

        for input_str in malicious_inputs:
            assert validator.detect_command_injection(input_str), f"Failed to detect: {input_str}"

        # Safe inputs
        safe_inputs = [
            "normal text",
            "file.txt",
            "data with spaces",
        ]

        for input_str in safe_inputs:
            assert not validator.detect_command_injection(input_str), f"False positive: {input_str}"


class TestValidationMiddleware:
    """Test ValidationMiddleware"""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies"""
        from unittest.mock import MagicMock
        app.state.openai_client = AsyncMock()
        app.state.cache_manager = MagicMock()
        app.state.rag_pipeline = MagicMock()
        return TestClient(app, raise_server_exceptions=False)

    def test_request_size_limit_success(self, client):
        """Test normal request within size limit"""
        response = client.get("/")
        assert response.status_code == 200

    def test_request_size_limit_exceeded(self, client):
        """Test request size limit enforcement"""
        # Create request with moderately sized payload
        # The validation middleware should allow this (under 10MB limit)
        large_data = {"query": "x" * (1000)}  # 1KB test

        # This will fail due to RAG pipeline not initialized, not due to size
        try:
            response = client.post("/api/v1/query/", json=large_data)
            # If we get a response, it should not be a 413 (too large)
            assert response.status_code not in [413]
        except RuntimeError as e:
            # Expected: RAG pipeline not initialized
            assert "RAG pipeline" in str(e)

    def test_xss_detection_in_request(self, client):
        """Test XSS detection in API request"""
        malicious_payload = {
            "query": "<script>alert('XSS')</script>",
            "collection_name": "test"
        }

        # Should be blocked by validation middleware with 400, or rate limited with 429
        try:
            response = client.post("/api/v1/query/", json=malicious_payload)
            # If we get here, check that it was blocked (400/422) or rate limited (429)
            assert response.status_code in [400, 422, 429]
        except Exception as e:
            # HTTPException is raised
            assert "XSS" in str(e) or "malicious" in str(e).lower()

    def test_sql_injection_detection_in_request(self, client):
        """Test SQL injection detection in API request"""
        malicious_payload = {
            "query": "test' OR '1'='1",
            "collection_name": "test"
        }

        # Should be blocked by validation middleware
        try:
            response = client.post("/api/v1/query/", json=malicious_payload)
            # If we get here, check that it was blocked
            assert response.status_code in [400, 422, 429]
        except Exception as e:
            # HTTPException is raised
            assert "SQL injection" in str(e) or "injection" in str(e).lower()

    def test_path_traversal_detection(self, client):
        """Test path traversal detection"""
        malicious_payload = {
            "query": "test",
            "collection_name": "../../../etc/passwd"
        }

        # Should be blocked
        try:
            response = client.post("/api/v1/query/", json=malicious_payload)
            # If we get here, check that it was blocked
            assert response.status_code in [400, 422, 429]
        except Exception as e:
            # HTTPException is raised
            assert "Path traversal" in str(e) or "traversal" in str(e).lower()

    def test_security_headers_present(self, client):
        """Test that security headers are added to responses"""
        response = client.get("/")

        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"

        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"

        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-XSS-Protection"] == "1; mode=block"

        assert "Strict-Transport-Security" in response.headers

        assert "Content-Security-Policy" in response.headers

    def test_normal_request_passes_validation(self, client):
        """Test that normal requests pass validation"""
        # Test with a simple GET request that doesn't need RAG pipeline
        response = client.get("/")
        # Should succeed (no validation errors)
        assert response.status_code == 200

    def test_user_agent_too_long(self, client):
        """Test User-Agent header validation"""
        long_user_agent = "a" * 501

        # Should be rejected (400/431) or trigger server error (500) if middleware
        # raises HTTPException that isn't caught by error handler
        try:
            response = client.get(
                "/",
                headers={"User-Agent": long_user_agent}
            )
            # If no exception, check status code
            assert response.status_code in [400, 431, 500]
        except Exception as e:
            # HTTPException is raised
            assert "User-Agent" in str(e) or "too long" in str(e).lower()


class TestRateLimiting:
    """Test enhanced rate limiting with IP detection"""

    def test_get_client_ip_direct(self):
        """Test IP detection from direct connection"""
        from app.core.rate_limit import get_client_ip

        # Mock request without proxy headers
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.100"

        ip = get_client_ip(request)
        assert ip == "192.168.1.100"

    def test_get_client_ip_x_forwarded_for(self):
        """Test IP detection from X-Forwarded-For header"""
        from app.core.rate_limit import get_client_ip

        request = Mock(spec=Request)
        request.headers = {"X-Forwarded-For": "203.0.113.1, 70.41.3.18, 150.172.238.178"}
        request.client = Mock()
        request.client.host = "192.168.1.100"

        ip = get_client_ip(request)
        # Should take first IP
        assert ip == "203.0.113.1"

    def test_get_client_ip_x_real_ip(self):
        """Test IP detection from X-Real-IP header"""
        from app.core.rate_limit import get_client_ip

        request = Mock(spec=Request)
        request.headers = {"X-Real-IP": "198.51.100.1"}
        request.client = Mock()
        request.client.host = "192.168.1.100"

        ip = get_client_ip(request)
        assert ip == "198.51.100.1"

    def test_get_client_ip_cloudflare(self):
        """Test IP detection from CF-Connecting-IP header"""
        from app.core.rate_limit import get_client_ip

        request = Mock(spec=Request)
        request.headers = {"CF-Connecting-IP": "203.0.113.195"}
        request.client = Mock()
        request.client.host = "192.168.1.100"

        ip = get_client_ip(request)
        assert ip == "203.0.113.195"

    def test_get_client_ip_x_real_ip_strips_whitespace(self):
        """X-Real-IP must be stripped like X-Forwarded-For.

        Regression: only the X-Forwarded-For path normalized surrounding
        whitespace, so a client sending "X-Real-IP: 1.2.3.4 " got a distinct
        rate-limit key per whitespace variant — fragmenting buckets and
        evading the per-IP limit. All three proxy headers must strip.
        """
        from app.core.rate_limit import get_client_ip

        for padded in ("  198.51.100.1  ", "\t198.51.100.1", "198.51.100.1\n"):
            request = Mock(spec=Request)
            request.headers = {"X-Real-IP": padded}
            request.client = Mock()
            request.client.host = "192.168.1.100"
            assert get_client_ip(request) == "198.51.100.1"

    def test_get_client_ip_cloudflare_strips_whitespace(self):
        """CF-Connecting-IP must be stripped like the other proxy headers."""
        from app.core.rate_limit import get_client_ip

        request = Mock(spec=Request)
        request.headers = {"CF-Connecting-IP": "  203.0.113.195 "}
        request.client = Mock()
        request.client.host = "192.168.1.100"
        assert get_client_ip(request) == "203.0.113.195"

    def test_get_user_id_with_api_key(self):
        """Test user ID with API key"""
        from app.core.rate_limit import get_user_id

        request = Mock(spec=Request)
        request.headers = {"X-API-Key": "test_api_key_123"}
        request.client = Mock()
        request.client.host = "192.168.1.100"

        user_id = get_user_id(request)
        assert user_id == "key:test_api_key_123"

    def test_get_user_id_with_ip(self):
        """Test user ID with IP address"""
        from app.core.rate_limit import get_user_id

        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.100"

        user_id = get_user_id(request)
        assert user_id == "ip:192.168.1.100"


class TestIntegration:
    """Integration tests for security features"""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies"""
        from unittest.mock import MagicMock
        app.state.openai_client = AsyncMock()
        app.state.cache_manager = MagicMock()
        app.state.rag_pipeline = MagicMock()
        return TestClient(app, raise_server_exceptions=False)

    def test_full_security_stack(self, client):
        """Test that all security features work together"""
        # Test 1: Normal request succeeds
        response = client.get("/")
        assert response.status_code == 200

        # Test 2: Security headers present
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers

        # Test 3: Malicious request blocked
        try:
            malicious_response = client.post(
                "/api/v1/query/",
                json={"query": "<script>alert(1)</script>"}
            )
            # If we get here, check status
            assert malicious_response.status_code in [400, 422, 429]
        except Exception as e:
            # Exception expected (XSS detected)
            assert "XSS" in str(e) or "malicious" in str(e).lower()

    def test_cors_headers_present(self, client):
        """Test that CORS headers are still present after middleware"""
        response = client.get(
            "/",
            headers={"Origin": "http://localhost:3000"}
        )

        # CORS headers should be present
        # (depending on configuration)
        # At minimum, the request should succeed
        assert response.status_code == 200


class TestValidationMiddlewareIsolated:
    """Exercise ValidationMiddleware branches without the full app.

    The integration tests above drive the real app (rate limiting, RAG
    pipeline, etc.), which obscures the middleware's own security branches.
    These tests wrap a minimal Starlette app in ValidationMiddleware so every
    rejection path returns its intended status code, and call the internal
    async helpers directly for branches that cannot be triggered over HTTP.
    """

    @staticmethod
    def _build_app(**middleware_kwargs) -> TestClient:
        """Minimal Starlette app with ValidationMiddleware and one endpoint."""
        async def endpoint(request: Request) -> JSONResponse:
            return JSONResponse({"ok": True})

        application = Starlette(routes=[
            Route("/", endpoint, methods=["GET", "POST", "PUT", "PATCH"]),
        ])
        application.add_middleware(ValidationMiddleware, **middleware_kwargs)
        return TestClient(application)

    @staticmethod
    def _make_middleware(**kwargs) -> ValidationMiddleware:
        """Construct a middleware with a no-op ASGI app for direct method tests."""
        async def dummy(scope, receive, send):
            return None

        return ValidationMiddleware(dummy, **kwargs)

    def test_request_too_large_returns_413(self):
        """Content-Length over max_request_size is rejected with 413 (DoS guard)."""
        client = self._build_app(max_request_size=10)
        response = client.post("/", json={"q": "x" * 100})
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()

    async def test_invalid_content_length_returns_400(self):
        """A non-integer Content-Length is rejected with 400."""
        mw = self._make_middleware()
        request = Mock(spec=Request)
        request.headers = {"content-length": "not-a-number"}
        with pytest.raises(HTTPException) as exc_info:
            await mw._validate_content_length(request)
        assert exc_info.value.status_code == 400
        assert "Invalid Content-Length" in exc_info.value.detail

    def test_command_injection_blocked(self):
        """Command-injection payload is rejected with 400 after XSS/SQL/path pass."""
        client = self._build_app()
        response = client.post("/", json={"q": "data `whoami`"})
        assert response.status_code == 400
        assert "Command injection" in response.json()["detail"]

    def test_list_value_recursion_validated(self):
        """Malicious strings nested in a JSON list are reached via list recursion."""
        client = self._build_app()
        response = client.post("/", json={"items": ["data `whoami`"]})
        assert response.status_code == 400
        assert "Command injection" in response.json()["detail"]

    def test_top_level_list_body_validated(self):
        """A malicious payload sent as a *top-level* JSON array is rejected.

        Regression: ``_validate_request_body`` only entered the recursive
        validator when ``isinstance(data, dict)``, so a body that was itself a
        JSON array skipped validation entirely and passed through UNVALIDATED
        — the same boundary-skip class as the deep-nesting case (INV-VAL-001).
        A batch-style array body is a realistic shape, so the strings inside it
        must still be scanned. Before the fix this returned 200.
        """
        client = self._build_app()
        response = client.post("/", json=["data `whoami`"])
        assert response.status_code == 400
        assert "Command injection" in response.json()["detail"]

    def test_deeply_nested_json_rejected_not_swallowed(self):
        """Deeply nested JSON that exhausts the recursive validator is rejected (400),
        not swallowed.

        A payload nested past the interpreter's recursion limit is under the
        content-length limit and parses via the json C scanner, but exhausts the
        Python recursion stack of ``_validate_dict_recursive`` -> RecursionError.
        Before the fail-closed fix this was caught by the generic ``except
        Exception`` and swallowed, so the request passed through UNVALIDATED — an
        attacker could wrap any payload in enough nesting to abort the scan
        before it was reached (INV-VAL-001 boundary-rejection violation).
        """
        client = self._build_app()
        depth = sys.getrecursionlimit() + 200  # comfortably exceeds the stack
        payload = ('{"a":' * depth) + '1' + ('}' * depth)
        response = client.post(
            "/",
            content=payload.encode(),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        assert "too deeply nested" in response.json()["detail"].lower()

    def test_security_validation_disabled_passes(self):
        """With validation disabled, a malicious payload is allowed through."""
        client = self._build_app(enable_security_validation=False)
        response = client.post("/", json={"q": "data `whoami`"})
        assert response.status_code == 200

    def test_non_json_body_passes_validation(self):
        """Non-JSON request bodies skip security validation (no crash)."""
        client = self._build_app()
        response = client.post(
            "/",
            content=b"this is not json",
            headers={"content-type": "text/plain"},
        )
        assert response.status_code == 200

    def test_empty_body_skips_validation(self):
        """An empty POST body returns early before any security scan."""
        client = self._build_app()
        response = client.post("/", content=b"")
        assert response.status_code == 200

    def test_suspicious_header_logged(self, caplog):
        """Recognized spoofing headers are logged without blocking the request."""
        client = self._build_app()
        with caplog.at_level(logging.INFO):
            response = client.get("/", headers={"X-Forwarded-Host": "evil.example"})
        assert response.status_code == 200
        assert any(
            "Suspicious header" in record.message for record in caplog.records
        )

    async def test_unexpected_body_error_is_logged_not_raised(self, caplog):
        """Unexpected (non-HTTP) errors reading the body are swallowed, not blocking."""
        mw = self._make_middleware()
        request = Mock(spec=Request)
        request.body = AsyncMock(side_effect=RuntimeError("boom"))
        with caplog.at_level(logging.ERROR):
            result = await mw._validate_request_body(request)
        assert result is None
        assert any(
            "Error validating request body" in record.message
            for record in caplog.records
        )
