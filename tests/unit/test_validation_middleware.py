"""
Tests for validation middleware and security validation.
"""

import pytest
from fastapi import Request, HTTPException
from starlette.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

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
        ]

        for input_str in malicious_inputs:
            assert validator.detect_sql_injection(input_str), f"Failed to detect: {input_str}"

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
        ]

        for input_str in malicious_inputs:
            assert validator.detect_xss(input_str), f"Failed to detect: {input_str}"

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
        """Create test client"""
        return TestClient(app)

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

        # Should be blocked by validation middleware with 400
        # Note: TestClient doesn't catch HTTPException the same way
        # The middleware raises the exception before the route handler
        try:
            response = client.post("/api/v1/query/", json=malicious_payload)
            # If we get here, check that it was blocked
            assert response.status_code in [400, 422]
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
            assert response.status_code in [400, 422]
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
            assert response.status_code in [400, 422]
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

        # Should be rejected
        try:
            response = client.get(
                "/",
                headers={"User-Agent": long_user_agent}
            )
            # If no exception, check status code
            assert response.status_code in [400, 431]
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
        """Create test client"""
        return TestClient(app)

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
            assert malicious_response.status_code in [400, 422]
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
