"""
Unit tests for custom exception handling system.

Tests all custom exceptions and global exception handlers.
"""

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from unittest.mock import Mock
from pydantic import ValidationError

from app.core.exceptions import (
    BaseRAGException,
    ValidationError as RAGValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    InternalServerError,
    ServiceUnavailableError,
    DatabaseError,
    VectorDBError,
    EmbeddingError,
    DocumentProcessingError,
    LLMError,
    CacheError,
)
from app.core.error_handlers import (
    rag_exception_handler,
    http_exception_handler,
    validation_error_handler,
    generic_exception_handler,
    rate_limit_exceeded_handler,
)
from fastapi import HTTPException
from slowapi.errors import RateLimitExceeded


class TestCustomExceptions:
    """Test custom exception classes"""

    def test_base_rag_exception(self):
        """Test BaseRAGException initialization and to_dict"""
        exc = BaseRAGException("Test error", details={"key": "value"})

        assert exc.message == "Test error"
        assert exc.details == {"key": "value"}
        assert str(exc) == "Test error"

        error_dict = exc.to_dict()
        assert error_dict["error"] == "BaseRAGException"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"] == {"key": "value"}

    def test_validation_error(self):
        """Test ValidationError"""
        exc = RAGValidationError("Invalid input")
        assert exc.message == "Invalid input"
        assert exc.to_dict()["error"] == "ValidationError"

    def test_authentication_error(self):
        """Test AuthenticationError"""
        exc = AuthenticationError("Invalid credentials")
        assert exc.message == "Invalid credentials"
        assert exc.to_dict()["error"] == "AuthenticationError"

    def test_authorization_error(self):
        """Test AuthorizationError"""
        exc = AuthorizationError("Access denied")
        assert exc.message == "Access denied"

    def test_not_found_error(self):
        """Test NotFoundError"""
        exc = NotFoundError("Resource not found")
        assert exc.message == "Resource not found"

    def test_conflict_error(self):
        """Test ConflictError"""
        exc = ConflictError("Resource already exists")
        assert exc.message == "Resource already exists"

    def test_rate_limit_error(self):
        """Test RateLimitError with retry_after"""
        exc = RateLimitError("Rate limit exceeded", retry_after=60)
        assert exc.message == "Rate limit exceeded"
        assert exc.retry_after == 60

        error_dict = exc.to_dict()
        assert "retry_after" in error_dict
        assert error_dict["retry_after"] == 60

    def test_rate_limit_error_without_retry(self):
        """Test RateLimitError without retry_after"""
        exc = RateLimitError("Rate limit exceeded")
        assert exc.retry_after is None

        error_dict = exc.to_dict()
        assert "retry_after" not in error_dict

    def test_database_error(self):
        """Test DatabaseError"""
        exc = DatabaseError("Connection failed")
        assert exc.message == "Connection failed"

    def test_vector_db_error(self):
        """Test VectorDBError"""
        exc = VectorDBError("Index not found")
        assert exc.message == "Index not found"

    def test_embedding_error(self):
        """Test EmbeddingError"""
        exc = EmbeddingError("Embedding generation failed")
        assert exc.message == "Embedding generation failed"

    def test_document_processing_error(self):
        """Test DocumentProcessingError"""
        exc = DocumentProcessingError("Failed to parse document")
        assert exc.message == "Failed to parse document"

    def test_llm_error(self):
        """Test LLMError"""
        exc = LLMError("LLM API error")
        assert exc.message == "LLM API error"

    def test_cache_error(self):
        """Test CacheError"""
        exc = CacheError("Cache connection failed")
        assert exc.message == "Cache connection failed"


class TestExceptionHandlers:
    """Test global exception handlers"""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request"""
        request = Mock(spec=Request)
        request.url.path = "/api/v1/test"
        request.method = "GET"
        return request

    @pytest.mark.asyncio
    async def test_rag_exception_handler(self, mock_request):
        """Test handler for custom RAG exceptions"""
        exc = NotFoundError("Document not found", details={"doc_id": "123"})

        response = await rag_exception_handler(mock_request, exc)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 404

        content = response.body.decode()
        assert "Document not found" in content

    @pytest.mark.asyncio
    async def test_rate_limit_exception_handler_with_retry(self, mock_request):
        """Test RateLimitError handler with retry_after"""
        exc = RateLimitError("Too many requests", retry_after=60)

        response = await rag_exception_handler(mock_request, exc)

        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "60"

    @pytest.mark.asyncio
    async def test_rate_limit_exception_handler_without_retry(self, mock_request):
        """Test RateLimitError handler without retry_after"""
        exc = RateLimitError("Too many requests")

        response = await rag_exception_handler(mock_request, exc)

        assert response.status_code == 429
        assert "Retry-After" not in response.headers

    @pytest.mark.asyncio
    async def test_http_exception_handler(self, mock_request):
        """Test handler for HTTPException"""
        exc = HTTPException(status_code=400, detail="Bad request")

        response = await http_exception_handler(mock_request, exc)

        assert response.status_code == 400
        content = response.body.decode()
        assert "Bad request" in content

    @pytest.mark.asyncio
    async def test_validation_error_handler(self, mock_request):
        """Test handler for Pydantic ValidationError"""
        # Create a Pydantic validation error
        try:
            # Trigger a Pydantic validation error
            from pydantic import BaseModel, Field

            class TestModel(BaseModel):
                name: str = Field(..., min_length=1)

            TestModel(name="")  # This should raise ValidationError
        except ValidationError as e:
            response = await validation_error_handler(mock_request, e)

            assert response.status_code == 422
            content = response.body.decode()
            assert "RequestValidationError" in content

    @pytest.mark.asyncio
    async def test_generic_exception_handler(self, mock_request):
        """Test fallback handler for unhandled exceptions"""
        exc = ValueError("Unexpected error")

        response = await generic_exception_handler(mock_request, exc)

        assert response.status_code == 500
        content = response.body.decode()
        assert "InternalServerError" in content

    @pytest.mark.asyncio
    async def test_slowapi_rate_limit_handler(self, mock_request):
        """Test handler for slowapi RateLimitExceeded"""
        # Create a mock RateLimitExceeded exception
        exc = Mock(spec=RateLimitExceeded)
        exc.retry_after = 30

        response = await rate_limit_exceeded_handler(mock_request, exc)

        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "30"


class TestExceptionStatusCodes:
    """Test that exceptions map to correct HTTP status codes"""

    def test_validation_error_status_code(self):
        """Test ValidationError maps to 400"""
        from app.core.error_handlers import EXCEPTION_STATUS_CODES
        from app.core.exceptions import ValidationError as RAGValidationError

        assert EXCEPTION_STATUS_CODES[RAGValidationError] == 400

    def test_authentication_error_status_code(self):
        """Test AuthenticationError maps to 401"""
        from app.core.error_handlers import EXCEPTION_STATUS_CODES
        from app.core.exceptions import AuthenticationError

        assert EXCEPTION_STATUS_CODES[AuthenticationError] == 401

    def test_not_found_error_status_code(self):
        """Test NotFoundError maps to 404"""
        from app.core.error_handlers import EXCEPTION_STATUS_CODES
        from app.core.exceptions import NotFoundError

        assert EXCEPTION_STATUS_CODES[NotFoundError] == 404

    def test_rate_limit_error_status_code(self):
        """Test RateLimitError maps to 429"""
        from app.core.error_handlers import EXCEPTION_STATUS_CODES
        from app.core.exceptions import RateLimitError

        assert EXCEPTION_STATUS_CODES[RateLimitError] == 429

    def test_internal_server_error_status_code(self):
        """Test InternalServerError maps to 500"""
        from app.core.error_handlers import EXCEPTION_STATUS_CODES
        from app.core.exceptions import InternalServerError

        assert EXCEPTION_STATUS_CODES[InternalServerError] == 500
