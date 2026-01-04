"""
Custom exception classes for Enterprise RAG System.

Provides a hierarchical exception structure for better error handling
and more meaningful error responses.
"""

from typing import Optional, Any


class BaseRAGException(Exception):
    """
    Base exception class for all RAG system errors.

    Attributes:
        message: Human-readable error message
        details: Additional error context (optional)
    """

    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary for JSON response."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class ValidationError(BaseRAGException):
    """
    Raised when input validation fails.

    Status: 400 Bad Request
    """

    pass


class AuthenticationError(BaseRAGException):
    """
    Raised when authentication fails.

    Status: 401 Unauthorized
    """

    pass


class AuthorizationError(BaseRAGException):
    """
    Raised when user lacks permission for an action.

    Status: 403 Forbidden
    """

    pass


class NotFoundError(BaseRAGException):
    """
    Raised when a requested resource is not found.

    Status: 404 Not Found
    """

    pass


class ConflictError(BaseRAGException):
    """
    Raised when a request conflicts with current state.

    Status: 409 Conflict
    """

    pass


class RateLimitError(BaseRAGException):
    """
    Raised when rate limit is exceeded.

    Status: 429 Too Many Requests
    """

    def __init__(self, message: str, retry_after: Optional[int] = None, details: Optional[Any] = None):
        super().__init__(message, details)
        self.retry_after = retry_after

    def to_dict(self) -> dict:
        """Convert exception to dictionary with retry_after."""
        data = super().to_dict()
        if self.retry_after:
            data["retry_after"] = self.retry_after
        return data


class InternalServerError(BaseRAGException):
    """
    Raised for unexpected server errors.

    Status: 500 Internal Server Error
    """

    pass


class ServiceUnavailableError(BaseRAGException):
    """
    Raised when a dependent service is unavailable.

    Status: 503 Service Unavailable
    """

    pass


class DatabaseError(BaseRAGException):
    """
    Raised for database-related errors.

    Status: 500 Internal Server Error
    """

    pass


class VectorDBError(BaseRAGException):
    """
    Raised for vector database errors.

    Status: 500 Internal Server Error
    """

    pass


class EmbeddingError(BaseRAGException):
    """
    Raised for embedding generation errors.

    Status: 500 Internal Server Error
    """

    pass


class DocumentProcessingError(BaseRAGException):
    """
    Raised for document processing failures.

    Status: 500 Internal Server Error
    """

    pass


class LLMError(BaseRAGException):
    """
    Raised for LLM-related errors.

    Status: 500 Internal Server Error
    """

    pass


class CacheError(BaseRAGException):
    """
    Raised for cache-related errors.

    Status: 500 Internal Server Error
    """

    pass
