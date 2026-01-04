"""
Global exception handlers for Enterprise RAG System.

Provides centralized error handling for all custom exceptions
and returns standardized JSON responses.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from slowapi.errors import RateLimitExceeded
import logging

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
from app.core.logging_config import get_logger

logger = get_logger(__name__)


# Status code mapping for custom exceptions
EXCEPTION_STATUS_CODES = {
    RAGValidationError: 400,
    AuthenticationError: 401,
    AuthorizationError: 403,
    NotFoundError: 404,
    ConflictError: 409,
    RateLimitError: 429,
    InternalServerError: 500,
    ServiceUnavailableError: 503,
    DatabaseError: 500,
    VectorDBError: 500,
    EmbeddingError: 500,
    DocumentProcessingError: 500,
    LLMError: 500,
    CacheError: 500,
}


async def rag_exception_handler(request: Request, exc: BaseRAGException) -> JSONResponse:
    """
    Handler for all custom RAG exceptions.

    Args:
        request: FastAPI request object
        exc: BaseRAGException or subclass

    Returns:
        JSONResponse with appropriate status code and error details
    """
    status_code = EXCEPTION_STATUS_CODES.get(type(exc), 500)
    error_dict = exc.to_dict()

    # Log error with context
    logger.error(
        f"{exc.__class__.__name__}: {exc.message}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": status_code,
            "details": exc.details
        },
        exc_info=True
    )

    # Add Retry-After header for RateLimitError
    headers = {}
    if isinstance(exc, RateLimitError) and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(
        status_code=status_code,
        content=error_dict,
        headers=headers
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handler for FastAPI HTTPException.

    Provides consistent error format for HTTPException.

    Args:
        request: FastAPI request object
        exc: HTTPException

    Returns:
        JSONResponse with standardized format
    """
    error_dict = {
        "error": "HTTPException",
        "message": exc.detail,
        "details": None
    }

    # Log warning for 4xx, error for 5xx
    if exc.status_code >= 500:
        logger.error(
            f"HTTPException {exc.status_code}: {exc.detail}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code
            }
        )
    else:
        logger.warning(
            f"HTTPException {exc.status_code}: {exc.detail}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code
            }
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_dict
    )


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Handler for Pydantic ValidationError.

    Converts Pydantic validation errors to standard format.

    Args:
        request: FastAPI request object
        exc: ValidationError

    Returns:
        JSONResponse with field-level error details
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"]
        })

    error_dict = {
        "error": "RequestValidationError",
        "message": "Request validation failed",
        "details": errors
    }

    logger.warning(
        f"ValidationError: {len(errors)} field(s)",
        extra={
            "path": request.url.path,
            "method": request.method,
            "errors": errors
        }
    )

    return JSONResponse(
        status_code=422,
        content=error_dict
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Fallback handler for unhandled exceptions.

    Catches all exceptions not handled by specific handlers.

    Args:
        request: FastAPI request object
        exc: Any Exception

    Returns:
        JSONResponse with generic error message
    """
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
        },
        exc_info=True
    )

    error_dict = {
        "error": "InternalServerError",
        "message": "An unexpected error occurred. Please try again later.",
        "details": str(exc) if logger.isEnabledFor(logging.DEBUG) else None
    }

    return JSONResponse(
        status_code=500,
        content=error_dict
    )


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Handler for slowapi RateLimitExceeded exception.

    Args:
        request: FastAPI request object
        exc: RateLimitExceeded exception

    Returns:
        JSONResponse with 429 status code
    """
    error_dict = {
        "error": "RateLimitExceeded",
        "message": "Too many requests. Please try again later.",
        "details": None
    }

    headers = {}
    if hasattr(exc, 'retry_after') and exc.retry_after:
        error_dict["retry_after"] = str(exc.retry_after)
        headers["Retry-After"] = str(exc.retry_after)

    logger.warning(
        f"Rate limit exceeded: {request.url.path}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "retry_after": getattr(exc, 'retry_after', None)
        }
    )

    return JSONResponse(
        status_code=429,
        content=error_dict,
        headers=headers
    )


def register_exception_handlers(app):
    """
    Register all exception handlers with the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    # Custom RAG exceptions
    for exc_class in [
        RAGValidationError,
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
    ]:
        app.add_exception_handler(exc_class, rag_exception_handler)

    # Built-in exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(ValidationError, validation_error_handler)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
