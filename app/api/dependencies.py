"""
Dependency Injection Setup for FastAPI

This module provides dependency injection functions for FastAPI routes,
allowing for clean separation of concerns and testability.
"""

from fastapi import Request, Header, HTTPException, status
from typing import Optional

from app.services.rag_pipeline import RAGPipeline
from app.models.api_key import APIKey
from app.core.auth import get_auth_manager


async def get_rag_pipeline(request: Request) -> RAGPipeline:
    """
    FastAPI dependency to get the RAG pipeline instance.

    This function retrieves the RAG pipeline from the application state,
    making it available to endpoints that need it via dependency injection.

    Args:
        request: The FastAPI request object

    Returns:
        The RAG pipeline instance

    Raises:
        RuntimeError: If the pipeline has not been initialized
    """
    try:
        return request.app.state.rag_pipeline
    except AttributeError:
        raise RuntimeError("RAG pipeline not initialized. Check application startup logs.")


async def get_api_key(
    x_api_key: Optional[str] = Header(None, description="API key for authentication")
) -> APIKey:
    """
    FastAPI dependency to verify API key from X-API-Key header.

    This function extracts and verifies the API key from the request header,
    making the authenticated API key available to protected endpoints.

    Args:
        x_api_key: The API key from X-API-Key header

    Returns:
        The verified APIKey instance

    Raises:
        HTTPException: If authentication fails (401) or key is invalid (403)
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing. Please provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    auth_manager = get_auth_manager()
    api_key = auth_manager.verify_api_key(x_api_key)

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired API key.",
        )

    return api_key


async def get_optional_api_key(
    x_api_key: Optional[str] = Header(None, description="Optional API key for authentication")
) -> Optional[APIKey]:
    """
    FastAPI dependency for optional API key authentication.

    This function attempts to verify the API key but doesn't require it.
    Useful for endpoints that work with or without authentication.

    Args:
        x_api_key: The API key from X-API-Key header (optional)

    Returns:
        The verified APIKey instance or None if not provided/invalid
    """
    if x_api_key is None:
        return None

    auth_manager = get_auth_manager()
    return auth_manager.verify_api_key(x_api_key)
