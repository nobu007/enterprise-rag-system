"""
Dependency Injection Setup for FastAPI

This module provides dependency injection functions for FastAPI routes,
allowing for clean separation of concerns and testability.
"""

from fastapi import Request
from app.services.rag_pipeline import RAGPipeline


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
