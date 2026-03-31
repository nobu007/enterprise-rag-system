"""
Dependency Injection Setup for FastAPI

This module provides dependency injection functions for FastAPI routes,
allowing for clean separation of concerns and testability.
"""

from fastapi import Request
from openai import AsyncOpenAI
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


async def get_llm_client(request: Request) -> AsyncOpenAI:
    """
    FastAPI dependency to get the LLM client instance.

    This function retrieves the AsyncOpenAI client from the RAG pipeline,
    making it available for streaming operations.

    Args:
        request: The FastAPI request object

    Returns:
        The AsyncOpenAI client instance

    Raises:
        RuntimeError: If the pipeline has not been initialized
    """
    try:
        pipeline = request.app.state.rag_pipeline
        return pipeline.llm_client
    except AttributeError:
        raise RuntimeError("LLM client not initialized. Check application startup logs.")
