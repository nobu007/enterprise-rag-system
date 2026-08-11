"""
Unit tests for FastAPI dependency injection helpers.

The existing route tests override ``get_rag_pipeline`` with a lambda, so the
real dependency bodies (including their ``RuntimeError`` paths) are never
exercised. These tests call the real functions with a lightweight request
double to cover both the happy path and the not-initialized error path.
"""

import types

import pytest
from unittest.mock import Mock

from app.api.dependencies import get_llm_client, get_rag_pipeline


def _make_request(pipeline=None, *, with_pipeline: bool = True):
    """Build a minimal request double exposing ``app.state``.

    ``get_rag_pipeline``/``get_llm_client`` only touch ``request.app.state``,
    so a plain namespace is sufficient and lets us omit ``rag_pipeline`` to
    trigger the ``AttributeError`` -> ``RuntimeError`` path.
    """
    state = types.SimpleNamespace()
    if with_pipeline:
        state.rag_pipeline = pipeline
    return types.SimpleNamespace(app=types.SimpleNamespace(state=state))


class TestGetRagPipeline:
    """get_rag_pipeline: app.state.rag_pipeline passthrough + error path."""

    @pytest.mark.asyncio
    async def test_returns_pipeline_when_initialized(self):
        pipeline = Mock(name="rag_pipeline")
        request = _make_request(pipeline=pipeline)

        result = await get_rag_pipeline(request)

        assert result is pipeline

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_missing(self):
        # No rag_pipeline attribute on state -> AttributeError -> RuntimeError
        request = _make_request(with_pipeline=False)

        with pytest.raises(RuntimeError, match="RAG pipeline not initialized"):
            await get_rag_pipeline(request)


class TestGetLlmClient:
    """get_llm_client: pipeline.llm_client passthrough + error path."""

    @pytest.mark.asyncio
    async def test_returns_llm_client_when_initialized(self):
        llm_client = Mock(name="llm_client")
        pipeline = Mock(name="rag_pipeline")
        pipeline.llm_client = llm_client
        request = _make_request(pipeline=pipeline)

        result = await get_llm_client(request)

        assert result is llm_client

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_pipeline_missing(self):
        request = _make_request(with_pipeline=False)

        with pytest.raises(RuntimeError, match="LLM client not initialized"):
            await get_llm_client(request)
