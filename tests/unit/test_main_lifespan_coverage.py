"""Tests for the FastAPI lifespan startup/shutdown in app.main.

The existing test suite builds ``TestClient(app)`` WITHOUT a ``with`` block,
so Starlette never runs the app's lifespan and lines 42-107 (component
initialization, the init-failure wrap, and shutdown) stay uncovered. These
tests drive the lifespan context manager directly -- the same code path
Starlette runs on real server start -- with every external constructor
mocked, so no live OpenAI/Redis/vector-DB dependency is required.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock

import app.main as main_module
from app.main import lifespan


class TestLifespan:
    """Cover lifespan startup (happy), init-failure (RuntimeError), shutdown."""

    @staticmethod
    def _patch_constructors(monkeypatch) -> dict:
        """Mock every external constructor lifespan calls; return the mocks."""
        mocks = {
            "openai": Mock(name="openai_client"),
            "cache": Mock(name="cache_manager"),
            "vdb": Mock(name="vector_db"),
            "embed": Mock(name="embedding_model"),
            "retriever": Mock(name="retriever"),
            "pipeline": Mock(name="rag_pipeline"),
            "concurrency": Mock(name="concurrency_limiter"),
        }
        monkeypatch.setattr(main_module, "AsyncOpenAI", Mock(return_value=mocks["openai"]))
        monkeypatch.setattr(main_module, "CacheManager", Mock(return_value=mocks["cache"]))
        monkeypatch.setattr(main_module, "get_vector_db", Mock(return_value=mocks["vdb"]))
        monkeypatch.setattr(main_module, "get_embedding_model", Mock(return_value=mocks["embed"]))
        monkeypatch.setattr(main_module, "HybridRetriever", Mock(return_value=mocks["retriever"]))
        monkeypatch.setattr(main_module, "RAGPipeline", Mock(return_value=mocks["pipeline"]))
        monkeypatch.setattr(main_module, "get_concurrency_limiter", Mock(return_value=mocks["concurrency"]))
        return mocks

    async def test_lifespan_startup_initializes_state_and_shutdown(self, monkeypatch):
        """Happy path: startup wires every component onto app.state; shutdown runs on exit."""
        mocks = self._patch_constructors(monkeypatch)
        fake_app = SimpleNamespace(state=SimpleNamespace())

        async with lifespan(fake_app):
            # Startup (lines 42-99) completed before yield: all components are on state.
            assert fake_app.state.openai_client is mocks["openai"]
            assert fake_app.state.cache_manager is mocks["cache"]
            assert fake_app.state.rag_pipeline is mocks["pipeline"]
            assert fake_app.state.concurrency_limiter is mocks["concurrency"]
        # Reaching here means shutdown (line 107) ran on context exit without error.

    async def test_lifespan_startup_failure_raises_runtime_error(self, monkeypatch):
        """A failing constructor is wrapped into RuntimeError("initialization failed")."""
        monkeypatch.setattr(
            main_module,
            "AsyncOpenAI",
            Mock(side_effect=ConnectionError("upstream unreachable")),
        )
        fake_app = SimpleNamespace(state=SimpleNamespace())

        with pytest.raises(RuntimeError, match="initialization failed"):
            async with lifespan(fake_app):
                pass
