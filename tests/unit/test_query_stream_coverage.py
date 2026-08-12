"""Coverage-focused tests for the /query/stream endpoint body in
app/api/routes/query.py (L387-488).

The streaming route (``stream_query``) had 0% route-level coverage: the
existing streaming tests (test_feature_16) exercise ``StreamingRAGService``
directly, never the FastAPI route. These tests drive the route through
``TestClient`` to cover:

- the SSE happy path (StreamingResponse + retriever_func + format_sse_stream)
- the reranking branch inside retriever_func
- the ValueError -> 400 handler (validate_stream_request)
- the generic Exception -> 500 handler (setup failure)
- the in-stream error SSE fallback (exception inside generate())
"""
import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from app.api.dependencies import get_llm_client, get_rag_pipeline
from app.api.routes.query import router
from app.services.retrieval import RetrievalResult


def _retrieval_result(document="RAG is an AI framework", score=0.9,
                      source="test.pdf"):
    return RetrievalResult(
        document=document,
        score=score,
        metadata={"source": source, "page": 1},
        source=source,
    )


def _llm_stream():
    """Fresh async generator mimicking an OpenAI streaming completion."""
    async def _gen():
        yield Mock(choices=[Mock(delta=Mock(content="RAG stands"))])
        yield Mock(choices=[Mock(delta=Mock(content=" for retrieval"))])
        yield Mock(choices=[], usage=Mock(completion_tokens=4))
    return _gen()


def _make_pipeline(retrieve_side_effect=None):
    p = Mock()
    p.retriever = Mock()
    if retrieve_side_effect is not None:
        p.retriever.retrieve = AsyncMock(side_effect=retrieve_side_effect)
    else:
        p.retriever.retrieve = AsyncMock(return_value=[_retrieval_result()])
    p.reranker = None  # no reranking by default
    return p


def _make_llm_client():
    client = AsyncMock(spec=AsyncOpenAI)
    # Fresh async generator per call so each request gets its own stream.
    client.chat.completions.create = AsyncMock(
        side_effect=lambda *a, **k: _llm_stream()
    )
    return client


@pytest.fixture
def app_with_overrides():
    """Build a FastAPI app with mocked pipeline + LLM dependencies.

    Yields ``(app, pipeline, llm)`` so tests can reconfigure the mocks
    (e.g. attach a reranker, flip a side effect) before issuing a request.
    """
    pipeline = _make_pipeline()
    llm = _make_llm_client()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    app.dependency_overrides[get_llm_client] = lambda: llm
    yield app, pipeline, llm
    app.dependency_overrides = {}


def _parse_sse(text):
    """Parse concatenated SSE body into a list of ``data:`` JSON dicts."""
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            try:
                out.append(json.loads(block[len("data: "):]))
            except json.JSONDecodeError:
                pass
    return out


class TestStreamQueryHappyPath:
    """Cover L387-477: StreamingResponse creation + SSE generation."""

    def test_stream_returns_sse_with_content(self, app_with_overrides):
        app, pipeline, llm = app_with_overrides
        client = TestClient(app)
        resp = client.post("/query/stream", json={"query": "What is RAG?"})

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        chunks = _parse_sse(resp.text)
        assert chunks, "expected at least one SSE data chunk"
        joined = "".join(c.get("content", "") for c in chunks)
        assert "RAG stands" in joined
        assert any(c.get("is_done") for c in chunks), "final chunk must signal done"
        # Default request (rerank=True) but pipeline.reranker is None -> the
        # rerank branch is skipped, so no error chunk should be emitted.
        assert not any("error" in c for c in chunks)


class TestStreamQueryRerankBranch:
    """Cover L418-443: reranking inside retriever_func.

    The route must use ``Reranker.rerank_results`` (result objects in,
    reordered result objects out) -- the same pattern ``RAGPipeline`` uses.
    Calling ``rerank`` with a list of dicts and then indexing the returned
    ``(index, score)`` tuples as ``r["document"]`` is a latent bug.
    """

    def test_stream_applies_reranking_without_error(self, app_with_overrides):
        app, pipeline, llm = app_with_overrides
        calls = []

        reranker = Mock()

        def _rerank_results(query, results, top_k=None, **kwargs):
            calls.append(list(results))
            return list(results)  # pass-through ordering

        reranker.rerank_results = _rerank_results
        pipeline.reranker = reranker

        client = TestClient(app)
        resp = client.post("/query/stream", json={
            "query": "What is RAG?",
            "rerank": True,
        })

        assert resp.status_code == 200
        chunks = _parse_sse(resp.text)
        # Reranking succeeded -> no error chunk in the stream.
        assert not any("error" in c for c in chunks), chunks
        # rerank_results was invoked with the retrieved result objects.
        assert len(calls) == 1
        assert calls[0] and isinstance(calls[0][0], RetrievalResult)


class TestStreamQueryErrorHandlers:
    """Cover L479-488 (ValueError->400, generic->500) and L462-466 (in-stream)."""

    def test_stream_returns_400_on_invalid_query(self, app_with_overrides):
        app, pipeline, llm = app_with_overrides
        client = TestClient(app)
        # Whitespace-only query passes pydantic min_length=1 but is rejected
        # by StreamingRAGService.validate_stream_request -> ValueError -> 400.
        resp = client.post("/query/stream", json={"query": "   "})

        assert resp.status_code == 400
        assert "Invalid request" in resp.json()["detail"]

    def test_stream_returns_500_on_setup_error(self, app_with_overrides,
                                               monkeypatch):
        app, pipeline, llm = app_with_overrides
        from app.services.streaming import StreamingRAGService

        def _raise(self, **kwargs):
            raise RuntimeError("setup blew up")

        monkeypatch.setattr(StreamingRAGService, "validate_stream_request", _raise)

        client = TestClient(app)
        resp = client.post("/query/stream", json={"query": "What is RAG?"})

        assert resp.status_code == 500
        assert "Streaming failed" in resp.json()["detail"]

    def test_stream_emits_error_sse_on_inner_failure(self, app_with_overrides):
        app, pipeline, llm = app_with_overrides
        # Retrieval raising inside generate() is caught and sent as an error
        # SSE chunk (status stays 200 because StreamingResponse is returned).
        pipeline.retriever.retrieve = AsyncMock(side_effect=RuntimeError("retrieve down"))

        client = TestClient(app)
        resp = client.post("/query/stream", json={"query": "What is RAG?"})

        assert resp.status_code == 200
        chunks = _parse_sse(resp.text)
        assert any("error" in c and c.get("is_done") for c in chunks), chunks

    def test_stream_generate_fallback_on_format_error(self, app_with_overrides,
                                                      monkeypatch):
        """generate()'s own except (L447-451) sends an error SSE when the SSE
        formatter itself raises -- the route's last-resort containment."""
        from app.api.routes import query as query_module

        async def _boom(chunk_generator):
            raise RuntimeError("format exploded")
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(query_module, "format_sse_stream", _boom)

        app, pipeline, llm = app_with_overrides
        client = TestClient(app)
        resp = client.post("/query/stream", json={"query": "What is RAG?"})

        assert resp.status_code == 200
        chunks = _parse_sse(resp.text)
        assert any("error" in c and c.get("is_done") for c in chunks), chunks

    def test_stream_error_sse_valid_json_with_special_chars(
        self, app_with_overrides, monkeypatch
    ):
        """The route's last-resort error SSE (generate() except) must emit
        valid JSON even when the exception message contains characters that
        break naive f-string JSON assembly: double-quotes, backslashes, and
        newlines (a newline also splits the SSE event prematurely).

        The two sibling SSE emitters -- StreamingChunk.to_sse and
        format_sse_stream -- build their payloads with json.dumps(); the
        route's hand-rolled f-string is the odd one out and yields malformed
        JSON for any error message containing a quote or newline. Parse every
        emitted data: block strictly (json.loads, no swallowing) and require
        the error message to round-trip intact.
        """
        from app.api.routes import query as query_module

        bad_message = 'format failed: "timeout" at \\node\nnext line'

        async def _boom(chunk_generator):
            raise RuntimeError(bad_message)
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(query_module, "format_sse_stream", _boom)

        app, pipeline, llm = app_with_overrides
        client = TestClient(app)
        resp = client.post("/query/stream", json={"query": "What is RAG?"})

        assert resp.status_code == 200

        # Strict-parse EVERY data: block; the f-string bug surfaces here as
        # json.JSONDecodeError (inner quotes left unescaped) or as a split
        # event (embedded newline). json.dumps round-trips the message intact.
        parsed = []
        for block in resp.text.split("\n\n"):
            block = block.strip()
            if block.startswith("data: "):
                parsed.append(json.loads(block[len("data: "):]))
        err_chunks = [c for c in parsed if "error" in c]
        assert err_chunks, f"expected an error chunk; body={resp.text!r}"
        assert err_chunks[0]["is_done"] is True
        assert err_chunks[0]["error"] == bad_message
