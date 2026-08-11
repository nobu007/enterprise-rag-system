"""Coverage-focused tests for StreamingRAGPipeline.stream_query.

Targets the previously-uncovered streaming path of
app/services/rag_pipeline.py (L365-434): the no-results branch, the
sources + LLM-token streaming happy path (including the falsy-content
skip in the chunk loop), and the in-stream error -> error-dict path.

stream_query is an ``async def`` generator; it calls the LLM client
directly (NOT via _call_llm / the circuit breaker), so a faithful
async-iterable mock for ``chat.completions.create`` is all that is
needed. Idiom mirrors tests/unit/test_query_stream_coverage.py.
"""
import pytest
from unittest.mock import AsyncMock, Mock

from app.services.rag_pipeline import StreamingRAGPipeline
from app.services.retrieval import RetrievalResult


@pytest.fixture
def retriever():
    return Mock()


@pytest.fixture
def llm_client():
    return Mock()


def _results():
    return [
        RetrievalResult(
            document="Relevant document text.",
            score=0.9,
            metadata={"filename": "doc.pdf", "page": 1},
            source="doc.pdf",
        ),
    ]


async def _chunk_stream(contents):
    """Async-iterable of mock LLM stream chunks (one per content)."""
    for content in contents:
        chunk = Mock()
        chunk.choices = [Mock()]
        chunk.choices[0].delta.content = content
        yield chunk


def _pipeline(retriever, llm_client):
    return StreamingRAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        llm_model="gpt-4",
    )


class TestStreamQueryNoResults:
    """Cover the no-retrieval-results branch (L376-382)."""

    @pytest.mark.asyncio
    async def test_no_results_yields_not_found(self, retriever, llm_client):
        retriever.retrieve.return_value = []
        pipeline = _pipeline(retriever, llm_client)

        items = [x async for x in pipeline.stream_query("q")]

        assert items == [
            {
                "type": "answer",
                "content": "I couldn't find any relevant information.",
                "done": True,
            }
        ]
        # No results -> LLM streaming never starts.
        llm_client.chat.completions.create.assert_not_called()


class TestStreamQueryHappyPath:
    """Cover sources + token streaming + final chunk (L384-427)."""

    @pytest.mark.asyncio
    async def test_streams_sources_and_tokens(self, retriever, llm_client):
        retriever.retrieve.return_value = _results()
        # A None-content chunk exercises the falsy-skip at L414.
        llm_client.chat.completions.create = AsyncMock(
            return_value=_chunk_stream(["Hello", None, " world"])
        )
        pipeline = _pipeline(retriever, llm_client)

        items = [x async for x in pipeline.stream_query("q")]

        # Sources emitted first.
        assert items[0]["type"] == "sources"
        assert items[0]["content"][0]["document"] == "doc.pdf"
        assert items[0]["content"][0]["score"] == 0.9

        answers = [it for it in items if it["type"] == "answer"]
        contents = [it["content"] for it in answers]
        # Two non-None tokens streamed (None chunk skipped).
        assert "Hello" in contents
        assert " world" in contents

        # Final done marker with latency.
        final = answers[-1]
        assert final["done"] is True
        assert final["content"] == ""
        assert "latency_ms" in final


class TestStreamQueryError:
    """Cover the in-stream error branch (L429-434)."""

    @pytest.mark.asyncio
    async def test_stream_error_yields_error_dict(
        self, retriever, llm_client
    ):
        retriever.retrieve.return_value = _results()
        llm_client.chat.completions.create = AsyncMock(
            side_effect=Exception("stream failed")
        )
        pipeline = _pipeline(retriever, llm_client)

        items = [x async for x in pipeline.stream_query("q")]

        # Sources are yielded before the LLM call fails.
        assert items[0]["type"] == "sources"
        err = items[-1]
        assert err["type"] == "error"
        assert "stream failed" in err["content"]
        assert err["done"] is True
