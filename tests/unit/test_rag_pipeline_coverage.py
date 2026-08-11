"""Coverage-focused tests for app/services/rag_pipeline.py.

Targets previously-uncovered branches of RAGPipeline: the cache
hit/miss/set paths of ``query`` (L218-230, L325-327), the
circuit-breaker-disabled (L75, L158) and circuit-open (L154-156)
branches of ``_call_llm``, the LLM-failure -> RuntimeError wrapping
(L147-148), and the empty-results confidence path (L166-167).

The existing test_rag_pipeline.py never wires a ``cache_manager`` and
always leaves the circuit breaker enabled (its default), so none of
these paths overlap with the suite already in place.
"""
import pytest
from unittest.mock import AsyncMock, Mock

from app.core.circuit_breaker import CircuitBreakerError
from app.services.rag_pipeline import RAGPipeline, RAGResponse
from app.services.retrieval import RetrievalResult


@pytest.fixture
def retriever():
    return Mock()


@pytest.fixture
def llm_client():
    return Mock()


@pytest.fixture
def chat_completion():
    """Successful LLM completion (mirrors test_rag_pipeline fixtures)."""
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = "Generated answer."
    resp.choices[0].finish_reason = "stop"
    resp.usage.total_tokens = 42
    resp.usage.prompt_tokens = 30
    resp.usage.completion_tokens = 12
    return resp


def _results():
    return [
        RetrievalResult(
            document="Relevant document text.",
            score=0.9,
            metadata={"filename": "doc.pdf", "page": 1},
            source="doc.pdf",
        ),
    ]


def _wire_llm(llm_client, completion):
    llm_client.chat.completions.create = AsyncMock(
        return_value=completion
    )


class TestCacheBranches:
    """Cover cache hit/miss/set paths of query() (L218-230, L325-327)."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_without_retrieval(
        self, retriever, llm_client, chat_completion
    ):
        _wire_llm(llm_client, chat_completion)
        cached = {
            "answer": "cached answer",
            "sources": [],
            "confidence": 0.5,
            "latency_ms": 7,
            "tokens_used": 3,
            "retrieval_results": [],
        }
        cache = Mock()
        cache.generate_key.return_value = "k"
        cache.get.return_value = cached

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            llm_model="gpt-4",
            cache_manager=cache,
        )
        response = await pipeline.query("any question")

        assert isinstance(response, RAGResponse)
        assert response.answer == "cached answer"
        # Cache hit short-circuits both retrieval and the LLM call.
        retriever.retrieve.assert_not_called()
        llm_client.chat.completions.create.assert_not_called()
        cache.get.assert_called_once_with("k")

    @pytest.mark.asyncio
    async def test_cache_miss_full_path_sets_cache(
        self, retriever, llm_client, chat_completion
    ):
        _wire_llm(llm_client, chat_completion)
        retriever.retrieve.return_value = _results()
        cache = Mock()
        cache.generate_key.return_value = "k"
        cache.get.return_value = None

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            llm_model="gpt-4",
            cache_manager=cache,
        )
        response = await pipeline.query("a question")

        assert isinstance(response, RAGResponse)
        assert response.answer == "Generated answer."
        # Miss -> full pipeline ran -> resulting response was cached.
        retriever.retrieve.assert_called_once()
        cache.set.assert_called_once()
        assert cache.set.call_args[0][1] is response


class TestCallLlmCircuitBreaker:
    """Cover the circuit-breaker branches of _call_llm (L75, L147-158)."""

    @pytest.mark.asyncio
    async def test_call_llm_without_circuit_breaker(
        self, retriever, llm_client, chat_completion
    ):
        _wire_llm(llm_client, chat_completion)
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            llm_model="gpt-4",
            enable_circuit_breaker=False,
        )
        # L75: breaker disabled -> None.
        assert pipeline.llm_circuit_breaker is None
        # L158: direct call path (no breaker wrapper).
        result = await pipeline._call_llm("prompt")
        assert result["answer"] == "Generated answer."

    @pytest.mark.asyncio
    async def test_llm_failure_wrapped_as_runtime_error(
        self, retriever, llm_client
    ):
        llm_client.chat.completions.create = AsyncMock(
            side_effect=Exception("provider down")
        )
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            llm_model="gpt-4",
            enable_circuit_breaker=False,
        )
        # L147-148: any provider exception -> RuntimeError("LLM call ...").
        with pytest.raises(RuntimeError, match="LLM call failed"):
            await pipeline._call_llm("prompt")

    @pytest.mark.asyncio
    async def test_circuit_opens_and_rejects_after_threshold(
        self, retriever, llm_client
    ):
        llm_client.chat.completions.create = AsyncMock(
            side_effect=Exception("provider down")
        )
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            llm_model="gpt-4",
        )
        # failure_threshold == 5: each call raises RuntimeError, counted.
        for _ in range(5):
            with pytest.raises(RuntimeError):
                await pipeline._call_llm("prompt")
        # Breaker now OPEN -> 6th call rejected before invocation.
        with pytest.raises(CircuitBreakerError):
            await pipeline._call_llm("prompt")
        assert (
            pipeline.llm_circuit_breaker.state.value == "open"
        )


class TestConfidenceEdge:
    """Cover empty-results branch of _calculate_confidence (L166-167)."""

    def test_confidence_zero_for_empty_results(
        self, retriever, llm_client
    ):
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            llm_model="gpt-4",
        )
        assert pipeline._calculate_confidence([], "answer") == 0.0
