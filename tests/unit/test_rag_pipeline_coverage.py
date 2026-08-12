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
import logging

import pytest
from unittest.mock import AsyncMock, Mock, patch

from app.core.cache import CacheManager
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


class TestCacheKeyFilterIsolation:
    """The cache key must distinguish queries that differ only by filter or
    search mode, else one request returns another's cached result.

    Reproduces a cross-filter data leak: with a REAL CacheManager (L1 active,
    Redis down) and a retriever whose result set depends on ``filter_dict``,
    a second query carrying a different filter used to hit the first query's
    cache entry (same question/collection/top_k/rerank -> same key) and was
    served the WRONG documents.
    """

    @staticmethod
    def _real_cache():
        with patch(
            "app.core.cache.redis.from_url", side_effect=ConnectionError("no redis")
        ):
            return CacheManager(enabled=True)

    @staticmethod
    def _filter_aware_retriever():
        retriever = Mock()

        def retrieve(query, top_k, use_hybrid, filter_dict, collection):
            dept = (filter_dict or {}).get("dept", "public")
            return [
                RetrievalResult(
                    document=f"{dept.upper()} confidential document",
                    score=0.9,
                    metadata={"filename": f"{dept}.pdf"},
                    source=f"{dept}.pdf",
                )
            ]

        retriever.retrieve = retrieve
        return retriever

    def _pipeline(self, cache, retriever, llm_client):
        return RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            cache_manager=cache,
            reranker=None,
            enable_circuit_breaker=False,
        )

    @pytest.mark.asyncio
    async def test_different_filter_does_not_leak_cached_result(
        self, llm_client, chat_completion
    ):
        _wire_llm(llm_client, chat_completion)
        cache = self._real_cache()
        pipeline = self._pipeline(cache, self._filter_aware_retriever(), llm_client)

        finance = await pipeline.query(
            "salary", top_k=5, use_hybrid=True,
            filter_dict={"dept": "finance"}, rerank=False,
        )
        engineering = await pipeline.query(
            "salary", top_k=5, use_hybrid=True,
            filter_dict={"dept": "engineering"}, rerank=False,
        )

        # Same question/collection/top_k/rerank, but a DIFFERENT filter must
        # NOT return the finance query's cached documents.
        assert finance.sources[0]["document"] == "finance.pdf"
        assert engineering.sources[0]["document"] == "engineering.pdf"

    @pytest.mark.asyncio
    async def test_different_search_mode_does_not_leak_cached_result(
        self, llm_client, chat_completion
    ):
        _wire_llm(llm_client, chat_completion)
        cache = self._real_cache()
        # Retriever that returns a different document per search mode.
        retriever = Mock()

        def retrieve(query, top_k, use_hybrid, filter_dict, collection):
            mode = "hybrid" if use_hybrid else "vector"
            return [
                RetrievalResult(
                    document=f"{mode} result",
                    score=0.9,
                    metadata={"filename": f"{mode}.pdf"},
                    source=f"{mode}.pdf",
                )
            ]

        retriever.retrieve = retrieve
        pipeline = self._pipeline(cache, retriever, llm_client)

        hybrid = await pipeline.query(
            "q", top_k=5, use_hybrid=True, filter_dict=None, rerank=False,
        )
        vector = await pipeline.query(
            "q", top_k=5, use_hybrid=False, filter_dict=None, rerank=False,
        )

        assert hybrid.sources[0]["document"] == "hybrid.pdf"
        assert vector.sources[0]["document"] == "vector.pdf"



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

    def test_confidence_clamped_when_top_score_negative(
        self, retriever, llm_client
    ):
        """A negative top retrieval score must not yield a negative confidence.

        Regression: ``_calculate_confidence`` weighted the raw top score by
        0.5 but only clamped the UPPER bound (``min(confidence, 1.0)``). The
        top score is a FAISS inner-product / cosine similarity over
        L2-normalized vectors, so it ranges over [-1, 1] (vectordb.py uses
        ``IndexFlatIP``). A query whose every retrieved doc is dissimilar
        (cosine < 0) produced a negative top_score, and ``0.5 * negative``
        drove ``confidence`` below zero — flowing a nonsensical negative
        confidence to the API response. The lower-bound clamp mirrors
        ranking.py's final clamp (``max(0.0, min(1.0, ...))``). Before the
        fix this returned -0.295.
        """
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            llm_model="gpt-4",
        )
        negative_results = [
            RetrievalResult(
                document="Dissimilar doc, opposite vector direction.",
                score=-0.6,  # realistic cosine for a dissimilar vector
                metadata={},
                source="doc.pdf",
            ),
        ]
        confidence = pipeline._calculate_confidence(
            negative_results, "short"
        )
        # top_score=-0.6, high_score_count=0, answer_length_factor=0.025:
        # pre-fix min(-0.295, 1.0) = -0.295 (negative!); post-fix -> 0.0.
        assert confidence == 0.0


class TestCollectionLogInjection:
    """CWE-117: a client-controlled ``collection`` must not forge log lines.

    ``collection`` is the ``QueryRequest.collection`` body field; the query
    route passes it straight through to ``RAGPipeline.query``
    (``collection=... or "default"``). The retrieval log line at L243
    interpolates it next to ``question``, so a CR/LF in a collection name
    terminates the real log line and starts a forged one -- the same
    log-injection class as the ``query`` body field (43166fa) and the
    ``X-Request-ID`` header (b88a5c5), but a distinct client-controlled field
    that the question-sweep missed. The validation middleware scans body
    values for XSS/SQL/path/command but NOT control chars, so CR/LF reaches
    this logger unfiltered.
    """

    @pytest.mark.asyncio
    async def test_collection_crlf_neutralised_in_retrieval_log(
        self, retriever, llm_client, chat_completion
    ):
        _wire_llm(llm_client, chat_completion)
        retriever.retrieve.return_value = _results()
        cache = Mock()
        cache.generate_key.return_value = "k"
        cache.get.return_value = None  # miss -> retrieval path -> L243 logs

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            llm_model="gpt-4",
            cache_manager=cache,
        )

        # Capture the rag_pipeline logger's DEBUG records directly (robust to
        # root-handler / level config, unlike caplog propagation).
        rag_logger = logging.getLogger("app.services.rag_pipeline")
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = _Capture(logging.DEBUG)
        rag_logger.addHandler(handler)
        rag_logger.setLevel(logging.DEBUG)
        try:
            await pipeline.query(
                "clean question",  # no control chars -> isolates `collection`
                collection="hr-policies\nFAKE LOG line\r",
            )
        finally:
            rag_logger.removeHandler(handler)

        retrieval_msgs = [
            r.getMessage()
            for r in captured
            if "Retrieving documents for" in r.getMessage()
        ]
        assert retrieval_msgs, "retrieval log line was not emitted"
        msg = retrieval_msgs[0]
        # No raw CR/LF survives -> the forged "FAKE LOG line" cannot start a
        # new log line. Pre-fix the raw chr(10)/chr(13) were present.
        assert "\n" not in msg
        assert "\r" not in msg
        # Escaped form is present -> value preserved, only log rep changed.
        assert "hr-policies\\nFAKE LOG line\\r" in msg
