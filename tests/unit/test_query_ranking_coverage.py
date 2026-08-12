"""Coverage-focused tests for the learning-to-rank branches of the
query/batch endpoints in app/api/routes/query.py (L145-155, L248-258).

The ``rank_results`` request flag defaults to False, so the existing
route tests never exercise the QueryResultRanker integration nor its
graceful fallback. These tests POST with rank_results=True (happy
path) and with the ranker forced to raise (fallback path).
"""
import pytest
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_rag_pipeline
from app.api.routes.query import router
from app.services.rag_pipeline import RAGResponse
from app.services.ranking import QueryResultRanker


def _make_response(relevance_score=0.9):
    # Source dict shape mirrors what RAGPipeline.query actually builds
    # (rag_pipeline.py): the score lives under ``relevance_score``, NOT
    # ``score``. The prior mock used ``score``, which QueryResultRanker reads
    # directly -- a faithless mock that masked the field-name mismatch: in
    # production the ranker read 0.0 for every source, so rank_results=True
    # was a silent no-op while this test happily asserted "ranking_score" was
    # present. See test_rank_results_reads_relevance_score_field for the
    # contract that fails pre-fix.
    return RAGResponse(
        answer="answer",
        sources=[{
            "document": "Machine learning basics",
            "relevance_score": relevance_score,
        }],
        confidence=0.8,
        latency_ms=10,
        tokens_used=5,
        retrieval_results=[],
    )


@pytest.fixture
def pipeline():
    """Mocked RAG pipeline (query + batch_query)."""
    p = Mock()
    p.query = AsyncMock(return_value=_make_response())
    p.batch_query = AsyncMock(return_value=[_make_response()])
    return p


@pytest.fixture
def client(pipeline):
    """Test client with a mocked RAG pipeline (no prefix on router)."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides = {}


def _force_rank_failure(monkeypatch):
    """Make QueryResultRanker.rank_results raise (exercise fallback)."""
    def _raise(self, *args, **kwargs):
        raise RuntimeError("forced ranking failure")
    monkeypatch.setattr(QueryResultRanker, "rank_results", _raise)


class TestQueryRankResults:
    """Cover rank_results branch + fallback in /query/ (L145-155)."""

    def test_rank_results_applies_ranking(self, client):
        resp = client.post("/query/", json={
            "query": "what is machine learning",
            "rank_results": True,
            "top_k": 5,
        })
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        assert sources
        assert "ranking_score" in sources[0]
        assert "ranking_features" in sources[0]
        # The ranker must read the score from the prod source-dict field
        # (``relevance_score``). With a single 0.9-relevance source,
        # max_score=0.9 so semantic_score == 0.9/0.9 == 1.0. Pre-fix (reading
        # the absent ``score`` key) this was 0.0 and rank_results=True was a
        # silent no-op.
        assert sources[0]["ranking_features"]["semantic_score"] == pytest.approx(1.0)

    def test_rank_results_fallback_on_error(self, client, monkeypatch):
        _force_rank_failure(monkeypatch)
        resp = client.post("/query/", json={
            "query": "what is machine learning",
            "rank_results": True,
            "top_k": 5,
        })
        # Graceful fallback -> 200, original sources retained
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        assert sources
        assert "ranking_score" not in sources[0]


class TestBatchQueryRankResults:
    """Cover rank_results branch + fallback in /query/batch (L248-258)."""

    def test_batch_rank_results_applies_ranking(self, client):
        resp = client.post("/query/batch", json={
            "queries": ["what is machine learning"],
            "rank_results": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body
        assert "ranking_score" in body[0]["sources"][0]
        # See test_rank_results_applies_ranking: the ranker must read the prod
        # ``relevance_score`` field, not the absent ``score`` key.
        assert body[0]["sources"][0]["ranking_features"]["semantic_score"] == pytest.approx(1.0)

    def test_batch_rank_results_fallback_on_error(self, client, monkeypatch):
        _force_rank_failure(monkeypatch)
        resp = client.post("/query/batch", json={
            "queries": ["what is machine learning"],
            "rank_results": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body
        assert "ranking_score" not in body[0]["sources"][0]


class TestRankResultsReadsRelevanceScore:
    """Contract: ``rank_results=True`` must actually REORDER prod-shaped
    source dicts, whose score lives under ``relevance_score`` (rag_pipeline.py)
    -- not the ``score`` key the ranker used to read alone. Pre-fix every
    source tied at an identical ranking_score (semantic_score 0.0 for all), so
    the stable sort left the original order intact: a silent no-op that only
    attached ``ranking_score``/``ranking_features`` and logged "Applied
    learning-to-rank". This reorder test fails pre-fix and passes post-fix."""

    def test_query_reorders_by_relevance_score(self, client, pipeline):
        # Lower-relevance doc listed FIRST; a working ranker must bubble the
        # higher-relevance doc to the top.
        pipeline.query = AsyncMock(return_value=RAGResponse(
            answer="answer",
            sources=[
                {"document": "low relevance", "relevance_score": 0.5},
                {"document": "high relevance", "relevance_score": 0.9},
            ],
            confidence=0.8,
            latency_ms=10,
            tokens_used=5,
            retrieval_results=[],
        ))
        resp = client.post("/query/", json={
            "query": "machine learning",
            "rank_results": True,
            "top_k": 5,
        })
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        # High-relevance doc ranked first (would remain second pre-fix, where
        # the absent ``score`` key made every semantic_score 0.0 -> tie).
        assert sources[0]["document"] == "high relevance"
        assert sources[0]["ranking_features"]["semantic_score"] == pytest.approx(1.0)
        assert sources[1]["ranking_features"]["semantic_score"] == pytest.approx(0.5 / 0.9)


class TestQueryEndpointErrorPath:
    """Cover the outer 500 handlers (L165-166, L271-272)."""

    def test_query_returns_500_on_pipeline_error(self, client, pipeline):
        pipeline.query.side_effect = RuntimeError("pipeline down")
        resp = client.post("/query/", json={"query": "x"})
        assert resp.status_code == 500
        assert "Query failed" in resp.json()["detail"]

    def test_batch_returns_500_on_pipeline_error(self, client, pipeline):
        pipeline.batch_query.side_effect = RuntimeError("pipeline down")
        resp = client.post("/query/batch", json={"queries": ["x"]})
        assert resp.status_code == 500
        assert "Batch query failed" in resp.json()["detail"]
