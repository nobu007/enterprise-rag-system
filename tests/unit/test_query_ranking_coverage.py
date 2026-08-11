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


def _make_response(score=0.9):
    return RAGResponse(
        answer="answer",
        sources=[{
            "document": "Machine learning basics",
            "score": score,
            "metadata": {
                "view_count": 100,
                "created_at": "2024-01-01",
            },
            "source": "test",
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
