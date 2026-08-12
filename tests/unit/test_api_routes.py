"""
Unit tests for Query API Routes

Tests for the /query endpoints including validation, error handling,
and response format verification.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient

from app.api.routes.query import router, QueryRequest, QueryResponse
from app.services.rag_pipeline import RAGResponse
from app.api.dependencies import get_rag_pipeline


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline instance"""
    pipeline = Mock()
    # Set up async mocks
    pipeline.query = AsyncMock()
    pipeline.batch_query = AsyncMock()
    return pipeline


@pytest.fixture
def sample_rag_response():
    """Sample RAG response for testing"""
    return RAGResponse(
        answer="This is a test answer based on the context.",
        sources=[
            {
                'index': 1,
                'document': 'test1.pdf',
                'page': 1,
                'relevance_score': 0.85,
                'text_preview': 'Sample document text...'
            }
        ],
        confidence=0.82,
        latency_ms=150,
        tokens_used=100,
        retrieval_results=[]
    )


@pytest.fixture
def client(mock_rag_pipeline, sample_rag_response):
    """Test client with mocked dependencies"""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    # Set up the default return value for query method
    mock_rag_pipeline.query.return_value = sample_rag_response
    mock_rag_pipeline.batch_query.return_value = [sample_rag_response]

    # Override the dependency
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag_pipeline

    yield TestClient(app)

    # Clean up
    app.dependency_overrides = {}


class TestQueryRequestValidation:
    """Test QueryRequest validation"""

    def test_valid_query_request(self):
        """Test valid query request creation"""
        request = QueryRequest(
            query="What is machine learning?",
            top_k=5,
            use_hybrid=True
        )
        assert request.query == "What is machine learning?"
        assert request.top_k == 5
        assert request.use_hybrid is True

    def test_query_request_min_length_validation(self):
        """Test that empty query is rejected"""
        with pytest.raises(Exception):
            QueryRequest(query="")

    def test_query_request_max_length_validation(self):
        """Test that overly long query is rejected.

        StreamingQueryRequest caps query at 1000 chars (max_length field +
        validate_stream_request's ``len(query) > 1000`` check) and both
        endpoint docstrings promise "1-1000 characters". The non-streaming
        QueryRequest must enforce the same cap so /query rejects an oversized
        query at the boundary instead of accepting unbounded text that would
        blow up embedding cost / the LLM context window.
        """
        # Boundary: exactly 1000 chars is valid
        QueryRequest(query="a" * 1000)
        # 1001 chars is rejected
        with pytest.raises(Exception):
            QueryRequest(query="a" * 1001)

    def test_query_request_top_k_bounds(self):
        """Test top_k bounds validation"""
        # Valid range: 1-20
        QueryRequest(query="Test?", top_k=1)
        QueryRequest(query="Test?", top_k=20)

        # Out of bounds
        with pytest.raises(Exception):
            QueryRequest(query="Test?", top_k=0)

        with pytest.raises(Exception):
            QueryRequest(query="Test?", top_k=21)

    def test_query_request_with_optional_fields(self):
        """Test query request with optional fields"""
        request = QueryRequest(
            query="Test query",
            collection="test_collection",
            top_k=10,
            use_hybrid=False,
            filters={"category": "tech"}
        )
        assert request.collection == "test_collection"
        assert request.use_hybrid is False
        assert request.filters == {"category": "tech"}

    def test_query_request_collection_max_length(self):
        """Collection name is capped at 1000 chars (sibling of query max_length).

        ``collection`` is the other client-controlled body string carried by
        all three request models. It reaches Prometheus labels
        (``cache_hits.labels(collection=...)`` etc.) as a raw value and is
        interpolated into retrieval log lines, so without a cap an oversized
        name bloats metric label values / log output. Mirrors the query cap
        (1000) established across QueryRequest / StreamingQueryRequest.
        """
        # Boundary: exactly 1000 chars is valid
        req = QueryRequest(query="q", collection="c" * 1000)
        assert req.collection == "c" * 1000
        # None still accepted (Optional)
        QueryRequest(query="q", collection=None)
        # 1001 chars is rejected
        with pytest.raises(Exception):
            QueryRequest(query="q", collection="c" * 1001)


class TestQueryEndpoint:
    """Test POST /query/ endpoint"""

    def test_query_endpoint_success(self, client, mock_rag_pipeline, sample_rag_response):
        """Test successful query execution"""
        mock_rag_pipeline.query.return_value = sample_rag_response

        response = client.post(
            "/query/",
            json={
                "query": "What is machine learning?",
                "top_k": 5,
                "use_hybrid": True
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == sample_rag_response.answer
        assert data["confidence"] == sample_rag_response.confidence
        assert data["latency_ms"] == sample_rag_response.latency_ms
        assert data["tokens_used"] == sample_rag_response.tokens_used
        assert len(data["sources"]) == 1

        # Verify pipeline was called correctly
        mock_rag_pipeline.query.assert_called_once_with(
            question="What is machine learning?",
            top_k=5,
            use_hybrid=True,
            filter_dict=None,
            rerank=True,
            collection='default'
        )

    def test_query_endpoint_with_filters(self, client, mock_rag_pipeline, sample_rag_response):
        """Test query with metadata filters"""
        mock_rag_pipeline.query.return_value = sample_rag_response

        response = client.post(
            "/query/",
            json={
                "query": "What is AI?",
                "top_k": 5,
                "filters": {"category": "tech", "year": 2024}
            }
        )

        assert response.status_code == 200
        mock_rag_pipeline.query.assert_called_once_with(
            question="What is AI?",
            top_k=5,
            use_hybrid=True,
            filter_dict={"category": "tech", "year": 2024},
            rerank=True,
            collection='default'
        )

    def test_query_endpoint_error_handling(self, client, mock_rag_pipeline):
        """Test query endpoint error handling"""
        # Simulate pipeline error
        mock_rag_pipeline.query.side_effect = Exception("Database connection failed")

        response = client.post(
            "/query/",
            json={
                "query": "What is machine learning?",
                "top_k": 5
            }
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Query failed" in data["detail"]

    def test_query_endpoint_collection_too_long_rejected(self, client):
        """An oversized collection name is rejected at the /query boundary (422).

        Mirrors the query max_length cap; collection reaches Prometheus labels
        and logs as a raw value, so cap it at the boundary.
        """
        response = client.post(
            "/query/",
            json={"query": "q", "collection": "c" * 1001}
        )
        assert response.status_code == 422


class TestBatchQueryEndpoint:
    """Test POST /query/batch endpoint"""

    def test_batch_query_success(self, client, mock_rag_pipeline, sample_rag_response):
        """Test successful batch query"""
        mock_rag_pipeline.batch_query.return_value = [
            sample_rag_response,
            sample_rag_response,
            sample_rag_response
        ]

        response = client.post(
            "/query/batch",
            json={
                "queries": [
                    "What is machine learning?",
                    "What is deep learning?",
                    "What is NLP?"
                ],
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert all("answer" in item for item in data)
        assert all("sources" in item for item in data)

        mock_rag_pipeline.batch_query.assert_called_once_with(
            questions=[
                "What is machine learning?",
                "What is deep learning?",
                "What is NLP?"
            ],
            top_k=5,
            collection='default'
        )

    def test_batch_query_empty_list(self, client, mock_rag_pipeline):
        """Test batch query with empty query list"""
        mock_rag_pipeline.batch_query.return_value = []

        response = client.post(
            "/query/batch",
            json={
                "queries": [],
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0

    def test_batch_query_with_partial_failure(self, client, mock_rag_pipeline, sample_rag_response):
        """Test batch query where some queries fail"""
        # Mix of successful and failed responses
        mock_rag_pipeline.batch_query.return_value = [
            sample_rag_response,
            RAGResponse(
                answer="Error: Invalid query",
                sources=[],
                confidence=0.0,
                latency_ms=0,
                tokens_used=0,
                retrieval_results=[]
            )
        ]

        response = client.post(
            "/query/batch",
            json={
                "queries": ["Valid query", "Invalid query"],
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_batch_query_validation(self, client):
        """Test batch query request validation"""
        # Missing required field 'queries'
        response = client.post(
            "/query/batch",
            json={"top_k": 5}
        )

        assert response.status_code == 422  # Validation error

    def test_batch_query_per_item_length_rejected(self, client):
        """An oversized individual batch query is rejected at the boundary.

        Mirrors the single-query max_length=1000 cap (QueryRequest /
        StreamingQueryRequest.query); without it one batch item can carry
        an arbitrarily long query -> unbounded embedding / LLM cost.
        """
        response = client.post(
            "/query/batch",
            json={"queries": ["ok", "a" * 1001], "top_k": 5}
        )

        assert response.status_code == 422  # per-item max_length exceeded

    def test_batch_query_per_item_max_length_boundary_accepted(
        self, client, mock_rag_pipeline, sample_rag_response
    ):
        """A batch item of exactly 1000 chars is accepted (boundary value)."""
        mock_rag_pipeline.batch_query.return_value = [sample_rag_response]

        response = client.post(
            "/query/batch",
            json={"queries": ["a" * 1000], "top_k": 5}
        )

        assert response.status_code == 200

    def test_batch_query_collection_too_long_rejected(self, client):
        """An oversized collection name is rejected at the /query/batch boundary (422)."""
        response = client.post(
            "/query/batch",
            json={"queries": ["q"], "collection": "c" * 1001}
        )
        assert response.status_code == 422


class TestHealthEndpoint:
    """Test GET /query/health endpoint"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/query/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "RAG Query API"


class TestResponseModels:
    """Test response model serialization"""

    def test_query_response_serialization(self, sample_rag_response):
        """Test QueryResponse can be properly serialized"""
        response = QueryResponse(
            answer=sample_rag_response.answer,
            sources=sample_rag_response.sources,
            confidence=sample_rag_response.confidence,
            latency_ms=sample_rag_response.latency_ms,
            tokens_used=sample_rag_response.tokens_used
        )

        assert response.answer is not None
        assert response.sources is not None
        assert isinstance(response.confidence, float)
        assert isinstance(response.latency_ms, int)
        assert isinstance(response.tokens_used, int)

    def test_batch_query_request_validation(self):
        """Test BatchQueryRequest validation"""
        from app.api.routes.query import BatchQueryRequest

        # Valid request
        request = BatchQueryRequest(
            queries=["Query 1", "Query 2"],
            top_k=10
        )
        assert len(request.queries) == 2
        assert request.top_k == 10

        # Test top_k bounds
        with pytest.raises(Exception):
            BatchQueryRequest(queries=["Test"], top_k=0)

        with pytest.raises(Exception):
            BatchQueryRequest(queries=["Test"], top_k=21)

    def test_batch_query_request_per_item_max_length(self):
        """Per-item query capped at 1000 (sibling of QueryRequest.query)."""
        from app.api.routes.query import BatchQueryRequest
        from pydantic import ValidationError

        # Boundary accepted: exactly 1000 chars per item, multiple items
        req = BatchQueryRequest(queries=["a" * 1000, "b" * 1000])
        assert len(req.queries) == 2

        # One oversized item among several is still rejected
        with pytest.raises(ValidationError):
            BatchQueryRequest(queries=["ok", "a" * 1001])

        # Single oversized item rejected
        with pytest.raises(ValidationError):
            BatchQueryRequest(queries=["a" * 1001])

    def test_batch_query_request_collection_max_length(self):
        """Collection capped at 1000 on BatchQueryRequest (sibling of per-item query cap)."""
        from app.api.routes.query import BatchQueryRequest
        from pydantic import ValidationError

        # Boundary accepted
        req = BatchQueryRequest(queries=["q"], collection="c" * 1000)
        assert req.collection == "c" * 1000
        # None accepted (Optional)
        BatchQueryRequest(queries=["q"], collection=None)
        # Oversized rejected
        with pytest.raises(ValidationError):
            BatchQueryRequest(queries=["q"], collection="c" * 1001)

    def test_batch_query_request_list_size_cap(self):
        """List size capped at 100: pipeline.batch_query fans each item out to a
        full pipeline.query with no internal bound, and the route limiter is
        per-request -- without this cap one body of many short queries bypasses
        per-query rate limiting (cost/DoS fan-out)."""
        from app.api.routes.query import BatchQueryRequest
        from pydantic import ValidationError

        # Boundary accepted: exactly 100 items, each at the per-item char cap
        req = BatchQueryRequest(queries=["q" * 1000] * 100)
        assert len(req.queries) == 100

        # One item over the cap is rejected at the boundary (422 in the API)
        with pytest.raises(ValidationError):
            BatchQueryRequest(queries=["q"] * 101)

    def test_streaming_query_request_collection_max_length(self):
        """Collection capped at 1000 on StreamingQueryRequest (sibling of query cap)."""
        from app.api.routes.query import StreamingQueryRequest
        from pydantic import ValidationError

        # Boundary accepted
        StreamingQueryRequest(query="q", collection="c" * 1000)
        # None accepted (Optional)
        StreamingQueryRequest(query="q", collection=None)
        # Oversized rejected
        with pytest.raises(ValidationError):
            StreamingQueryRequest(query="q", collection="c" * 1001)
