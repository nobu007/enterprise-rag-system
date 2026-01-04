"""
Unit tests for Query API Routes

Tests for the /query endpoints including validation, error handling,
and response format verification.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.api.routes.query import router, QueryRequest, QueryResponse
from app.services.rag_pipeline import RAGResponse


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

    # Mock the get_rag_pipeline function - patch at import location
    with patch('app.main.get_rag_pipeline', return_value=mock_rag_pipeline):
        # Set up the default return value for query method
        mock_rag_pipeline.query.return_value = sample_rag_response
        mock_rag_pipeline.batch_query.return_value = [sample_rag_response]
        yield TestClient(app)


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
            filter_dict=None
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
            filter_dict={"category": "tech", "year": 2024}
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
            top_k=5
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
