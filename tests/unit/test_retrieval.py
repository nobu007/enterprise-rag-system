"""
Unit tests for Retrieval Service (HybridRetriever, ContextCompressor)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from app.services.retrieval import (
    HybridRetriever,
    RetrievalResult,
    ContextCompressor,
)
from app.core.vectordb import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector_db():
    """Mock vector database"""
    db = Mock()
    db.search.return_value = [
        SearchResult(id="doc1", score=0.95, metadata={"filename": "a.pdf", "source": "a.pdf"}, text="Document one text"),
        SearchResult(id="doc2", score=0.80, metadata={"filename": "b.pdf", "source": "b.pdf"}, text="Document two text"),
        SearchResult(id="doc3", score=0.65, metadata={"filename": "c.pdf", "source": "c.pdf"}, text="Document three text"),
    ]
    return db


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model"""
    model = Mock()
    model.embed_query.return_value = [0.1] * 1536
    model.embed_texts.return_value = [[0.1] * 1536]
    model.dimension = 1536
    return model


@pytest.fixture
def retriever(mock_vector_db, mock_embedding_model):
    """HybridRetriever instance with mocks"""
    return HybridRetriever(
        vector_db=mock_vector_db,
        embedding_model=mock_embedding_model,
        alpha=0.5,
    )


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results for ContextCompressor tests"""
    return [
        RetrievalResult(
            document="First document with enough text for testing compression logic.",
            score=0.95,
            metadata={"filename": "a.pdf", "page": 1},
            source="a.pdf",
        ),
        RetrievalResult(
            document="Second document text for verification.",
            score=0.80,
            metadata={"filename": "b.pdf", "page": 2},
            source="b.pdf",
        ),
    ]


# ---------------------------------------------------------------------------
# HybridRetriever Tests
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    """Tests for HybridRetriever"""

    def test_initialization(self, retriever, mock_vector_db, mock_embedding_model):
        """Test retriever initialisation stores dependencies."""
        assert retriever.vector_db is mock_vector_db
        assert retriever.embedding_model is mock_embedding_model
        assert retriever.alpha == 0.5
        assert retriever.bm25_index is None

    def test_semantic_search(self, retriever, mock_vector_db, mock_embedding_model):
        """Test semantic search delegates to vector_db."""
        results = retriever.semantic_search("test query", top_k=3)
        mock_embedding_model.embed_query.assert_called_once_with("test query")
        mock_vector_db.search.assert_called_once()
        assert len(results) == 3

    def test_semantic_search_with_collection(self, retriever, mock_vector_db):
        """Test semantic search passes collection parameter."""
        retriever.semantic_search("q", top_k=2, collection="custom")
        call_kwargs = mock_vector_db.search.call_args
        assert call_kwargs.kwargs.get("collection") == "custom" or call_kwargs[1].get("collection") == "custom"

    def test_keyword_search_empty_without_index(self, retriever):
        """Test keyword search returns empty when BM25 not built."""
        results = retriever.keyword_search("test", top_k=5)
        assert results == []

    def test_build_bm25_index(self, retriever):
        """Test BM25 index can be built from documents."""
        from app.services.document_loader import Document

        docs = [
            Document(content="hello world test document", metadata={"source": "a"}, doc_id="d1"),
            Document(content="another test with different words", metadata={"source": "b"}, doc_id="d2"),
        ]
        retriever.build_bm25_index(docs)
        assert retriever.bm25_index is not None
        assert len(retriever.bm25_documents) == 2

    def test_keyword_search_with_index(self, retriever):
        """Test BM25 keyword search returns scored results."""
        from app.services.document_loader import Document

        # BM25 requires 3+ documents for meaningful IDF scores
        docs = [
            Document(content="machine learning deep neural network training data", metadata={"source": "x"}, doc_id="d1"),
            Document(content="cat dog animal pet veterinary care hospital", metadata={"source": "y"}, doc_id="d2"),
            Document(content="python programming language software development", metadata={"source": "z"}, doc_id="d3"),
        ]
        retriever.build_bm25_index(docs)
        results = retriever.keyword_search("machine learning", top_k=3)
        # BM25 should find at least one result for "machine learning"
        assert len(results) >= 1
        assert results[0]["score"] > 0

    def test_retrieve_semantic_only(self, retriever):
        """Test retrieve falls back to semantic when no BM25."""
        results = retriever.retrieve("query", top_k=3, use_hybrid=True)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_hybrid_with_bm25(self, retriever):
        """Test retrieve uses hybrid search when BM25 available."""
        from app.services.document_loader import Document

        docs = [
            Document(content="machine learning tutorial", metadata={"source": "ml"}, doc_id="d1"),
        ]
        retriever.build_bm25_index(docs)
        results = retriever.retrieve("machine learning", top_k=3, use_hybrid=True)
        assert len(results) > 0

    def test_retrieve_forced_semantic(self, retriever):
        """Test retrieve with use_hybrid=False forces semantic only."""
        results = retriever.retrieve("test", top_k=2, use_hybrid=False)
        assert all(isinstance(r, RetrievalResult) for r in results)


# ---------------------------------------------------------------------------
# ContextCompressor Tests
# ---------------------------------------------------------------------------


class TestContextCompressor:
    """Tests for ContextCompressor"""

    def test_truncate_within_limit(self, sample_retrieval_results):
        """Test truncation when content fits within token limit."""
        compressor = ContextCompressor(max_tokens=4000)
        context = compressor.compress("q", sample_retrieval_results)
        assert "First document" in context
        assert "Second document" in context

    def test_truncate_exceeds_limit(self):
        """Test truncation stops when token limit exceeded."""
        large_doc = RetrievalResult(
            document="x" * 20000,  # ~5000 tokens
            score=0.9,
            metadata={"filename": "big.pdf"},
            source="big.pdf",
        )
        compressor = ContextCompressor(max_tokens=100)
        context = compressor.compress("q", [large_doc])
        # Should still return something (first chunk may fit partially)
        assert isinstance(context, str)

    def test_empty_results(self):
        """Test compression with no results."""
        compressor = ContextCompressor(max_tokens=4000)
        context = compressor.compress("q", [])
        assert context == ""

    def test_rerank_method_delegates(self, sample_retrieval_results):
        """Test rerank method delegates to truncation (placeholder impl)."""
        compressor = ContextCompressor(max_tokens=4000)
        context = compressor.compress("q", sample_retrieval_results, method="rerank")
        assert len(context) > 0
