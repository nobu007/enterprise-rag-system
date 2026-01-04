"""
Unit tests for Reranker service
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys


# Create mock module before importing
mock_sentence_transformers = MagicMock()
mock_sentence_transformers.CrossEncoder = MagicMock
sys.modules['sentence_transformers'] = mock_sentence_transformers

from app.services.reranker import Reranker


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder model"""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.7, 0.3, 0.5, 0.8]
    return mock_model


@pytest.fixture
def reranker(mock_cross_encoder):
    """Create a Reranker instance with mocked model"""
    with patch.object(mock_sentence_transformers, 'CrossEncoder', return_value=mock_cross_encoder):
        reranker = Reranker(model_name="test-model")
        yield reranker


class TestRerankerInit:
    """Tests for Reranker initialization"""

    def test_init_with_default_model(self):
        """Test initialization with default model"""
        mock_model = MagicMock()
        with patch.object(mock_sentence_transformers, 'CrossEncoder', return_value=mock_model):
            reranker = Reranker()

            assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
            assert reranker.model == mock_model

    def test_init_with_custom_model(self):
        """Test initialization with custom model"""
        mock_model = MagicMock()
        with patch.object(mock_sentence_transformers, 'CrossEncoder', return_value=mock_model):
            reranker = Reranker(model_name="custom-model")

            assert reranker.model_name == "custom-model"
            assert reranker.model == mock_model

    def test_init_with_env_var(self):
        """Test initialization with environment variable"""
        mock_model = MagicMock()
        with patch.dict('os.environ', {'RERANKER_MODEL': 'env-model'}):
            with patch.object(mock_sentence_transformers, 'CrossEncoder', return_value=mock_model):
                reranker = Reranker()

                assert reranker.model_name == "env-model"

    def test_init_import_error(self):
        """Test initialization fails gracefully without sentence-transformers"""
        with patch.object(mock_sentence_transformers, 'CrossEncoder', side_effect=ImportError):
            with pytest.raises(ImportError) as exc_info:
                Reranker()

            assert "sentence-transformers" in str(exc_info.value)


class TestRerank:
    """Tests for rerank method"""

    def test_rerank_basic(self, reranker, mock_cross_encoder):
        """Test basic reranking functionality"""
        query = "test query"
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        result = reranker.rerank(query, documents)

        # Verify model.predict was called with correct pairs
        mock_cross_encoder.predict.assert_called_once()
        call_args = mock_cross_encoder.predict.call_args[0][0]
        assert len(call_args) == 5
        assert all(pair[0] == query for pair in call_args)

        # Verify results are sorted correctly
        assert len(result) == 5
        # Mock scores: [0.9, 0.7, 0.3, 0.5, 0.8]
        # Sorted indices: [0, 4, 1, 3, 2]
        assert result[0] == (0, 0.9)  # doc1 has highest score
        assert result[1] == (4, 0.8)  # doc5 has second highest
        assert result[2] == (1, 0.7)
        assert result[3] == (3, 0.5)
        assert result[4] == (2, 0.3)

    def test_rerank_with_top_k(self, reranker, mock_cross_encoder):
        """Test reranking with top_k limit"""
        query = "test query"
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        result = reranker.rerank(query, documents, top_k=3)

        assert len(result) == 3
        assert result[0] == (0, 0.9)
        assert result[1] == (4, 0.8)
        assert result[2] == (1, 0.7)

    def test_rerank_empty_documents(self, reranker):
        """Test reranking with empty document list"""
        result = reranker.rerank("test query", [])

        assert result == []

    def test_rerank_empty_query(self, reranker):
        """Test reranking with empty query"""
        documents = ["doc1", "doc2", "doc3"]
        result = reranker.rerank("", documents)

        # Should return original order with zero scores
        assert len(result) == 3
        assert result == [(0, 0.0), (1, 0.0), (2, 0.0)]

    def test_rerank_model_error(self, reranker, mock_cross_encoder):
        """Test reranking handles model errors gracefully"""
        mock_cross_encoder.predict.side_effect = Exception("Model error")

        query = "test query"
        documents = ["doc1", "doc2", "doc3"]

        result = reranker.rerank(query, documents)

        # Should return default scores on error
        assert result == [(0, 0.0), (1, 0.0), (2, 0.0)]


class TestRerankResults:
    """Tests for rerank_results method"""

    def test_rerank_results_basic(self):
        """Test reranking retrieval results"""
        from app.services.retrieval import RetrievalResult

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.3]

        with patch.object(mock_sentence_transformers, 'CrossEncoder', return_value=mock_model):
            reranker = Reranker(model_name="test-model")

            query = "test query"
            results = [
                RetrievalResult(document="doc1", score=0.5, metadata={}, source="test"),
                RetrievalResult(document="doc2", score=0.6, metadata={}, source="test"),
                RetrievalResult(document="doc3", score=0.4, metadata={}, source="test"),
            ]

            reranked = reranker.rerank_results(query, results, top_k=2)

            assert len(reranked) == 2
            # Mock scores: [0.9, 0.7, 0.3]
            # Top 2: doc1 (index 0), doc2 (index 1)
            assert reranked[0].document == "doc1"
            assert reranked[1].document == "doc2"

    def test_rerank_results_empty(self, reranker):
        """Test reranking empty results list"""
        result = reranker.rerank_results("test query", [])

        assert result == []

    def test_rerank_results_custom_attr(self):
        """Test reranking with custom document attribute"""
        class CustomResult:
            def __init__(self, text):
                self.text = text

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8]

        with patch.object(mock_sentence_transformers, 'CrossEncoder', return_value=mock_model):
            reranker = Reranker(model_name="test-model")

            query = "test query"
            results = [
                CustomResult("doc1"),
                CustomResult("doc2"),
            ]

            reranked = reranker.rerank_results(query, results, doc_text_attr="text")

            assert len(reranked) == 2
            assert reranked[0].text == "doc1"
            assert reranked[1].text == "doc2"

    def test_rerank_results_invalid_attr(self, reranker):
        """Test reranking with invalid attribute returns original results"""
        class CustomResult:
            def __init__(self, text):
                self.text = text

        query = "test query"
        results = [CustomResult("doc1"), CustomResult("doc2")]

        reranked = reranker.rerank_results(query, results, doc_text_attr="invalid_attr")

        # Should return original results on error
        assert len(reranked) == 2
        assert reranked == results
