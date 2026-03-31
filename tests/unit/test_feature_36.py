"""
Unit tests for Query Result Ranking (Feature 36)

Tests cover:
- Ranker initialization and configuration
- Feature extraction from search results
- Ranking score calculation
- Result ranking and reordering
- Edge cases and error handling
- Integration with RAG results
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import sys


# Mock numpy before importing
sys.modules['numpy'] = MagicMock()

from app.services.ranking import (
    QueryResultRanker,
    get_ranker,
    reset_ranker
)


@pytest.fixture
def sample_results():
    """Create sample search results for testing"""
    return [
        {
            'score': 0.85,
            'document': 'Document about machine learning and AI',
            'metadata': {
                'created_at': '2024-01-15',
                'view_count': 150,
                'freshness': 0.8
            },
            'source': 'test'
        },
        {
            'score': 0.65,
            'document': 'Short text about data science',
            'metadata': {
                'created_at': '2024-02-01',
                'view_count': 300,
                'freshness': 0.9
            },
            'source': 'test',
            'doc_length': 50
        },
        {
            'score': 0.75,
            'document': 'A very long document with extensive content about deep learning neural networks and their applications in various fields including computer vision natural language processing and reinforcement learning',
            'metadata': {
                'created_at': '2023-12-01',
                'view_count': 50,
                'freshness': 0.3
            },
            'source': 'test',
            'doc_length': 500
        },
        {
            'score': 0.90,
            'document': 'Medium length document on neural networks',
            'metadata': {
                'created_at': '2024-01-20',
                'view_count': 200,
                'freshness': 0.7
            },
            'source': 'test',
            'doc_length': 150
        }
    ]


class TestQueryResultRankerInit:
    """Tests for QueryResultRanker initialization"""

    def test_init_with_default_weights(self):
        """Test initialization with default weights"""
        ranker = QueryResultRanker()

        assert ranker.semantic_weight == 0.4
        assert ranker.keyword_weight == 0.3
        assert ranker.freshness_weight == 0.1
        assert ranker.popularity_weight == 0.2

    def test_init_with_custom_weights(self):
        """Test initialization with custom weights"""
        ranker = QueryResultRanker(
            semantic_weight=0.5,
            keyword_weight=0.2,
            freshness_weight=0.15,
            popularity_weight=0.15
        )

        assert ranker.semantic_weight == 0.5
        assert ranker.keyword_weight == 0.2
        assert ranker.freshness_weight == 0.15
        assert ranker.popularity_weight == 0.15

    def test_init_normalizes_weights(self):
        """Test that weights are normalized if they don't sum to 1.0"""
        ranker = QueryResultRanker(
            semantic_weight=0.8,
            keyword_weight=0.8,
            freshness_weight=0.2,
            popularity_weight=0.2
        )

        # Should be normalized to sum to ~1.0
        total = (
            ranker.semantic_weight +
            ranker.keyword_weight +
            ranker.freshness_weight +
            ranker.popularity_weight
        )
        assert 0.99 <= total <= 1.01


class TestExtractFeatures:
    """Tests for feature extraction"""

    def test_extract_basic_features(self, sample_results):
        """Test basic feature extraction"""
        ranker = QueryResultRanker()

        features = ranker.extract_features(
            sample_results[0],
            query_length=5
        )

        assert 'semantic_score' in features
        assert 'keyword_score' in features
        assert 'length_penalty' in features
        assert 'freshness' in features
        assert 'popularity' in features
        assert 'ql_ratio' in features

    def test_extract_features_with_metadata(self, sample_results):
        """Test feature extraction with full metadata"""
        ranker = QueryResultRanker()

        features = ranker.extract_features(
            sample_results[1],
            query_length=3
        )

        # Check semantic score normalization
        assert 0.0 <= features['semantic_score'] <= 1.0

        # Check freshness is calculated from date (not from metadata value)
        # With date in 2024, it will be somewhat old but still valid
        assert 0.0 <= features['freshness'] <= 1.0

        # Check popularity (view_count / 100)
        assert features['popularity'] == 3.0  # 300 / 100

    def test_extract_features_without_metadata(self):
        """Test feature extraction with minimal metadata"""
        ranker = QueryResultRanker()

        result = {
            'score': 0.7,
            'document': 'Test document',
            'metadata': {}
        }

        features = ranker.extract_features(
            result,
            query_length=2
        )

        # Should have default values
        assert features['freshness'] == 0.5
        assert features['popularity'] == 0.0

    def test_extract_features_length_penalty(self):
        """Test length penalty calculation"""
        ranker = QueryResultRanker()

        # Short document
        short_result = {'score': 0.5, 'document': 'Short', 'doc_length': 10}
        short_features = ranker.extract_features(short_result, query_length=2)
        assert short_features['length_penalty'] > 0.9

        # Long document
        long_result = {'score': 0.5, 'document': 'X' * 2000, 'doc_length': 2000}
        long_features = ranker.extract_features(long_result, query_length=2)
        assert long_features['length_penalty'] < 0.5


class TestCalculateRankingScore:
    """Tests for ranking score calculation"""

    def test_calculate_score_basic(self):
        """Test basic score calculation"""
        ranker = QueryResultRanker()

        features = {
            'semantic_score': 0.8,
            'keyword_score': 0.7,
            'freshness': 0.9,
            'popularity': 0.5,
            'length_penalty': 1.0
        }

        score = ranker.calculate_ranking_score(features)

        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0

        # Should be high with good features
        assert score > 0.5

    def test_calculate_score_with_length_penalty(self):
        """Test that length penalty affects score"""
        ranker = QueryResultRanker()

        features_high = {
            'semantic_score': 0.8,
            'keyword_score': 0.7,
            'freshness': 0.9,
            'popularity': 0.5,
            'length_penalty': 1.0  # No penalty
        }

        features_low = {
            'semantic_score': 0.8,
            'keyword_score': 0.7,
            'freshness': 0.9,
            'popularity': 0.5,
            'length_penalty': 0.5  # High penalty
        }

        score_high = ranker.calculate_ranking_score(features_high)
        score_low = ranker.calculate_ranking_score(features_low)

        assert score_high > score_low

    def test_calculate_score_clamps_to_range(self):
        """Test that scores are clamped to [0, 1] range"""
        ranker = QueryResultRanker()

        # Very high features
        features_high = {
            'semantic_score': 2.0,
            'keyword_score': 2.0,
            'freshness': 2.0,
            'popularity': 2.0,
            'length_penalty': 1.0
        }

        # Very low/negative features
        features_low = {
            'semantic_score': -1.0,
            'keyword_score': -1.0,
            'freshness': -1.0,
            'popularity': -1.0,
            'length_penalty': 1.0
        }

        score_high = ranker.calculate_ranking_score(features_high)
        score_low = ranker.calculate_ranking_score(features_low)

        assert 0.0 <= score_high <= 1.0
        assert 0.0 <= score_low <= 1.0


class TestRankResults:
    """Tests for result ranking"""

    def test_rank_results_basic(self, sample_results):
        """Test basic result ranking"""
        ranker = QueryResultRanker()

        ranked = ranker.rank_results(
            query="machine learning",
            results=sample_results
        )

        # Should return all results
        assert len(ranked) == len(sample_results)

        # Results should have ranking metadata
        assert 'ranking_score' in ranked[0]
        assert 'ranking_features' in ranked[0]

        # Should be sorted by ranking_score (descending)
        scores = [r['ranking_score'] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_results_with_top_k(self, sample_results):
        """Test ranking with top_k limit"""
        ranker = QueryResultRanker()

        ranked = ranker.rank_results(
            query="test",
            results=sample_results,
            top_k=2
        )

        # Should only return top 2
        assert len(ranked) == 2

        # Top 2 should have highest scores
        scores = [r['ranking_score'] for r in ranked]
        # Scores should be sorted in descending order
        assert scores[0] >= scores[1]

        # All returned results should have ranking metadata
        assert all('ranking_score' in r for r in ranked)

    def test_rank_results_empty(self):
        """Test ranking with empty results"""
        ranker = QueryResultRanker()

        ranked = ranker.rank_results(
            query="test",
            results=[]
        )

        assert ranked == []

    def test_rank_results_preserves_content(self, sample_results):
        """Test that ranking preserves original content"""
        ranker = QueryResultRanker()

        ranked = ranker.rank_results(
            query="test",
            results=sample_results
        )

        # All original content should be in ranked results
        original_docs = {r['document'] for r in sample_results}
        ranked_docs = {r['document'] for r in ranked}
        assert original_docs == ranked_docs

        # Original content should be preserved (just reordered)
        for ranked_item in ranked:
            # Find the original result
            original = next(
                (r for r in sample_results if r['document'] == ranked_item['document']),
                None
            )
            assert original is not None
            assert ranked_item['document'] == original['document']
            assert ranked_item['source'] == original['source']
            assert ranked_item['metadata'] == original['metadata']

    def test_rank_results_handles_errors(self, sample_results):
        """Test that ranking handles individual result errors gracefully"""
        ranker = QueryResultRanker()

        # Add a malformed result (dict with missing required fields)
        bad_results = sample_results + [{'invalid': 'result'}]

        # Should not raise exception
        ranked = ranker.rank_results(
            query="test",
            results=bad_results
        )

        # Should have at least the valid results
        assert len(ranked) >= len(sample_results)

        # All ranked results should have ranking_score
        assert all('ranking_score' in r for r in ranked)


class TestRankRagResults:
    """Tests for RAG-specific result ranking"""

    def test_rank_rag_results_basic(self):
        """Test ranking RAG pipeline results"""
        # Create mock RetrievalResult objects
        class MockRetrievalResult:
            def __init__(self, document, score, metadata, source):
                self.document = document
                self.score = score
                self.metadata = metadata
                self.source = source

        results = [
            MockRetrievalResult(
                document="Doc 1",
                score=0.7,
                metadata={'view_count': 100},
                source="test"
            ),
            MockRetrievalResult(
                document="Doc 2",
                score=0.9,
                metadata={'view_count': 50},
                source="test"
            ),
        ]

        ranker = QueryResultRanker()
        ranked = ranker.rank_rag_results(
            query="test query",
            results=results
        )

        # Should return same type
        assert len(ranked) == 2
        assert hasattr(ranked[0], 'document')
        assert hasattr(ranked[0], 'score')

        # Scores should be updated
        assert ranked[0].score != results[0].score or \
               ranked[1].score != results[1].score

    def test_rank_rag_results_empty(self):
        """Test ranking empty RAG results"""
        ranker = QueryResultRanker()

        ranked = ranker.rank_rag_results(
            query="test",
            results=[]
        )

        assert ranked == []


class TestWeightManagement:
    """Tests for weight management methods"""

    def test_get_weights(self):
        """Test getting current weights"""
        ranker = QueryResultRanker(
            semantic_weight=0.5,
            keyword_weight=0.3,
            freshness_weight=0.1,
            popularity_weight=0.1
        )

        weights = ranker.get_weights()

        assert weights['semantic'] == 0.5
        assert weights['keyword'] == 0.3
        assert weights['freshness'] == 0.1
        assert weights['popularity'] == 0.1

    def test_update_weights(self):
        """Test updating weights"""
        ranker = QueryResultRanker()

        ranker.update_weights({
            'semantic': 0.6,
            'keyword': 0.2
        })

        weights = ranker.get_weights()

        assert weights['semantic'] == 0.6
        assert weights['keyword'] == 0.2
        # Other weights should remain unchanged
        assert weights['freshness'] == 0.1
        assert weights['popularity'] == 0.2


class TestGlobalRanker:
    """Tests for global ranker instance management"""

    def test_get_ranker_singleton(self):
        """Test that get_ranker returns singleton instance"""
        reset_ranker()

        ranker1 = get_ranker()
        ranker2 = get_ranker()

        assert ranker1 is ranker2

    def test_reset_ranker(self):
        """Test resetting the global ranker"""
        ranker1 = get_ranker()

        reset_ranker()

        ranker2 = get_ranker()

        # Should be different instances after reset
        assert ranker1 is not ranker2


class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_rank_results_with_zero_max_score(self):
        """Test ranking when all scores are zero"""
        ranker = QueryResultRanker()

        results = [
            {'score': 0.0, 'document': 'Doc 1', 'metadata': {}},
            {'score': 0.0, 'document': 'Doc 2', 'metadata': {}},
        ]

        ranked = ranker.rank_results(
            query="test",
            results=results
        )

        # Should handle gracefully
        assert len(ranked) == 2
        assert all('ranking_score' in r for r in ranked)

    def test_rank_results_with_missing_fields(self):
        """Test ranking with missing optional fields"""
        ranker = QueryResultRanker()

        results = [
            {'score': 0.5, 'document': 'Doc'},  # Minimal result
        ]

        # Should not raise exception
        ranked = ranker.rank_results(
            query="test",
            results=results
        )

        assert len(ranked) == 1

    def test_extract_features_with_zero_query_length(self):
        """Test feature extraction with zero query length"""
        ranker = QueryResultRanker()

        result = {
            'score': 0.5,
            'document': 'Test document',
            'doc_length': 100,
            'metadata': {}
        }

        features = ranker.extract_features(
            result,
            query_length=0
        )

        # Should handle gracefully
        assert 'ql_ratio' in features
        assert features['ql_ratio'] == 0.0
