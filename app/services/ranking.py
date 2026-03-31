"""
Learning-to-Rank Service for Query Result Optimization

This module implements machine learning-based ranking for optimizing
the order of query results in RAG systems.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class QueryResultRanker:
    """
    Learning-to-rank service for optimizing query result ordering.

    Uses multiple features to score and reorder retrieved documents
    for improved relevance and user satisfaction.
    """

    def __init__(
        self,
        semantic_weight: float = 0.4,
        keyword_weight: float = 0.3,
        freshness_weight: float = 0.1,
        popularity_weight: float = 0.2
    ):
        """
        Initialize the ranker with feature weights.

        Args:
            semantic_weight: Weight for semantic similarity scores
            keyword_weight: Weight for keyword/BM25 scores
            freshness_weight: Weight for document recency
            popularity_weight: Weight for document access frequency
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.freshness_weight = freshness_weight
        self.popularity_weight = popularity_weight

        # Validate weights sum to approximately 1.0
        total_weight = (
            semantic_weight +
            keyword_weight +
            freshness_weight +
            popularity_weight
        )
        if not (0.9 <= total_weight <= 1.1):
            logger.warning(
                f"Weights sum to {total_weight}, expected ~1.0. "
                "Normalizing weights."
            )
            self.semantic_weight /= total_weight
            self.keyword_weight /= total_weight
            self.freshness_weight /= total_weight
            self.popularity_weight /= total_weight

        logger.info(
            f"QueryResultRanker initialized with weights: "
            f"semantic={self.semantic_weight:.2f}, "
            f"keyword={self.keyword_weight:.2f}, "
            f"freshness={self.freshness_weight:.2f}, "
            f"popularity={self.popularity_weight:.2f}"
        )

    def extract_features(
        self,
        result: Dict[str, Any],
        query_length: int,
        max_score: float = 1.0
    ) -> Dict[str, float]:
        """
        Extract ranking features from a search result.

        Args:
            result: Search result dictionary with metadata
            query_length: Length of the query for normalization
            max_score: Maximum possible score for normalization

        Returns:
            Dictionary of feature values
        """
        features = {}

        # Semantic similarity score (normalized)
        features['semantic_score'] = (
            result.get('score', 0.0) / max_score
            if max_score > 0 else 0.0
        )

        # Keyword match score (if available)
        features['keyword_score'] = (
            result.get('keyword_score', 0.0)
        )

        # Document length penalty (prefer concise, relevant docs)
        doc_length = result.get('doc_length', 0)
        features['length_penalty'] = (
            1.0 / (1.0 + doc_length / 1000.0)
            if doc_length > 0 else 0.5
        )

        # Freshness score (if metadata available)
        metadata = result.get('metadata', {})
        if 'created_at' in metadata or 'last_updated' in metadata:
            # Normalize to 0-1 range (recent = higher score)
            # This is a simplified version - in production use actual dates
            features['freshness'] = metadata.get('freshness', 0.5)
        else:
            features['freshness'] = 0.5

        # Popularity score (if metadata available)
        features['popularity'] = (
            metadata.get('view_count', 0) / 100.0
        ) if 'view_count' in metadata else 0.0

        # Query-document length ratio
        if doc_length > 0:
            features['ql_ratio'] = min(query_length / doc_length, 1.0)
        else:
            features['ql_ratio'] = 0.0

        return features

    def calculate_ranking_score(
        self,
        features: Dict[str, float]
    ) -> float:
        """
        Calculate final ranking score from features.

        Args:
            features: Dictionary of feature values

        Returns:
            Combined ranking score
        """
        score = (
            self.semantic_weight * features.get('semantic_score', 0.0) +
            self.keyword_weight * features.get('keyword_score', 0.0) +
            self.freshness_weight * features.get('freshness', 0.5) +
            self.popularity_weight * features.get('popularity', 0.0)
        )

        # Apply small penalty for very long documents
        length_penalty = features.get('length_penalty', 1.0)
        score *= length_penalty

        return max(0.0, min(1.0, score))

    def rank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank and reorder query results using learned features.

        Args:
            query: The search query (for feature extraction)
            results: List of search results with scores and metadata
            top_k: Number of top results to return (None = all)

        Returns:
            Re-ordered list of results ranked by relevance
        """
        if not results:
            logger.warning("No results provided for ranking")
            return []

        query_length = len(query.split())

        # Find max score for normalization
        max_score = max(
            (r.get('score', 0.0) for r in results),
            default=1.0
        )
        if max_score == 0:
            max_score = 1.0

        # Extract features and calculate scores
        scored_results = []
        for idx, result in enumerate(results):
            try:
                features = self.extract_features(
                    result,
                    query_length,
                    max_score
                )
                ranking_score = self.calculate_ranking_score(features)

                # Create a copy with ranking metadata
                ranked_result = result.copy()
                ranked_result['ranking_score'] = ranking_score
                ranked_result['ranking_features'] = features

                scored_results.append((ranking_score, idx, ranked_result))

            except Exception as e:
                logger.error(
                    f"Error ranking result at index {idx}: {e}"
                )
                # Keep original result with low ranking score
                result['ranking_score'] = 0.0
                scored_results.append((0.0, idx, result))

        # Sort by ranking score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Apply top_k limit
        if top_k is not None and top_k > 0:
            scored_results = scored_results[:top_k]

        # Return results in ranked order
        ranked_results = [item[2] for item in scored_results]

        logger.info(
            f"Ranked {len(ranked_results)} results "
            f"(top score: {scored_results[0][0]:.3f})"
        )

        return ranked_results

    def rank_rag_results(
        self,
        query: str,
        results: List[Any],
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        Rank RAG pipeline results (works with RetrievalResult objects).

        Args:
            query: The search query
            results: List of RetrievalResult objects
            top_k: Number of top results to return

        Returns:
            Re-ordered list of RetrievalResult objects
        """
        if not results:
            return []

        # Convert results to dict format for ranking
        dict_results = []
        for result in results:
            try:
                dict_result = {
                    'score': getattr(result, 'score', 0.0),
                    'document': getattr(result, 'document', ''),
                    'metadata': getattr(result, 'metadata', {}),
                    'source': getattr(result, 'source', ''),
                    'doc_length': len(getattr(result, 'document', '')),
                    'keyword_score': getattr(result, 'keyword_score', 0.0)
                }
                dict_results.append(dict_result)
            except Exception as e:
                logger.error(f"Error converting result: {e}")
                continue

        # Rank the dict results
        ranked_dicts = self.rank_results(query, dict_results, top_k)

        # Convert back to original result type
        if results and hasattr(results[0], '__class__'):
            ResultClass = results[0].__class__
            ranked_results = []
            for ranked_dict in ranked_dicts:
                try:
                    # Create new instance with ranked data
                    ranked_result = ResultClass(
                        document=ranked_dict['document'],
                        score=ranked_dict['ranking_score'],
                        metadata=ranked_dict['metadata'],
                        source=ranked_dict.get('source', '')
                    )
                    ranked_results.append(ranked_result)
                except Exception as e:
                    logger.error(f"Error creating result instance: {e}")
                    continue

            return ranked_results

        return ranked_dicts

    def update_weights(
        self,
        weights: Dict[str, float]
    ) -> None:
        """
        Update ranking feature weights.

        Use this for online learning or A/B testing scenarios.

        Args:
            weights: Dictionary of new weight values
        """
        if 'semantic' in weights:
            self.semantic_weight = weights['semantic']
        if 'keyword' in weights:
            self.keyword_weight = weights['keyword']
        if 'freshness' in weights:
            self.freshness_weight = weights['freshness']
        if 'popularity' in weights:
            self.popularity_weight = weights['popularity']

        logger.info(f"Updated ranking weights: {weights}")

    def get_weights(self) -> Dict[str, float]:
        """
        Get current ranking weights.

        Returns:
            Dictionary of current weight values
        """
        return {
            'semantic': self.semantic_weight,
            'keyword': self.keyword_weight,
            'freshness': self.freshness_weight,
            'popularity': self.popularity_weight
        }


# Global ranker instance (can be configured via environment)
_ranker: Optional[QueryResultRanker] = None


def get_ranker() -> QueryResultRanker:
    """
    Get or create the global ranker instance.

    Returns:
        QueryResultRanker instance
    """
    global _ranker
    if _ranker is None:
        _ranker = QueryResultRanker()
    return _ranker


def reset_ranker() -> None:
    """Reset the global ranker instance (mainly for testing)."""
    global _ranker
    _ranker = None
