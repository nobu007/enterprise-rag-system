"""
Cross-Encoder Re-ranking Service

This module implements cross-encoder based re-ranking for improved retrieval accuracy.
"""

from typing import List, Tuple, Optional
import os
from app.core.logging_config import get_logger


logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder based reranker for improving retrieval accuracy.

    Uses sentence-transformers CrossEncoder models to re-rank retrieved documents
    based on their relevance to the query.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the reranker with a cross-encoder model.

        Args:
            model_name: Name of the cross-encoder model to use.
                       Defaults to RERANKER_MODEL env var or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        """
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise ImportError(
                "sentence-transformers is required for Reranker. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise RuntimeError(f"Failed to initialize reranker: {e}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents based on their relevance to the query.

        Args:
            query: The search query
            documents: List of document texts to re-rank
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of tuples (original_index, score) sorted by relevance in descending order
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided for reranking")
            return list(enumerate([0.0] * len(documents)))

        try:
            # Prepare query-document pairs for cross-encoder
            pairs = [[query, doc] for doc in documents]

            # Get cross-encoder scores
            logger.debug(f"Reranking {len(documents)} documents")
            scores = self.model.predict(pairs)

            # Create list of (index, score) tuples
            indexed_scores = list(enumerate(scores))

            # Sort by score in descending order
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Apply top_k limit if specified
            if top_k is not None and top_k > 0:
                indexed_scores = indexed_scores[:top_k]

            logger.debug(f"Reranking complete. Top score: {indexed_scores[0][1]:.4f}")
            return indexed_scores

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original order on error
            return list(enumerate([0.0] * len(documents)))

    def rerank_results(
        self,
        query: str,
        results: List,
        top_k: Optional[int] = None,
        doc_text_attr: str = "document"
    ) -> List:
        """
        Re-rank retrieval results and return sorted results.

        Args:
            query: The search query
            results: List of retrieval result objects
            top_k: Number of top results to return
            doc_text_attr: Attribute name for document text in result objects

        Returns:
            List of re-ranked result objects (same type as input)
        """
        if not results:
            return []

        # Extract document texts
        try:
            documents = [getattr(result, doc_text_attr) for result in results]
        except AttributeError as e:
            logger.error(f"Result objects don't have '{doc_text_attr}' attribute: {e}")
            return results

        # Rerank
        reranked_indices = self.rerank(query, documents, top_k=top_k)

        # Return re-ordered results
        return [results[idx] for idx, score in reranked_indices]
