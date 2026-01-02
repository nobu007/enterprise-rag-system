"""
Retrieval Service for RAG System

This module implements hybrid search and retrieval logic.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.core.vectordb import VectorDB, SearchResult
from app.core.embeddings import EmbeddingModel
from app.services.document_loader import Document


@dataclass
class RetrievalResult:
    """Result from retrieval system"""
    document: str
    score: float
    metadata: Dict[str, Any]
    source: str


class HybridRetriever:
    """Hybrid retrieval using semantic + keyword search"""
    
    def __init__(
        self,
        vector_db: VectorDB,
        embedding_model: EmbeddingModel,
        alpha: float = 0.5
    ):
        """
        Initialize hybrid retriever
        
        Args:
            vector_db: Vector database instance
            embedding_model: Embedding model instance
            alpha: Weight for semantic vs keyword search (0=keyword only, 1=semantic only)
        """
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.bm25_index = None
    
    def build_bm25_index(self, documents: List[Document]) -> None:
        """Build BM25 index for keyword search"""
        try:
            from rank_bm25 import BM25Okapi
            import re
            
            # Tokenize documents
            tokenized_docs = []
            for doc in documents:
                # Simple tokenization
                tokens = re.findall(r'\w+', doc.content.lower())
                tokenized_docs.append(tokens)
            
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.bm25_documents = documents
            
            print(f"✅ Built BM25 index with {len(documents)} documents")
        
        except ImportError:
            print("⚠️  rank-bm25 not installed. Keyword search disabled.")
            print("   Install with: pip install rank-bm25")
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform semantic search using embeddings"""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search vector database
        results = self.vector_db.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search"""
        if not self.bm25_index:
            return []
        
        import re
        
        # Tokenize query
        query_tokens = re.findall(r'\w+', query.lower())
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    'document': self.bm25_documents[idx],
                    'score': scores[idx],
                    'index': idx
                })
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform hybrid search combining semantic and keyword search"""
        # Perform both searches
        semantic_results = self.semantic_search(query, top_k=top_k * 2, filter_dict=filter_dict)
        keyword_results = self.keyword_search(query, top_k=top_k * 2) if self.bm25_index else []
        
        # Combine results using Reciprocal Rank Fusion (RRF)
        combined_scores = {}
        
        # Add semantic search scores
        for rank, result in enumerate(semantic_results):
            doc_id = result.id
            rrf_score = 1.0 / (rank + 60)  # RRF formula
            combined_scores[doc_id] = {
                'score': self.alpha * rrf_score,
                'text': result.text,
                'metadata': result.metadata,
                'semantic_rank': rank + 1
            }
        
        # Add keyword search scores
        if keyword_results:
            for rank, result in enumerate(keyword_results):
                doc = result['document']
                doc_id = doc.doc_id
                rrf_score = 1.0 / (rank + 60)
                
                if doc_id in combined_scores:
                    combined_scores[doc_id]['score'] += (1 - self.alpha) * rrf_score
                    combined_scores[doc_id]['keyword_rank'] = rank + 1
                else:
                    combined_scores[doc_id] = {
                        'score': (1 - self.alpha) * rrf_score,
                        'text': doc.content,
                        'metadata': doc.metadata,
                        'keyword_rank': rank + 1
                    }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:top_k]
        
        # Convert to RetrievalResult objects
        final_results = []
        for doc_id, data in sorted_results:
            final_results.append(RetrievalResult(
                document=data['text'],
                score=data['score'],
                metadata=data['metadata'],
                source=data['metadata'].get('source', 'unknown')
            ))
        
        return final_results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Main retrieval method"""
        if use_hybrid and self.bm25_index:
            return self.hybrid_search(query, top_k=top_k, filter_dict=filter_dict)
        else:
            # Fallback to semantic search only
            semantic_results = self.semantic_search(query, top_k=top_k, filter_dict=filter_dict)
            return [
                RetrievalResult(
                    document=r.text,
                    score=r.score,
                    metadata=r.metadata,
                    source=r.metadata.get('source', 'unknown')
                )
                for r in semantic_results
            ]


class ContextCompressor:
    """Compress retrieved context to fit LLM context window"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
    
    def compress(
        self,
        query: str,
        results: List[RetrievalResult],
        method: str = "truncate"
    ) -> str:
        """Compress retrieved results into context string"""
        if method == "truncate":
            return self._truncate_context(results)
        elif method == "rerank":
            return self._rerank_and_truncate(query, results)
        else:
            return self._truncate_context(results)
    
    def _truncate_context(self, results: List[RetrievalResult]) -> str:
        """Simple truncation to fit context window"""
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results):
            # Rough token estimation (1 token ≈ 4 characters)
            estimated_tokens = len(result.document) // 4
            
            if total_length + estimated_tokens > self.max_tokens:
                break
            
            source = result.metadata.get('filename', 'unknown')
            page = result.metadata.get('page', '')
            page_info = f" (page {page})" if page else ""
            
            context_parts.append(
                f"[Source {i+1}: {source}{page_info}]\n{result.document}\n"
            )
            total_length += estimated_tokens
        
        return "\n---\n".join(context_parts)
    
    def _rerank_and_truncate(self, query: str, results: List[RetrievalResult]) -> str:
        """Re-rank results before truncation (placeholder for future implementation)"""
        # For now, just use truncation
        # TODO: Implement cross-encoder re-ranking
        return self._truncate_context(results)
