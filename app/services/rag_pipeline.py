"""
RAG Pipeline Implementation

This module orchestrates the complete RAG workflow.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.services.retrieval import HybridRetriever, RetrievalResult, ContextCompressor
from app.services.reranker import Reranker
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.cache import CacheManager


settings = get_settings()
logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    latency_ms: int
    tokens_used: int
    retrieval_results: List[RetrievalResult]


class RAGPipeline:
    """Complete RAG pipeline orchestration"""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: AsyncOpenAI,
        llm_model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        reranker: Optional[Reranker] = None,
        cache_manager: Optional["CacheManager"] = None
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.llm_model = llm_model or settings.llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.compressor = ContextCompressor(max_tokens=4000)
        self.reranker = reranker
        self.cache = cache_manager
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM"""
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.

Context information is below:
---
{context}
---

Instructions:
- Answer the question using ONLY the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Cite your sources by mentioning the source number [Source X]
- Be concise but comprehensive
- If you're uncertain, acknowledge it

Question: {query}

Answer:"""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM with the prompt"""
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate answers based on given context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return {
                'answer': response.choices[0].message.content,
                'tokens_used': response.usage.total_tokens,
                'finish_reason': response.choices[0].finish_reason
            }

        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")
    
    def _calculate_confidence(
        self,
        retrieval_results: List[RetrievalResult],
        answer: str
    ) -> float:
        """Calculate confidence score for the answer"""
        if not retrieval_results:
            return 0.0
        
        # Simple confidence calculation based on:
        # 1. Top retrieval score
        # 2. Number of high-scoring results
        # 3. Answer length (very short answers might indicate uncertainty)
        
        top_score = retrieval_results[0].score if retrieval_results else 0.0
        high_score_count = sum(1 for r in retrieval_results if r.score > 0.7)
        answer_length_factor = min(len(answer) / 200, 1.0)  # Normalize to 0-1
        
        confidence = (
            0.5 * top_score +
            0.3 * (high_score_count / len(retrieval_results)) +
            0.2 * answer_length_factor
        )
        
        return min(confidence, 1.0)
    
    async def query(
        self,
        question: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
        collection: str = "default"
    ) -> RAGResponse:
        """
        Execute complete RAG pipeline

        Args:
            question: User's question
            top_k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid search
            filter_dict: Optional metadata filters
            rerank: Whether to apply cross-encoder re-ranking
            collection: Collection to search in

        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()

        # Check cache first
        cache_key = None
        if self.cache:
            cache_key = self.cache.generate_key(question, collection, top_k, rerank)
            cached_result = self.cache.get(cache_key)

            if cached_result:
                logger.info(f"Cache hit for query: {question[:50]}...")
                # Reconstruct RAGResponse from cached dict
                return RAGResponse(**cached_result)
            else:
                logger.debug(f"Cache miss for query: {question[:50]}...")

        # Step 1: Retrieve relevant documents
        logger.debug(f"Retrieving documents for: {question} from collection: {collection}")

        # If reranking is enabled, retrieve more candidates (top 50)
        retrieval_candidates = self.retriever.retrieve(
            query=question,
            top_k=50 if rerank and self.reranker else top_k,
            use_hybrid=use_hybrid,
            filter_dict=filter_dict,
            collection=collection
        )

        # Step 1.5: Re-rank if enabled and reranker is available
        if rerank and self.reranker and retrieval_candidates:
            logger.debug("Applying cross-encoder re-ranking")
            try:
                reranked_results = self.reranker.rerank_results(
                    query=question,
                    results=retrieval_candidates,
                    top_k=top_k,
                    doc_text_attr="document"
                )
                retrieval_results = reranked_results
            except Exception as e:
                logger.warning(f"Re-ranking failed, using original results: {e}")
                retrieval_results = retrieval_candidates[:top_k]
        else:
            retrieval_results = retrieval_candidates[:top_k]

        if not retrieval_results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_used=0,
                retrieval_results=[]
            )

        logger.debug(f"Retrieved {len(retrieval_results)} documents")

        # Step 2: Compress context
        context = self.compressor.compress(question, retrieval_results)

        # Step 3: Build prompt
        prompt = self._build_prompt(question, context)

        # Step 4: Call LLM
        logger.debug(f"Generating answer with {self.llm_model}")
        llm_response = await self._call_llm(prompt)

        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(retrieval_results, llm_response['answer'])

        # Step 6: Prepare sources
        sources = []
        for i, result in enumerate(retrieval_results):
            sources.append({
                'index': i + 1,
                'document': result.metadata.get('filename', 'unknown'),
                'page': result.metadata.get('page'),
                'relevance_score': round(result.score, 3),
                'text_preview': result.document[:200] + "..." if len(result.document) > 200 else result.document
            })

        latency_ms = int((time.time() - start_time) * 1000)

        logger.debug(f"Generated answer in {latency_ms}ms")

        response = RAGResponse(
            answer=llm_response['answer'],
            sources=sources,
            confidence=round(confidence, 2),
            latency_ms=latency_ms,
            tokens_used=llm_response['tokens_used'],
            retrieval_results=retrieval_results
        )

        # Cache the response
        if self.cache and cache_key:
            self.cache.set(cache_key, response)
            logger.debug(f"Cached response for query: {question[:50]}...")

        return response
    
    async def batch_query(self, questions: List[str], top_k: int = 5, collection: str = "default", **kwargs) -> List[RAGResponse]:
        """Process multiple questions"""
        responses = []

        for question in questions:
            try:
                response = await self.query(question, top_k=top_k, collection=collection, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                responses.append(RAGResponse(
                    answer=f"Error: {str(e)}",
                    sources=[],
                    confidence=0.0,
                    latency_ms=0,
                    tokens_used=0,
                    retrieval_results=[]
                ))

        return responses


class StreamingRAGPipeline(RAGPipeline):
    """RAG Pipeline with streaming support"""

    async def stream_query(
        self,
        question: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection: str = "default"
    ):
        """Stream response tokens as they're generated"""
        start_time = time.time()

        # Retrieve documents
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            filter_dict=filter_dict,
            collection=collection
        )

        if not retrieval_results:
            yield {
                'type': 'answer',
                'content': "I couldn't find any relevant information.",
                'done': True
            }
            return

        # Compress context and build prompt
        context = self.compressor.compress(question, retrieval_results)
        prompt = self._build_prompt(question, context)

        # Yield sources first
        yield {
            'type': 'sources',
            'content': [
                {
                    'document': r.metadata.get('filename', 'unknown'),
                    'score': round(r.score, 3)
                }
                for r in retrieval_results
            ]
        }

        # Stream LLM response
        try:
            stream = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {
                        'type': 'answer',
                        'content': chunk.choices[0].delta.content,
                        'done': False
                    }

            # Final chunk
            yield {
                'type': 'answer',
                'content': '',
                'done': True,
                'latency_ms': int((time.time() - start_time) * 1000)
            }

        except Exception as e:
            yield {
                'type': 'error',
                'content': str(e),
                'done': True
            }
