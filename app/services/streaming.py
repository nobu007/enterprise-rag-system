"""
Streaming Response Service

This module provides streaming response capabilities for the RAG system,
allowing real-time delivery of LLM-generated content to clients.
"""

import json
import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
from dataclasses import dataclass
import time

from openai import AsyncOpenAI
from app.core.logging_config import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class StreamingChunk:
    """A single chunk of streaming response"""
    content: str
    is_done: bool = False
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format"""
        data = {
            "content": self.content,
            "is_done": self.is_done
        }
        if self.sources:
            data["sources"] = self.sources
        if self.metadata:
            data["metadata"] = self.metadata
        return f"data: {json.dumps(data)}\n\n"


class StreamingResponseError(Exception):
    """Custom exception for streaming response errors"""
    pass


class StreamingRAGService:
    """
    Service for handling streaming RAG responses.

    This service provides async generators that yield chunks of LLM responses
    as they are generated, enabling real-time streaming to clients.
    """

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        llm_model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        chunk_timeout: float = 30.0
    ):
        """
        Initialize streaming service.

        Args:
            llm_client: Async OpenAI client
            llm_model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            chunk_timeout: Timeout for each chunk in seconds
        """
        self.llm_client = llm_client
        self.llm_model = llm_model or settings.llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunk_timeout = chunk_timeout
        logger.info(
            f"StreamingRAGService initialized with model={self.llm_model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build prompt for LLM with context and query.

        Args:
            query: User's question
            context: Retrieved context documents

        Returns:
            Formatted prompt string
        """
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

    async def stream_query_response(
        self,
        query: str,
        context: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream LLM response for a query with context.

        Args:
            query: User's question
            context: Retrieved context to inform the response
            sources: Source documents to include with final chunk
            metadata: Additional metadata to include

        Yields:
            StreamingChunk objects with incremental content

        Raises:
            StreamingResponseError: If streaming fails
        """
        try:
            prompt = self._build_prompt(query, context)

            logger.info(f"Starting streaming response for query: {query[:100]}...")

            # Track token usage
            total_tokens = 0
            start_time = time.time()

            # Create streaming completion
            stream = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options={"include_usage": True}
            )

            async for chunk in stream:
                try:
                    # Check for timeout
                    if time.time() - start_time > self.chunk_timeout:
                        logger.warning(f"Chunk timeout after {self.chunk_timeout}s")
                        raise StreamingResponseError(
                            f"Chunk timeout exceeded: {self.chunk_timeout}s"
                        )

                    # Extract content delta
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            total_tokens += 1
                            yield StreamingChunk(
                                content=delta.content,
                                is_done=False,
                                metadata={"tokens_so_far": total_tokens}
                            )

                    # Check for completion
                    if hasattr(chunk, 'usage') and chunk.usage:
                        latency_ms = int((time.time() - start_time) * 1000)
                        logger.info(
                            f"Streaming completed: {total_tokens} tokens, "
                            f"{latency_ms}ms"
                        )

                        # Final chunk with metadata
                        yield StreamingChunk(
                            content="",
                            is_done=True,
                            sources=sources,
                            metadata={
                                "total_tokens": total_tokens,
                                "latency_ms": latency_ms,
                                **(metadata or {})
                            }
                        )
                        break

                except Exception as e:
                    logger.error(f"Error processing stream chunk: {e}")
                    raise StreamingResponseError(f"Chunk processing error: {e}")

        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            raise StreamingResponseError(f"Failed to stream response: {e}")

    async def stream_query_with_retrieval(
        self,
        query: str,
        retriever_func,
        top_k: int = 5,
        use_hybrid: bool = True,
        rerank: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection: str = "default"
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Complete RAG streaming: retrieval + generation streaming.

        This is a high-level method that handles both retrieval and streaming
        generation, suitable for direct use in API endpoints.

        Args:
            query: User's question
            retriever_func: Async function that performs retrieval
            top_k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid search
            rerank: Whether to apply re-ranking
            filter_dict: Optional metadata filters
            collection: Collection name to search

        Yields:
            StreamingChunk objects with incremental content

        Raises:
            StreamingResponseError: If retrieval or streaming fails
        """
        try:
            logger.info(f"Starting RAG streaming for query: {query[:100]}...")

            # Step 1: Perform retrieval
            logger.info("Step 1: Retrieving relevant documents...")
            retrieval_start = time.time()

            retrieval_results = await retriever_func(
                query=query,
                top_k=top_k,
                use_hybrid=use_hybrid,
                filter_dict=filter_dict,
                rerank=rerank,
                collection=collection
            )

            retrieval_latency = int((time.time() - retrieval_start) * 1000)
            logger.info(f"Retrieval completed in {retrieval_latency}ms")

            # Step 2: Build context from retrieved documents
            context_parts = []
            sources = []

            for idx, result in enumerate(retrieval_results):
                context_parts.append(f"[Source {idx + 1}] {result.document}")
                sources.append({
                    "id": idx + 1,
                    "content": result.document[:200] + "...",
                    "score": float(result.score),
                    "metadata": result.metadata
                })

            context = "\n\n".join(context_parts)

            # Step 3: Stream LLM response
            logger.info("Step 2: Streaming LLM response...")
            async for chunk in self.stream_query_response(
                query=query,
                context=context,
                sources=sources,
                metadata={"retrieval_latency_ms": retrieval_latency}
            ):
                yield chunk

        except Exception as e:
            logger.error(f"RAG streaming failed: {e}", exc_info=True)
            raise StreamingResponseError(f"Failed to stream RAG response: {e}")

    def validate_stream_request(
        self,
        query: str,
        top_k: int,
        max_tokens: int
    ) -> None:
        """
        Validate streaming request parameters.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens to generate

        Raises:
            ValueError: If parameters are invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if len(query) > 1000:
            raise ValueError("Query too long (max 1000 characters)")

        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")

        if max_tokens < 100 or max_tokens > 4096:
            raise ValueError("max_tokens must be between 100 and 4096")

        logger.debug("Streaming request parameters validated")


async def format_sse_stream(
    chunk_generator: AsyncGenerator[StreamingChunk, None]
) -> AsyncGenerator[str, None]:
    """
    Format streaming chunks as Server-Sent Events.

    Args:
        chunk_generator: Generator of StreamingChunk objects

    Yields:
        Formatted SSE strings
    """
    try:
        async for chunk in chunk_generator:
            yield chunk.to_sse()
    except Exception as e:
        logger.error(f"SSE formatting error: {e}")
        # Send error message as SSE
        error_data = {
            "error": str(e),
            "is_done": True
        }
        yield f"data: {json.dumps(error_data)}\n\n"
