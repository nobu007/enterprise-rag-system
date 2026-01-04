"""
Query API Routes

This module defines API endpoints for querying the RAG system.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.services.rag_pipeline import RAGResponse


router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The question to ask", min_length=1)
    collection: Optional[str] = Field(None, description="Collection/namespace to search in")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    use_hybrid: bool = Field(True, description="Use hybrid search (semantic + keyword)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    latency_ms: int
    tokens_used: int


class BatchQueryRequest(BaseModel):
    """Request model for batch query endpoint"""
    queries: List[str] = Field(..., description="List of questions to ask")
    collection: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)


@router.post("/", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system with a question

    Args:
        request: Query request with question and parameters

    Returns:
        QueryResponse with answer and sources
    """
    try:
        # Get RAG pipeline instance (should be injected via dependency)
        from app.main import get_rag_pipeline

        pipeline = get_rag_pipeline()

        # Execute query
        result = await pipeline.query(
            question=request.query,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            filter_dict=request.filters
        )

        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/batch", response_model=List[QueryResponse])
async def batch_query(request: BatchQueryRequest) -> List[QueryResponse]:
    """
    Query the RAG system with multiple questions

    Args:
        request: Batch query request

    Returns:
        List of QueryResponse objects
    """
    try:
        from app.main import get_rag_pipeline

        pipeline = get_rag_pipeline()

        # Execute batch query
        results = await pipeline.batch_query(
            questions=request.queries,
            top_k=request.top_k
        )

        responses = []
        for result in results:
            responses.append(QueryResponse(
                answer=result.answer,
                sources=result.sources,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                tokens_used=result.tokens_used
            ))

        return responses

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch query failed: {str(e)}"
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Query API"
    }
