"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """Query request schema"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    collection: str = Field(default="default", description="Document collection name")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    include_sources: bool = Field(default=True, description="Include source documents")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is our company policy on remote work?",
                "collection": "hr-policies",
                "top_k": 5,
                "include_sources": True
            }
        }


class Source(BaseModel):
    """Source document schema"""
    document: str
    page: Optional[int] = None
    relevance_score: float
    text: str


class QueryResponse(BaseModel):
    """Query response schema"""
    answer: str
    sources: List[Source]
    confidence: float = Field(ge=0, le=1)
    latency_ms: int
    tokens_used: int
    cached: bool = False


class IngestRequest(BaseModel):
    """Document ingestion request"""
    source_path: str
    collection: str = "default"
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)


class IngestResponse(BaseModel):
    """Document ingestion response"""
    status: str
    documents_processed: int
    chunks_created: int
    collection: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: dict
