"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class ErrorDetail(BaseModel):
    """Error detail schema"""

    field: str = Field(..., description="エラーフィールド名 / Error field name")
    message: str = Field(..., description="エラーメッセージ / Error message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "field": "query",
                    "message": "ensure this value has at least 1 characters"
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response schema / 標準エラーレスポンス"""

    error: str = Field(..., description="エラータイプ / Error type")
    message: str = Field(..., description="エラーメッセージ / Error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="詳細エラー情報 / Detailed error information")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "ValidationError",
                    "message": "Request validation failed",
                    "details": [
                        {
                            "field": "query",
                            "message": "ensure this value has at least 1 characters"
                        }
                    ]
                },
                {
                    "error": "RateLimitExceeded",
                    "message": "Too many requests. Please try again later.",
                    "details": None
                }
            ]
        }
    }


class QueryRequest(BaseModel):
    """Query request schema / クエリリクエスト"""

    query: str = Field(
        ...,
        description="検索クエリテキスト / Search query text",
        min_length=1,
        max_length=1000,
        examples=["What is RAG?", "How does vector search work?", "What is our remote work policy?"]
    )
    collection: str = Field(
        default="default",
        description="ドキュメントコレクション名 / Document collection name",
        examples=["default", "hr-policies", "tech-docs"]
    )
    top_k: int = Field(
        default=5,
        description="返却する関連ドキュメント数 / Number of relevant documents to return",
        ge=1,
        le=20,
        examples=[5, 10]
    )
    include_sources: bool = Field(
        default=True,
        description="ソースドキュメントを含めるかどうか / Whether to include source documents",
        examples=[True, False]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is our company policy on remote work?",
                    "collection": "hr-policies",
                    "top_k": 5,
                    "include_sources": True
                }
            ]
        }
    }


class Source(BaseModel):
    """Source document schema / ソースドキュメント"""

    document: str = Field(..., description="ドキュメント名 / Document name")
    page: Optional[int] = Field(None, description="ページ番号 / Page number")
    relevance_score: float = Field(..., description="関連度スコア / Relevance score")
    text: str = Field(..., description="ドキュメントテキスト / Document text")


class QueryResponse(BaseModel):
    """Query response schema / クエリレスポンス"""

    answer: str = Field(..., description="生成された回答 / Generated answer")
    sources: List[Source] = Field(..., description="参照したソースドキュメント / Referenced source documents")
    confidence: float = Field(..., description="回答の信頼度 (0-1) / Answer confidence (0-1)", ge=0, le=1)
    latency_ms: int = Field(..., description="レイテンシ (ミリ秒) / Latency (milliseconds)")
    tokens_used: int = Field(..., description="使用トークン数 / Tokens used")
    cached: bool = Field(default=False, description="キャッシュヒットかどうか / Whether the response was cached")


class IngestRequest(BaseModel):
    """Document ingestion request / ドキュメントインジェストリクエスト"""

    source_path: str = Field(..., description="ドキュメントのソースパス / Source path to documents")
    collection: str = Field(default="default", description="コレクション名 / Collection name", examples=["default", "tech-docs"])
    chunk_size: int = Field(
        default=1000,
        description="チャンクサイズ / Chunk size for splitting",
        ge=100,
        le=4000,
        examples=[1000, 1500]
    )
    chunk_overlap: int = Field(
        default=200,
        description="チャンクオーバーラップ / Chunk overlap",
        ge=0,
        le=500,
        examples=[200]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_path": "./data/documents",
                    "collection": "hr-policies",
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }
            ]
        }
    }


class IngestResponse(BaseModel):
    """Document ingestion response / ドキュメントインジェストレスポンス"""

    status: str = Field(..., description="ステータス / Status")
    documents_processed: int = Field(..., description="処理されたドキュメント数 / Number of documents processed")
    chunks_created: int = Field(..., description="作成されたチャンク数 / Number of chunks created")
    collection: str = Field(..., description="コレクション名 / Collection name")
    timestamp: datetime = Field(..., description="タイムスタンプ / Timestamp")


class HealthResponse(BaseModel):
    """Health check response / ヘルスチェックレスポンス"""

    status: str = Field(..., description="ステータス / Status")
    version: str = Field(..., description="バージョン / Version")
    services: Dict[str, str] = Field(..., description="サービスごとのステータス / Status by service")
