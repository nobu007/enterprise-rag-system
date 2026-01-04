"""
Query API Routes

This module defines API endpoints for querying the RAG system.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.services.rag_pipeline import RAGResponse, RAGPipeline
from app.api.dependencies import get_rag_pipeline, get_api_key
from app.core.rate_limit import limiter


router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The question to ask", min_length=1)
    collection: Optional[str] = Field(None, description="Collection/namespace to search in")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    use_hybrid: bool = Field(True, description="Use hybrid search (semantic + keyword)")
    rerank: bool = Field(True, description="Apply cross-encoder re-ranking for better accuracy")
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


@router.post(
    "/",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute RAG Query / RAGクエリ実行",
    description="Perform semantic search and retrieve relevant context, then generate answer using LLM / セマンティック検索を行い関連コンテキストを取得した後、LLMを使用して回答を生成します",
    response_description="Generated answer with retrieved context and metadata / 取得したコンテキストとメタデータを含む生成された回答",
    responses={
        200: {"description": "Successful query / クエリ成功"},
        400: {"description": "Invalid request parameters / 不正なリクエストパラメータ"},
        401: {"description": "Authentication required / 認証が必要"},
        422: {"description": "Validation error / バリデーションエラー"},
        429: {"description": "Rate limit exceeded / レート制限超過"},
        500: {"description": "Internal server error / サーバー内部エラー"}
    },
    tags=["Query"]
)
@limiter.limit("60/minute")
async def query(
    request: Request,
    query_req: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    api_key=Depends(get_api_key),  # Require authentication
) -> QueryResponse:
    """
    Query the RAG system with a question / RAGシステムに質問をクエリします

    ## Features / 機能

    - **Semantic Search**: Vector similarity search for relevant documents / セマンティック検索: 関連ドキュメントのベクトル類似度検索
    - **Hybrid Search**: Combines semantic and keyword search / ハイブリッド検索: セマンティック検索とキーワード検索の組み合わせ
    - **Re-ranking**: Cross-encoder re-ranking for better accuracy / 再ランク付け: より高い精度のためのクロスエンコーダーによる再ランク付け
    - **Multi-collection**: Search across different document collections / マルチコレクション: 異なるドキュメントコレクションの検索

    ## Parameters / パラメータ

    - **query**: Search query text (1-1000 characters) / 検索クエリテキスト (1-1000文字)
    - **collection**: Target collection name (default: "default") / 対象コレクション名 (デフォルト: "default")
    - **top_k**: Number of results to return (1-20) / 返却する結果数 (1-20)
    - **use_hybrid**: Enable hybrid search (default: true) / ハイブリッド検索を有効化 (デフォルト: true)
    - **rerank**: Apply re-ranking (default: true) / 再ランク付けを適用 (デフォルト: true)
    - **filters**: Optional metadata filters / オプションのメタデータフィルター

    ## Example / 例

    ```json
    {
      "query": "What is Retrieval-Augmented Generation?",
      "collection": "default",
      "top_k": 5,
      "use_hybrid": true,
      "rerank": true,
      "filters": null
    }
    ```

    Args:
        request: FastAPI Request object
        query_req: Query request with question and parameters
        pipeline: RAG pipeline injected via dependency injection

    Returns:
        QueryResponse with answer and sources
    """
    try:
        # Execute query
        result = await pipeline.query(
            question=query_req.query,
            top_k=query_req.top_k,
            use_hybrid=query_req.use_hybrid,
            filter_dict=query_req.filters,
            rerank=query_req.rerank,
            collection=query_req.collection or "default"
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


@router.post(
    "/batch",
    response_model=List[QueryResponse],
    summary="Execute Batch RAG Queries / バッチRAGクエリ実行",
    description="Query the RAG system with multiple questions in a single request / 単一のリクエストで複数の質問をRAGシステムにクエリします",
    response_description="List of generated answers with retrieved context / 取得したコンテキストを含む生成された回答のリスト",
    responses={
        200: {"description": "Successful batch query / バッチクエリ成功"},
        400: {"description": "Invalid request parameters / 不正なリクエストパラメータ"},
        401: {"description": "Authentication required / 認証が必要"},
        422: {"description": "Validation error / バリデーションエラー"},
        429: {"description": "Rate limit exceeded / レート制限超過"},
        500: {"description": "Internal server error / サーバー内部エラー"}
    },
    tags=["Query"]
)
@limiter.limit("60/minute")
async def batch_query(
    request: Request,
    batch_req: BatchQueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    api_key=Depends(get_api_key),  # Require authentication
) -> List[QueryResponse]:
    """
    Query the RAG system with multiple questions / 複数の質問をRAGシステムにクエリします

    ## Use Cases / 使用例

    - **Bulk Processing**: Process multiple questions efficiently / 一括処理: 複数の質問を効率的に処理
    - **Comparison**: Compare answers for similar questions / 比較: 類似した質問の回答を比較
    - **Testing**: Validate system behavior with multiple inputs / テスト: 複数の入力でシステムの動作を検証

    ## Parameters / パラメータ

    - **queries**: List of search query texts / 検索クエリテキストのリスト
    - **collection**: Target collection name / 対象コレクション名
    - **top_k**: Number of results per query / クエリごとの結果数

    ## Example / 例

    ```json
    {
      "queries": [
        "What is RAG?",
        "How does vector search work?",
        "Explain cross-encoder re-ranking"
      ],
      "collection": "default",
      "top_k": 5
    }
    ```

    Args:
        request: FastAPI Request object
        batch_req: Batch query request
        pipeline: RAG pipeline injected via dependency injection

    Returns:
        List of QueryResponse objects
    """
    try:
        # Execute batch query
        results = await pipeline.batch_query(
            questions=batch_req.queries,
            top_k=batch_req.top_k,
            collection=batch_req.collection or "default"
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


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Query Service Health Check / クエリサービスヘルスチェック",
    description="Check the health status of the Query service / Queryサービスのヘルス状態を確認します",
    response_description="Service health status / サービスのヘルス状態",
    responses={
        200: {"description": "Service is healthy / サービスが正常"}
    },
    tags=["Query"]
)
async def health_check() -> Dict[str, str]:
    """Health check endpoint / ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "service": "RAG Query API"
    }
