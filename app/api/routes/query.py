"""
Query API Routes

This module defines API endpoints for querying the RAG system.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio

from app.services.rag_pipeline import RAGResponse, RAGPipeline
from app.services.ranking import QueryResultRanker
from app.services.streaming import StreamingRAGService, format_sse_stream
from app.api.dependencies import get_rag_pipeline, get_llm_client
from app.core.rate_limit import limiter
from app.core.logging_config import get_logger

logger = get_logger(__name__)


router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The question to ask", min_length=1)
    collection: Optional[str] = Field(None, description="Collection/namespace to search in")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    use_hybrid: bool = Field(True, description="Use hybrid search (semantic + keyword)")
    rerank: bool = Field(True, description="Apply cross-encoder re-ranking for better accuracy")
    rank_results: bool = Field(False, description="Apply learning-to-rank for optimized result ordering")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    latency_ms: int
    tokens_used: int


class StreamingQueryRequest(BaseModel):
    """Request model for streaming query endpoint"""
    query: str = Field(..., description="The question to ask", min_length=1, max_length=1000)
    collection: Optional[str] = Field(None, description="Collection/namespace to search in")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    use_hybrid: bool = Field(True, description="Use hybrid search (semantic + keyword)")
    rerank: bool = Field(True, description="Apply cross-encoder re-ranking for better accuracy")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    max_tokens: int = Field(2048, description="Maximum tokens to generate", ge=100, le=4096)


class BatchQueryRequest(BaseModel):
    """Request model for batch query endpoint"""
    queries: List[str] = Field(..., description="List of questions to ask")
    collection: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)
    use_hybrid: bool = Field(True, description="Use hybrid search (semantic + keyword)")
    rerank: bool = Field(True, description="Apply cross-encoder re-ranking")
    rank_results: bool = Field(False, description="Apply feature-based ranking for result optimization")


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
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> QueryResponse:
    """
    Query the RAG system with a question / RAGシステムに質問をクエリします

    ## Features / 機能

    - **Semantic Search**: Vector similarity search for relevant documents / セマンティック検索: 関連ドキュメントのベクトル類似度検索
    - **Hybrid Search**: Combines semantic and keyword search / ハイブリッド検索: セマンティック検索とキーワード検索の組み合わせ
    - **Re-ranking**: Cross-encoder re-ranking for better accuracy / 再ランク付け: より高い精度のためのクロスエンコーダーによる再ランク付け
    - **Learning-to-Rank**: ML-based result optimization / ランキング: 機械学習ベースの結果最適化
    - **Multi-collection**: Search across different document collections / マルチコレクション: 異なるドキュメントコレクションの検索

    ## Parameters / パラメータ

    - **query**: Search query text (1-1000 characters) / 検索クエリテキスト (1-1000文字)
    - **collection**: Target collection name (default: "default") / 対象コレクション名 (デフォルト: "default")
    - **top_k**: Number of results to return (1-20) / 返却する結果数 (1-20)
    - **use_hybrid**: Enable hybrid search (default: true) / ハイブリッド検索を有効化 (デフォルト: true)
    - **rerank**: Apply re-ranking (default: true) / 再ランク付けを適用 (デフォルト: true)
    - **rank_results**: Apply learning-to-rank optimization (default: false) / ランキング最適化を適用 (デフォルト: false)
    - **filters**: Optional metadata filters / オプションのメタデータフィルター

    ## Example / 例

    ```json
    {
      "query": "What is Retrieval-Augmented Generation?",
      "collection": "default",
      "top_k": 5,
      "use_hybrid": true,
      "rerank": true,
      "rank_results": true,
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

        # Apply learning-to-rank if requested
        sources = result.sources
        if query_req.rank_results:
            try:
                ranker = QueryResultRanker()
                sources = ranker.rank_results(
                    query=query_req.query,
                    results=sources,
                    top_k=query_req.top_k
                )
                logger.info(f"Applied learning-to-rank to {len(sources)} results")
            except Exception as e:
                logger.error(f"Ranking failed, using original order: {e}")

        return QueryResponse(
            answer=result.answer,
            sources=sources,
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
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
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
    - **use_hybrid**: Enable hybrid search (default: true) / ハイブリッド検索を有効化
    - **rerank**: Apply cross-encoder re-ranking (default: true) / 再ランク付けを適用
    - **rank_results**: Apply feature-based ranking (default: false) / 特徴量ランク付けを適用

    ## Example / 例

    ```json
    {
      "queries": [
        "What is RAG?",
        "How does vector search work?",
        "Explain cross-encoder re-ranking"
      ],
      "collection": "default",
      "top_k": 5,
      "use_hybrid": true,
      "rerank": true,
      "rank_results": true
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
        for idx, result in enumerate(results):
            sources = result.sources

            # Apply feature-based ranking if requested
            if batch_req.rank_results:
                try:
                    ranker = QueryResultRanker()
                    sources = ranker.rank_results(
                        query=batch_req.queries[idx],
                        results=sources,
                        top_k=batch_req.top_k
                    )
                    logger.info(f"Applied feature-based ranking to query {idx}")
                except Exception as e:
                    logger.error(f"Ranking failed for query {idx}: {e}")

            responses.append(QueryResponse(
                answer=result.answer,
                sources=sources,
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


@router.post(
    "/stream",
    summary="Stream RAG Query Response / RAGクエリレスポンスのストリーミング",
    description="Stream RAG query responses in real-time using Server-Sent Events (SSE) / Server-Sent Events (SSE) を使用してRAGクエリレスポンスをリアルタイムにストリーミングします",
    response_description="Server-Sent Events stream with incremental response chunks / 増分レスポンスチャンクを含むServer-Sent Eventsストリーム",
    responses={
        200: {
            "description": "Streaming response / ストリーミングレスポンス",
            "content": {
                "text/event-stream": {
                    "example": """data: {"content": "Based on", "is_done": false}

data: {"content": " the provided", "is_done": false}

data: {"content": " context...", "is_done": true, "sources": [...]}"""
                }
            }
        },
        400: {"description": "Invalid request parameters / 不正なリクエストパラメータ"},
        422: {"description": "Validation error / バリデーションエラー"},
        429: {"description": "Rate limit exceeded / レート制限超過"},
        500: {"description": "Internal server error / サーバー内部エラー"}
    },
    tags=["Query"]
)
@limiter.limit("60/minute")
async def stream_query(
    request: Request,
    stream_req: StreamingQueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    llm_client = Depends(get_llm_client)
) -> StreamingResponse:
    """
    Stream RAG query response using Server-Sent Events (SSE) / Server-Sent Events (SSE) を使用してRAGクエリレスポンスをストリーミングします

    ## Features / 機能

    - **Real-time Streaming**: Deliver LLM responses as they're generated / リアルタイムストリーミング: 生成されるLLMレスポンスをリアルタイムに配信
    - **Server-Sent Events**: Standard SSE protocol for easy client integration / Server-Sent Events: クライアント統合が容易な標準SSEプロトコル
    - **RAG Pipeline**: Full retrieval-augmented generation with context / RAGパイプライン: コンテキストを含む完全な検索拡張生成
    - **Backward Compatible**: Non-streaming endpoint remains available / 後方互換: 非ストリーミングエンドポイントも引き続き利用可能

    ## Parameters / パラメータ

    - **query**: Search query text (1-1000 characters) / 検索クエリテキスト (1-1000文字)
    - **collection**: Target collection name (default: "default") / 対象コレクション名 (デフォルト: "default")
    - **top_k**: Number of results to return (1-20) / 返却する結果数 (1-20)
    - **use_hybrid**: Enable hybrid search (default: true) / ハイブリッド検索を有効化 (デフォルト: true)
    - **rerank**: Apply re-ranking (default: true) / 再ランク付けを適用 (デフォルト: true)
    - **filters**: Optional metadata filters / オプションのメタデータフィルター
    - **max_tokens**: Maximum tokens to generate (100-4096, default: 2048) / 生成する最大トークン数 (100-4096, デフォルト: 2048)

    ## Client Integration Example / クライアント統合例

    ### JavaScript/TypeScript:
    ```javascript
    const eventSource = new EventSource('/query/stream?' + new URLSearchParams({
        query: 'What is RAG?',
        top_k: 5,
        use_hybrid: true
    }));

    let fullResponse = '';
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.content) {
            fullResponse += data.content;
            console.log('Chunk:', data.content);
        }
        if (data.is_done) {
            console.log('Complete:', fullResponse);
            console.log('Sources:', data.sources);
            eventSource.close();
        }
    };
    ```

    ### Python:
    ```python
    import requests
    import json

    response = requests.get(
        'http://localhost:8000/query/stream',
        params={'query': 'What is RAG?', 'top_k': 5},
        stream=True
    )

    full_response = ''
    for line in response.iter_lines():
        if line.startswith(b'data: '):
            data = json.loads(line[6:])
            if data.get('content'):
                full_response += data['content']
            if data.get('is_done'):
                print('Complete:', full_response)
                print('Sources:', data.get('sources'))
                break
    ```

    Args:
        request: FastAPI Request object
        stream_req: Streaming query request with parameters
        pipeline: RAG pipeline injected via dependency injection
        llm_client: Async OpenAI client for streaming

    Returns:
        StreamingResponse with SSE formatted chunks
    """
    try:
        # Initialize streaming service
        streaming_service = StreamingRAGService(
            llm_client=llm_client,
            max_tokens=stream_req.max_tokens
        )

        # Validate request parameters
        streaming_service.validate_stream_request(
            query=stream_req.query,
            top_k=stream_req.top_k,
            max_tokens=stream_req.max_tokens
        )

        # Create async generator for streaming
        async def generate() -> AsyncGenerator[str, None]:
            """Generate SSE stream chunks"""
            try:
                # Create retriever function with reranking support
                async def retriever_func(query, top_k, use_hybrid, filter_dict, rerank, collection):
                    """Wrapper for pipeline's retrieval logic with optional reranking"""
                    # Perform initial retrieval
                    results = await pipeline.retriever.retrieve(
                        query=query,
                        top_k=top_k,
                        use_hybrid=use_hybrid,
                        filter_dict=filter_dict,
                        collection=collection
                    )

                    # Apply reranking if requested and reranker is available
                    if rerank and pipeline.reranker:
                        from app.services.retrieval import RetrievalResult
                        # Convert to format expected by reranker
                        rerank_inputs = [
                            {
                                "document": r.document,
                                "score": r.score,
                                "metadata": r.metadata
                            }
                            for r in results
                        ]
                        reranked = pipeline.reranker.rerank(
                            query=query,
                            documents=rerank_inputs
                        )
                        # Convert back to RetrievalResult
                        results = [
                            RetrievalResult(
                                document=r["document"],
                                score=r["score"],
                                metadata=r["metadata"],
                                source=r["metadata"].get("source", "unknown")
                            )
                            for r in reranked
                        ]
                        logger.info(f"Applied reranking to {len(results)} results")

                    return results

                # Stream response with retrieval
                chunk_generator = streaming_service.stream_query_with_retrieval(
                    query=stream_req.query,
                    retriever_func=retriever_func,
                    top_k=stream_req.top_k,
                    use_hybrid=stream_req.use_hybrid,
                    rerank=stream_req.rerank,
                    filter_dict=stream_req.filters,
                    collection=stream_req.collection or "default"
                )

                # Format as SSE
                async for sse_chunk in format_sse_stream(chunk_generator):
                    yield sse_chunk

            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                # Send error as SSE
                error_sse = f'data: {{"error": "{str(e)}", "is_done": true}}\n\n'
                yield error_sse

        # Return SSE streaming response
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming failed: {str(e)}"
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
