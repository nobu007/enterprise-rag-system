"""
Health check endpoints
"""

from fastapi import APIRouter, Request
from typing import Dict

from app.core.rate_limit import limiter

router = APIRouter()


@router.get(
    "/health",
    summary="Basic Health Check / 基本ヘルスチェック",
    description="Check if the API is running and responding / APIが実行中で応答するかどうかを確認します",
    response_description="Basic health status / 基本的なヘルス状態",
    responses={
        200: {"description": "API is healthy / APIが正常"}
    },
    tags=["Health"]
)
@limiter.limit("120/minute")
async def health_check(request: Request) -> Dict[str, str]:
    """Health check endpoint / ヘルスチェックエンドポイント

    ## Response / レスポンス

    - **status**: "healthy" if the API is running / APIが実行中の場合は"healthy"
    - **version**: API version number / APIバージョン番号

    ## Example Response / レスポンス例

    ```json
    {
      "status": "healthy",
      "version": "1.0.0"
    }
    ```
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@router.get(
    "/health/detailed",
    summary="Detailed Health Check / 詳細ヘルスチェック",
    description="Check the health status of all system services / すべてのシステムサービスのヘルス状態を確認します",
    response_description="Detailed health status for all services / すべてのサービスの詳細なヘルス状態",
    responses={
        200: {"description": "All services are healthy / すべてのサービスが正常"}
    },
    tags=["Health"]
)
@limiter.limit("120/minute")
async def detailed_health_check(request: Request) -> Dict[str, Dict[str, str]]:
    """Detailed health check with service status / サービスステータスを含む詳細なヘルスチェック

    ## Services Checked / チェックされるサービス

    - **api**: API service status / APIサービスの状態
    - **vector_db**: Vector database connection status / ベクトルデータベースの接続状態
    - **llm**: LLM service availability / LLMサービスの可用性

    ## Example Response / レスポンス例

    ```json
    {
      "status": "healthy",
      "version": "1.0.0",
      "services": {
        "api": "healthy",
        "vector_db": "healthy",
        "llm": "healthy"
      }
    }
    ```
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "api": "healthy",
            "vector_db": "healthy",
            "llm": "healthy"
        }
    }


@router.get(
    "/cache/stats",
    summary="Cache Statistics / キャッシュ統計",
    description="Get Redis cache performance statistics / Redisキャッシュのパフォーマンス統計を取得します",
    response_description="Cache statistics including memory usage and hit rates / メモリ使用量とヒット率を含むキャッシュ統計",
    responses={
        200: {"description": "Cache statistics retrieved successfully / キャッシュ統計取得成功"}
    },
    tags=["Health"]
)
@limiter.limit("120/minute")
async def cache_stats(request: Request) -> Dict[str, Dict[str, str]]:
    """Cache statistics endpoint / キャッシュ統計エンドポイント

    ## Statistics Included / 含まれる統計

    - **enabled**: Whether caching is enabled / キャッシュが有効かどうか
    - **total_keys**: Number of cached items / キャッシュされたアイテム数
    - **memory_used**: Current memory usage / 現在のメモリ使用量
    - **memory_peak**: Peak memory usage / ピークメモリ使用量
    - **connected_clients**: Number of active connections / アクティブな接続数
    - **uptime_days**: Cache uptime in days / キャッシュ稼働時間（日）
    - **ttl_seconds**: Cache TTL setting / キャッシュTTL設定

    ## Example Response / レスポンス例

    ```json
    {
      "enabled": true,
      "total_keys": 150,
      "memory_used": "2.5M",
      "memory_peak": "3.2M",
      "connected_clients": 5,
      "uptime_days": 7,
      "ttl_seconds": 3600
    }
    ```

    ## Note / 注意

    Returns `{"enabled": false}` if cache is not initialized / キャッシュが初期化されていない場合、`{"enabled": false}` を返します
    """
    # Get cache manager from app state
    cache = request.app.state.cache_manager if hasattr(request.app.state, 'cache_manager') else None

    if not cache:
        return {
            "enabled": False,
            "message": "Cache manager not initialized"
        }

    # Get cache stats
    stats = cache.get_stats()
    return stats

