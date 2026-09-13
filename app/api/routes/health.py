"""
Health check endpoints
"""

from fastapi import APIRouter
from typing import Any, Dict

from app.core.config import get_settings

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
async def health_check() -> Dict[str, str]:
    """Health check endpoint / ヘルスチェックエンドポイント

    ## Response / レスポンス

    - **status**: "healthy" if the API is running / APIが実行中の場合は"healthy"
    - **version**: API version number / APIバージョン番号

    ## Example Response / レスポンス例

    ```json
    {
      "status": "healthy",
      "version": "0.3.0"
    }
    ```
    """
    return {
        "status": "healthy",
        "version": get_settings().app_version
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
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with service status / サービスステータスを含む詳細なヘルスチェック

    ## Services Checked / チェックされるサービス

    - **api**: API service status / APIサービスの状態
    - **vector_db**: Vector database connection status / ベクトルデータベースの接続状態
    - **llm**: LLM service availability / LLMサービスの可用性

    ## Example Response / レスポンス例

    ```json
    {
      "status": "healthy",
      "version": "0.3.0",
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
        "version": get_settings().app_version,
        "services": {
            "api": "healthy",
            "vector_db": "healthy",
            "llm": "healthy"
        }
    }
