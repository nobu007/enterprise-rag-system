"""
API Key Management Routes

This module defines API endpoints for managing API keys.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

from app.models.api_key import APIKey, APIKeyRole
from app.core.auth import get_auth_manager, APIKeyAuth
from app.api.dependencies import get_api_key


router = APIRouter(prefix="/auth/api-keys", tags=["Authentication"])


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating an API key"""

    name: str = Field(..., description="Name for the API key", min_length=1, max_length=100)
    role: APIKeyRole = Field(
        default=APIKeyRole.USER, description="Access role for the API key"
    )
    expires_in_days: Optional[int] = Field(
        default=None,
        description="Expiration in days (None = no expiration)",
        ge=1,
        le=3650,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Production API Key",
                    "role": "user",
                    "expires_in_days": 365,
                }
            ]
        }
    }


class CreateAPIKeyResponse(BaseModel):
    """Response model for creating an API key"""

    key_id: str = Field(..., description="Unique key identifier")
    key_value: str = Field(..., description="The API key value (only shown once)")
    name: str = Field(..., description="Key name")
    role: str = Field(..., description="Key role")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")


class APIKeyResponse(BaseModel):
    """Response model for API key details"""

    key_id: str = Field(..., description="Unique key identifier")
    name: str = Field(..., description="Key name")
    role: str = Field(..., description="Key role")
    is_active: bool = Field(..., description="Whether the key is active")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
    last_used_at: Optional[str] = Field(None, description="Last used timestamp")
    revoked_at: Optional[str] = Field(None, description="Revocation timestamp")


@router.post(
    "/",
    response_model=CreateAPIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API Key / APIキー作成",
    description="Create a new API key for authentication / 認証用の新しいAPIキーを作成します",
    response_description="Created API key with value / 作成されたAPIキーと値",
    responses={
        201: {"description": "API key created successfully / APIキー作成成功"},
        400: {"description": "Invalid request parameters / 不正なリクエストパラメータ"},
        401: {"description": "Authentication required / 認証が必要"},
        403: {"description": "Insufficient permissions / 権限不足"},
    },
)
async def create_api_key(
    request: CreateAPIKeyRequest,
    auth_key: APIKey = Depends(get_api_key),
) -> CreateAPIKeyResponse:
    """
    Create a new API key / 新しいAPIキーを作成します

    ## Permissions / 権限

    - **ADMIN** role: Can create keys with any role / あらゆるロールのキーを作成可能
    - **USER** role: Can only create USER and READ_ONLY keys / USERとREAD_ONLYキーのみ作成可能

    ## Parameters / パラメータ

    - **name**: Human-readable name for the key (required) / キーの識別名（必須）
    - **role**: Access role (default: "user") / アクセスロール（デフォルト: "user"）
    - **expires_in_days**: Optional expiration in days / 有効期限（日数）

    ## Example / 例

    ```json
    {
      "name": "Production App Key",
      "role": "user",
      "expires_in_days": 365
    }
    ```

    **Important**: Save the key_value securely. It will not be shown again.
    **重要**: key_valueを安全に保存してください。再表示されることはありません。

    Args:
        request: API key creation request
        auth_key: Authenticated API key from dependency

    Returns:
        CreateAPIKeyResponse with the new key

    Raises:
        HTTPException: If user lacks permission
    """
    # Check permissions
    if auth_key.role == APIKeyRole.READ_ONLY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="READ_ONLY keys cannot create new API keys",
        )

    if auth_key.role == APIKeyRole.USER and request.role == APIKeyRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="USER keys cannot create ADMIN keys",
        )

    # Create the key
    auth_manager = get_auth_manager()
    api_key = auth_manager.create_api_key(
        name=request.name,
        role=request.role,
        created_by=auth_key.key_id,
        expires_in_days=request.expires_in_days,
    )

    return CreateAPIKeyResponse(
        key_id=api_key.key_id,
        key_value=api_key.key_value,
        name=api_key.name,
        role=api_key.role.value,
        created_at=api_key.created_at.isoformat(),
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
    )


@router.get(
    "/",
    response_model=List[APIKeyResponse],
    summary="List API Keys / APIキー一覧",
    description="List all API keys (excluding revoked keys) / すべてのAPIキーを一覧表示します（取り消されたキーを除く）",
    response_description="List of API keys / APIキーのリスト",
    responses={
        200: {"description": "Success / 成功"},
        401: {"description": "Authentication required / 認証が必要"},
        403: {"description": "Insufficient permissions / 権限不足"},
    },
)
async def list_api_keys(
    auth_key: APIKey = Depends(get_api_key),
    include_revoked: bool = False,
) -> List[APIKeyResponse]:
    """
    List API keys / APIキーを一覧表示します

    ## Permissions / 権限

    - **ADMIN**: Can see all keys / すべてのキーを表示可能
    - **USER/READ_ONLY**: Can only see their own keys / 自分のキーのみ表示可能

    ## Parameters / パラメータ

    - **include_revoked**: Include revoked keys in the list / 取り消されたキーを含めるかどうか

    Args:
        auth_key: Authenticated API key
        include_revoked: Whether to include revoked keys

    Returns:
        List of APIKeyResponse objects
    """
    auth_manager = get_auth_manager()

    # Get all keys
    all_keys = auth_manager.list_api_keys(include_revoked=include_revoked)

    # Filter based on role
    if auth_key.role == APIKeyRole.ADMIN:
        # Admins can see all keys
        filtered_keys = all_keys
    else:
        # Non-admins can only see keys they created
        filtered_keys = [key for key in all_keys if key.created_by == auth_key.key_id]

    # Convert to response models
    return [
        APIKeyResponse(
            key_id=key.key_id,
            name=key.name,
            role=key.role.value,
            is_active=key.is_active,
            created_at=key.created_at.isoformat(),
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            revoked_at=key.revoked_at.isoformat() if key.revoked_at else None,
        )
        for key in filtered_keys
    ]


@router.get(
    "/{key_id}",
    response_model=APIKeyResponse,
    summary="Get API Key Details / APIキー詳細取得",
    description="Get details of a specific API key / 特定のAPIキーの詳細を取得します",
    response_description="API key details / APIキーの詳細",
    responses={
        200: {"description": "Success / 成功"},
        401: {"description": "Authentication required / 認証が必要"},
        403: {"description": "Insufficient permissions / 権限不足"},
        404: {"description": "API key not found / APIキーが見つかりません"},
    },
)
async def get_api_key_details(
    key_id: str,
    auth_key: APIKey = Depends(get_api_key),
) -> APIKeyResponse:
    """
    Get API key details / APIキーの詳細を取得します

    ## Permissions / 権限

    - **ADMIN**: Can view any key / 任意のキーを表示可能
    - **USER/READ_ONLY**: Can only view their own keys / 自分のキーのみ表示可能

    Args:
        key_id: The ID of the API key
        auth_key: Authenticated API key

    Returns:
        APIKeyResponse with key details

    Raises:
        HTTPException: If key not found or permission denied
    """
    auth_manager = get_auth_manager()
    api_key = auth_manager.get_api_key(key_id)

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    # Check permissions
    if auth_key.role != APIKeyRole.ADMIN and api_key.created_by != auth_key.key_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own API keys",
        )

    return APIKeyResponse(
        key_id=api_key.key_id,
        name=api_key.name,
        role=api_key.role.value,
        is_active=api_key.is_active,
        created_at=api_key.created_at.isoformat(),
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
        last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        revoked_at=api_key.revoked_at.isoformat() if api_key.revoked_at else None,
    )


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API Key / APIキー取り消し",
    description="Revoke an API key (cannot be undone) / APIキーを取り消します（元に戻せません）",
    responses={
        204: {"description": "API key revoked successfully / APIキー取り消し成功"},
        401: {"description": "Authentication required / 認証が必要"},
        403: {"description": "Insufficient permissions / 権限不足"},
        404: {"description": "API key not found / APIキーが見つかりません"},
    },
)
async def revoke_api_key(
    key_id: str,
    auth_key: APIKey = Depends(get_api_key),
):
    """
    Revoke an API key / APIキーを取り消します

    ## Permissions / 権限

    - **ADMIN**: Can revoke any key / 任意のキーを取り消し可能
    - **USER/READ_ONLY**: Can only revoke their own keys / 自分のキーのみ取り消し可能

    **Warning**: This action cannot be undone. The key will immediately become invalid.
    **警告**: この操作は元に戻せません。キーは直ちに無効になります。

    Args:
        key_id: The ID of the API key to revoke
        auth_key: Authenticated API key

    Raises:
        HTTPException: If key not found or permission denied
    """
    auth_manager = get_auth_manager()
    api_key = auth_manager.get_api_key(key_id)

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    # Check permissions
    if auth_key.role != APIKeyRole.ADMIN and api_key.created_by != auth_key.key_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only revoke your own API keys",
        )

    # Revoke the key
    success = auth_manager.revoke_api_key(key_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key",
        )
