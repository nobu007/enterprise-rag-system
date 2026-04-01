"""
Tenant middleware for FastAPI applications

This middleware provides automatic tenant identification and context injection
for multi-tenant applications.
"""

from typing import Optional, Callable
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

from app.core.tenant import (
    TenantContext,
    Tenant,
    TenantManager,
    get_tenant_manager,
    TenantIsolationError,
)
from app.core.logging_config import get_logger


logger = get_logger(__name__)


# Header names for tenant identification
TENANT_ID_HEADER = "X-Tenant-ID"
API_KEY_HEADER = "X-API-Key"


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware to identify and inject tenant context into requests

    This middleware extracts tenant information from headers and validates
    tenant status before allowing the request to proceed.

    Tenant identification methods (in order of priority):
    1. X-API-Key header: API key authentication
    2. X-Tenant-ID header: Direct tenant ID

    The middleware adds a `tenant_context` attribute to the request state.
    """

    def __init__(
        self,
        app: ASGIApp,
        tenant_manager: Optional[TenantManager] = None,
        require_tenant: bool = False,
        default_tenant_id: str = "default",
        enable_isolation: bool = True,
    ) -> None:
        """
        Initialize tenant middleware

        Args:
            app: ASGI application
            tenant_manager: Tenant manager instance (uses global if None)
            require_tenant: Whether to require valid tenant for all requests
            default_tenant_id: Default tenant ID if none specified
            enable_isolation: Whether to enforce tenant isolation
        """
        super().__init__(app)
        self.tenant_manager = tenant_manager or get_tenant_manager()
        self.require_tenant = require_tenant
        self.default_tenant_id = default_tenant_id
        self.enable_isolation = enable_isolation

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request and inject tenant context

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response from downstream handler

        Raises:
            HTTPException: If tenant is invalid or not found
        """
        # Extract tenant identification from headers
        api_key = request.headers.get(API_KEY_HEADER)
        tenant_id = request.headers.get(TENANT_ID_HEADER)

        # Determine tenant ID
        resolved_tenant_id = self._resolve_tenant_id(api_key, tenant_id)

        # Get tenant object
        tenant = None
        if resolved_tenant_id and resolved_tenant_id != self.default_tenant_id:
            tenant = self.tenant_manager.get_tenant(resolved_tenant_id)

        # Validate tenant
        if self.require_tenant and not tenant:
            logger.warning(
                f"Tenant required but not found: {resolved_tenant_id}",
                extra={"tenant_id": resolved_tenant_id, "path": request.url.path},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Valid tenant authentication required",
            )

        # Check tenant status
        if tenant and not tenant.is_active():
            logger.warning(
                f"Inactive tenant access attempt: {resolved_tenant_id}",
                extra={"tenant_id": resolved_tenant_id, "status": tenant.status},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Tenant is not active (status: {tenant.status.value})",
            )

        # Create tenant context
        context = TenantContext(
            tenant_id=resolved_tenant_id or self.default_tenant_id,
            tenant=tenant,
            is_isolated=self.enable_isolation,
            request_id=request.headers.get("X-Request-ID"),
        )

        # Inject context into request state
        request.state.tenant_context = context

        # Log tenant access
        logger.info(
            f"Request processed for tenant: {context.tenant_id}",
            extra={
                "tenant_id": context.tenant_id,
                "path": request.url.path,
                "method": request.method,
                "is_isolated": context.is_isolated,
            },
        )

        # Process request
        try:
            response = await call_next(request)
            return response
        except TenantIsolationError as e:
            logger.error(
                f"Tenant isolation violation: {str(e)}",
                extra={"tenant_id": context.tenant_id},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant isolation violation",
            )

    def _resolve_tenant_id(
        self,
        api_key: Optional[str],
        tenant_id: Optional[str],
    ) -> Optional[str]:
        """
        Resolve tenant ID from various identification methods

        Args:
            api_key: API key from header
            tenant_id: Tenant ID from header

        Returns:
            Resolved tenant ID or None
        """
        # Priority 1: API key authentication
        if api_key:
            resolved_id = self.tenant_manager.validate_api_key(api_key)
            if resolved_id:
                return resolved_id
            logger.warning(f"Invalid API key provided: {api_key[:10]}...")

        # Priority 2: Direct tenant ID
        if tenant_id:
            return tenant_id

        # Fallback: default tenant
        if not self.require_tenant:
            return self.default_tenant_id

        return None


def get_tenant_context(request: Request) -> TenantContext:
    """
    Extract tenant context from request

    This is a helper function to be used in route handlers.

    Args:
        request: FastAPI request object

    Returns:
        TenantContext from request state

    Raises:
        HTTPException: If tenant context is not found
    """
    context = getattr(request.state, "tenant_context", None)
    if not context:
        logger.error("Tenant context not found in request state")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tenant context not initialized",
        )
    return context


def require_tenant(
    request: Request,
    allow_default: bool = False,
) -> Tenant:
    """
    Get authenticated tenant from request

    This helper ensures that a valid, authenticated tenant is present.

    Args:
        request: FastAPI request object
        allow_default: Whether to allow default tenant

    Returns:
        Tenant object

    Raises:
        HTTPException: If tenant is not authenticated
    """
    context = get_tenant_context(request)

    if not allow_default and context.tenant_id == "default":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant authentication required",
        )

    if not context.tenant:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant not found",
        )

    return context.tenant


def require_active_tenant(
    request: Request,
    allow_default: bool = False,
) -> Tenant:
    """
    Get active tenant from request

    This helper ensures that a valid, active tenant is present.

    Args:
        request: FastAPI request object
        allow_default: Whether to allow default tenant

    Returns:
        Active Tenant object

    Raises:
        HTTPException: If tenant is not active
    """
    tenant = require_tenant(request, allow_default=allow_default)

    if not tenant.is_active():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tenant is not active (status: {tenant.status.value})",
        )

    return tenant


# Dependency function for FastAPI dependency injection
async def get_current_tenant(
    request: Request,
) -> Optional[Tenant]:
    """
    FastAPI dependency to get current tenant

    Usage in route handlers:
        @app.get("/api/endpoint")
        async def endpoint(tenant: Optional[Tenant] = Depends(get_current_tenant)):
            ...

    Args:
        request: FastAPI request object

    Returns:
        Current tenant or None
    """
    context = get_tenant_context(request)
    return context.tenant


async def get_required_tenant(
    request: Request,
) -> Tenant:
    """
    FastAPI dependency to get required authenticated tenant

    Usage in route handlers:
        @app.get("/api/endpoint")
        async def endpoint(tenant: Tenant = Depends(get_required_tenant)):
            ...

    Args:
        request: FastAPI request object

    Returns:
        Current tenant

    Raises:
        HTTPException: If tenant is not authenticated
    """
    return require_tenant(request)


async def get_active_tenant(
    request: Request,
) -> Tenant:
    """
    FastAPI dependency to get required active tenant

    Usage in route handlers:
        @app.get("/api/endpoint")
        async def endpoint(tenant: Tenant = Depends(get_active_tenant)):
            ...

    Args:
        request: FastAPI request object

    Returns:
        Current active tenant

    Raises:
        HTTPException: If tenant is not active
    """
    return require_active_tenant(request)
