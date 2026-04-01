"""
Multi-tenant support module for Enterprise RAG System

This module provides tenant isolation and management capabilities,
ensuring data segregation between different tenants/organizations.
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime
import hashlib
import secrets
from pydantic import BaseModel, Field, field_validator


class TenantStatus(str, Enum):
    """Tenant status enumeration"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    ARCHIVED = "archived"


class Tenant(BaseModel):
    """
    Tenant model representing an organization/customer

    Attributes:
        tenant_id: Unique identifier for the tenant
        name: Human-readable tenant name
        status: Current status of the tenant
        config: Tenant-specific configuration
        metadata: Additional tenant metadata
        created_at: Timestamp when tenant was created
        updated_at: Timestamp when tenant was last updated
    """

    tenant_id: str = Field(..., description="Unique tenant identifier")
    name: str = Field(..., min_length=1, max_length=255, description="Tenant name")
    status: TenantStatus = Field(default=TenantStatus.PENDING, description="Tenant status")
    config: Dict[str, Any] = Field(default_factory=dict, description="Tenant configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Tenant metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

    @field_validator('tenant_id')
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        """Validate tenant ID format"""
        if not v or not v.strip():
            raise ValueError("tenant_id cannot be empty")
        # Ensure tenant_id is safe (no special characters that could cause issues)
        if not all(c.isalnum() or c in ('-', '_') for c in v):
            raise ValueError("tenant_id can only contain alphanumeric characters, hyphens, and underscores")
        return v.strip()

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate tenant name"""
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()

    def is_active(self) -> bool:
        """Check if tenant is active"""
        return self.status == TenantStatus.ACTIVE

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value for this tenant

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def update_config(self, key: str, value: Any) -> None:
        """
        Update a configuration value for this tenant

        Args:
            key: Configuration key
            value: New value
        """
        self.config[key] = value
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert tenant to dictionary representation"""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "status": self.status.value,
            "config": self.config,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class TenantContext(BaseModel):
    """
    Tenant context for request-scoped tenant information

    This context is attached to each request and contains
    tenant-specific information for the duration of the request.
    """

    tenant_id: str = Field(..., description="Tenant identifier for this request")
    tenant: Optional[Tenant] = Field(default=None, description="Full tenant object")
    is_isolated: bool = Field(default=True, description="Whether tenant isolation is enforced")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")

    def get_tenant_id(self) -> str:
        """Get tenant ID with fallback"""
        return self.tenant_id or "default"

    def is_authenticated(self) -> bool:
        """Check if tenant context is authenticated"""
        return bool(self.tenant_id and self.tenant_id != "default")


class TenantIsolationError(Exception):
    """Exception raised when tenant isolation is violated"""
    pass


class TenantManager:
    """
    Manager class for tenant operations

    Handles tenant CRUD operations, validation, and isolation enforcement.
    """

    def __init__(self):
        """Initialize tenant manager with in-memory storage"""
        # In production, this would use a database
        self._tenants: Dict[str, Tenant] = {}
        self._api_keys: Dict[str, str] = {}  # api_key -> tenant_id mapping

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        status: TenantStatus = TenantStatus.PENDING,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """
        Create a new tenant

        Args:
            tenant_id: Unique tenant identifier
            name: Human-readable name
            status: Initial status
            config: Optional configuration
            metadata: Optional metadata

        Returns:
            Created tenant object

        Raises:
            ValueError: If tenant already exists or validation fails
        """
        if tenant_id in self._tenants:
            raise ValueError(f"Tenant {tenant_id} already exists")

        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            status=status,
            config=config or {},
            metadata=metadata or {},
        )

        self._tenants[tenant_id] = tenant
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """
        Get tenant by ID

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant object or None if not found
        """
        return self._tenants.get(tenant_id)

    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        status: Optional[TenantStatus] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tenant]:
        """
        Update an existing tenant

        Args:
            tenant_id: Tenant identifier
            name: New name (optional)
            status: New status (optional)
            config: New config (optional)
            metadata: New metadata (optional)

        Returns:
            Updated tenant or None if not found
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        if name is not None:
            tenant.name = name
        if status is not None:
            tenant.status = status
        if config is not None:
            tenant.config.update(config)
        if metadata is not None:
            tenant.metadata.update(metadata)

        tenant.updated_at = datetime.utcnow()
        return tenant

    def delete_tenant(self, tenant_id: str) -> bool:
        """
        Delete a tenant

        Args:
            tenant_id: Tenant identifier

        Returns:
            True if deleted, False if not found
        """
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            # Remove associated API keys
            self._api_keys = {
                k: v for k, v in self._api_keys.items() if v != tenant_id
            }
            return True
        return False

    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        limit: int = 100,
    ) -> List[Tenant]:
        """
        List tenants with optional filtering

        Args:
            status: Filter by status (optional)
            limit: Maximum number of results

        Returns:
            List of tenants
        """
        tenants = list(self._tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status]

        return tenants[:limit]

    def generate_api_key(self, tenant_id: str) -> str:
        """
        Generate API key for a tenant

        Args:
            tenant_id: Tenant identifier

        Returns:
            Generated API key

        Raises:
            ValueError: If tenant doesn't exist
        """
        if tenant_id not in self._tenants:
            raise ValueError(f"Tenant {tenant_id} does not exist")

        # Generate secure API key
        api_key = f"rag_{secrets.token_urlsafe(32)}"
        self._api_keys[api_key] = tenant_id

        return api_key

    def validate_api_key(self, api_key: str) -> Optional[str]:
        """
        Validate API key and return tenant ID

        Args:
            api_key: API key to validate

        Returns:
            Tenant ID if valid, None otherwise
        """
        return self._api_keys.get(api_key)

    def get_tenant_count(self) -> int:
        """Get total number of tenants"""
        return len(self._tenants)

    def get_active_tenant_count(self) -> int:
        """Get number of active tenants"""
        return sum(1 for t in self._tenants.values() if t.is_active())


# Global tenant manager instance
_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """
    Get global tenant manager instance

    Returns:
        TenantManager instance
    """
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager


def create_tenant_context(
    tenant_id: str,
    tenant: Optional[Tenant] = None,
    is_isolated: bool = True,
    request_id: Optional[str] = None,
) -> TenantContext:
    """
    Create a tenant context

    Args:
        tenant_id: Tenant identifier
        tenant: Full tenant object (optional)
        is_isolated: Whether isolation is enforced
        request_id: Request tracking ID

    Returns:
        TenantContext object
    """
    return TenantContext(
        tenant_id=tenant_id,
        tenant=tenant,
        is_isolated=is_isolated,
        request_id=request_id,
    )


def validate_tenant_isolation(
    context: TenantContext,
    target_tenant_id: str,
) -> bool:
    """
    Validate that an operation respects tenant isolation

    Args:
        context: Current tenant context
        target_tenant_id: Target tenant for the operation

    Returns:
        True if isolation is respected

    Raises:
        TenantIsolationError: If isolation would be violated
    """
    if not context.is_isolated:
        return True

    if context.tenant_id != target_tenant_id:
        raise TenantIsolationError(
            f"Tenant isolation violation: context tenant {context.tenant_id} "
            f"attempting to access tenant {target_tenant_id}"
        )

    return True
