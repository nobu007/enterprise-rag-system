"""
Unit tests for multi-tenant support (Feature 34)

Tests cover:
- Tenant model creation and validation
- Tenant context management
- Tenant manager operations
- Tenant middleware functionality
- Tenant isolation enforcement
- API key generation and validation
- Error handling and edge cases
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request, HTTPException
from starlette.types import ASGIApp

from app.core.tenant import (
    Tenant,
    TenantStatus,
    TenantContext,
    TenantManager,
    TenantIsolationError,
    get_tenant_manager,
    create_tenant_context,
    validate_tenant_isolation,
)
from app.middleware.tenant import (
    TenantMiddleware,
    get_tenant_context,
    require_tenant,
    require_active_tenant,
    get_current_tenant,
    get_required_tenant,
    get_active_tenant,
    TENANT_ID_HEADER,
    API_KEY_HEADER,
)


# ============================================================================
# Test Tenant Model
# ============================================================================

class TestTenantModel:
    """Test tenant model functionality"""

    def test_create_tenant_with_valid_data(self):
        """Test creating a tenant with valid data"""
        tenant = Tenant(
            tenant_id="tenant-001",
            name="Acme Corporation",
            status=TenantStatus.ACTIVE,
            config={"max_documents": 1000},
            metadata={"industry": "Technology"},
        )

        assert tenant.tenant_id == "tenant-001"
        assert tenant.name == "Acme Corporation"
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.config["max_documents"] == 1000
        assert tenant.metadata["industry"] == "Technology"
        assert isinstance(tenant.created_at, datetime)
        assert isinstance(tenant.updated_at, datetime)

    def test_create_tenant_with_defaults(self):
        """Test creating a tenant with default values"""
        tenant = Tenant(tenant_id="tenant-002", name="Default Corp")

        assert tenant.tenant_id == "tenant-002"
        assert tenant.name == "Default Corp"
        assert tenant.status == TenantStatus.PENDING
        assert tenant.config == {}
        assert tenant.metadata == {}

    def test_tenant_id_validation_empty(self):
        """Test that empty tenant_id raises validation error"""
        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            Tenant(tenant_id="", name="Test")

    def test_tenant_id_validation_invalid_chars(self):
        """Test that invalid characters in tenant_id raise validation error"""
        with pytest.raises(ValueError, match="can only contain alphanumeric"):
            Tenant(tenant_id="tenant@001", name="Test")

    def test_tenant_id_validation_whitespace(self):
        """Test that whitespace-only tenant_id raises validation error"""
        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            Tenant(tenant_id="   ", name="Test")

    def test_name_validation_empty(self):
        """Test that empty name raises validation error"""
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            Tenant(tenant_id="tenant-001", name="")

    def test_name_validation_whitespace(self):
        """Test that whitespace-only name raises validation error"""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Tenant(tenant_id="tenant-001", name="   ")

    def test_is_active_method(self):
        """Test the is_active method"""
        active_tenant = Tenant(tenant_id="t1", name="Active", status=TenantStatus.ACTIVE)
        suspended_tenant = Tenant(tenant_id="t2", name="Suspended", status=TenantStatus.SUSPENDED)
        pending_tenant = Tenant(tenant_id="t3", name="Pending", status=TenantStatus.PENDING)

        assert active_tenant.is_active() is True
        assert suspended_tenant.is_active() is False
        assert pending_tenant.is_active() is False

    def test_get_config_value(self):
        """Test getting configuration values"""
        tenant = Tenant(
            tenant_id="t1",
            name="Test",
            config={"max_docs": 100, "enable_cache": True},
        )

        assert tenant.get_config_value("max_docs") == 100
        assert tenant.get_config_value("enable_cache") is True
        assert tenant.get_config_value("nonexistent", "default") == "default"
        assert tenant.get_config_value("nonexistent") is None

    def test_update_config(self):
        """Test updating configuration values"""
        tenant = Tenant(tenant_id="t1", name="Test")

        initial_updated_at = tenant.updated_at
        tenant.update_config("new_key", "new_value")

        assert tenant.config["new_key"] == "new_value"
        assert tenant.updated_at > initial_updated_at

    def test_to_dict(self):
        """Test converting tenant to dictionary"""
        tenant = Tenant(
            tenant_id="t1",
            name="Test Tenant",
            status=TenantStatus.ACTIVE,
            config={"key": "value"},
        )

        tenant_dict = tenant.to_dict()

        assert tenant_dict["tenant_id"] == "t1"
        assert tenant_dict["name"] == "Test Tenant"
        assert tenant_dict["status"] == "active"
        assert tenant_dict["config"] == {"key": "value"}
        assert "created_at" in tenant_dict
        assert "updated_at" in tenant_dict


# ============================================================================
# Test TenantContext
# ============================================================================

class TestTenantContext:
    """Test tenant context functionality"""

    def test_create_tenant_context(self):
        """Test creating a tenant context"""
        context = TenantContext(
            tenant_id="tenant-001",
            is_isolated=True,
            request_id="req-123",
        )

        assert context.tenant_id == "tenant-001"
        assert context.is_isolated is True
        assert context.request_id == "req-123"
        assert context.tenant is None

    def test_create_tenant_context_with_tenant(self):
        """Test creating a tenant context with full tenant object"""
        tenant = Tenant(tenant_id="t1", name="Test")
        context = TenantContext(tenant_id="t1", tenant=tenant)

        assert context.tenant_id == "t1"
        assert context.tenant == tenant

    def test_get_tenant_id_with_fallback(self):
        """Test get_tenant_id with default fallback"""
        context1 = TenantContext(tenant_id="t1")
        assert context1.get_tenant_id() == "t1"

        context2 = TenantContext(tenant_id="")
        # Empty string returns "default" per the get_tenant_id logic
        assert context2.get_tenant_id() == "default"

        context3 = TenantContext(tenant_id="default")
        assert context3.get_tenant_id() == "default"

        context4 = TenantContext(tenant_id="   ")
        # Whitespace returns empty after strip
        assert context4.get_tenant_id().strip() == ""

    def test_is_authenticated(self):
        """Test is_authenticated method"""
        context1 = TenantContext(tenant_id="tenant-001")
        assert context1.is_authenticated() is True

        context2 = TenantContext(tenant_id="default")
        assert context2.is_authenticated() is False

        context3 = TenantContext(tenant_id="")
        # Empty string is falsy, so it should return False
        assert context3.is_authenticated() is False


# ============================================================================
# Test TenantManager
# ============================================================================

class TestTenantManager:
    """Test tenant manager functionality"""

    @pytest.fixture
    def manager(self):
        """Create a fresh tenant manager for each test"""
        return TenantManager()

    def test_create_tenant(self, manager):
        """Test creating a new tenant"""
        tenant = manager.create_tenant(
            tenant_id="t1",
            name="Test Tenant",
            status=TenantStatus.ACTIVE,
        )

        assert tenant.tenant_id == "t1"
        assert tenant.name == "Test Tenant"
        assert tenant.status == TenantStatus.ACTIVE

    def test_create_duplicate_tenant_raises_error(self, manager):
        """Test that creating duplicate tenant raises error"""
        manager.create_tenant(tenant_id="t1", name="First")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_tenant(tenant_id="t1", name="Duplicate")

    def test_get_tenant(self, manager):
        """Test retrieving a tenant"""
        created = manager.create_tenant(tenant_id="t1", name="Test")
        retrieved = manager.get_tenant("t1")

        assert retrieved is not None
        assert retrieved.tenant_id == "t1"
        assert retrieved.name == "Test"

    def test_get_nonexistent_tenant_returns_none(self, manager):
        """Test that getting nonexistent tenant returns None"""
        result = manager.get_tenant("nonexistent")
        assert result is None

    def test_update_tenant(self, manager):
        """Test updating a tenant"""
        manager.create_tenant(tenant_id="t1", name="Original")
        updated = manager.update_tenant(
            tenant_id="t1",
            name="Updated",
            status=TenantStatus.ACTIVE,
            config={"new_key": "new_value"},
        )

        assert updated is not None
        assert updated.name == "Updated"
        assert updated.status == TenantStatus.ACTIVE
        assert updated.config["new_key"] == "new_value"

    def test_update_nonexistent_tenant_returns_none(self, manager):
        """Test that updating nonexistent tenant returns None"""
        result = manager.update_tenant(tenant_id="nonexistent", name="Test")
        assert result is None

    def test_delete_tenant(self, manager):
        """Test deleting a tenant"""
        manager.create_tenant(tenant_id="t1", name="Test")
        result = manager.delete_tenant("t1")

        assert result is True
        assert manager.get_tenant("t1") is None

    def test_delete_nonexistent_tenant_returns_false(self, manager):
        """Test that deleting nonexistent tenant returns False"""
        result = manager.delete_tenant("nonexistent")
        assert result is False

    def test_list_tenants(self, manager):
        """Test listing tenants"""
        manager.create_tenant(tenant_id="t1", name="Tenant 1", status=TenantStatus.ACTIVE)
        manager.create_tenant(tenant_id="t2", name="Tenant 2", status=TenantStatus.SUSPENDED)
        manager.create_tenant(tenant_id="t3", name="Tenant 3", status=TenantStatus.ACTIVE)

        all_tenants = manager.list_tenants()
        assert len(all_tenants) == 3

        active_tenants = manager.list_tenants(status=TenantStatus.ACTIVE)
        assert len(active_tenants) == 2

        suspended_tenants = manager.list_tenants(status=TenantStatus.SUSPENDED)
        assert len(suspended_tenants) == 1

    def test_list_tenants_with_limit(self, manager):
        """Test listing tenants with limit"""
        for i in range(10):
            manager.create_tenant(tenant_id=f"t{i}", name=f"Tenant {i}")

        tenants = manager.list_tenants(limit=5)
        assert len(tenants) == 5

    def test_generate_api_key(self, manager):
        """Test generating API key for tenant"""
        manager.create_tenant(tenant_id="t1", name="Test")
        api_key = manager.generate_api_key("t1")

        assert api_key.startswith("rag_")
        assert len(api_key) > 10

    def test_generate_api_key_for_nonexistent_tenant(self, manager):
        """Test that generating API key for nonexistent tenant raises error"""
        with pytest.raises(ValueError, match="does not exist"):
            manager.generate_api_key("nonexistent")

    def test_validate_api_key(self, manager):
        """Test validating API key"""
        manager.create_tenant(tenant_id="t1", name="Test")
        api_key = manager.generate_api_key("t1")

        tenant_id = manager.validate_api_key(api_key)
        assert tenant_id == "t1"

    def test_validate_invalid_api_key(self, manager):
        """Test that invalid API key returns None"""
        result = manager.validate_api_key("invalid_key")
        assert result is None

    def test_delete_tenant_removes_api_keys(self, manager):
        """Test that deleting tenant also removes API keys"""
        manager.create_tenant(tenant_id="t1", name="Test")
        api_key = manager.generate_api_key("t1")

        manager.delete_tenant("t1")

        # API key should no longer be valid
        assert manager.validate_api_key(api_key) is None

    def test_get_tenant_count(self, manager):
        """Test getting tenant count"""
        assert manager.get_tenant_count() == 0

        manager.create_tenant(tenant_id="t1", name="Tenant 1")
        manager.create_tenant(tenant_id="t2", name="Tenant 2")
        assert manager.get_tenant_count() == 2

    def test_get_active_tenant_count(self, manager):
        """Test getting active tenant count"""
        manager.create_tenant(tenant_id="t1", name="Tenant 1", status=TenantStatus.ACTIVE)
        manager.create_tenant(tenant_id="t2", name="Tenant 2", status=TenantStatus.SUSPENDED)
        manager.create_tenant(tenant_id="t3", name="Tenant 3", status=TenantStatus.ACTIVE)

        assert manager.get_active_tenant_count() == 2


# ============================================================================
# Test Tenant Isolation
# ============================================================================

class TestTenantIsolation:
    """Test tenant isolation functionality"""

    def test_validate_isolation_same_tenant(self):
        """Test that same tenant access is allowed"""
        context = TenantContext(tenant_id="t1", is_isolated=True)

        result = validate_tenant_isolation(context, "t1")
        assert result is True

    def test_validate_isolation_different_tenant_raises_error(self):
        """Test that cross-tenant access raises error"""
        context = TenantContext(tenant_id="t1", is_isolated=True)

        with pytest.raises(TenantIsolationError, match="Tenant isolation violation"):
            validate_tenant_isolation(context, "t2")

    def test_validate_isolation_when_disabled(self):
        """Test that isolation can be disabled"""
        context = TenantContext(tenant_id="t1", is_isolated=False)

        # Should not raise even for different tenant
        result = validate_tenant_isolation(context, "t2")
        assert result is True


# ============================================================================
# Test Tenant Middleware
# ============================================================================

class TestTenantMiddleware:
    """Test tenant middleware functionality"""

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app"""
        async def app(scope, receive, send):
            pass
        return app

    @pytest.fixture
    def manager(self):
        """Create tenant manager with test data"""
        manager = TenantManager()
        manager.create_tenant(tenant_id="t1", name="Tenant 1", status=TenantStatus.ACTIVE)
        manager.create_tenant(tenant_id="t2", name="Tenant 2", status=TenantStatus.SUSPENDED)
        manager.generate_api_key("t1")  # Generate API key for testing
        return manager

    @pytest.mark.asyncio
    async def test_middleware_with_api_key(self, mock_app, manager):
        """Test middleware identifies tenant from API key"""
        middleware = TenantMiddleware(mock_app, tenant_manager=manager)

        # Get the API key that was generated
        api_key = list(manager._api_keys.keys())[0]

        # Create mock request
        request = MagicMock(spec=Request)
        request.headers = {API_KEY_HEADER: api_key}
        request.url.path = "/test"
        request.method = "GET"
        request.state = MagicMock()

        # Create mock call_next
        async def call_next(req):
            response = MagicMock()
            return response

        response = await middleware.dispatch(request, call_next)

        # Verify tenant context was set
        assert hasattr(request.state, "tenant_context")
        assert request.state.tenant_context.tenant_id == "t1"

    @pytest.mark.asyncio
    async def test_middleware_with_tenant_id_header(self, mock_app, manager):
        """Test middleware identifies tenant from X-Tenant-ID header"""
        middleware = TenantMiddleware(mock_app, tenant_manager=manager)

        request = MagicMock(spec=Request)
        request.headers = {TENANT_ID_HEADER: "t1"}
        request.url.path = "/test"
        request.method = "GET"
        request.state = MagicMock()

        async def call_next(req):
            return MagicMock()

        await middleware.dispatch(request, call_next)

        assert request.state.tenant_context.tenant_id == "t1"

    @pytest.mark.asyncio
    async def test_middleware_with_inactive_tenant(self, mock_app, manager):
        """Test middleware rejects inactive tenant"""
        middleware = TenantMiddleware(
            mock_app,
            tenant_manager=manager,
            require_tenant=True,
        )

        request = MagicMock(spec=Request)
        request.headers = {TENANT_ID_HEADER: "t2"}  # Suspended tenant
        request.url.path = "/test"
        request.method = "GET"
        request.state = MagicMock()

        async def call_next(req):
            return MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next)

        assert exc_info.value.status_code == 403
        assert "not active" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_middleware_requires_tenant(self, mock_app, manager):
        """Test middleware requires tenant when require_tenant=True"""
        middleware = TenantMiddleware(
            mock_app,
            tenant_manager=manager,
            require_tenant=True,
        )

        request = MagicMock(spec=Request)
        request.headers = {}
        request.url.path = "/test"
        request.method = "GET"
        request.state = MagicMock()

        async def call_next(req):
            return MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next)

        assert exc_info.value.status_code == 401


# ============================================================================
# Test Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Test middleware helper functions"""

    def test_get_tenant_context_from_request(self):
        """Test extracting tenant context from request"""
        context = TenantContext(tenant_id="t1")
        request = MagicMock()
        request.state.tenant_context = context

        result = get_tenant_context(request)
        assert result.tenant_id == "t1"

    def test_get_tenant_context_not_found_raises_error(self):
        """Test that missing tenant context raises error"""
        # Create a simple object without tenant_context attribute
        class SimpleState:
            pass

        class SimpleRequest:
            def __init__(self):
                self.state = SimpleState()

        request = SimpleRequest()

        with pytest.raises(HTTPException) as exc_info:
            get_tenant_context(request)

        assert exc_info.value.status_code == 500

    def test_require_tenant_with_valid_tenant(self):
        """Test require_tenant with valid tenant"""
        tenant = Tenant(tenant_id="t1", name="Test")
        context = TenantContext(tenant_id="t1", tenant=tenant)
        request = MagicMock()
        request.state.tenant_context = context

        result = require_tenant(request)
        assert result.tenant_id == "t1"

    def test_require_tenant_with_default_raises_error(self):
        """Test require_tenant rejects default tenant"""
        context = TenantContext(tenant_id="default")
        request = MagicMock()
        request.state.tenant_context = context

        with pytest.raises(HTTPException) as exc_info:
            require_tenant(request)

        assert exc_info.value.status_code == 401

    def test_require_active_tenant(self):
        """Test require_active_tenant with active tenant"""
        tenant = Tenant(tenant_id="t1", name="Test", status=TenantStatus.ACTIVE)
        context = TenantContext(tenant_id="t1", tenant=tenant)
        request = MagicMock()
        request.state.tenant_context = context

        result = require_active_tenant(request)
        assert result.tenant_id == "t1"

    def test_require_active_tenant_with_inactive_raises_error(self):
        """Test require_active_tenant rejects inactive tenant"""
        tenant = Tenant(tenant_id="t1", name="Test", status=TenantStatus.SUSPENDED)
        context = TenantContext(tenant_id="t1", tenant=tenant)
        request = MagicMock()
        request.state.tenant_context = context

        with pytest.raises(HTTPException) as exc_info:
            require_active_tenant(request)

        assert exc_info.value.status_code == 403


# ============================================================================
# Test Global Functions
# ============================================================================

class TestGlobalFunctions:
    """Test global utility functions"""

    def test_get_tenant_manager_singleton(self):
        """Test that get_tenant_manager returns singleton"""
        manager1 = get_tenant_manager()
        manager2 = get_tenant_manager()

        assert manager1 is manager2

    def test_create_tenant_context_utility(self):
        """Test create_tenant_context utility function"""
        tenant = Tenant(tenant_id="t1", name="Test")
        context = create_tenant_context(
            tenant_id="t1",
            tenant=tenant,
            is_isolated=True,
            request_id="req-123",
        )

        assert context.tenant_id == "t1"
        assert context.tenant == tenant
        assert context.is_isolated is True
        assert context.request_id == "req-123"
