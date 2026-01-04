"""
Unit tests for database connection pooling module.

Tests cover:
- Pool initialization and configuration
- Connection lifecycle management
- Query execution methods
- Error handling
- Health checks
- Global pool management
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys


# Mock asyncpg module
class MockPool:
    """Mock asyncpg pool for testing."""
    def __init__(self):
        self._closed = False
        self._min_size = 10
        self._max_size = 50
        self._size = 10
        self._idle_size = 8
        self._connection = MockConnection()

    def get_min_size(self):
        return self._min_size

    def get_max_size(self):
        return self._max_size

    def get_size(self):
        return self._size

    def get_idle_size(self):
        return self._idle_size

    async def close(self):
        self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def acquire(self):
        """Mock acquire that returns a mock connection."""
        class AcquireContext:
            async def __aenter__(self2):
                return self._connection

            async def __aexit__(self2, *args):
                pass

            def __init__(self2, connection):
                self2._connection = connection

        return AcquireContext(self._connection)


# Create a mock connection
class MockConnection:
    """Mock database connection."""
    async def execute(self, query, *args, **kwargs):
        return "SELECT 1"

    async def fetch(self, query, *args, **kwargs):
        return []

    async def fetchrow(self, query, *args, **kwargs):
        return None

    async def fetchval(self, query, *args, **kwargs):
        return 1


# Setup mock module
mock_asyncpg = MagicMock()

async def mock_create_pool(*args, **kwargs):
    """Mock create_pool that returns a new mock pool instance."""
    return MockPool()

mock_asyncpg.create_pool = mock_create_pool
sys.modules['asyncpg'] = mock_asyncpg

# Now import after mocking
from app.core.database import DatabasePool


@pytest.fixture
def db_config():
    """Sample database configuration."""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_pass",
        "min_size": 5,
        "max_size": 20,
    }


@pytest.fixture
async def initialized_pool(db_config):
    """Create and initialize a pool for testing."""
    pool = DatabasePool(**db_config)
    await pool.initialize()
    yield pool
    # Cleanup
    if pool.pool is not None:
        await pool.close()


class TestDatabasePoolConfiguration:
    """Test pool configuration and initialization."""

    def test_pool_initialization_with_config(self, db_config):
        """Test that pool is created with correct configuration."""
        pool = DatabasePool(**db_config)

        assert pool.host == "localhost"
        assert pool.port == 5432
        assert pool.database == "test_db"
        assert pool.user == "test_user"
        assert pool.password == "test_pass"
        assert pool.min_size == 5
        assert pool.max_size == 20

    def test_default_configuration(self):
        """Test pool with default configuration."""
        pool = DatabasePool()

        assert pool.host == "localhost"
        assert pool.port == 5432
        assert pool.database == "enterprise_rag"
        assert pool.user == "postgres"
        assert pool.min_size == 10
        assert pool.max_size == 50

    def test_pool_info_retrieval(self, db_config):
        """Test getting pool configuration information."""
        pool = DatabasePool(**db_config)
        info = pool.get_pool_info()

        assert info["host"] == "localhost"
        assert info["port"] == 5432
        assert info["database"] == "test_db"
        assert info["min_size"] == 5
        assert info["max_size"] == 20
        assert info["initialized"] is False


class TestPoolLifecycle:
    """Test pool lifecycle management."""

    @pytest.mark.asyncio
    async def test_pool_initialization(self, db_config):
        """Test successful pool initialization."""
        pool = DatabasePool(**db_config)

        await pool.initialize()

        assert pool.is_initialized()
        assert pool.pool is not None

    @pytest.mark.asyncio
    async def test_pool_double_initialization(self, initialized_pool):
        """Test that double initialization doesn't cause issues."""
        # Should not raise an exception
        await initialized_pool.initialize()

        assert initialized_pool.is_initialized()

    @pytest.mark.asyncio
    async def test_pool_closure(self, db_config):
        """Test successful pool closure."""
        pool = DatabasePool(**db_config)
        await pool.initialize()

        assert pool.is_initialized()

        await pool.close()

        assert pool._initialized is False

    @pytest.mark.asyncio
    async def test_pool_close_when_not_initialized(self, db_config):
        """Test closing a pool that was never initialized."""
        pool = DatabasePool(**db_config)

        # Should not raise an exception
        await pool.close()

        assert pool._initialized is False


class TestPoolStatus:
    """Test pool status and health checks."""

    @pytest.mark.asyncio
    async def test_is_initialized_before_init(self, db_config):
        """Test is_initialized returns False before initialization."""
        pool = DatabasePool(**db_config)

        assert pool.is_initialized() is False

    @pytest.mark.asyncio
    async def test_is_initialized_after_init(self, db_config):
        """Test is_initialized returns True after initialization."""
        pool = DatabasePool(**db_config)
        await pool.initialize()

        assert pool.is_initialized() is True

    @pytest.mark.asyncio
    async def test_health_check_when_not_initialized(self, db_config):
        """Test health check when pool is not initialized."""
        pool = DatabasePool(**db_config)

        health = await pool.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health

    @pytest.mark.asyncio
    async def test_health_check_when_initialized(self, db_config):
        """Test health check when pool is initialized."""
        pool = DatabasePool(**db_config)
        await pool.initialize()

        health = await pool.health_check()

        assert health["status"] == "healthy"
        assert "min_size" in health
        assert "max_size" in health
        assert "available" in health


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_execute_when_not_initialized(self, db_config):
        """Test executing query when pool is not initialized."""
        pool = DatabasePool(**db_config)

        with pytest.raises(RuntimeError) as exc_info:
            await pool.execute("SELECT 1")

        assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_when_not_initialized(self, db_config):
        """Test fetching when pool is not initialized."""
        pool = DatabasePool(**db_config)

        with pytest.raises(RuntimeError) as exc_info:
            await pool.fetch("SELECT 1")

        assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_acquire_when_not_initialized(self, db_config):
        """Test acquiring connection when pool is not initialized."""
        pool = DatabasePool(**db_config)

        with pytest.raises(RuntimeError) as exc_info:
            async with pool.acquire():
                pass

        assert "not initialized" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_min_pool_size(self):
        """Test pool with zero minimum size."""
        pool = DatabasePool(min_size=0, max_size=10)

        assert pool.min_size == 0
        assert pool.max_size == 10

    def test_equal_min_max_pool_size(self):
        """Test pool with equal min and max size."""
        pool = DatabasePool(min_size=10, max_size=10)

        assert pool.min_size == 10
        assert pool.max_size == 10

    def test_empty_password(self):
        """Test pool with empty password."""
        pool = DatabasePool(password="")

        assert pool.password == ""

    def test_large_command_timeout(self):
        """Test pool with very large command timeout."""
        pool = DatabasePool(command_timeout=3600)

        assert pool.command_timeout == 3600

    def test_custom_port(self):
        """Test pool with custom port."""
        pool = DatabasePool(port=5433)

        assert pool.port == 5433


class TestMethodSignatures:
    """Test that public methods have correct signatures."""

    def test_initialize_method_exists(self):
        """Test that initialize method exists."""
        pool = DatabasePool()
        assert hasattr(pool, 'initialize')
        assert callable(pool.initialize)

    def test_close_method_exists(self):
        """Test that close method exists."""
        pool = DatabasePool()
        assert hasattr(pool, 'close')
        assert callable(pool.close)

    def test_execute_method_exists(self):
        """Test that execute method exists."""
        pool = DatabasePool()
        assert hasattr(pool, 'execute')
        assert callable(pool.execute)

    def test_fetch_method_exists(self):
        """Test that fetch method exists."""
        pool = DatabasePool()
        assert hasattr(pool, 'fetch')
        assert callable(pool.fetch)

    def test_health_check_method_exists(self):
        """Test that health_check method exists."""
        pool = DatabasePool()
        assert hasattr(pool, 'health_check')
        assert callable(pool.health_check)

    def test_acquire_method_exists(self):
        """Test that acquire method exists."""
        pool = DatabasePool()
        assert hasattr(pool, 'acquire')
        assert callable(pool.acquire)
