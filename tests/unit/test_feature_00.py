"""
Unit tests for Database Connection Pooling (Feature 00)

This test suite verifies the PostgreSQL connection pooling implementation
including pool creation, connection management, and query execution.
"""

import pytest
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from contextlib import asynccontextmanager

# Set test environment variables before importing database module
os.environ.setdefault("POSTGRESQL_PASSWORD", "test-password")
os.environ.setdefault("POSTGRESQL_HOST", "localhost")
os.environ.setdefault("POSTGRESQL_PORT", "5432")
os.environ.setdefault("POSTGRESQL_USER", "test_user")
os.environ.setdefault("POSTGRESQL_DATABASE", "test_db")

from app.core.database import DatabasePool, get_db_pool, init_database, close_database


@pytest.fixture
def mock_connection():
    """Create a mock database connection"""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="SELECT 1")
    conn.executemany = AsyncMock()
    return conn


@pytest.fixture
def mock_pool(mock_connection):
    """Create a mock asyncpg pool with proper context manager"""
    pool = AsyncMock()
    pool._size = 10

    # Create a proper async context manager for acquire()
    @asynccontextmanager
    async def mock_acquire():
        yield mock_connection

    pool.acquire = mock_acquire

    # Create a proper transaction context manager
    @asynccontextmanager
    async def mock_transaction():
        yield AsyncMock()

    pool.transaction = mock_transaction

    return pool


@pytest.fixture
def db_pool():
    """Create a DatabasePool instance for testing"""
    return DatabasePool()


class TestDatabasePoolInit:
    """Test DatabasePool initialization and basic properties"""

    def test_initialization(self, db_pool):
        """Test that DatabasePool initializes correctly"""
        assert db_pool._pool is None
        assert not db_pool.is_initialized
        assert db_pool.pool_size == 0
        assert db_pool.settings is not None

    def test_settings_loaded(self, db_pool):
        """Test that configuration settings are loaded"""
        assert db_pool.settings.postgresql_host == "localhost"
        assert db_pool.settings.postgresql_port == 5432
        assert db_pool.settings.postgresql_user == "test_user"
        assert db_pool.settings.postgresql_database == "test_db"


class TestCreatePool:
    """Test connection pool creation"""

    @pytest.mark.asyncio
    async def test_create_pool_success(self, db_pool, mock_pool):
        """Test successful pool creation"""
        with patch("app.core.database.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)) as mock_create:
            await db_pool.create_pool()

            assert db_pool.is_initialized
            assert db_pool._pool == mock_pool
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_pool_failure(self, db_pool):
        """Test pool creation failure handling"""
        with patch("app.core.database.asyncpg.create_pool", new=AsyncMock(side_effect=Exception("Connection failed"))) as mock_create:
            with pytest.raises(Exception) as exc_info:
                await db_pool.create_pool()

            assert "Connection failed" in str(exc_info.value)
            assert not db_pool.is_initialized
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_pool_parameters(self, db_pool, mock_pool):
        """Test that pool creation uses correct parameters"""
        with patch("app.core.database.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)) as mock_create:
            await db_pool.create_pool()

            call_args = mock_create.call_args
            assert call_args is not None

            # Verify key parameters are passed
            kwargs = call_args.kwargs if call_args.kwargs else call_args[1]
            assert kwargs.get("host") == "localhost"
            assert kwargs.get("port") == 5432
            assert kwargs.get("user") == "test_user"
            assert kwargs.get("database") == "test_db"
            assert "min_size" in kwargs
            assert "max_size" in kwargs


class TestClosePool:
    """Test pool closure"""

    @pytest.mark.asyncio
    async def test_close_pool_initialized(self, db_pool, mock_pool):
        """Test closing an initialized pool"""
        db_pool._pool = mock_pool
        mock_pool.close = AsyncMock()

        await db_pool.close_pool()

        assert db_pool._pool is None
        assert not db_pool.is_initialized
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_pool_not_initialized(self, db_pool):
        """Test closing a pool that was never initialized"""
        # Should not raise an exception
        await db_pool.close_pool()

        assert db_pool._pool is None
        assert not db_pool.is_initialized


class TestAcquireConnection:
    """Test connection acquisition from pool"""

    @pytest.mark.asyncio
    async def test_acquire_connection_success(self, db_pool, mock_pool, mock_connection):
        """Test successful connection acquisition"""
        db_pool._pool = mock_pool
        mock_pool.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.__aexit__ = AsyncMock()
        mock_pool.acquire = MagicMock(return_value=mock_pool)

        async with db_pool.acquire_connection() as conn:
            assert conn == mock_connection

    @pytest.mark.asyncio
    async def test_acquire_connection_pool_not_initialized(self, db_pool):
        """Test acquiring connection when pool is not initialized"""
        with pytest.raises(RuntimeError) as exc_info:
            async with db_pool.acquire_connection() as conn:
                pass

        assert "Database pool not initialized" in str(exc_info.value)


class TestExecute:
    """Test execute method for non-query statements"""

    @pytest.mark.asyncio
    async def test_execute_success(self, db_pool, mock_pool, mock_connection):
        """Test successful execute operation"""
        db_pool._pool = mock_pool
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        result = await db_pool.execute("INSERT INTO users (name) VALUES ($1)", "Alice")

        assert result == "INSERT 0 1"
        mock_connection.execute.assert_called_once_with("INSERT INTO users (name) VALUES ($1)", "Alice", timeout=None)

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, db_pool, mock_pool, mock_connection):
        """Test execute operation with custom timeout"""
        db_pool._pool = mock_pool
        mock_connection.execute = AsyncMock(return_value="UPDATE 1")

        result = await db_pool.execute(
            "UPDATE users SET name = $1",
            "Bob",
            timeout=10.0
        )

        assert result == "UPDATE 1"
        mock_connection.execute.assert_called_once_with(
            "UPDATE users SET name = $1", "Bob", timeout=10.0
        )


class TestFetch:
    """Test fetch methods for query operations"""

    @pytest.mark.asyncio
    async def test_fetch(self, db_pool, mock_pool, mock_connection):
        """Test fetch method returns all results"""
        db_pool._pool = mock_pool

        mock_records = [MagicMock(id=1), MagicMock(id=2)]
        mock_connection.fetch = AsyncMock(return_value=mock_records)

        result = await db_pool.fetch("SELECT * FROM users")

        assert result == mock_records
        mock_connection.fetch.assert_called_once_with("SELECT * FROM users", timeout=None)

    @pytest.mark.asyncio
    async def test_fetchrow(self, db_pool, mock_pool, mock_connection):
        """Test fetchrow method returns single result"""
        db_pool._pool = mock_pool

        mock_record = MagicMock(id=1, name="Alice")
        mock_connection.fetchrow = AsyncMock(return_value=mock_record)

        result = await db_pool.fetchrow("SELECT * FROM users WHERE id = $1", 1)

        assert result == mock_record
        mock_connection.fetchrow.assert_called_once_with("SELECT * FROM users WHERE id = $1", 1, timeout=None)

    @pytest.mark.asyncio
    async def test_fetchrow_no_results(self, db_pool, mock_pool, mock_connection):
        """Test fetchrow returns None when no results"""
        db_pool._pool = mock_pool

        mock_connection.fetchrow = AsyncMock(return_value=None)

        result = await db_pool.fetchrow("SELECT * FROM users WHERE id = $1", 999)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetchval(self, db_pool, mock_pool, mock_connection):
        """Test fetchval method returns single value"""
        db_pool._pool = mock_pool

        mock_connection.fetchval = AsyncMock(return_value=42)

        result = await db_pool.fetchval("SELECT COUNT(*) FROM users")

        assert result == 42
        mock_connection.fetchval.assert_called_once_with("SELECT COUNT(*) FROM users", timeout=None, column=0)


class TestExecutemany:
    """Test executemany for batch operations"""

    @pytest.mark.asyncio
    async def test_executemany(self, db_pool, mock_pool, mock_connection):
        """Test executemany for batch inserts"""
        db_pool._pool = mock_pool

        mock_connection.executemany = AsyncMock()

        await db_pool.executemany(
            "INSERT INTO users (name, age) VALUES ($1, $2)",
            [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
        )

        mock_connection.executemany.assert_called_once()

        call_args = mock_connection.executemany.call_args
        assert call_args[0][0] == "INSERT INTO users (name, age) VALUES ($1, $2)"
        assert len(call_args[0][1]) == 3


class TestTransaction:
    """Test transaction management"""

    @pytest.mark.asyncio
    async def test_transaction(self, db_pool, mock_pool):
        """Test transaction creation"""
        db_pool._pool = mock_pool
        mock_transaction = AsyncMock()
        mock_pool.transaction = MagicMock(return_value=mock_transaction)

        result = await db_pool.transaction()

        assert result == mock_transaction
        mock_pool.transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_transaction_pool_not_initialized(self, db_pool):
        """Test transaction when pool is not initialized"""
        with pytest.raises(RuntimeError) as exc_info:
            await db_pool.transaction()

        assert "Database pool not initialized" in str(exc_info.value)


class TestGlobalPool:
    """Test global database pool functions"""

    @pytest.mark.asyncio
    async def test_get_db_pool(self):
        """Test get_db_pool returns global instance"""
        pool = await get_db_pool()
        assert isinstance(pool, DatabasePool)

        # Should return the same instance on subsequent calls
        pool2 = await get_db_pool()
        assert pool is pool2

    @pytest.mark.asyncio
    async def test_init_database(self):
        """Test init_database function"""
        with patch("app.core.database.db_pool.create_pool", new=AsyncMock()) as mock_create:
            await init_database()
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_database(self):
        """Test close_database function"""
        with patch("app.core.database.db_pool.close_pool", new=AsyncMock()) as mock_close:
            await close_database()
            mock_close.assert_called_once()


class TestPoolProperties:
    """Test pool property methods"""

    def test_pool_size_initialized(self, db_pool, mock_pool):
        """Test pool_size property when pool is initialized"""
        db_pool._pool = mock_pool
        assert db_pool.pool_size == 10

    def test_pool_size_not_initialized(self, db_pool):
        """Test pool_size property when pool is not initialized"""
        assert db_pool.pool_size == 0

    def test_is_initialized_true(self, db_pool, mock_pool):
        """Test is_initialized property when pool exists"""
        db_pool._pool = mock_pool
        assert db_pool.is_initialized is True

    def test_is_initialized_false(self, db_pool):
        """Test is_initialized property when pool doesn't exist"""
        assert db_pool.is_initialized is False
