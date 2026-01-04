"""
PostgreSQL connection pool manager for Enterprise RAG System

This module provides connection pooling functionality using asyncpg.
Features:
- Async connection pool management
- Automatic connection lifecycle management
- Connection health checks
- Graceful shutdown handling
"""

import asyncpg
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class DatabasePool:
    """
    Manages PostgreSQL connection pool using asyncpg.

    Features:
    - Connection pooling with configurable size
    - Automatic connection acquisition and release
    - Connection health monitoring
    - Graceful shutdown and cleanup
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "enterprise_rag",
        user: str = "postgres",
        password: str = "",
        min_size: int = 10,
        max_size: int = 50,
        command_timeout: int = 60,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0,
    ):
        """
        Initialize database pool configuration.

        Args:
            host: Database host address
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_size: Minimum number of connections in the pool
            max_size: Maximum number of connections in the pool
            command_timeout: Command execution timeout in seconds
            max_queries: Maximum queries per connection before recycling
            max_inactive_connection_lifetime: Maximum inactive lifetime in seconds
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime

        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the connection pool.

        Raises:
            Exception: If pool creation fails
        """
        if self._initialized:
            logger.warning("Database pool already initialized")
            return

        try:
            # Build DSN (Data Source Name)
            dsn = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

            # Create connection pool
            self.pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=self.command_timeout,
                max_queries=self.max_queries,
                max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
            )

            self._initialized = True
            logger.info(
                f"Database pool initialized: {self.host}:{self.port}/{self.database} "
                f"(min_size={self.min_size}, max_size={self.max_size})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self) -> None:
        """
        Close the connection pool gracefully.

        This should be called during application shutdown.
        """
        if self.pool and not self.pool._closed:
            try:
                await self.pool.close()
                self._initialized = False
                logger.info("Database pool closed successfully")
            except Exception as e:
                logger.error(f"Error closing database pool: {e}")
                raise
        else:
            logger.warning("Database pool already closed or not initialized")

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.

        Yields:
            asyncpg.Connection: Database connection

        Raises:
            Exception: If pool is not initialized or acquisition fails

        Example:
            >>> async with pool.acquire() as conn:
            ...     result = await conn.fetchval("SELECT 1")
        """
        if not self._initialized or not self.pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")

        async with self.pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args, timeout: Optional[float] = None) -> str:
        """
        Execute an SQL command (no results returned).

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds (overrides default)

        Returns:
            Execution status string

        Raises:
            Exception: If query execution fails
        """
        if not self._initialized or not self.pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *args, timeout=timeout)
            return result

    async def fetch(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> List[asyncpg.Record]:
        """
        Fetch rows from the database.

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds (overrides default)

        Returns:
            List of database records

        Raises:
            Exception: If query execution fails
        """
        if not self._initialized or not self.pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")

        async with self.pool.acquire() as conn:
            result = await conn.fetch(query, *args, timeout=timeout)
            return result

    async def fetchrow(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[asyncpg.Record]:
        """
        Fetch a single row from the database.

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds (overrides default)

        Returns:
            Single database record or None

        Raises:
            Exception: If query execution fails
        """
        if not self._initialized or not self.pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(query, *args, timeout=timeout)
            return result

    async def fetchval(
        self,
        query: str,
        *args,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Fetch a single value from the database.

        Args:
            query: SQL query to execute
            *args: Query parameters
            column: Column index to fetch
            timeout: Query timeout in seconds (overrides default)

        Returns:
            Single value from the query result

        Raises:
            Exception: If query execution fails
        """
        if not self._initialized or not self.pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")

        async with self.pool.acquire() as conn:
            result = await conn.fetchval(query, *args, column=column, timeout=timeout)
            return result

    async def transaction(self):
        """
        Start a database transaction.

        Returns:
            Transaction context manager

        Raises:
            Exception: If pool is not initialized

        Example:
            >>> async with pool.transaction() as conn:
            ...     await conn.execute("INSERT INTO ...")
            ...     await conn.execute("UPDATE ...")
        """
        if not self._initialized or not self.pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    def is_initialized(self) -> bool:
        """
        Check if the pool is initialized and ready.

        Returns:
            True if pool is initialized, False otherwise
        """
        return self._initialized and self.pool is not None and not self.pool._closed

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection.

        Returns:
            Dictionary with health status information

        Example:
            >>> health = await pool.health_check()
            >>> print(health)
            {
                'status': 'healthy',
                'pool_size': 10,
                'max_size': 50,
                'available_connections': 8
            }
        """
        if not self.is_initialized():
            return {
                "status": "unhealthy",
                "error": "Pool not initialized"
            }

        try:
            # Execute a simple query to test connectivity
            result = await self.fetchval("SELECT 1")

            # Get pool statistics
            pool_stats = {
                "status": "healthy" if result == 1 else "unhealthy",
                "min_size": self.pool.get_min_size(),
                "max_size": self.pool.get_max_size(),
                "current_size": self.pool.get_size(),
                "available": self.pool.get_idle_size(),
            }

            return pool_stats

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def get_pool_info(self) -> Dict[str, Any]:
        """
        Get information about the connection pool configuration.

        Returns:
            Dictionary with pool configuration
        """
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "command_timeout": self.command_timeout,
            "max_queries": self.max_queries,
            "max_inactive_connection_lifetime": self.max_inactive_connection_lifetime,
            "initialized": self._initialized,
        }


# Global pool instance
_global_pool: Optional[DatabasePool] = None


async def get_database_pool() -> DatabasePool:
    """
    Get the global database pool instance.

    Returns:
        DatabasePool instance

    Raises:
        RuntimeError: If pool has not been initialized
    """
    global _global_pool

    if _global_pool is None:
        raise RuntimeError(
            "Database pool not initialized. Call init_database_pool() first."
        )

    return _global_pool


async def init_database_pool(config: Dict[str, Any]) -> DatabasePool:
    """
    Initialize the global database pool.

    Args:
        config: Dictionary containing database configuration

    Returns:
        Initialized DatabasePool instance

    Example:
        >>> config = {
        ...     "host": "localhost",
        ...     "port": 5432,
        ...     "database": "mydb",
        ...     "user": "user",
        ...     "password": "pass"
        ... }
        >>> pool = await init_database_pool(config)
    """
    global _global_pool

    if _global_pool is not None:
        logger.warning("Database pool already initialized, returning existing instance")
        return _global_pool

    _global_pool = DatabasePool(
        host=config.get("host", "localhost"),
        port=config.get("port", 5432),
        database=config.get("database", "enterprise_rag"),
        user=config.get("user", "postgres"),
        password=config.get("password", ""),
        min_size=config.get("min_size", 10),
        max_size=config.get("max_size", 50),
        command_timeout=config.get("command_timeout", 60),
        max_queries=config.get("max_queries", 50000),
        max_inactive_connection_lifetime=config.get("max_inactive_connection_lifetime", 300.0),
    )

    await _global_pool.initialize()
    return _global_pool


async def close_database_pool() -> None:
    """
    Close the global database pool.

    This should be called during application shutdown.
    """
    global _global_pool

    if _global_pool:
        await _global_pool.close()
        _global_pool = None
        logger.info("Global database pool closed")
    else:
        logger.warning("No global database pool to close")
