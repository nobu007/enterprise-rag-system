"""
Database connection pooling for Enterprise RAG System

This module provides async PostgreSQL connection pooling using asyncpg
for improved performance under load in production environments.
"""

import asyncpg
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import logging

from .config import get_settings

logger = logging.getLogger(__name__)


class DatabasePool:
    """
    Async PostgreSQL connection pool manager

    Provides efficient connection pooling for database operations,
    with automatic connection management and error handling.
    """

    def __init__(self):
        """Initialize the database pool manager"""
        self._pool: Optional[asyncpg.Pool] = None
        self.settings = get_settings()

    async def create_pool(self) -> None:
        """
        Create the connection pool

        Establishes a connection pool to PostgreSQL with configured settings.
        Raises an exception if pool creation fails.
        """
        try:
            self._pool = await asyncpg.create_pool(
                host=self.settings.postgresql_host,
                port=self.settings.postgresql_port,
                user=self.settings.postgresql_user,
                password=self.settings.postgresql_password,
                database=self.settings.postgresql_database,
                min_size=self.settings.postgresql_pool_min,
                max_size=self.settings.postgresql_pool_max,
                command_timeout=self.settings.postgresql_timeout,
                max_queries=self.settings.postgresql_max_queries,
                max_inactive_connection_lifetime=self.settings.postgresql_max_lifetime,
                connection_kwargs={
                    "server_settings": {
                        "application_name": self.settings.app_name,
                        "timezone": "UTC"
                    }
                }
            )
            logger.info(
                f"Created database pool: {self.settings.postgresql_host}:"
                f"{self.settings.postgresql_port}/{self.settings.postgresql_database}"
            )
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise

    async def close_pool(self) -> None:
        """
        Close the connection pool

        Gracefully closes all connections in the pool.
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire_connection(self):
        """
        Acquire a connection from the pool

        Yields:
            asyncpg.Connection: A database connection

        Example:
            async with pool.acquire_connection() as conn:
                result = await conn.fetchval("SELECT NOW()")
        """
        if not self._pool:
            raise RuntimeError("Database pool not initialized. Call create_pool() first.")

        async with self._pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args, timeout: Optional[float] = None) -> str:
        """
        Execute a SQL query that doesn't return data (INSERT, UPDATE, DELETE)

        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Query timeout in seconds (overrides default)

        Returns:
            str: Status message from PostgreSQL

        Example:
            await pool.execute("INSERT INTO users (name) VALUES ($1)", "Alice")
        """
        async with self.acquire_connection() as conn:
            return await conn.execute(query, *args, timeout=timeout)

    async def fetch(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> List[asyncpg.Record]:
        """
        Execute a query and return all results

        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Query timeout in seconds

        Returns:
            List[asyncpg.Record]: Query results

        Example:
            rows = await pool.fetch("SELECT * FROM users WHERE age > $1", 18)
        """
        async with self.acquire_connection() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[asyncpg.Record]:
        """
        Execute a query and return the first result

        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Query timeout in seconds

        Returns:
            Optional[asyncpg.Record]: First row or None if no results

        Example:
            user = await pool.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        """
        async with self.acquire_connection() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)

    async def fetchval(
        self,
        query: str,
        *args,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a query and return a single value

        Args:
            query: SQL query string
            *args: Query parameters
            column: Column index to return (default: 0)
            timeout: Query timeout in seconds

        Returns:
            Any: Single value from the query result

        Example:
            count = await pool.fetchval("SELECT COUNT(*) FROM users")
        """
        async with self.acquire_connection() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)

    async def executemany(
        self,
        command: str,
        args: List[tuple],
        timeout: Optional[float] = None
    ) -> None:
        """
        Execute a command multiple times with different parameter sets

        Args:
            command: SQL command string
            args: List of parameter tuples
            timeout: Query timeout in seconds

        Example:
            await pool.executemany(
                "INSERT INTO users (name, age) VALUES ($1, $2)",
                [("Alice", 30), ("Bob", 25)]
            )
        """
        async with self.acquire_connection() as conn:
            await conn.executemany(command, args, timeout=timeout)

    async def transaction(self):
        """
        Start a database transaction

        Returns:
            asyncpg.transaction: Transaction context manager

        Example:
            async with pool.transaction() as tx:
                await pool.execute("INSERT INTO ...")
                await pool.execute("UPDATE ...")
        """
        if not self._pool:
            raise RuntimeError("Database pool not initialized. Call create_pool() first.")

        return self._pool.transaction()

    @property
    def pool_size(self) -> int:
        """
        Get the current size of the connection pool

        Returns:
            int: Number of connections currently in the pool
        """
        return self._pool._size if self._pool else 0

    @property
    def is_initialized(self) -> bool:
        """
        Check if the pool is initialized

        Returns:
            bool: True if pool is initialized and ready
        """
        return self._pool is not None


# Global database pool instance
db_pool = DatabasePool()


async def get_db_pool() -> DatabasePool:
    """
    Get the global database pool instance

    Returns:
        DatabasePool: The global database pool

    Example:
        pool = await get_db_pool()
        if not pool.is_initialized:
            await pool.create_pool()
    """
    return db_pool


async def init_database() -> None:
    """
    Initialize the database connection pool

    This should be called during application startup.
    """
    try:
        await db_pool.create_pool()
        logger.info("Database pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


async def close_database() -> None:
    """
    Close the database connection pool

    This should be called during application shutdown.
    """
    await db_pool.close_pool()
    logger.info("Database pool closed")
