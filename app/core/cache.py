"""
Redis-based cache manager for RAG responses

This module provides caching functionality to improve performance and reduce costs.
"""

import redis
import json
import hashlib
import logging
from typing import Optional, Any, Dict
from dataclasses import asdict

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Manages Redis-based caching for RAG query responses.

    Provides caching functionality with automatic key generation,
    TTL management, and error handling.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        ttl: int = 3600,
        enabled: bool = True
    ):
        """
        Initialize CacheManager.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Optional Redis password
            ttl: Default time-to-live for cache entries (seconds)
            enabled: Whether caching is enabled
        """
        self.ttl = ttl
        self.enabled = enabled

        try:
            if enabled:
                # Build Redis URL
                if redis_password:
                    redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
                else:
                    redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )

                # Test connection
                self.redis_client.ping()
                logger.info(f"Cache connected to Redis at {redis_host}:{redis_port}")
            else:
                self.redis_client = None
                logger.info("Caching disabled")

        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching will be disabled.")
            self.redis_client = None
            self.enabled = False

    def generate_key(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 5,
        rerank: bool = True
    ) -> str:
        """
        Generate a unique cache key from query parameters.

        Args:
            query: User query
            collection: Collection name
            top_k: Number of results
            rerank: Whether re-ranking is enabled

        Returns:
            SHA256 hash of the parameters
        """
        # Create a normalized string from parameters
        params = f"{query}:{collection}:{top_k}:{rerank}"
        return hashlib.sha256(params.encode()).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached value by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/error
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                logger.debug(f"Cache hit for key: {key[:16]}...")
                return json.loads(value)
            else:
                logger.debug(f"Cache miss for key: {key[:16]}...")
                return None

        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (uses default if not specified)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            # Convert value to JSON-serializable format if it's a dataclass
            if hasattr(value, '__dataclass_fields__'):
                value = asdict(value)

            serialized = json.dumps(value)
            self.redis_client.setex(key, ttl or self.ttl, serialized)
            logger.debug(f"Cached value for key: {key[:16]}...")
            return True

        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete specific key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            self.redis_client.delete(key)
            logger.debug(f"Deleted cache key: {key[:16]}...")
            return True

        except Exception as e:
            logger.warning(f"Cache delete failed: {e}")
            return False

    def clear_collection(self, collection_prefix: str) -> bool:
        """
        Clear all cache keys for a specific collection.

        Note: This is a potentially expensive operation as it scans all keys.

        Args:
            collection_prefix: Prefix to match keys

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            # Scan for keys matching the pattern
            keys = []
            for key in self.redis_client.scan_iter(match=f"{collection_prefix}*"):
                keys.append(key)

            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache keys for collection: {collection_prefix}")

            return True

        except Exception as e:
            logger.warning(f"Cache clear collection failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics from Redis.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled or not self.redis_client:
            return {
                "enabled": False,
                "error": "Cache is disabled or Redis unavailable"
            }

        try:
            info = self.redis_client.info()
            db_stats = info.get(f"db{self.redis_client.db}", {})

            return {
                "enabled": True,
                "total_keys": db_stats.get("keys", 0),
                "memory_used": info.get("used_memory_human", "N/A"),
                "memory_peak": info.get("used_memory_peak_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0),
                "ttl_seconds": self.ttl
            }

        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {
                "enabled": True,
                "error": str(e)
            }

    def is_available(self) -> bool:
        """
        Check if cache is available and connected.

        Returns:
            True if cache is enabled and connected, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False

    def flush_all(self) -> bool:
        """
        Flush all cache entries (use with caution).

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            self.redis_client.flushdb()
            logger.warning("Cache flushed completely")
            return True

        except Exception as e:
            logger.warning(f"Cache flush failed: {e}")
            return False
