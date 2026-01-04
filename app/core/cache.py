"""
Redis-based cache manager for RAG responses

This module provides caching functionality to improve performance and reduce costs.
Features:
- Hierarchical caching (L1: Memory, L2: Redis)
- Improved semantic cache keys with normalization
- Cache versioning and invalidation strategy
"""

import redis
import json
import hashlib
import logging
import re
import time
from typing import Optional, Any, Dict, Tuple
from dataclasses import asdict
from functools import lru_cache
from threading import Lock

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Manages hierarchical caching for RAG query responses.

    Features:
    - L1: In-memory cache (fast, limited size)
    - L2: Redis cache (shared, persistent)
    - Improved cache key generation with normalization
    - Version-based cache invalidation
    """

    # Cache version for invalidation
    CACHE_VERSION = "v2"

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        ttl: int = 3600,
        enabled: bool = True,
        l1_size: int = 128  # L1 cache size
    ):
        """
        Initialize CacheManager with hierarchical caching.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Optional Redis password
            ttl: Default time-to-live for cache entries (seconds)
            enabled: Whether caching is enabled
            l1_size: Maximum number of entries in L1 cache
        """
        self.ttl = ttl
        self.enabled = enabled
        self.l1_size = l1_size
        self.l1_lock = Lock()

        # Initialize L1 (memory) cache
        self.l1_cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)

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
                logger.info(f"Cache connected to Redis at {redis_host}:{redis_port} (L1: {l1_size} entries)")
            else:
                self.redis_client = None
                logger.info("Caching disabled")

        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. L2 caching disabled, L1 still active.")
            self.redis_client = None
            # Keep L1 cache enabled even if Redis fails

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for better cache hit rates.

        Normalization steps:
        1. Convert to lowercase
        2. Remove extra whitespace
        3. Remove special characters (except alphanumeric and spaces)
        4. Strip leading/trailing whitespace

        Args:
            query: Raw query string

        Returns:
            Normalized query string
        """
        # Convert to lowercase
        query = query.lower()

        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)

        # Remove special characters (keep alphanumeric and spaces)
        query = re.sub(r'[^a-z0-9\s]', '', query)

        # Strip whitespace
        query = query.strip()

        return query

    def generate_key(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 5,
        rerank: bool = True
    ) -> str:
        """
        Generate a unique cache key from query parameters with improved normalization.

        Args:
            query: User query
            collection: Collection name
            top_k: Number of results
            rerank: Whether re-ranking is enabled

        Returns:
            SHA256 hash of the normalized parameters with version prefix
        """
        # Normalize the query for better cache hits
        normalized_query = self._normalize_query(query)

        # Create a normalized string from parameters
        params = f"{self.CACHE_VERSION}:{normalized_query}:{collection}:{top_k}:{rerank}"

        # Generate hash
        hash_key = hashlib.sha256(params.encode()).hexdigest()

        return f"rag:{hash_key}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached value by key with hierarchical caching (L1 -> L2).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/error
        """
        if not self.enabled:
            return None

        # Try L1 cache first (fast, in-memory)
        with self.l1_lock:
            if key in self.l1_cache:
                value, timestamp = self.l1_cache[key]
                # Check if entry is still valid (within TTL)
                if time.time() - timestamp < self.ttl:
                    logger.debug(f"L1 cache hit for key: {key[:16]}...")
                    return value
                else:
                    # Entry expired, remove from L1
                    del self.l1_cache[key]

        # Try L2 cache (Redis)
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    logger.debug(f"L2 cache hit for key: {key[:16]}...")
                    parsed_value = json.loads(value)

                    # Promote to L1 cache
                    with self.l1_lock:
                        self._l1_add(key, parsed_value)

                    return parsed_value
                else:
                    logger.debug(f"Cache miss for key: {key[:16]}...")
                    return None

            except Exception as e:
                logger.warning(f"L2 cache get failed: {e}")
                return None

        return None

    def _l1_add(self, key: str, value: Any) -> None:
        """
        Add entry to L1 cache with eviction policy.

        Args:
            key: Cache key
            value: Value to cache
        """
        # If cache is full, remove oldest entry
        if len(self.l1_cache) >= self.l1_size:
            # Find oldest entry
            oldest_key = min(self.l1_cache.keys(), key=lambda k: self.l1_cache[k][1])
            del self.l1_cache[oldest_key]

        # Add new entry
        self.l1_cache[key] = (value, time.time())

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store value in cache with hierarchical approach (L1 + L2).

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (uses default if not specified)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        # Convert value to JSON-serializable format if it's a dataclass
        if hasattr(value, '__dataclass_fields__'):
            value = asdict(value)

        # Store in L1 cache (always)
        with self.l1_lock:
            self._l1_add(key, value)

        # Store in L2 cache (Redis)
        if self.redis_client:
            try:
                serialized = json.dumps(value)
                self.redis_client.setex(key, ttl or self.ttl, serialized)
                logger.debug(f"Cached value in L1+L2 for key: {key[:16]}...")
                return True

            except Exception as e:
                logger.warning(f"L2 cache set failed: {e}")
                # Return True if L1 succeeded
                return True

        return True

    def delete(self, key: str) -> bool:
        """
        Delete specific key from both L1 and L2 cache.

        Args:
            key: Cache key to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        # Delete from L1
        with self.l1_lock:
            if key in self.l1_cache:
                del self.l1_cache[key]

        # Delete from L2
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                logger.debug(f"Deleted cache key from L1+L2: {key[:16]}...")
                return True

            except Exception as e:
                logger.warning(f"L2 cache delete failed: {e}")
                return False

        return True

    def invalidate_version(self, old_version: str) -> bool:
        """
        Invalidate all cache entries from a specific version.

        This is called when the cache version changes to invalidate old entries.

        Args:
            old_version: Version to invalidate

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        # Clear L1 cache entirely (simplest approach)
        with self.l1_lock:
            self.l1_cache.clear()
            logger.info(f"Cleared L1 cache for version change to {self.CACHE_VERSION}")

        # For L2, we rely on the version prefix in keys
        # Old entries will naturally expire, but we can force clear if needed
        if self.redis_client:
            try:
                # Scan for keys with old version prefix
                pattern = f"*{old_version}*"
                keys = []
                for key in self.redis_client.scan_iter(match=pattern):
                    keys.append(key)

                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} old version cache entries")

                return True

            except Exception as e:
                logger.warning(f"Version invalidation failed: {e}")
                return False

        return True

    def clear_collection(self, collection_prefix: str) -> bool:
        """
        Clear all cache keys for a specific collection from both L1 and L2.

        Note: This is a potentially expensive operation as it scans all keys.

        Args:
            collection_prefix: Prefix to match keys

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        # Clear from L1 (filter by prefix)
        with self.l1_lock:
            keys_to_delete = [k for k in self.l1_cache.keys() if k.startswith(collection_prefix)]
            for key in keys_to_delete:
                del self.l1_cache[key]
            logger.info(f"Cleared {len(keys_to_delete)} L1 cache keys for collection: {collection_prefix}")

        # Clear from L2
        if self.redis_client:
            try:
                # Scan for keys matching the pattern
                keys = []
                for key in self.redis_client.scan_iter(match=f"{collection_prefix}*"):
                    keys.append(key)

                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} L2 cache keys for collection: {collection_prefix}")

                return True

            except Exception as e:
                logger.warning(f"L2 cache clear collection failed: {e}")
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics from both L1 and L2 caches.

        Returns:
            Dictionary with cache statistics for both layers
        """
        if not self.enabled:
            return {
                "enabled": False,
                "error": "Cache is disabled"
            }

        stats = {
            "enabled": True,
            "cache_version": self.CACHE_VERSION,
            "ttl_seconds": self.ttl,
            "l1": {
                "size": len(self.l1_cache),
                "max_size": self.l1_size,
                "usage_percent": round(len(self.l1_cache) / self.l1_size * 100, 2)
            }
        }

        # L2 stats (Redis)
        if self.redis_client:
            try:
                info = self.redis_client.info()
                db_stats = info.get(f"db{self.redis_client.db}", {})

                stats["l2"] = {
                    "total_keys": db_stats.get("keys", 0),
                    "memory_used": info.get("used_memory_human", "N/A"),
                    "memory_peak": info.get("used_memory_peak_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                    "uptime_days": info.get("uptime_in_days", 0)
                }

            except Exception as e:
                logger.warning(f"Failed to get L2 cache stats: {e}")
                stats["l2"] = {
                    "error": str(e)
                }
        else:
            stats["l2"] = {
                "enabled": False
            }

        return stats

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
        Flush all cache entries from both L1 and L2 (use with caution).

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        # Clear L1
        with self.l1_lock:
            self.l1_cache.clear()
            logger.warning("L1 cache flushed completely")

        # Clear L2
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.warning("L2 cache flushed completely")
                return True

            except Exception as e:
                logger.warning(f"L2 cache flush failed: {e}")
                return False

        return True
