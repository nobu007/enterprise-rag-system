"""
Unit tests for Cache Manager
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock, patch

from app.core.cache import CacheManager


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.setex.return_value = True
    mock.delete.return_value = True
    mock.db = 0  # Set db attribute
    mock.info.return_value = {
        "db0": {"keys": 10},
        "used_memory_human": "1.5M",
        "used_memory_peak_human": "2.0M",
        "connected_clients": 5,
        "uptime_in_days": 1
    }
    return mock


@pytest.fixture
def cache_manager(mock_redis):
    """Cache manager with mocked Redis"""
    with patch('app.core.cache.redis.from_url', return_value=mock_redis):
        manager = CacheManager(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            ttl=3600,
            enabled=True
        )
        manager.redis_client = mock_redis
        return manager


@pytest.fixture
def disabled_cache_manager():
    """Disabled cache manager"""
    return CacheManager(
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        ttl=3600,
        enabled=False
    )


class TestCacheManagerInit:
    """Tests for CacheManager initialization"""

    def test_init_enabled(self, mock_redis):
        """Test initialization with cache enabled"""
        with patch('app.core.cache.redis.from_url', return_value=mock_redis):
            manager = CacheManager(
                redis_host="localhost",
                redis_port=6379,
                enabled=True
            )

            assert manager.enabled is True
            assert manager.ttl == 3600
            assert manager.redis_client is not None

    def test_init_disabled(self):
        """Test initialization with cache disabled"""
        manager = CacheManager(
            redis_host="localhost",
            redis_port=6379,
            enabled=False
        )

        assert manager.enabled is False
        assert manager.redis_client is None

    def test_init_redis_connection_failure(self):
        """Test handling of Redis connection failure"""
        with patch('app.core.cache.redis.from_url', side_effect=Exception("Connection failed")):
            manager = CacheManager(
                redis_host="localhost",
                redis_port=6379,
                enabled=True
            )

            # Should keep L1 cache enabled even if Redis fails
            assert manager.enabled is True  # L1 cache still active
            assert manager.redis_client is None  # Redis unavailable


class TestGenerateKey:
    """Tests for cache key generation"""

    def test_generate_key_basic(self, cache_manager):
        """Test basic key generation"""
        key1 = cache_manager.generate_key("test query", "default", 5, True)
        key2 = cache_manager.generate_key("test query", "default", 5, True)

        # Same parameters should generate same key
        assert key1 == key2
        # Key format: "rag:" + 64-char hash
        assert key1.startswith("rag:")
        assert len(key1) == 68  # "rag:" prefix + 64-char hash

    def test_generate_key_different_params(self, cache_manager):
        """Test key generation with different parameters"""
        key1 = cache_manager.generate_key("test query", "default", 5, True)
        key2 = cache_manager.generate_key("test query", "default", 10, True)
        key3 = cache_manager.generate_key("test query", "collection2", 5, True)
        key4 = cache_manager.generate_key("different query", "default", 5, True)

        # Different parameters should generate different keys
        assert key1 != key2
        assert key1 != key3
        assert key1 != key4

    def test_generate_key_consistency(self, cache_manager):
        """Test key generation is consistent"""
        key1 = cache_manager.generate_key("What is RAG?", "docs", 10, False)
        key2 = cache_manager.generate_key("What is RAG?", "docs", 10, False)

        assert key1 == key2


class TestGetSetDelete:
    """Tests for cache get/set/delete operations"""

    def test_set_and_get(self, cache_manager, mock_redis):
        """Test setting and getting a value"""
        test_value = {"answer": "test answer", "confidence": 0.9}

        # Set value
        result = cache_manager.set("test_key", test_value)
        assert result is True

        # Mock Redis to return the value
        mock_redis.get.return_value = json.dumps(test_value)

        # Get value
        retrieved = cache_manager.get("test_key")
        assert retrieved == test_value

    def test_get_miss(self, cache_manager, mock_redis):
        """Test cache miss"""
        mock_redis.get.return_value = None

        result = cache_manager.get("nonexistent_key")
        assert result is None

    def test_get_disabled_cache(self, disabled_cache_manager):
        """Test get with disabled cache"""
        result = disabled_cache_manager.get("test_key")
        assert result is None

    def test_set_disabled_cache(self, disabled_cache_manager):
        """Test set with disabled cache"""
        result = disabled_cache_manager.set("test_key", {"test": "value"})
        assert result is False

    def test_delete(self, cache_manager, mock_redis):
        """Test deleting a key"""
        result = cache_manager.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")

    def test_delete_disabled_cache(self, disabled_cache_manager):
        """Test delete with disabled cache"""
        result = disabled_cache_manager.delete("test_key")
        assert result is False

    def test_set_with_custom_ttl(self, cache_manager, mock_redis):
        """Test set with custom TTL"""
        test_value = {"test": "value"}

        cache_manager.set("test_key", test_value, ttl=7200)

        # Check that setex was called with correct TTL
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]  # Positional args: (key, ttl, value)
        assert call_args[1] == 7200  # TTL is the 2nd positional arg


class TestGetStats:
    """Tests for cache statistics"""

    def test_get_stats_enabled(self, cache_manager):
        """Test getting stats when cache is enabled"""
        stats = cache_manager.get_stats()

        assert stats["enabled"] is True
        assert stats["cache_version"] == "v2"
        # L2 stats
        assert "l2" in stats
        assert stats["l2"]["total_keys"] == 10
        assert "memory_used" in stats["l2"]
        assert "connected_clients" in stats["l2"]

    def test_get_stats_disabled(self, disabled_cache_manager):
        """Test getting stats when cache is disabled"""
        stats = disabled_cache_manager.get_stats()

        assert stats["enabled"] is False
        assert "error" in stats

    def test_get_stats_redis_error(self, cache_manager, mock_redis):
        """Test handling of Redis error when getting stats"""
        mock_redis.info.side_effect = Exception("Redis error")

        stats = cache_manager.get_stats()

        assert stats["enabled"] is True
        # Error should be in l2 section
        assert "error" in stats["l2"]


class TestIsAvailable:
    """Tests for availability check"""

    def test_is_available_true(self, cache_manager, mock_redis):
        """Test when cache is available"""
        mock_redis.ping.return_value = True

        assert cache_manager.is_available() is True

    def test_is_available_false(self, cache_manager, mock_redis):
        """Test when cache is unavailable"""
        mock_redis.ping.side_effect = Exception("Connection lost")

        assert cache_manager.is_available() is False

    def test_is_available_disabled(self, disabled_cache_manager):
        """Test when cache is disabled"""
        assert disabled_cache_manager.is_available() is False


class TestClearCollection:
    """Tests for collection clearing"""

    def test_clear_collection(self, cache_manager, mock_redis):
        """Test clearing a collection"""
        # Mock scan_iter to return some keys
        mock_keys = ["key1", "key2", "key3"]
        mock_redis.scan_iter.return_value = iter(mock_keys)

        result = cache_manager.clear_collection("collection_prefix")

        assert result is True
        mock_redis.delete.assert_called_once_with(*mock_keys)

    def test_clear_collection_no_keys(self, cache_manager, mock_redis):
        """Test clearing collection with no keys"""
        mock_redis.scan_iter.return_value = iter([])

        result = cache_manager.clear_collection("collection_prefix")

        assert result is True
        # delete should not be called
        mock_redis.delete.assert_not_called()

    def test_clear_collection_disabled(self, disabled_cache_manager):
        """Test clearing collection with disabled cache"""
        result = disabled_cache_manager.clear_collection("collection_prefix")
        assert result is False


class TestFlushAll:
    """Tests for flushing all cache"""

    def test_flush_all(self, cache_manager, mock_redis):
        """Test flushing all cache entries"""
        result = cache_manager.flush_all()

        assert result is True
        mock_redis.flushdb.assert_called_once()

    def test_flush_all_disabled(self, disabled_cache_manager):
        """Test flush with disabled cache"""
        result = disabled_cache_manager.flush_all()
        assert result is False


class TestErrorHandling:
    """Tests for error handling"""

    def test_get_exception_handling(self, cache_manager, mock_redis):
        """Test exception handling in get"""
        mock_redis.get.side_effect = Exception("Redis error")

        result = cache_manager.get("test_key")
        assert result is None

    def test_set_exception_handling(self, cache_manager, mock_redis):
        """Test exception handling in set"""
        mock_redis.setex.side_effect = Exception("Redis error")

        result = cache_manager.set("test_key", {"test": "value"})
        # Should still succeed because L1 cache is active
        assert result is True

    def test_delete_exception_handling(self, cache_manager, mock_redis):
        """Test exception handling in delete"""
        mock_redis.delete.side_effect = Exception("Redis error")

        result = cache_manager.delete("test_key")
        assert result is False


class TestQueryNormalization:
    """Tests for query normalization"""

    def test_normalize_query_basic(self, cache_manager):
        """Test basic query normalization"""
        query1 = "  What is RAG?  "
        query2 = "what is rag"
        query3 = "WHAT IS RAG!"

        normalized1 = cache_manager._normalize_query(query1)
        normalized2 = cache_manager._normalize_query(query2)
        normalized3 = cache_manager._normalize_query(query3)

        # All should normalize to the same string
        assert normalized1 == normalized2 == normalized3
        assert normalized1 == "what is rag"

    def test_normalize_query_with_special_chars(self, cache_manager):
        """Test normalization with special characters"""
        query = "Hello, @World! #Test$"
        normalized = cache_manager._normalize_query(query)

        assert normalized == "hello world test"

    def test_normalize_query_extra_whitespace(self, cache_manager):
        """Test normalization with extra whitespace"""
        query = "This    is   a   test"
        normalized = cache_manager._normalize_query(query)

        assert normalized == "this is a test"


class TestHierarchicalCaching:
    """Tests for L1/L2 hierarchical caching"""

    def test_l1_cache_hit(self, cache_manager):
        """Test L1 cache hit"""
        test_value = {"answer": "test", "confidence": 0.9}
        cache_manager.set("test_key", test_value)

        # First get should hit L1
        result = cache_manager.get("test_key")
        assert result == test_value

        # Check L1 cache directly
        assert "test_key" in cache_manager.l1_cache

    def test_l1_cache_eviction(self, cache_manager):
        """Test L1 cache eviction when full"""
        # Create cache manager with small L1 size
        with patch('app.core.cache.redis.from_url', return_value=MagicMock()):
            small_cache = CacheManager(l1_size=3, enabled=True)
            small_cache.redis_client = MagicMock()  # Mock Redis

        # Fill L1 cache
        for i in range(5):
            small_cache.set(f"key_{i}", {"value": i})

        # L1 should only have 3 entries (max size)
        assert len(small_cache.l1_cache) == 3

        # Oldest entries should be evicted
        assert "key_0" not in small_cache.l1_cache
        assert "key_1" not in small_cache.l1_cache

    def test_l2_to_l1_promotion(self, cache_manager, mock_redis):
        """Test promotion from L2 to L1 cache"""
        test_value = {"answer": "test", "confidence": 0.9}

        # Clear L1 cache
        cache_manager.l1_cache.clear()

        # Mock L2 to return value
        mock_redis.get.return_value = json.dumps(test_value)

        # Get should promote to L1
        result = cache_manager.get("test_key")
        assert result == test_value
        assert "test_key" in cache_manager.l1_cache

    def test_l1_cache_expiration(self, cache_manager):
        """Test L1 cache entry expiration"""
        test_value = {"answer": "test"}

        # Set a very short TTL
        cache_manager.ttl = 0.1  # 100ms
        cache_manager.set("test_key", test_value)

        # Should be in L1
        assert "test_key" in cache_manager.l1_cache

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired
        result = cache_manager.get("test_key")
        assert result is None  # Expired from L1
        assert "test_key" not in cache_manager.l1_cache


class TestCacheVersioning:
    """Tests for cache versioning and invalidation"""

    def test_cache_key_includes_version(self, cache_manager):
        """Test that cache keys include version"""
        key = cache_manager.generate_key("test query", "default", 5, True)

        assert key.startswith("rag:")
        assert cache_manager.CACHE_VERSION == "v2"

    def test_invalidate_version(self, cache_manager, mock_redis):
        """Test version invalidation"""
        # Add some entries to L1
        cache_manager.set("key1", {"value": 1})
        cache_manager.set("key2", {"value": 2})

        # Invalidate old version
        result = cache_manager.invalidate_version("v1")

        # L1 should be cleared
        assert len(cache_manager.l1_cache) == 0
        assert result is True

    def test_version_mismatch_different_keys(self, cache_manager):
        """Test that different versions generate different keys"""
        # Temporarily change version
        old_version = cache_manager.CACHE_VERSION
        cache_manager.CACHE_VERSION = "v1"

        key_v1 = cache_manager.generate_key("test query", "default", 5, True)

        # Change version
        cache_manager.CACHE_VERSION = "v2"

        key_v2 = cache_manager.generate_key("test query", "default", 5, True)

        # Keys should be different
        assert key_v1 != key_v2

        # Restore version
        cache_manager.CACHE_VERSION = old_version


class TestImprovedStats:
    """Tests for improved cache statistics"""

    def test_stats_include_l1(self, cache_manager):
        """Test that stats include L1 information"""
        cache_manager.set("key1", {"value": 1})
        cache_manager.set("key2", {"value": 2})

        stats = cache_manager.get_stats()

        assert "l1" in stats
        assert stats["l1"]["size"] == 2
        assert stats["l1"]["max_size"] == cache_manager.l1_size
        assert "usage_percent" in stats["l1"]

    def test_stats_include_l2(self, cache_manager):
        """Test that stats include L2 information"""
        stats = cache_manager.get_stats()

        assert "l2" in stats
        assert stats["cache_version"] == "v2"

    def test_stats_disabled_cache(self, disabled_cache_manager):
        """Test stats when cache is disabled"""
        stats = disabled_cache_manager.get_stats()

        assert stats["enabled"] is False


class TestDeleteOperations:
    """Tests for delete operations with hierarchical cache"""

    def test_delete_from_both_layers(self, cache_manager, mock_redis):
        """Test deletion from both L1 and L2"""
        test_value = {"answer": "test"}

        # Set in both layers
        cache_manager.set("test_key", test_value)

        # Verify it's in L1
        assert "test_key" in cache_manager.l1_cache

        # Delete
        cache_manager.delete("test_key")

        # Should be removed from both
        assert "test_key" not in cache_manager.l1_cache
        mock_redis.delete.assert_called_once()

    def test_clear_collection_both_layers(self, cache_manager, mock_redis):
        """Test clearing collection from both layers"""
        # Add entries with same prefix
        cache_manager.set("coll:key1", {"value": 1})
        cache_manager.set("coll:key2", {"value": 2})
        cache_manager.set("other:key3", {"value": 3})

        # Mock scan_iter
        mock_redis.scan_iter.return_value = iter(["coll:key1", "coll:key2"])

        # Clear collection
        cache_manager.clear_collection("coll")

        # Collection keys should be removed from L1
        assert "coll:key1" not in cache_manager.l1_cache
        assert "coll:key2" not in cache_manager.l1_cache
        # Other keys should remain
        assert "other:key3" in cache_manager.l1_cache


class TestFlushAll:
    """Tests for flush_all with hierarchical cache"""

    def test_flush_all_layers(self, cache_manager, mock_redis):
        """Test flushing both L1 and L2"""
        # Add entries
        cache_manager.set("key1", {"value": 1})
        cache_manager.set("key2", {"value": 2})

        # Flush all
        cache_manager.flush_all()

        # L1 should be empty
        assert len(cache_manager.l1_cache) == 0

        # L2 should be flushed
        mock_redis.flushdb.assert_called_once()

