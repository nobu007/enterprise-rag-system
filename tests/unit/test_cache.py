"""
Unit tests for Cache Manager
"""

import pytest
import json
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

            # Should fallback to disabled
            assert manager.enabled is False
            assert manager.redis_client is None


class TestGenerateKey:
    """Tests for cache key generation"""

    def test_generate_key_basic(self, cache_manager):
        """Test basic key generation"""
        key1 = cache_manager.generate_key("test query", "default", 5, True)
        key2 = cache_manager.generate_key("test query", "default", 5, True)

        # Same parameters should generate same key
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex length

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
        assert stats["total_keys"] == 10
        assert "memory_used" in stats
        assert "connected_clients" in stats

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
        assert "error" in stats


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
        assert result is False

    def test_delete_exception_handling(self, cache_manager, mock_redis):
        """Test exception handling in delete"""
        mock_redis.delete.side_effect = Exception("Redis error")

        result = cache_manager.delete("test_key")
        assert result is False
