"""Coverage-focused tests for app/core/cache.py.

Targets branches the existing test_cache.py does not reach: __init__
password-URL construction; the enabled-but-Redis-unavailable state
(L1 active, L2 None) for get/set/delete/get_stats; set() dataclass
conversion; and the whole invalidate_version method (disabled, matched
keys, no keys, redis error, L1-only) plus the clear_collection and
flush_all redis-error + L1-only branches.
"""
import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from app.core.cache import CacheManager


@pytest.fixture
def mock_redis():
    mock = MagicMock()
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.setex.return_value = True
    mock.delete.return_value = True
    mock.flushdb.return_value = True
    mock.db = 0
    mock.info.return_value = {
        "db0": {"keys": 10},
        "used_memory_human": "1M",
    }
    return mock


@pytest.fixture
def enabled_cache(mock_redis):
    with patch("app.core.cache.redis.from_url", return_value=mock_redis):
        manager = CacheManager(enabled=True)
        manager.redis_client = mock_redis
        return manager


@pytest.fixture
def no_redis_cache():
    # enabled=True but Redis unavailable -> L1 only (redis_client=None).
    with patch(
        "app.core.cache.redis.from_url",
        side_effect=ConnectionError("no redis"),
    ):
        manager = CacheManager(enabled=True)
    assert manager.redis_client is None
    return manager


@pytest.fixture
def disabled_cache():
    return CacheManager(enabled=False)


@dataclass
class _Sample:
    a: int = 1
    b: str = "x"


class TestInitPasswordUrl:
    def test_password_embedded_in_redis_url(self, mock_redis):
        with patch(
            "app.core.cache.redis.from_url", return_value=mock_redis
        ) as fu:
            CacheManager(redis_password="secret", enabled=True)
        url = fu.call_args[0][0]
        assert ":secret@" in url


class TestL1OnlyState:
    """enabled + redis_client=None (L1 active, L2 down)."""

    def test_get_returns_none_on_miss(self, no_redis_cache):
        assert no_redis_cache.get("absent_key") is None

    def test_set_stores_in_l1_only(self, no_redis_cache):
        assert no_redis_cache.set("k", {"v": 1}) is True
        value, _ts = no_redis_cache.l1_cache["k"]
        assert value == {"v": 1}

    def test_delete_clears_l1_only(self, no_redis_cache):
        no_redis_cache.set("k", {"v": 1})
        assert no_redis_cache.delete("k") is True
        assert "k" not in no_redis_cache.l1_cache

    def test_get_stats_marks_l2_disabled(self, no_redis_cache):
        stats = no_redis_cache.get_stats()
        assert stats["l2"] == {"enabled": False}


class TestSetDataclass:
    def test_set_converts_dataclass_to_dict(
        self, enabled_cache, mock_redis
    ):
        enabled_cache.set("dc", _Sample(a=5, b="z"))
        payload = json.loads(mock_redis.setex.call_args[0][2])
        assert payload == {"a": 5, "b": "z"}


class TestInvalidateVersion:
    def test_disabled_returns_false(self, disabled_cache):
        assert disabled_cache.invalidate_version("v1") is False

    def test_deletes_matched_keys(self, enabled_cache, mock_redis):
        mock_redis.scan_iter.return_value = iter(["k1", "k2"])
        assert enabled_cache.invalidate_version("v1") is True
        mock_redis.delete.assert_called_once_with("k1", "k2")

    def test_no_keys_skips_delete(self, enabled_cache, mock_redis):
        mock_redis.scan_iter.return_value = iter([])
        assert enabled_cache.invalidate_version("v1") is True
        mock_redis.delete.assert_not_called()

    def test_redis_error_returns_false(self, enabled_cache, mock_redis):
        mock_redis.scan_iter.side_effect = Exception("scan fail")
        assert enabled_cache.invalidate_version("v1") is False

    def test_l1_only_clears_and_returns_true(self, no_redis_cache):
        no_redis_cache.set("k", {"v": 1})
        assert no_redis_cache.invalidate_version("v1") is True
        assert no_redis_cache.l1_cache == {}


class TestClearCollectionBranches:
    def test_redis_error_returns_false(self, enabled_cache, mock_redis):
        mock_redis.scan_iter.side_effect = Exception("scan fail")
        assert enabled_cache.clear_collection("p") is False

    def test_l1_only_filters_by_prefix(self, no_redis_cache):
        no_redis_cache.set("p:1", {"v": 1})
        no_redis_cache.set("other", {"v": 2})
        assert no_redis_cache.clear_collection("p:") is True
        assert "p:1" not in no_redis_cache.l1_cache
        assert "other" in no_redis_cache.l1_cache


class TestFlushAllBranches:
    def test_redis_error_returns_false(self, enabled_cache, mock_redis):
        mock_redis.flushdb.side_effect = Exception("flush fail")
        assert enabled_cache.flush_all() is False

    def test_l1_only_clears_and_returns_true(self, no_redis_cache):
        no_redis_cache.set("k", {"v": 1})
        assert no_redis_cache.flush_all() is True
        assert no_redis_cache.l1_cache == {}


class TestNormalizeQueryUnicode:
    """Non-Latin queries must not collapse to the same cache key.

    The normalizer may strip punctuation, but it must preserve the
    *content* of queries written in other scripts (CJK, Arabic,
    accented Latin). If every non-ASCII-alphanumeric character is
    deleted, distinct queries like "東京タワー" and "大阪城" both
    normalize to "" and produce the *same* cache key — a cache-poisoning
    bug where one query can return another's cached answer.
    """

    def _manager(self):
        # _normalize_query / generate_key never touch Redis; disabled
        # avoids needing a connection.
        return CacheManager(enabled=False)

    def test_distinct_cjk_queries_normalize_differently(self):
        mgr = self._manager()
        n1 = mgr._normalize_query("東京タワー")
        n2 = mgr._normalize_query("大阪城")
        assert n1 == "東京タワー"
        assert n2 == "大阪城"
        assert n1 != n2

    def test_distinct_cjk_queries_get_distinct_cache_keys(self):
        mgr = self._manager()
        k1 = mgr.generate_key("東京タワー")
        k2 = mgr.generate_key("大阪城")
        assert k1 != k2

    def test_accented_latin_preserved_not_truncated(self):
        # "café" must not be silently reduced to "caf" (losing the é),
        # which would collide it with a hypothetical "caf" query.
        mgr = self._manager()
        assert mgr._normalize_query("Café") == "café"
        assert mgr._normalize_query("naïve") == "naïve"


class TestCacheKeyFilterIsolation:
    """A query's result set depends on its filters and search mode, so the
    cache key must reflect them.

    ``generate_key`` historically keyed only on (query, collection, top_k,
    rerank) even though ``query()`` also feeds ``filter_dict`` and
    ``use_hybrid`` into retrieval. Two requests identical in the keyed
    params but differing in filter (or search mode) therefore produced the
    *same* key, so the second request returned the FIRST request's cached
    answer — a cross-filter data leak / cache-poisoning bug, the
    param-omission sibling of ``TestNormalizeQueryUnicode`` above.
    """

    def _manager(self):
        return CacheManager(enabled=False)

    def test_distinct_filters_get_distinct_cache_keys(self):
        mgr = self._manager()
        k_fin = mgr.generate_key(
            "salary", "default", 5, True, True, {"dept": "finance"}
        )
        k_eng = mgr.generate_key(
            "salary", "default", 5, True, True, {"dept": "engineering"}
        )
        assert k_fin != k_eng

    def test_filter_present_vs_absent_get_distinct_keys(self):
        mgr = self._manager()
        k_filtered = mgr.generate_key(
            "salary", "default", 5, True, True, {"dept": "finance"}
        )
        k_unfiltered = mgr.generate_key(
            "salary", "default", 5, True, True, None
        )
        assert k_filtered != k_unfiltered

    def test_use_hybrid_distinct_keys(self):
        mgr = self._manager()
        k_hybrid = mgr.generate_key("q", "default", 5, True, True, None)
        k_vector = mgr.generate_key("q", "default", 5, True, False, None)
        assert k_hybrid != k_vector

    def test_none_and_empty_filter_collapse_to_same_key(self):
        # Both None and {} mean "no restriction", so they must share a key
        # (otherwise equivalent queries needlessly bypass the cache).
        mgr = self._manager()
        k_none = mgr.generate_key("q", "default", 5, True, True, None)
        k_empty = mgr.generate_key("q", "default", 5, True, True, {})
        assert k_none == k_empty

    def test_filter_key_order_independent(self):
        # Semantically equal filters expressed in different key order must
        # not fragment the cache key.
        mgr = self._manager()
        k_ab = mgr.generate_key("q", "default", 5, True, True, {"a": 1, "b": 2})
        k_ba = mgr.generate_key("q", "default", 5, True, True, {"b": 2, "a": 1})
        assert k_ab == k_ba

