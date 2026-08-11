"""Coverage-focused tests for app/api/routes/health.py.

Targets the previously-uncovered ``/cache/stats`` endpoint (both the
cache-present and cache-absent branches) and the ``/health/detailed``
response body. Uses the real app (limiter is wired at import time in
app.main) and mutates ``app.state.cache_manager`` per test, restoring
the prior value afterward. TestClient is used without a context
manager so the app lifespan (which would try to initialize external
services) does not run.
"""
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import app

_MISSING = object()


class TestCacheStatsEndpoint:
    """Cover /cache/stats (health.py L135-145)."""

    def _restore_setup(self):
        """Snapshot app.state.cache_manager for restore."""
        return getattr(app.state, "cache_manager", _MISSING)

    def _restore(self, prior):
        if prior is _MISSING:
            if hasattr(app.state, "cache_manager"):
                del app.state.cache_manager
        else:
            app.state.cache_manager = prior

    def test_cache_stats_when_manager_absent(self):
        """No cache_manager -> enabled:False message (L135-141)."""
        prior = self._restore_setup()
        try:
            if hasattr(app.state, "cache_manager"):
                del app.state.cache_manager
            client = TestClient(app)
            resp = client.get("/cache/stats")
        finally:
            self._restore(prior)

        assert resp.status_code == 200
        assert resp.json() == {
            "enabled": False,
            "message": "Cache manager not initialized",
        }

    def test_cache_stats_when_manager_present(self):
        """cache_manager.get_stats() returned as-is (L135, L144-145)."""
        prior = self._restore_setup()
        try:
            stats = {
                "enabled": True,
                "total_keys": 42,
                "memory_used": "1.5M",
            }
            cache = Mock()
            cache.get_stats.return_value = stats
            app.state.cache_manager = cache
            client = TestClient(app)
            resp = client.get("/cache/stats")
        finally:
            self._restore(prior)

        assert resp.status_code == 200
        assert resp.json() == stats
        cache.get_stats.assert_called_once()

    def test_cache_stats_when_manager_falsy(self):
        """A falsy cache_manager (None) -> enabled:False (L137)."""
        prior = self._restore_setup()
        try:
            app.state.cache_manager = None
            client = TestClient(app)
            resp = client.get("/cache/stats")
        finally:
            self._restore(prior)

        assert resp.status_code == 200
        assert resp.json()["enabled"] is False


class TestDetailedHealthEndpoint:
    """Cover /health/detailed response body (health.py L81-89)."""

    def test_detailed_health_returns_services(self):
        client = TestClient(app)
        resp = client.get("/health/detailed")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["version"]
        assert set(body["services"]) == {"api", "vector_db", "llm"}
