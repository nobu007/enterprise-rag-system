"""Coverage-focused tests for app/api/routes/health.py.

Covers the basic and detailed health endpoints, including the version
source (settings, not a hardcoded literal).
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app as production_app
from app.core.config import get_settings


@pytest.fixture
def client():
    with TestClient(production_app, backend_options={"use_uvloop": True}) as c:
        yield c


def test_basic_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"] == get_settings().app_version


def test_detailed_health_returns_all_services(client):
    response = client.get("/health/detailed")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"] == get_settings().app_version
    assert body["services"] == {
        "api": "healthy",
        "vector_db": "healthy",
        "llm": "healthy",
    }
