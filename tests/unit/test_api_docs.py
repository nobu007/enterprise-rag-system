"""
Unit tests for API documentation consistency

Verifies that the OpenAPI schema and README.md describe the API that is
actually mounted — no ghost endpoints, no claimed-but-removed features.
Only the repository-root README.md is scanned (stale worktree copies are
explicitly excluded).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app as production_app

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client():
    with TestClient(production_app, backend_options={"use_uvloop": True}) as c:
        yield c


@pytest.fixture(scope="module")
def schema(client):
    return client.get("/openapi.json").json()


@pytest.fixture(scope="module")
def readme():
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


class TestOpenAPISchema:
    def test_openapi_schema_exists(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.json()["openapi"].startswith("3.")

    def test_openapi_schema_has_info(self, schema):
        assert schema["info"]["title"]
        assert schema["info"]["version"]

    def test_openapi_schema_has_paths(self, schema):
        assert len(schema["paths"]) > 0

    def test_all_endpoints_documented(self, schema):
        for path, methods in schema["paths"].items():
            for method, operation in methods.items():
                assert (
                    "description" in operation or "summary" in operation
                ), f"{method.upper()} {path} lacks documentation"

    def test_all_endpoints_have_tags(self, schema):
        for path, methods in schema["paths"].items():
            for method, operation in methods.items():
                assert operation.get("tags"), f"{method.upper()} {path} lacks tags"

    def test_mounted_paths_match_kept_surface(self, schema):
        expected = {
            "/health",
            "/health/detailed",
            "/api/v1/documents/ingest",
            "/api/v1/documents/upload",
            "/api/v1/documents/stats",
            "/",
        }
        assert set(schema["paths"].keys()) == expected

    def test_removed_features_not_mounted(self, schema):
        ghost_fragments = (
            "/query", "/batch", "/relationships", "/cache",
            "/versions", "/metrics", "/ingest/status",
        )
        for path in schema["paths"]:
            for fragment in ghost_fragments:
                assert fragment not in path, (
                    f"removed feature resurfaced in mounted path: {path}"
                )

    def test_api_tags_defined(self, schema):
        tag_names = {t["name"] for t in schema.get("tags", [])}
        mounted = {
            tag
            for methods in schema["paths"].values()
            for op in methods.values()
            for tag in op.get("tags", [])
        }
        assert mounted <= tag_names

    def test_contact_and_license_info(self, schema):
        assert schema["info"].get("contact")
        assert schema["info"].get("license")

    def test_swagger_ui_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_accessible(self, client):
        response = client.get("/redoc")
        assert response.status_code == 200


class TestREADMEConsistency:
    """README must describe only what the code actually provides."""

    def test_readme_exists_at_repo_root(self, readme):
        assert readme.strip(), "README.md is empty"

    def test_readme_documents_the_ingest_endpoints(self, readme):
        for endpoint in ("/documents/ingest", "/documents/upload", "/documents/stats"):
            assert endpoint in readme, f"README does not document {endpoint}"

    def test_readme_does_not_claim_removed_features(self, readme):
        # Features removed in the 2026-09 slim-down must not be advertised
        # as available. Match on feature words that only ever appeared for
        # the removed subsystems.
        forbidden = [
            "streaming", "rerank", "hybrid search", "redis", "celery",
            "rate limit", "confluence", "notion",
        ]
        lowered = readme.lower()
        for word in forbidden:
            assert word not in lowered, (
                f"README still claims removed feature: {word!r}"
            )

    def test_readme_vector_stores_match_implemented_backends(self, readme):
        implemented = ("faiss", "pinecone")
        # Any capitalized Vector-DB product name mentioned must be one of
        # the implemented backends (Weaviate/Chroma etc. were never wired).
        mentioned_products = [
            name for name in ("FAISS", "Pinecone", "Weaviate", "Chroma", "Milvus")
            if name.lower() in readme.lower()
        ]
        assert set(m.lower() for m in mentioned_products) <= set(implemented), (
            f"README names non-implemented vector stores: {mentioned_products}"
        )

    def test_readme_test_command_matches_suite(self, readme):
        assert ".venv310/bin/python -m pytest" in readme or "pytest" in readme
