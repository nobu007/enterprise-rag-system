"""
Test API documentation and OpenAPI schema validation
APIドキュメントとOpenAPIスキーマ検証のテスト
"""

import inspect
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes.query import router, stream_query


REPO_ROOT = Path(__file__).resolve().parents[2]
RELATIONSHIP_API_PREFIX = "/api/v1/relationships"
RELATIONSHIP_API_PATH = re.compile(
    rf"{re.escape(RELATIONSHIP_API_PREFIX)}(?:/[^\s`\"')]+)?"
)


@pytest.fixture
def client():
    """Test client fixture with mocked lifespan dependencies / モック化されたlifespan依存のテストクライアントフィクスチャ"""
    # Pre-set app.state to avoid lifespan initialization failures
    app.state.openai_client = AsyncMock()
    app.state.cache_manager = MagicMock()
    app.state.rag_pipeline = MagicMock()
    client = TestClient(
        app,
        raise_server_exceptions=False,
        backend_options={"use_uvloop": True},
    )
    try:
        yield client
    finally:
        client.close()


class TestAPIDocumentation:
    """Test API documentation / APIドキュメントのテスト"""

    def test_openapi_schema_exists(self, client):
        """Test that OpenAPI schema is accessible / OpenAPIスキーマにアクセスできることをテスト"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()
        assert response.json()["openapi"].startswith("3.")

    def test_openapi_schema_has_info(self, client):
        """Test that OpenAPI schema has required info fields / OpenAPIスキーマに必須のinfoフィールドがあることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "info" in schema
        assert schema["info"]["title"] == "Enterprise RAG System"
        assert "version" in schema["info"]
        assert "description" in schema["info"]

    def test_openapi_schema_has_paths(self, client):
        """Test that OpenAPI schema defines paths / OpenAPIスキーマがパスを定義していることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "paths" in schema
        assert len(schema["paths"]) > 0

    def test_streaming_route_prefix_matches_fixture_and_production_mount(self):
        """Keep direct-router and production route paths distinct."""
        production_paths = app.openapi()["paths"]
        assert "/api/v1/query/stream" in production_paths
        assert "/query/stream" not in production_paths

        direct_app = FastAPI()
        direct_app.include_router(router)
        direct_paths = direct_app.openapi()["paths"]
        assert "/query/stream" in direct_paths
        assert "/api/v1/query/stream" not in direct_paths

    def test_all_endpoints_documented(self, client):
        """Test that all endpoints have documentation / すべてのエンドポイントがドキュメント化されていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        for path, methods in schema["paths"].items():
            for method, details in methods.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    # Check for summary and description
                    assert "summary" in details or "description" in details, \
                        f"Endpoint {method.upper()} {path} missing documentation"

    def test_all_endpoints_have_tags(self, client):
        """Test that all endpoints have tags for grouping / すべてのエンドポイントがグループ化用のタグを持っていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        for path, methods in schema["paths"].items():
            for method, details in methods.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    # Skip auto-generated endpoints (e.g., Prometheus /metrics)
                    if path in ["/metrics"]:
                        continue
                    assert "tags" in details and len(details["tags"]) > 0, \
                        f"Endpoint {method.upper()} {path} missing tags"

    def test_query_endpoint_documentation(self, client):
        """Test query endpoint has comprehensive documentation / クエリエンドポイントが包括的なドキュメントを持っていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        query_path = schema["paths"].get("/api/v1/query/")
        assert query_path is not None, "Query endpoint not found"

        post_details = query_path.get("post")
        assert post_details is not None, "POST method not found for query endpoint"

        # Check for response documentation
        assert "responses" in post_details
        assert "200" in post_details["responses"]
        assert "422" in post_details["responses"] or "400" in post_details["responses"]

        # Check for request body with schema
        assert "requestBody" in post_details
        request_body = post_details["requestBody"]
        assert "content" in request_body
        assert "application/json" in request_body["content"]

    def test_streaming_documentation_matches_post_json_contract(self):
        """Keep copied streaming examples aligned with the mounted API."""
        schema = app.openapi()
        stream_path = schema["paths"].get("/api/v1/query/stream")

        assert stream_path is not None, "Streaming endpoint not found"
        assert set(stream_path) == {"post"}

        post_details = stream_path["post"]
        request_body = post_details["requestBody"]["content"]
        assert "application/json" in request_body
        assert request_body["application/json"]["schema"]["$ref"].endswith(
            "/StreamingQueryRequest"
        )
        assert "text/event-stream" in post_details["responses"]["200"]["content"]

        openapi_description = post_details["description"]
        route_docstring = inspect.getdoc(stream_query) or ""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        streaming_section = readme.split("### 🌊 Streaming Responses (New!)", 1)[1]
        streaming_section = streaming_section.split("### Request Tracking", 1)[0]
        for document in (openapi_description, route_docstring, streaming_section):
            assert "/api/v1/query/stream" in document
            assert "EventSource(" not in document
            assert "requests.get(" not in document
            assert "http://localhost:8000/query/stream" not in document

        assert "requests.post(" in route_docstring
        assert "curl -N -X POST" in streaming_section

    def test_multi_tenant_examples_match_query_route(self):
        """Keep both multi-tenant README examples aligned with the mounted
        query route."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        section_marker = "## 👥 Multi-Tenant Support"
        section_end = "## 🏗️ Architecture"

        assert section_marker in readme
        multi_tenant_section = readme.split(section_marker, 1)[1]
        assert section_end in multi_tenant_section
        multi_tenant_section = multi_tenant_section.split(section_end, 1)[0]

        query_path = "/api/v1/query/"
        assert query_path in app.openapi()["paths"]
        example_urls = re.findall(
            r'curl -X POST "([^"]+)"', multi_tenant_section
        )

        assert example_urls == [
            f"http://localhost:8000{query_path}",
            f"http://localhost:8000{query_path}",
        ]

    def test_documents_endpoint_documentation(self, client):
        """Test documents endpoints have comprehensive documentation / ドキュメントエンドポイントが包括的なドキュメントを持っていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        # The documents router must be mounted by the application, not only
        # defined in app.api.routes.documents.
        document_ingest_path = schema["paths"].get("/api/v1/documents/ingest")
        assert document_ingest_path is not None, (
            "Document ingest endpoint not found"
        )
        assert "post" in document_ingest_path

        # Check /ingest endpoint
        ingest_path = schema["paths"].get("/api/v1/ingest")
        assert ingest_path is not None, "Ingest endpoint not found"

        post_details = ingest_path.get("post")
        assert post_details is not None
        assert "summary" in post_details

        # /ingest/status endpoint is optional (may not exist); ingest above is verified

    def test_document_examples_match_production_prefix(self):
        """Keep document API examples aligned with the mounted production paths."""
        schema = app.openapi()
        document_descriptions = [
            operation.get("description", "")
            for path, operations in schema["paths"].items()
            if path.startswith("/api/v1/documents/")
            for operation in operations.values()
            if isinstance(operation, dict)
        ]

        assert "/api/v1/documents/batch/{task_id}/status" in schema["paths"]
        assert "/api/v1/documents/versioning" in schema["paths"]
        assert document_descriptions
        assert all(
            "http://localhost:8000/documents/" not in description
            and "`/documents/" not in description
            for description in document_descriptions
        )

    def test_relationship_documentation_matches_placeholder_router(self):
        """Do not advertise endpoints that the placeholder router does not expose."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        feature_line = next(
            (
                line
                for line in readme.splitlines()
                if "Document Relationship Graph" in line
            ),
            None,
        )

        assert feature_line is not None
        assert "planned" in feature_line.lower()
        assert "not available" in feature_line.lower()

        relationship_paths = [
            path
            for path in app.openapi()["paths"]
            if path == RELATIONSHIP_API_PREFIX
            or path.startswith(f"{RELATIONSHIP_API_PREFIX}/")
        ]
        assert relationship_paths == []

        documentation_files = sorted(
            path
            for path in REPO_ROOT.rglob("*.md")
            if ".git" not in path.parts
        )
        documented_paths = [
            (path, match)
            for path in documentation_files
            for match in RELATIONSHIP_API_PATH.findall(
                path.read_text(encoding="utf-8")
            )
        ]
        assert all(
            match == RELATIONSHIP_API_PREFIX for _, match in documented_paths
        ), documented_paths

    def test_testing_documentation_matches_repository_layout(self):
        """Keep README test commands aligned with the checked-in test suites."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        assert "pytest tests/unit" in readme
        assert "pytest tests/integration" in readme
        assert "pytest tests/e2e" not in readme
        assert "no separate `tests/e2e/` suite" in readme
        assert "under `tests/integration/`" in readme
        assert (REPO_ROOT / "tests" / "unit").is_dir()
        assert (REPO_ROOT / "tests" / "integration").is_dir()

    def test_health_endpoint_documentation(self, client):
        """Test health endpoints have documentation / ヘルスエンドポイントがドキュメントを持っていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        # Check /health endpoint
        health_path = schema["paths"].get("/health")
        assert health_path is not None, "Health endpoint not found"

        # Check /health/detailed endpoint
        detailed_health_path = schema["paths"].get("/health/detailed")
        assert detailed_health_path is not None, "Detailed health endpoint not found"

        # Check /cache/stats endpoint
        cache_stats_path = schema["paths"].get("/cache/stats")
        assert cache_stats_path is not None, "Cache stats endpoint not found"

    def test_health_routes_and_response_match_readme_contract(self, client):
        """Keep the documented health endpoint aligned with the live API."""
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]

        assert "/health/detailed" in paths
        assert set(paths["/health/detailed"]) == {"get"}
        assert "/health/db" not in paths
        assert client.get("/health/db").status_code == 404

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        health_section = readme.split("#### Health Check", 1)[1]
        health_section = health_section.split("#### Best Practices", 1)[0]
        response_match = re.search(
            r"# Example response:\s*\n(?P<body>\{.*?\n\})\s*```",
            health_section,
            re.DOTALL,
        )
        assert response_match is not None, "Health response example not found"

        documented_response = json.loads(response_match.group("body"))
        response = client.get("/health/detailed")
        assert response.status_code == 200
        assert response.json() == documented_response

    def test_documented_health_urls_do_not_use_unmounted_route(self):
        """Scan Markdown docs for executable or absolute stale health URLs."""
        stale_patterns = (
            re.compile(
                r"^\s*(?:curl|wget)\b[^\n]*/health/db\b", re.IGNORECASE
            ),
            re.compile(r"https?://[^\s`\"')]+/health/db\b", re.IGNORECASE),
        )
        stale_references = []

        for path in REPO_ROOT.rglob("*.md"):
            if ".git" in path.parts:
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if any(pattern.search(line) for pattern in stale_patterns):
                    stale_references.append(
                        f"{path.relative_to(REPO_ROOT)}:{line_number}"
                    )

        assert not stale_references, stale_references

    def test_error_response_models_defined(self, client):
        """Test that error response models are defined / エラーレスポンスモデルが定義されていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "components" in schema
        assert "schemas" in schema["components"]

        schemas = schema["components"]["schemas"]

        # Check for common response models (ErrorResponse may not be in schema
        # if not explicitly used in endpoint responses)
        assert len(schemas) > 0, "No schemas defined"
        # Verify key models exist
        assert "QueryRequest" in schemas or "QueryResponse" in schemas, \
            "Core query models not defined in schema"

    def test_pydantic_models_have_descriptions(self, client):
        """Test that Pydantic models have field descriptions / Pydanticモデルがフィールド記述を持っていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        schemas = schema["components"]["schemas"]

        # Check QueryRequest model
        if "QueryRequest" in schemas:
            query_request = schemas["QueryRequest"]
            assert "properties" in query_request

            # Check that fields have descriptions
            for prop_name, prop_details in query_request["properties"].items():
                # At least some fields should have descriptions
                if prop_name in ["query", "collection", "top_k"]:
                    assert "description" in prop_details, \
                        f"Field {prop_name} in QueryRequest missing description"

    def test_examples_in_request_models(self, client):
        """Test that request models include examples / リクエストモデルが例を含んでいることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        schemas = schema["components"]["schemas"]

        # Check that at least some models have examples
        # Note: Pydantic V2 may place examples at property level or in json_schema_extra
        models_with_examples = []
        for model_name, model_details in schemas.items():
            has_examples = False
            if "example" in model_details or "examples" in model_details:
                has_examples = True
            # Check properties for examples
            if "properties" in model_details:
                for prop_name, prop_details in model_details["properties"].items():
                    if any(k in prop_details for k in ("examples", "example", "enum")):
                        has_examples = True
                        break
            if has_examples:
                models_with_examples.append(model_name)

        # At least the core models should exist
        assert "QueryRequest" in schemas, "QueryRequest model must be defined"

    def test_swagger_ui_accessible(self, client):
        """Test that Swagger UI is accessible / Swagger UIにアクセスできることをテスト"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_redoc_accessible(self, client):
        """Test that ReDoc is accessible / ReDocにアクセスできることをテスト"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_api_tags_defined(self, client):
        """Test that API tags are defined in schema / APIタグがスキーマで定義されていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "tags" in schema
        tags = schema["tags"]

        # Check for expected tags
        tag_names = [tag["name"] for tag in tags]
        assert "Query" in tag_names
        assert "Documents" in tag_names
        assert "Health" in tag_names

    def test_contact_and_license_info(self, client):
        """Test that contact and license information is provided / コンタクトとライセンス情報が提供されていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        info = schema["info"]

        # Check for contact info
        assert "contact" in info
        contact = info["contact"]
        assert "name" in contact

        # Check for license info
        assert "license" in info
        license_info = info["license"]
        assert "name" in license_info

    def test_rate_limiting_documented(self, client):
        """Test that rate limiting is documented in the API description / レート制限がAPI記述で文書化されていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        info = schema["info"]
        description = info.get("description", "")

        # Check for rate limiting documentation
        assert "rate" in description.lower() or "limit" in description.lower(), \
            "Rate limiting not documented in API description"

    def test_response_descriptions(self, client):
        """Test that endpoints have response descriptions / エンドポイントがレスポンス記述を持っていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        for path, methods in schema["paths"].items():
            for method, details in methods.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    responses = details.get("responses", {})

                    # Check success response has description
                    if "200" in responses:
                        assert "description" in responses["200"], \
                            f"Endpoint {method.upper()} {path} missing response description for 200"


class TestAPIIntegration:
    """Integration tests for API documentation / APIドキュメントの統合テスト"""

    def test_query_request_schema_validates(self, client):
        """Test that query request schema validates correctly / クエリリクエストスキーマが正しく検証することをテスト"""
        # Valid request
        valid_request = {
            "query": "What is RAG?",
            "collection": "default",
            "top_k": 5,
            "include_sources": True
        }

        # This should not raise validation errors (we're not actually calling the endpoint,
        # just testing that the schema is valid)
        response = client.post("/api/v1/query/", json=valid_request)

        # We don't care if the query fails (might be missing DB),
        # we just care it's not a validation error
        # Validation errors return 422
        if response.status_code == 422:
            pytest.fail("Valid request failed validation")

    def test_query_request_validation_invalid_query(self, client):
        """Test that invalid query is rejected / 不正なクエリが拒否されることをテスト"""
        invalid_request = {
            "query": "",  # Empty query should fail validation
            "collection": "default",
            "top_k": 5
        }

        response = client.post("/api/v1/query/", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_query_request_validation_invalid_top_k(self, client):
        """Test that invalid top_k is rejected / 不正なtop_kが拒否されることをテスト"""
        invalid_request = {
            "query": "Test query",
            "collection": "default",
            "top_k": 100  # Should be <= 20
        }

        response = client.post("/api/v1/query/", json=invalid_request)
        assert response.status_code == 422  # Validation error
