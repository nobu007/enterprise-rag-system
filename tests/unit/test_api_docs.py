"""
Test API documentation and OpenAPI schema validation
APIドキュメントとOpenAPIスキーマ検証のテスト
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Test client fixture / テストクライアントフィクスチャ"""
    return TestClient(app)


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

    def test_documents_endpoint_documentation(self, client):
        """Test documents endpoints have comprehensive documentation / ドキュメントエンドポイントが包括的なドキュメントを持っていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        # Check /ingest endpoint
        ingest_path = schema["paths"].get("/api/v1/documents/ingest")
        assert ingest_path is not None, "Ingest endpoint not found"

        post_details = ingest_path.get("post")
        assert post_details is not None
        assert "summary" in post_details

        # Check /stats endpoint
        stats_path = schema["paths"].get("/api/v1/documents/stats")
        assert stats_path is not None, "Stats endpoint not found"

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

    def test_error_response_models_defined(self, client):
        """Test that error response models are defined / エラーレスポンスモデルが定義されていることをテスト"""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "components" in schema
        assert "schemas" in schema["components"]

        schemas = schema["components"]["schemas"]

        # Check for ErrorResponse model
        assert "ErrorResponse" in schemas or "ErrorDetail" in schemas, \
            "Error response models not defined"

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
        models_with_examples = []
        for model_name, model_details in schemas.items():
            if "example" in model_details or "examples" in model_details:
                models_with_examples.append(model_name)

        # We expect at least a few models to have examples
        assert len(models_with_examples) > 0, \
            "No request models have examples. Consider adding examples for better UX."

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
