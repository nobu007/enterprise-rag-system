"""
Unit tests for Document API Routes

Tests for the /documents endpoints including validation, error handling,
and response format verification, plus the mounted production app's
health/root routes.
"""

import pytest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.documents import router as documents_router
from app.main import app as production_app
from app.core.config import get_settings


class FakeEmbeddingModel:
    """Deterministic offline embedding model for route tests."""

    dimension = 8

    def embed_texts(self, texts):
        import hashlib

        return [
            list(hashlib.sha256(t.encode()).digest()[:8]) for t in texts
        ]


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Test client with the documents router mounted on a bare app.

    Embeddings are faked and the FAISS index path is redirected to a
    temporary directory so no real API call or repo-side artifact happens.
    """
    import app.core.embeddings as embeddings_module

    monkeypatch.setattr(
        embeddings_module, "get_embedding_model", lambda: FakeEmbeddingModel()
    )
    monkeypatch.setattr(
        get_settings(), "faiss_index_path", str(tmp_path / "faiss_index.bin")
    )

    app = FastAPI()
    app.include_router(documents_router, prefix="/api/v1")
    yield TestClient(app, backend_options={"use_uvloop": True})


@pytest.fixture
def prod_client():
    """Test client for the full production app (lifespan runs)."""
    with TestClient(production_app, backend_options={"use_uvloop": True}) as c:
        yield c


@pytest.fixture
def sample_docs_dir(tmp_path):
    """Directory with a couple of valid text documents."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(
        "Enterprise RAG systems retrieve relevant passages for a query "
        "and feed them to a language model.",
        encoding="utf-8",
    )
    (docs / "b.md").write_text(
        "# Title\n\nMarkdown body long enough to clear the validator's "
        "minimum content length requirement.",
        encoding="utf-8",
    )
    return str(docs)


class TestDocumentsIngestEndpoint:
    """POST /api/v1/documents/ingest"""

    def test_ingest_success(self, client, sample_docs_dir):
        response = client.post(
            "/api/v1/documents/ingest",
            json={"source_path": sample_docs_dir, "collection": "it"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["documents_processed"] == 2
        assert body["chunks_created"] >= 2
        assert body["collection"] == "it"

    def test_ingest_missing_directory_is_404(self, client):
        response = client.post(
            "/api/v1/documents/ingest",
            json={"source_path": "/nonexistent/path/xyz"},
        )
        assert response.status_code == 404

    def test_ingest_empty_directory_is_400(self, client, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        response = client.post(
            "/api/v1/documents/ingest",
            json={"source_path": str(empty)},
        )
        assert response.status_code == 400

    def test_ingest_control_chars_neutralised_in_logs(
        self, client, tmp_path, caplog
    ):
        """CRLF in client-supplied paths must not forge log lines.

        The source_path — both the directory name and the file names it
        contains — is client-controlled and used to flow raw into the
        route's info/warning lines and the loader's per-file debug line
        (the CWE-117 class pinned for vectordb collection logs in
        Issues 11-13). The bad file also fails validation so the
        failed-validation warning (metadata source + error messages)
        is exercised; the XSS error message embeds a content slice,
        making error_messages client-carried too.
        """
        import logging

        forged_dir = tmp_path / "src\n2000-01-01 INFO admin login ok"
        forged_dir.mkdir()
        (forged_dir / "good.txt").write_text(
            "Enterprise RAG systems retrieve relevant passages for a "
            "query and feed them to a language model.",
            encoding="utf-8",
        )
        (forged_dir / "bad\n2000-01-01 INFO root ok.txt").write_text(
            "<script>alert(1)</script> trailing padding so the file "
            "looks like an ordinary document body",
            encoding="utf-8",
        )

        with caplog.at_level(logging.DEBUG):
            response = client.post(
                "/api/v1/documents/ingest",
                json={"source_path": str(forged_dir)},
            )

        assert response.status_code == 200
        assert response.json()["success"] is True
        messages = [record.getMessage() for record in caplog.records]
        assert not any("\n" in m or "\r" in m for m in messages)
        assert any("\\n2000-01-01 INFO admin login ok" in m for m in messages)


class TestDocumentsUploadEndpoint:
    """POST /api/v1/documents/upload"""

    def test_upload_txt_success(self, client, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text(
            "Uploaded note content for the vector store, long enough to "
            "clear the validator minimum.",
            encoding="utf-8",
        )
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("note.txt", f.read_bytes(), "text/plain")},
            data={"collection": "uploads"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["documents_processed"] == 1
        assert body["collection"] == "uploads"

    def test_upload_unsupported_extension_is_400(self, client, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02")
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("data.bin", f.read_bytes(), "application/octet-stream")},
        )
        assert response.status_code == 400


class TestHealthAndRoot:
    """Production app health and root routes"""

    def test_health_check(self, prod_client):
        response = prod_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["version"] == production_app.version

    def test_detailed_health_check(self, prod_client):
        response = prod_client.get("/health/detailed")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["services"] == {
            "api": "healthy",
            "vector_db": "healthy",
            "llm": "healthy",
        }

    def test_root_endpoint(self, prod_client):
        response = prod_client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "running"
        assert body["docs"] == "/docs"
