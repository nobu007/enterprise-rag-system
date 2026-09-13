"""Regression tests for the command-line document ingestion pipeline."""

import logging
import sys
from types import SimpleNamespace

import pytest

from app.services.document_loader import Document
import scripts.ingest as ingest


def _patch_ingest_dependencies(monkeypatch, documents):
    split_inputs = []
    embedding_inputs = []
    upsert_calls = []
    database_calls = []

    class FakeSplitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def split_documents(self, documents_to_split):
            split_inputs.append(documents_to_split)
            return documents_to_split

    class FakeEmbeddingModel:
        dimension = 3

        def embed_texts(self, texts):
            embedding_inputs.append(texts)
            return [[0.1, 0.2, 0.3] for _ in texts]

    class FakeVectorDB:
        index = None

        def connect(self):
            database_calls.append("connect")

        def create_index(self, dimension):
            database_calls.append(("create_index", dimension))
            self.index = object()

        def upsert(self, **kwargs):
            upsert_calls.append(kwargs)

        def save(self, path):
            database_calls.append(("save", path))

        def get_stats(self):
            return {"total_vectors": 1, "dimension": 3}

    monkeypatch.setattr(
        ingest.DocumentLoader,
        "load_directory",
        staticmethod(lambda directory_path, recursive: documents),
    )
    monkeypatch.setattr(ingest, "TextSplitter", FakeSplitter)
    monkeypatch.setattr(
        ingest, "get_embedding_model", lambda: FakeEmbeddingModel()
    )
    monkeypatch.setattr(
        ingest,
        "get_vector_db",
        lambda **kwargs: FakeVectorDB(),
    )
    monkeypatch.setattr(
        ingest,
        "get_settings",
        lambda: SimpleNamespace(embedding_model="test-model"),
    )

    return split_inputs, embedding_inputs, upsert_calls, database_calls


def test_main_skips_invalid_documents_before_processing(monkeypatch, caplog):
    """CLI ingestion must apply the same validation gate as the API path."""
    valid = Document(
        content=(
            "This document contains enough useful text to pass validation."
        ),
        metadata={"source": "valid.txt", "file_type": "txt"},
    )
    invalid = Document(
        content="too short",
        metadata={
            "source": "invalid.txt",
            "file_type": "txt",
        },
    )
    patched = _patch_ingest_dependencies(monkeypatch, [valid, invalid])
    split_inputs, embedding_inputs, upsert_calls, _ = patched
    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest.py", "--source", "documents", "--collection", "hr"],
    )
    caplog.set_level(logging.INFO)

    ingest.main()

    assert split_inputs == [[valid]]
    assert embedding_inputs == [[valid.content]]
    assert len(upsert_calls) == 1
    assert upsert_calls[0]["ids"] == [valid.doc_id]
    assert upsert_calls[0]["metadata"] == [
        {"source": "valid.txt", "file_type": "txt", "collection": "hr"}
    ]
    assert any(
        "Skipped 1 invalid document" in record.getMessage()
        for record in caplog.records
    )


def test_main_stops_when_all_documents_fail_validation(monkeypatch, caplog):
    """CLI ingestion must not embed or store an entirely invalid batch."""
    invalid_documents = [
        Document(
            content="short",
            metadata={"source": "one.txt", "file_type": "txt"},
        ),
        Document(
            content="<script>alert('x')</script>",
            metadata={"source": "two.txt", "file_type": "txt"},
        ),
    ]
    split_inputs, embedding_inputs, upsert_calls, database_calls = (
        _patch_ingest_dependencies(monkeypatch, invalid_documents)
    )
    monkeypatch.setattr(sys, "argv", ["ingest.py", "--source", "documents"])
    caplog.set_level(logging.INFO)

    with pytest.raises(SystemExit) as exc_info:
        ingest.main()

    assert exc_info.value.code == 1
    assert split_inputs == []
    assert embedding_inputs == []
    assert upsert_calls == []
    assert database_calls == []
    assert any(
        "No valid documents found after validation" in record.getMessage()
        for record in caplog.records
    )
