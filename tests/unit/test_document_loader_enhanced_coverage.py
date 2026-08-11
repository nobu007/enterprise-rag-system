"""Coverage-focused tests for app/services/document_loader_enhanced.py.

Targets the previously-uncovered EnhancedDocumentLoader branches:
load_pdf (FileNotFoundError, advanced-parse happy path, advanced-failure
fallback, advanced-disabled, extract_tables=False) and load_directory
(default extensions, recursive/non-recursive enumeration, the .pdf/.md
table-extraction/.txt branches, and the per-file error handler).

Parser + base-loader calls are controlled with Mocks so the tests do
not depend on pypdf/pdfplumber (neither is installed here).
"""
from unittest.mock import Mock

import pytest

from app.services.document_loader import Document
from app.services.document_loader_enhanced import EnhancedDocumentLoader


def _parsed(text="full text", tables=None, charts=None):
    """A stand-in for a DocumentParser.ParseResult."""
    parsed = Mock()
    parsed.tables = tables if tables is not None else []
    parsed.charts = charts if charts is not None else []
    parsed.get_full_text.return_value = text
    return parsed


def _patch_basic_load_pdf(monkeypatch, docs):
    monkeypatch.setattr(
        "app.services.document_loader.DocumentLoader.load_pdf",
        Mock(return_value=docs),
    )


class TestLoadPdf:
    """Cover load_pdf (L59-92)."""

    def test_raises_file_not_found(self):
        loader = EnhancedDocumentLoader()
        with pytest.raises(FileNotFoundError, match="File not found"):
            loader.load_pdf("missing.pdf")

    def test_advanced_parse_happy_path(self, tmp_path):
        loader = EnhancedDocumentLoader()
        loader.parser = Mock()
        loader.parser.parse_pdf.return_value = _parsed(
            "content with table", tables=["t1"], charts=["c1"]
        )
        target = tmp_path / "doc.pdf"
        target.write_bytes(b"%PDF-1.4")

        docs = loader.load_pdf(str(target))
        assert len(docs) == 1
        assert docs[0].content == "content with table"
        assert docs[0].metadata["tables_extracted"] == 1
        assert docs[0].metadata["charts_detected"] == 1
        assert docs[0].metadata["parsing_method"] == "enhanced"

    def test_advanced_failure_falls_back_to_basic(
        self, tmp_path, monkeypatch
    ):
        loader = EnhancedDocumentLoader()
        loader.parser = Mock()
        loader.parser.parse_pdf.side_effect = Exception("parse boom")
        sentinel = Document(content="basic", metadata={})
        _patch_basic_load_pdf(monkeypatch, [sentinel])
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")

        assert loader.load_pdf(str(tmp_path / "doc.pdf")) == [sentinel]

    def test_basic_loader_when_advanced_disabled(self, tmp_path, monkeypatch):
        loader = EnhancedDocumentLoader(enable_advanced_parsing=False)
        assert loader.parser is None
        sentinel = Document(content="basic2", metadata={})
        _patch_basic_load_pdf(monkeypatch, [sentinel])
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")

        assert loader.load_pdf(str(tmp_path / "doc.pdf")) == [sentinel]

    def test_basic_loader_when_extract_tables_false(
        self, tmp_path, monkeypatch
    ):
        loader = EnhancedDocumentLoader()
        sentinel = Document(content="basic3", metadata={})
        _patch_basic_load_pdf(monkeypatch, [sentinel])
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")

        docs = loader.load_pdf(
            str(tmp_path / "doc.pdf"), extract_tables=False
        )
        assert docs == [sentinel]


class TestLoadDirectory:
    """Cover load_directory (L119-166)."""

    def test_raises_when_directory_missing(self):
        loader = EnhancedDocumentLoader()
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            loader.load_directory("/no/such/dir/xyz")

    def test_recursive_loads_subdirectory_files(self, tmp_path):
        loader = EnhancedDocumentLoader()
        loader.load_text_file = Mock(
            return_value=Document(content="t", metadata={})
        )
        (tmp_path / "top.txt").write_text("top")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep")

        docs = loader.load_directory(str(tmp_path))  # recursive=True
        assert len(docs) == 2  # rglob found both

    def test_non_recursive_ignores_subdirectories(self, tmp_path):
        loader = EnhancedDocumentLoader()
        loader.load_text_file = Mock(
            return_value=Document(content="t", metadata={})
        )
        (tmp_path / "top.txt").write_text("top")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep")

        docs = loader.load_directory(str(tmp_path), recursive=False)
        assert len(docs) == 1  # glob found only top-level

    def test_pdf_branch_extends_documents(self, tmp_path):
        loader = EnhancedDocumentLoader()
        pdf_doc = Document(content="pdf", metadata={"file_type": "pdf"})
        loader.load_pdf = Mock(return_value=[pdf_doc])
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")

        assert loader.load_directory(str(tmp_path)) == [pdf_doc]

    def test_md_table_extraction_enriches_content(self, tmp_path):
        loader = EnhancedDocumentLoader()
        loader.load_markdown = Mock(
            return_value=Document(
                content="raw md", metadata={"file_type": "md"}
            )
        )
        loader.parser = Mock()
        loader.parser.parse_text.return_value = _parsed(
            text="md with table", tables=["t1", "t2"]
        )
        (tmp_path / "t.md").write_text("# t")

        docs = loader.load_directory(str(tmp_path))
        assert docs[0].content == "md with table"
        assert docs[0].metadata["tables_extracted"] == 2

    def test_per_file_error_is_swallowed(self, tmp_path):
        loader = EnhancedDocumentLoader()
        loader.load_pdf = Mock(side_effect=Exception("boom"))
        loader.load_text_file = Mock(
            return_value=Document(content="txt", metadata={})
        )
        (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "ok.txt").write_text("ok")

        docs = loader.load_directory(str(tmp_path))
        # .pdf raised (logged + skipped), .txt loaded.
        assert len(docs) == 1
        assert docs[0].content == "txt"
