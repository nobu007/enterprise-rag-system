"""Coverage-focused tests for app/services/document_loader.py.

Targets previously-uncovered branches: Document.encrypt_content/
decrypt_content error paths, DocumentLoader.load_pdf (ImportError when
pypdf absent + happy path via fake pypdf), load_markdown with an
encryptor, the load_directory .pdf branch + per-file error handler,
and the two TextSplitter.split_text branches (oversized first part;
fixed-size fallback when no configured separator matches).
"""
import sys
from unittest.mock import Mock

import pytest

from app.core.encryption import EncryptionError
from app.services.document_loader import (
    Document,
    DocumentLoader,
    TextSplitter,
)


class TestDocumentEncryptionErrors:
    """Cover encrypt/decrypt except paths (L53-55, L83-85)."""

    def test_encrypt_content_reraises_encryption_error(self):
        doc = Document(content="secret", metadata={"source": "s"})
        encryptor = Mock()
        encryptor.encrypt.side_effect = EncryptionError("nope")
        with pytest.raises(EncryptionError):
            doc.encrypt_content(encryptor)

    def test_decrypt_content_reraises_on_decrypt_failure(self):
        doc = Document(content="x", metadata={"source": "s"})
        doc.encrypted_content = "ciphertext"
        encryptor = Mock()
        encryptor.decrypt.side_effect = Exception("decrypt boom")
        with pytest.raises(Exception, match="decrypt boom"):
            doc.decrypt_content(encryptor)


def _install_pypdf(monkeypatch, pages):
    reader = Mock()
    reader.pages = pages
    fake = Mock()
    fake.PdfReader = Mock(return_value=reader)
    monkeypatch.setitem(sys.modules, "pypdf", fake)
    return fake


class TestLoadPdf:
    """Cover load_pdf ImportError + happy path (L124-156)."""

    def test_raises_import_error_when_pypdf_absent(self):
        # pypdf is not installed in this environment.
        with pytest.raises(ImportError, match="pypdf not installed"):
            DocumentLoader.load_pdf("anything.pdf")

    def test_loads_non_empty_pages_and_skips_blank(
        self, monkeypatch, tmp_path
    ):
        full = Mock()
        full.extract_text.return_value = "real page text"
        blank = Mock()
        blank.extract_text.return_value = "   "  # stripped -> skipped
        _install_pypdf(monkeypatch, [full, blank])

        target = tmp_path / "doc.pdf"
        target.write_bytes(b"%PDF-1.4")

        docs = DocumentLoader.load_pdf(str(target))
        assert len(docs) == 1
        assert docs[0].content == "real page text"
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["total_pages"] == 2

    def test_raises_file_not_found_when_path_missing(self, monkeypatch):
        _install_pypdf(monkeypatch, [])
        with pytest.raises(FileNotFoundError, match="File not found"):
            DocumentLoader.load_pdf("does_not_exist.pdf")

    def test_encrypts_pages_when_encryptor_given(self, monkeypatch, tmp_path):
        page = Mock()
        page.extract_text.return_value = "page text"
        _install_pypdf(monkeypatch, [page])
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")

        result = Mock()
        result.encrypted_data = "ENC"
        result.nonce = "N"
        result.salt = "S"
        result.tag = "T"
        encryptor = Mock()
        encryptor.encrypt.return_value = result

        docs = DocumentLoader.load_pdf(str(tmp_path / "doc.pdf"), encryptor)
        assert docs[0].encrypted_content == "ENC"
        assert docs[0].metadata["encrypted"] is True


class TestLoadMarkdownEncryption:
    """Cover load_markdown with an encryptor (L180)."""

    def test_load_markdown_encrypts_when_encryptor_given(self, tmp_path):
        target = tmp_path / "note.md"
        target.write_text("# Title\nbody", encoding="utf-8")

        result = Mock()
        result.encrypted_data = "ENC"
        result.nonce = "N"
        result.salt = "S"
        result.tag = "T"
        encryptor = Mock()
        encryptor.encrypt.return_value = result

        doc = DocumentLoader.load_markdown(str(target), encryptor=encryptor)
        assert doc.encrypted_content == "ENC"
        assert doc.metadata["encrypted"] is True
        encryptor.encrypt.assert_called_once()


class TestLoadDirectoryPdf:
    """Cover the .pdf branch + per-file error handler (L218-219, L229-230)."""

    def test_pdf_branch_extends_documents(self, monkeypatch, tmp_path):
        full = Mock()
        full.extract_text.return_value = "pdf text"
        _install_pypdf(monkeypatch, [full])
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")

        docs = DocumentLoader.load_directory(str(tmp_path))
        assert any(d.metadata.get("file_type") == "pdf" for d in docs)

    def test_per_file_error_is_swallowed(self, tmp_path):
        # .pdf present but pypdf absent -> load_pdf raises -> logged+skip.
        (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4")
        assert DocumentLoader.load_directory(str(tmp_path)) == []


class TestTextSplitterBranches:
    """Cover oversized-first-part + fixed-size fallback (L268, L283-286)."""

    def test_oversized_first_part_becomes_its_own_chunk(self):
        # First part already exceeds chunk_size with empty current_chunk.
        splitter = TextSplitter(chunk_size=5, chunk_overlap=2)
        chunks = splitter.split_text("xxxxxxxxxx y")
        assert chunks
        assert any("xxxxxxxxxx" in c for c in chunks)

    def test_fixed_size_fallback_when_no_separator_matches(self):
        # Custom separators without "" -> none match -> fixed-size path.
        splitter = TextSplitter(
            chunk_size=10, chunk_overlap=2, separators=["\n\n"]
        )
        chunks = splitter.split_text("hello world")
        assert chunks
        assert "hello" in chunks[0]
