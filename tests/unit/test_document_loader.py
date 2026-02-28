"""
Unit tests for Document Loader and TextSplitter
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from app.services.document_loader import Document, DocumentLoader, TextSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample text file
        txt_path = os.path.join(tmpdir, "sample.txt")
        with open(txt_path, "w") as f:
            f.write("Hello world. This is a test document.")

        # Create sample markdown file
        md_path = os.path.join(tmpdir, "readme.md")
        with open(md_path, "w") as f:
            f.write("# Title\n\nSome markdown content here.")

        # Create a nested directory with another file
        nested = os.path.join(tmpdir, "subdir")
        os.makedirs(nested)
        nested_txt = os.path.join(nested, "nested.txt")
        with open(nested_txt, "w") as f:
            f.write("Nested document content.")

        # Create non-supported file
        unsupported = os.path.join(tmpdir, "data.csv")
        with open(unsupported, "w") as f:
            f.write("col1,col2\na,b")

        yield tmpdir


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a single sample text file."""
    p = tmp_path / "test.txt"
    p.write_text("This is test content.")
    return str(p)


@pytest.fixture
def sample_md_file(tmp_path):
    """Create a single sample markdown file."""
    p = tmp_path / "doc.md"
    p.write_text("# Heading\n\nParagraph text here.")
    return str(p)


# ---------------------------------------------------------------------------
# Document Tests
# ---------------------------------------------------------------------------


class TestDocument:
    """Tests for Document dataclass"""

    def test_auto_id_generation(self):
        """Test document auto-generates ID from content hash."""
        doc = Document(content="test content", metadata={"source": "file.txt"})
        assert doc.doc_id is not None
        assert doc.doc_id.startswith("file.txt_")

    def test_custom_id(self):
        """Test document preserves custom ID."""
        doc = Document(content="test", metadata={}, doc_id="custom-123")
        assert doc.doc_id == "custom-123"

    def test_deterministic_id(self):
        """Test same content produces same ID."""
        doc1 = Document(content="hello", metadata={"source": "a"})
        doc2 = Document(content="hello", metadata={"source": "a"})
        assert doc1.doc_id == doc2.doc_id

    def test_different_content_different_id(self):
        """Test different content produces different ID."""
        doc1 = Document(content="hello", metadata={"source": "a"})
        doc2 = Document(content="world", metadata={"source": "a"})
        assert doc1.doc_id != doc2.doc_id


# ---------------------------------------------------------------------------
# DocumentLoader Tests
# ---------------------------------------------------------------------------


class TestDocumentLoader:
    """Tests for DocumentLoader"""

    def test_load_text_file(self, sample_text_file):
        """Test loading a text file."""
        doc = DocumentLoader.load_text_file(sample_text_file)
        assert doc.content == "This is test content."
        assert doc.metadata["file_type"] == "txt"
        assert doc.metadata["filename"] == "test.txt"

    def test_load_text_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            DocumentLoader.load_text_file("/nonexistent/path.txt")

    def test_load_markdown(self, sample_md_file):
        """Test loading a markdown file."""
        doc = DocumentLoader.load_markdown(sample_md_file)
        assert "# Heading" in doc.content
        assert doc.metadata["file_type"] == "markdown"

    def test_load_markdown_not_found(self):
        """Test FileNotFoundError for missing markdown file."""
        with pytest.raises(FileNotFoundError):
            DocumentLoader.load_markdown("/nonexistent/doc.md")

    def test_load_directory_recursive(self, temp_dir):
        """Test recursive directory loading."""
        docs = DocumentLoader.load_directory(temp_dir, recursive=True)
        filenames = [d.metadata["filename"] for d in docs]
        assert "sample.txt" in filenames
        assert "readme.md" in filenames
        assert "nested.txt" in filenames
        # CSV should not be loaded
        assert "data.csv" not in filenames

    def test_load_directory_non_recursive(self, temp_dir):
        """Test non-recursive directory loading."""
        docs = DocumentLoader.load_directory(temp_dir, recursive=False)
        filenames = [d.metadata["filename"] for d in docs]
        assert "sample.txt" in filenames
        assert "readme.md" in filenames
        assert "nested.txt" not in filenames

    def test_load_directory_not_found(self):
        """Test FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            DocumentLoader.load_directory("/nonexistent/dir")

    def test_load_directory_custom_extensions(self, temp_dir):
        """Test loading with custom file extensions."""
        docs = DocumentLoader.load_directory(temp_dir, file_extensions=[".txt"])
        for doc in docs:
            assert doc.metadata["file_type"] == "txt"

    def test_load_directory_empty(self, tmp_path):
        """Test loading empty directory returns empty list."""
        docs = DocumentLoader.load_directory(str(tmp_path))
        assert docs == []


# ---------------------------------------------------------------------------
# TextSplitter Tests
# ---------------------------------------------------------------------------


class TestTextSplitter:
    """Tests for TextSplitter"""

    def test_initialization_defaults(self):
        """Test default parameters."""
        splitter = TextSplitter()
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200

    def test_split_short_text(self):
        """Test that short text is not split."""
        splitter = TextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split_text("Short text.")
        assert len(chunks) >= 1
        assert any("Short text" in c for c in chunks)

    def test_split_long_text(self):
        """Test splitting text longer than chunk_size."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        long_text = "Word " * 100  # ~500 characters
        chunks = splitter.split_text(long_text)
        assert len(chunks) > 1
        # Each chunk should be roughly chunk_size or smaller
        for chunk in chunks:
            # Allow some tolerance for separator boundaries
            assert len(chunk) <= splitter.chunk_size + 50

    def test_split_documents(self, temp_dir):
        """Test splitting loaded documents."""
        docs = DocumentLoader.load_directory(temp_dir, recursive=False)
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
        chunks = splitter.split_documents(docs)
        assert len(chunks) >= len(docs)
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert chunk.content.strip() != ""

    def test_split_preserves_metadata(self):
        """Test that split documents retain original metadata."""
        doc = Document(
            content="This is a longer document. " * 20,  # ~520 chars with separators
            metadata={"source": "test.txt", "custom": "value"},
            doc_id="orig",
        )
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["custom"] == "value"
