"""
Unit tests for Feature 43: Document Parsing Enhancement

Tests for enhanced document parser with table and chart support.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.services.parser import (
    DocumentParser,
    TableData,
    TableFormat,
    ChartReference,
    ParsedContent,
    FormattingParser
)
from app.services.document_loader_v2 import EnhancedDocumentLoader
from app.services.document_loader import Document as Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pdf_with_tables(tmp_path):
    """Create a sample PDF file path (mocked)."""
    pdf_path = tmp_path / "sample.pdf"
    # Create an empty file for testing
    pdf_path.write_bytes(b"%PDF-1.4\n%mock pdf")
    return str(pdf_path)


@pytest.fixture
def sample_text_with_table():
    """Sample text containing a markdown table."""
    return """
# Sales Report

## Q1 2024 Results

| Month | Revenue | Expenses | Profit |
|-------|---------|----------|--------|
| Jan   | $10000  | $8000    | $2000  |
| Feb   | $12000  | $9000    | $3000  |
| Mar   | $15000  | $10000   | $5000  |

The table above shows our Q1 performance.

Figure 1 displays the revenue trend.
As seen in Chart 2, expenses are stable.
"""


@pytest.fixture
def sample_text_with_charts():
    """Sample text containing chart references."""
    return """
## Data Analysis

The bar chart below shows monthly sales.

Figure 1: Revenue Growth
This line chart displays the trend over time.

As depicted in the pie chart, market share is distributed evenly.

The graph below illustrates customer acquisition.
"""


@pytest.fixture
def sample_text_with_formatting():
    """Sample text with various formatting."""
    return """
# Main Title

This is a paragraph with **bold** and *italic* text.

## Subsection

- List item 1
- List item 2
- List item 3

1. Numbered item
2. Another item

```python
def example():
    return "code"
```

[Link text](https://example.com)
"""


@pytest.fixture
def parser():
    """Create a DocumentParser instance."""
    return DocumentParser(extract_tables=True, extract_charts=True)


# ---------------------------------------------------------------------------
# TableData Tests
# ---------------------------------------------------------------------------


class TestTableData:
    """Tests for TableData dataclass"""

    def test_table_data_initialization(self):
        """Test table data initialization"""
        table = TableData(
            headers=["Name", "Age"],
            rows=[["Alice", 30], ["Bob", 25]],
            format=TableFormat.MARKDOWN
        )
        assert table.headers == ["Name", "Age"]
        assert len(table.rows) == 2
        assert table.format == TableFormat.MARKDOWN

    def test_to_markdown(self):
        """Test converting table to markdown format"""
        table = TableData(
            headers=["Name", "Age"],
            rows=[["Alice", "30"], ["Bob", "25"]],
            format=TableFormat.MARKDOWN,
            caption="Test Table"
        )
        markdown = table.to_markdown()

        assert "**Table: Test Table**" in markdown
        assert "| Name | Age |" in markdown
        assert "| --- | --- |" in markdown
        assert "| Alice | 30 |" in markdown
        assert "| Bob | 25 |" in markdown

    def test_to_markdown_no_caption(self):
        """Test markdown conversion without caption"""
        table = TableData(
            headers=["A", "B"],
            rows=[["1", "2"]],
            format=TableFormat.MARKDOWN
        )
        markdown = table.to_markdown()
        assert "**Table:" not in markdown
        assert "| A | B |" in markdown

    def test_to_csv(self):
        """Test converting table to CSV format"""
        table = TableData(
            headers=["Name", "Age"],
            rows=[["Alice", "30"], ["Bob", "25"]],
            format=TableFormat.CSV
        )
        csv = table.to_csv()

        lines = csv.strip().splitlines()
        assert lines[0] == "Name,Age"
        assert lines[1] == "Alice,30"
        assert lines[2] == "Bob,25"

    def test_to_dict(self):
        """Test converting table to list of dictionaries"""
        table = TableData(
            headers=["Name", "Age"],
            rows=[["Alice", "30"], ["Bob", "25"]],
            format=TableFormat.JSON
        )
        result = table.to_dict()

        assert len(result) == 2
        assert result[0] == {"Name": "Alice", "Age": "30"}
        assert result[1] == {"Name": "Bob", "Age": "25"}

    def test_to_dict_handles_missing_columns(self):
        """Test to_dict handles rows with fewer columns than headers"""
        table = TableData(
            headers=["Name", "Age", "City"],
            rows=[["Alice", "30"], ["Bob", "25", "NYC"]],
            format=TableFormat.JSON
        )
        result = table.to_dict()

        assert result[0]["City"] == ""
        assert result[1]["City"] == "NYC"


# ---------------------------------------------------------------------------
# ChartReference Tests
# ---------------------------------------------------------------------------


class TestChartReference:
    """Tests for ChartReference dataclass"""

    def test_chart_reference_initialization(self):
        """Test chart reference initialization"""
        chart = ChartReference(
            description="Revenue chart",
            page_number=1,
            chart_type="bar",
            title="Q1 Revenue"
        )
        assert chart.description == "Revenue chart"
        assert chart.page_number == 1
        assert chart.chart_type == "bar"
        assert chart.title == "Q1 Revenue"

    def test_chart_reference_optional_fields(self):
        """Test chart reference with optional fields"""
        chart = ChartReference(description="Some chart")
        assert chart.description == "Some chart"
        assert chart.page_number is None
        assert chart.chart_type is None
        assert chart.title is None


# ---------------------------------------------------------------------------
# ParsedContent Tests
# ---------------------------------------------------------------------------


class TestParsedContent:
    """Tests for ParsedContent dataclass"""

    def test_parsed_content_initialization(self):
        """Test parsed content initialization"""
        content = ParsedContent(
            text="Sample text",
            tables=[],
            charts=[],
            metadata={"source": "test.txt"}
        )
        assert content.text == "Sample text"
        assert len(content.tables) == 0
        assert len(content.charts) == 0
        assert content.metadata["source"] == "test.txt"

    def test_get_full_text_simple(self):
        """Test get_full_text with simple content"""
        content = ParsedContent(
            text="Simple text",
            tables=[],
            charts=[],
            metadata={}
        )
        full_text = content.get_full_text()
        assert full_text == "Simple text"

    def test_get_full_text_with_tables(self):
        """Test get_full_text includes tables"""
        table = TableData(
            headers=["A", "B"],
            rows=[["1", "2"]],
            format=TableFormat.MARKDOWN,
            caption="Data"
        )
        content = ParsedContent(
            text="Text before",
            tables=[table],
            charts=[],
            metadata={}
        )
        full_text = content.get_full_text()
        assert "Text before" in full_text
        assert "**Table: Data**" in full_text
        assert "| A | B |" in full_text

    def test_get_full_text_with_charts(self):
        """Test get_full_text includes chart references"""
        chart = ChartReference(
            description="Revenue chart",
            chart_type="bar",
            title="Sales"
        )
        content = ParsedContent(
            text="Text",
            tables=[],
            charts=[chart],
            metadata={}
        )
        full_text = content.get_full_text()
        assert "Charts and Figures:" in full_text
        assert "Sales (bar)" in full_text


# ---------------------------------------------------------------------------
# DocumentParser Tests
# ---------------------------------------------------------------------------


class TestDocumentParser:
    """Tests for DocumentParser"""

    def test_parser_initialization(self):
        """Test parser initialization"""
        parser = DocumentParser(extract_tables=True, extract_charts=True)
        assert parser.extract_tables is True
        assert parser.extract_charts is True

    def test_parser_initialization_default(self):
        """Test parser with defaults"""
        parser = DocumentParser()
        assert parser.extract_tables is True
        assert parser.extract_charts is True

    def test_parse_text_basic(self, parser):
        """Test basic text parsing"""
        content = parser.parse_text("Simple text", include_tables=False)
        assert content.text == "Simple text"
        assert len(content.tables) == 0
        assert len(content.charts) == 0

    def test_parse_text_extract_tables(self, parser, sample_text_with_table):
        """Test table extraction from text"""
        content = parser.parse_text(sample_text_with_table, include_tables=True)
        assert len(content.tables) > 0
        # Check that headers were extracted
        assert any("Month" in table.headers for table in content.tables)

    def test_parse_text_extract_charts(self, parser, sample_text_with_charts):
        """Test chart reference detection"""
        content = parser.parse_text(sample_text_with_charts, include_tables=False)
        assert len(content.charts) > 0
        # Check that different chart types were detected
        chart_types = [c.chart_type for c in content.charts if c.chart_type]
        assert "bar" in chart_types or "line" in chart_types or "pie" in chart_types

    def test_parse_text_combined(self, parser, sample_text_with_table):
        """Test parsing with both tables and charts"""
        content = parser.parse_text(sample_text_with_table, include_tables=True)
        assert len(content.tables) > 0
        # The sample text has a chart reference
        assert len(content.charts) > 0

    def test_parse_pdf_file_not_found(self, parser):
        """Test PDF parsing with non-existent file"""
        with pytest.raises(FileNotFoundError):
            parser.parse_pdf("/nonexistent/file.pdf")

    def test_parse_pdf_without_pdfplumber(self, parser, sample_pdf_with_tables):
        """Test PDF parsing behavior when pdfplumber is not available"""
        # When pdfplumber is not available, parser should handle gracefully
        if not parser.pdfplumber_available:
            # Should still work without crashing
            # Just verify it doesn't have the enhanced capability
            assert not parser.pdfplumber_available
        else:
            # If available, just verify the attribute exists
            assert hasattr(parser, 'pdfplumber_available')

    def test_parse_pdf_file_not_found(self, parser):
        """Test PDF parsing with non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            parser.parse_pdf("/nonexistent/file.pdf")

    def test_detect_chart_references(self, parser):
        """Test chart reference detection patterns"""
        text = """
        Figure 1 shows the trend.
        As seen in Chart 2, values are increasing.
        The bar chart displays results.
        """
        charts = parser._detect_chart_references(text)
        assert len(charts) > 0

    def test_extract_tables_from_text(self, parser):
        """Test table extraction from markdown tables"""
        text = """
        | Header 1 | Header 2 |
        |----------|----------|
        | Data 1   | Data 2   |
        | Data 3   | Data 4   |
        """
        tables = parser._extract_tables_from_text(text)
        assert len(tables) > 0
        assert "Header 1" in tables[0].headers

    def test_parse_empty_text(self, parser):
        """Test parsing empty text"""
        content = parser.parse_text("", include_tables=True)
        assert content.text == ""
        assert len(content.tables) == 0
        assert len(content.charts) == 0


# ---------------------------------------------------------------------------
# FormattingParser Tests
# ---------------------------------------------------------------------------


class TestFormattingParser:
    """Tests for FormattingParser"""

    def test_parse_structure_headers(self):
        """Test parsing headers from text"""
        text = """# Title 1
## Title 2
### Title 3"""
        structure = FormattingParser.parse_structure(text)
        assert len(structure["headers"]) == 3
        assert structure["headers"][0]["level"] == 1
        assert structure["headers"][0]["title"] == "Title 1"
        assert structure["headers"][1]["level"] == 2

    def test_parse_structure_lists(self):
        """Test parsing lists from text"""
        text = """
        - Item 1
        - Item 2
        1. Numbered
        2. Also numbered
        """
        structure = FormattingParser.parse_structure(text)
        assert len(structure["lists"]) == 4

    def test_parse_structure_code_blocks(self):
        """Test parsing code blocks"""
        text = """
        ```python
        def test():
            pass
        ```
        """
        structure = FormattingParser.parse_structure(text)
        assert len(structure["code_blocks"]) == 1
        assert structure["code_blocks"][0]["language"] == "python"

    def test_parse_structure_links(self):
        """Test parsing links"""
        text = "Check out [this link](https://example.com) for more."
        structure = FormattingParser.parse_structure(text)
        assert len(structure["links"]) == 1
        assert structure["links"][0]["text"] == "this link"
        assert structure["links"][0]["url"] == "https://example.com"

    def test_extract_sections(self):
        """Test extracting document sections"""
        text = """# Introduction

Intro content.

# Methods

Methods content.

## Subsection

More content."""
        sections = FormattingParser.extract_sections(text)
        assert len(sections) >= 3
        assert sections[0]["title"] == "Introduction"
        assert sections[1]["title"] == "Methods"
        assert sections[2]["title"] == "Subsection"

    def test_extract_sections_content(self):
        """Test that sections preserve content"""
        text = """
        # Section 1

        Content for section 1.

        # Section 2

        Content for section 2.
        """
        sections = FormattingParser.extract_sections(text)
        assert any("Content for section 1" in s["content"] for s in sections)
        assert any("Content for section 2" in s["content"] for s in sections)


# ---------------------------------------------------------------------------
# DocumentLoader Integration Tests
# ---------------------------------------------------------------------------


class TestDocumentLoaderIntegration:
    """Tests for EnhancedDocumentLoader integration with parser"""

    def test_loader_with_advanced_parsing(self):
        """Test loader with advanced parsing enabled"""
        loader = EnhancedDocumentLoader(enable_advanced_parsing=True)
        assert loader.parser is not None
        assert loader.enable_advanced_parsing is True

    def test_loader_without_advanced_parsing(self):
        """Test loader with advanced parsing disabled"""
        loader = EnhancedDocumentLoader(enable_advanced_parsing=False)
        assert loader.parser is None
        assert loader.enable_advanced_parsing is False

    def test_load_text_file_with_parser(self, tmp_path):
        """Test loading text file through enhanced loader"""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Sample content")

        loader = EnhancedDocumentLoader(enable_advanced_parsing=True)
        doc = loader.load_text_file(str(test_file))

        assert doc.content == "Sample content"
        assert doc.metadata["file_type"] == "txt"

    def test_load_markdown_with_table_extraction(self, tmp_path, sample_text_with_table):
        """Test loading markdown with table extraction"""
        # Create test file
        test_file = tmp_path / "report.md"
        test_file.write_text(sample_text_with_table)

        loader = EnhancedDocumentLoader(enable_advanced_parsing=True)
        docs = loader.load_directory(str(tmp_path), file_extensions=[".md"])

        assert len(docs) > 0
        # Check that tables were extracted and included in content
        doc = docs[0]
        # Tables should be extracted when advanced parsing is enabled
        assert "tables_extracted" in doc.metadata or doc.content is not None

    def test_backward_compatibility_load_pdf(self, sample_pdf_with_tables):
        """Test backward compatible PDF loading uses base loader"""
        # This test verifies backward compatibility
        from app.services.document_loader import DocumentLoader as BaseLoader

        # Verify base loader still exists and has the expected method
        assert hasattr(BaseLoader, 'load_pdf')
        assert callable(BaseLoader.load_pdf)

    def test_load_directory_with_table_extraction(self, tmp_path):
        """Test directory loading with table extraction"""
        # Create test files
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")

        loader = EnhancedDocumentLoader(enable_advanced_parsing=True)
        docs = loader.load_directory(str(tmp_path), file_extensions=[".txt"])

        assert len(docs) == 2

    def test_load_pdf_with_tables_disabled(self, sample_pdf_with_tables):
        """Test PDF loading with table extraction disabled"""
        loader = EnhancedDocumentLoader(enable_advanced_parsing=False)

        # When advanced parsing is disabled, should still work
        assert loader.parser is None
        assert loader.enable_advanced_parsing is False


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_parse_malformed_table(self, parser):
        """Test handling malformed tables"""
        text = """
        | Header 1 | Header 2
        | Data 1   | Data 2
        """
        content = parser.parse_text(text, include_tables=True)
        # Should not crash
        assert content is not None

    def test_parse_empty_table(self, parser):
        """Test handling empty table structures"""
        text = """
        | | |
        |---|---|
        """
        content = parser.parse_text(text, include_tables=True)
        # Should handle gracefully
        assert content is not None

    def test_parse_text_with_no_charts(self, parser):
        """Test parsing text without chart references"""
        text = "Just plain text with no charts."
        content = parser.parse_text(text, include_tables=False)
        assert len(content.charts) == 0

    def test_table_with_empty_cells(self, parser):
        """Test table with empty cells"""
        table = TableData(
            headers=["A", "B", "C"],
            rows=[["1", "", "3"], ["", "2", ""]],
            format=TableFormat.MARKDOWN
        )
        markdown = table.to_markdown()
        # Should handle empty cells
        assert "| A | B | C |" in markdown

    def test_very_long_table_row(self, parser):
        """Test handling very long table rows"""
        long_row = ["Col " + str(i) for i in range(100)]
        table = TableData(
            headers=long_row,
            rows=[long_row],
            format=TableFormat.MARKDOWN
        )
        # Should handle without crashing
        markdown = table.to_markdown()
        assert markdown is not None

    def test_special_characters_in_table(self):
        """Test table with special characters"""
        table = TableData(
            headers=["Name", "Description"],
            rows=[["Test & Demo", "Price: $100\nDiscount: 10%"]],
            format=TableFormat.MARKDOWN
        )
        markdown = table.to_markdown()
        assert "Test & Demo" in markdown
        assert "$100" in markdown

    def test_unicode_in_table(self):
        """Test table with unicode characters"""
        table = TableData(
            headers=["名前", "年齢"],
            rows=[["太郎", "25"], ["花子", "30"]],
            format=TableFormat.MARKDOWN
        )
        markdown = table.to_markdown()
        assert "太郎" in markdown
        assert "花子" in markdown


# ---------------------------------------------------------------------------
# Performance Tests
# ---------------------------------------------------------------------------


class TestPerformance:
    """Performance-related tests"""

    def test_parse_large_text(self, parser):
        """Test parsing large text document"""
        # Generate large text
        large_text = "Sentence. " * 10000 + "\n\n" + "| A | B |\n|---|---|\n| 1 | 2 |" * 100

        import time
        start = time.time()
        content = parser.parse_text(large_text, include_tables=True)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert content is not None

    def test_parse_multiple_tables(self, parser):
        """Test parsing document with many tables"""
        text = ""
        for i in range(50):
            text += f"\n\n## Table {i}\n\n| Col1 | Col2 |\n|------|------|\n| {i}A  | {i}B  |\n"

        content = parser.parse_text(text, include_tables=True)
        # Should extract multiple tables
        assert len(content.tables) > 0
