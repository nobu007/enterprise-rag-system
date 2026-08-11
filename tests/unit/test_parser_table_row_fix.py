"""
Regression tests for DocumentParser._parse_table_row ("|" branch).

The markdown ("|") branch used ``cells.index(c)`` to drop boundary empty
cells, but ``index()`` returns the *first* matching value -- so any empty
cell whose first occurrence sat at index 0 or the last index was dropped
even when it sat in the MIDDLE of a row. Sparse rows like
``| Alice | | Smith |`` parsed to ``['Alice', 'Smith']``, shifting every
later column under the wrong header (data corruption on the live ingestion
path: EnhancedDocumentLoader -> DocumentParser.parse_text / parse_pdf ->
_extract_tables_from_text -> _parse_table_row).

These tests exercise the "|" branch specifically (the TSV and multi-space
branches are covered in test_parser_coverage.py). They must pass only
after the boundary-only stripping fix.
"""

from app.services.parser import DocumentParser


def _parser() -> DocumentParser:
    return DocumentParser(extract_tables=True, extract_charts=True)


class TestParseTableRowMarkdownBranch:
    """Cover the "|" delimiter branch of _parse_table_row."""

    def test_middle_empty_cell_is_preserved(self):
        """Empty cells between values must survive (regression for the bug)."""
        # Before the fix this returned ['a', 'c'], misaligning columns.
        assert _parser()._parse_table_row("| a |   | c |") == ["a", "", "c"]

    def test_leading_and_trailing_empties_are_stripped(self):
        """split('|') boundary empties are removed, values kept."""
        assert _parser()._parse_table_row("| a | b |") == ["a", "b"]

    def test_row_without_boundary_pipes_keeps_all_cells(self):
        assert _parser()._parse_table_row("a | b | c") == ["a", "b", "c"]

    def test_multiple_middle_empties_preserved(self):
        assert _parser()._parse_table_row("| a | | b | | c |") == ["a", "", "b", "", "c"]

    def test_row_of_only_pipes_yields_no_cells(self):
        assert _parser()._parse_table_row("|") == []

    def test_sparse_table_keeps_column_alignment(self):
        """End-to-end: a sparse markdown table preserves column meaning."""
        text = (
            "| Name | Mid | Last |\n"
            "|------|-----|------|\n"
            "| Alice|     | Smith|\n"
        )
        tables = _parser()._extract_tables_from_text(text)
        assert len(tables) == 1
        assert tables[0].headers == ["Name", "Mid", "Last"]
        # "Smith" must stay under "Last", not shift into "Mid".
        assert tables[0].rows == [["Alice", "", "Smith"]]
