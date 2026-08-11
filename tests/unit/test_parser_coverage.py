"""Coverage-focused tests for app/services/parser.py.

Targets previously-uncovered branches: TableData.to_markdown/to_dict
edge cases, _parse_table_row TSV/multi-space branches,
_guess_chart_type aspect-ratio branches, and the two PDF parsing
methods (_parse_pdf_with_pdfplumber, _parse_pdf_with_pypdf) plus the
parse_pdf dispatch + __init__ pdfplumber-available branch.

Neither pdfplumber nor pypdf is installed in this environment, so fake
modules are injected via monkeypatch. There is no existing test_parser
file; these paths are 0%-covered (current parser coverage comes
indirectly from document_loader tests exercising parse_text only).
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from app.services.parser import DocumentParser, TableData, TableFormat


# --------------------------------------------------------------------------
# TableData branches (L40, L85)
# --------------------------------------------------------------------------


class TestTableDataBranches:
    def test_to_markdown_empty_when_no_headers_or_rows(self):
        assert TableData([], [], TableFormat.MARKDOWN).to_markdown() == ""
        # Headers present but no rows is also empty.
        assert (
            TableData(["a"], [], TableFormat.MARKDOWN).to_markdown() == ""
        )

    def test_to_dict_without_headers_uses_indexed_keys(self):
        table = TableData([], [["a", "b"]], TableFormat.MARKDOWN)
        assert table.to_dict() == [{"col_0": "a", "col_1": "b"}]


# --------------------------------------------------------------------------
# _parse_table_row branches (L379-385)
# --------------------------------------------------------------------------


class TestParseTableRowDelimiters:
    def test_tsv_row_splits_on_tabs(self):
        parser = DocumentParser()
        assert parser._parse_table_row("a\tb\tc") == ["a", "b", "c"]

    def test_multi_space_row_splits_on_double_plus_spaces(self):
        parser = DocumentParser()
        assert parser._parse_table_row("alpha    beta") == ["alpha", "beta"]

    def test_tsv_row_strips_boundary_empties(self):
        """Regression: TSV rows with leading/trailing tabs must not gain
        phantom boundary cells (only the ``|`` branch stripped them)."""
        parser = DocumentParser()
        # leading + trailing tab -> no phantom ['', ..., '']
        assert parser._parse_table_row("\ta\tb\t") == ["a", "b"]
        # middle empty cell preserved for column alignment
        assert parser._parse_table_row("a\t\tb") == ["a", "", "b"]

    def test_multi_space_row_strips_leading_empty(self):
        """Regression: a space-aligned row with leading indentation must not
        gain a phantom leading cell (re.split on the leading run yields one)."""
        parser = DocumentParser()
        assert parser._parse_table_row("  a  b") == ["a", "b"]
        assert parser._parse_table_row("  alpha    beta") == ["alpha", "beta"]

    def test_all_delimiter_branches_strip_boundary_consistently(self):
        """All three delimiter branches must strip boundary empties the same
        way -- the ``|`` branch had the fix, TSV/space did not (probe: sibling
        branches with inconsistent normalization)."""
        parser = DocumentParser()
        md = parser._parse_table_row("|a|b|")
        tsv = parser._parse_table_row("\ta\tb\t")
        spc = parser._parse_table_row("  a  b")
        assert md == tsv == spc == ["a", "b"]


# --------------------------------------------------------------------------
# _guess_chart_type branches (L435-447)
# --------------------------------------------------------------------------


class TestGuessChartType:
    def test_wide_image_is_bar(self):
        parser = DocumentParser()
        img = SimpleNamespace(width=300, height=100)  # ratio 3.0
        assert parser._guess_chart_type(img) == "bar"

    def test_square_image_is_pie(self):
        parser = DocumentParser()
        img = SimpleNamespace(width=100, height=100)  # ratio 1.0
        assert parser._guess_chart_type(img) == "pie"

    def test_zero_height_uses_fallback_aspect(self):
        parser = DocumentParser()
        img = SimpleNamespace(width=200, height=0)  # ratio -> 1.0
        assert parser._guess_chart_type(img) == "pie"

    def test_tall_image_is_unknown(self):
        parser = DocumentParser()
        img = SimpleNamespace(width=100, height=300)  # ratio 0.33
        assert parser._guess_chart_type(img) == "unknown"

    def test_image_without_dimensions_is_unknown(self):
        parser = DocumentParser()
        assert parser._guess_chart_type(object()) == "unknown"


# --------------------------------------------------------------------------
# PDF parsing (L152-153, L182-186, L195-243, L262-287)
# --------------------------------------------------------------------------


def _page(text=None, tables=None, images=None):
    page = Mock()
    page.extract_text.return_value = text
    page.extract_tables.return_value = tables or []
    page.images = images or []
    return page


def _install_pdfplumber(monkeypatch, pages):
    pdf = Mock()
    pdf.pages = pages
    cm = MagicMock()
    cm.__enter__.return_value = pdf
    cm.__exit__.return_value = False
    fake = Mock()
    fake.open.return_value = cm
    monkeypatch.setitem(sys.modules, "pdfplumber", fake)
    return fake


class TestParsePdfWithPdfplumber:
    def test_extracts_text_tables_and_charts(
        self, monkeypatch, tmp_path
    ):
        image = SimpleNamespace(width=300, height=100)  # -> "bar"
        # First page: text + a real table + an empty (filtered) table
        # + an image; second page: None text + no tables.
        p1 = _page(
            text="Page one text",
            tables=[
                [["h1", "h2"], ["v1", "v2"]],
                [["", None], ["", ""]],  # all-empty -> filtered out
            ],
            images=[image],
        )
        p2 = _page(text=None, tables=[], images=[])
        _install_pdfplumber(monkeypatch, [p1, p2])

        parser = DocumentParser()  # __init__ sees fake -> available
        assert parser.pdfplumber_available is True

        target = tmp_path / "doc.pdf"
        target.write_bytes(b"%PDF-1.4")

        result = parser.parse_pdf(str(target), include_tables=True)

        assert "Page one text" in result.text
        # Only the non-empty table survives the filter.
        assert len(result.tables) == 1
        assert result.tables[0].headers == ["h1", "h2"]
        assert result.tables[0].rows == [["v1", "v2"]]
        assert len(result.charts) == 1
        assert result.charts[0].chart_type == "bar"
        assert result.metadata["total_tables"] == 1
        assert result.metadata["total_charts"] == 1


class TestParsePdfWithPypdf:
    def test_falls_back_to_pypdf_when_pdfplumber_absent(
        self, monkeypatch, tmp_path
    ):
        p1 = Mock()
        p1.extract_text.return_value = "Alpha"
        p2 = Mock()
        p2.extract_text.return_value = "Beta"
        reader = Mock()
        reader.pages = [p1, p2]

        fake_pypdf = Mock()
        fake_pypdf.PdfReader = Mock(return_value=reader)
        monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

        parser = DocumentParser()  # pdfplumber not installed -> False
        assert parser.pdfplumber_available is False

        target = tmp_path / "doc.pdf"
        target.write_bytes(b"%PDF-1.4")

        result = parser.parse_pdf(str(target), include_tables=True)

        assert "Alpha" in result.text and "Beta" in result.text
        assert result.tables == []  # no table-separator pattern
        assert result.charts == []
        assert result.metadata["total_pages"] == 2
        assert result.metadata["total_tables"] == 0
