"""
Enhanced Document Parser with Table and Chart Support

This module provides advanced parsing capabilities for enterprise documents,
including table extraction, chart detection, and formatting preservation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path

from app.core.logging_config import get_logger


logger = get_logger(__name__)


class TableFormat(Enum):
    """Supported table formats"""
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"
    HTML = "html"


@dataclass
class TableData:
    """Extracted table data"""
    headers: List[str]
    rows: List[List[str]]
    format: TableFormat
    page_number: Optional[int] = None
    caption: Optional[str] = None

    def to_markdown(self) -> str:
        """Convert table to markdown format"""
        if not self.headers or not self.rows:
            return ""

        lines = []
        if self.caption:
            lines.append(f"**Table: {self.caption}**\n")

        # Header row
        lines.append("| " + " | ".join(self.headers) + " |")
        # Separator row
        lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        # Data rows
        for row in self.rows:
            # Ensure row has same number of columns as headers
            padded_row = row + [""] * (len(self.headers) - len(row))
            lines.append("| " + " | ".join(str(cell) for cell in padded_row) + " |")

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Convert table to CSV format"""
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        if self.headers:
            writer.writerow(self.headers)
        for row in self.rows:
            writer.writerow(row)

        return output.getvalue()

    def to_dict(self) -> List[Dict[str, str]]:
        """Convert table to list of dictionaries"""
        result = []
        for row in self.rows:
            if self.headers:
                row_dict = {
                    self.headers[i]: row[i] if i < len(row) else ""
                    for i in range(len(self.headers))
                }
                result.append(row_dict)
            else:
                result.append({"col_" + str(i): cell for i, cell in enumerate(row)})
        return result


@dataclass
class ChartReference:
    """Reference to a detected chart/image"""
    description: str
    page_number: Optional[int] = None
    chart_type: Optional[str] = None  # bar, line, pie, etc.
    title: Optional[str] = None


@dataclass
class ParsedContent:
    """Parsed document content with structured elements"""
    text: str
    tables: List[TableData]
    charts: List[ChartReference]
    metadata: Dict[str, Any]

    def get_full_text(self) -> str:
        """Get full text including tables in markdown format"""
        parts = [self.text]

        # Add tables
        for i, table in enumerate(self.tables):
            if table.caption:
                parts.append(f"\n\n{table.to_markdown()}\n")
            else:
                parts.append(f"\n\nTable {i + 1}:\n{table.to_markdown()}\n")

        # Add chart references
        if self.charts:
            parts.append("\n\nCharts and Figures:\n")
            for chart in self.charts:
                if chart.title:
                    parts.append(f"- {chart.title} ({chart.chart_type})")
                else:
                    parts.append(f"- {chart.description}")

        return "\n".join(parts)


class DocumentParser:
    """
    Enhanced document parser with table and chart support.

    This parser can extract tables from PDF and text documents,
    detect chart references, and preserve formatting.
    """

    def __init__(self, extract_tables: bool = True, extract_charts: bool = True):
        """
        Initialize parser.

        Args:
            extract_tables: Whether to extract tables from documents
            extract_charts: Whether to detect chart references
        """
        self.extract_tables = extract_tables
        self.extract_charts = extract_charts

        # Try to import pdfplumber for advanced PDF parsing
        self.pdfplumber_available = False
        try:
            import pdfplumber  # noqa: F401 (probe import: tests availability)
            self.pdfplumber_available = True
            logger.info("pdfplumber is available for enhanced PDF parsing")
        except ImportError:
            logger.warning(
                "pdfplumber not installed. Table extraction from PDFs will be limited. "
                "Install with: pip install pdfplumber"
            )

    def parse_pdf(
        self,
        file_path: str,
        include_tables: bool = True,
        table_format: TableFormat = TableFormat.MARKDOWN
    ) -> ParsedContent:
        """
        Parse PDF file with enhanced features.

        Args:
            file_path: Path to PDF file
            include_tables: Whether to extract and include tables
            table_format: Format for table output

        Returns:
            ParsedContent with text, tables, and charts
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try pdfplumber first for better table extraction
        if self.pdfplumber_available:
            return self._parse_pdf_with_pdfplumber(path, include_tables, table_format)
        else:
            # Fall back to pypdf
            return self._parse_pdf_with_pypdf(path, include_tables, table_format)

    def _parse_pdf_with_pdfplumber(
        self,
        path: Path,
        include_tables: bool,
        table_format: TableFormat
    ) -> ParsedContent:
        """Parse PDF using pdfplumber for better table extraction."""
        import pdfplumber

        all_text = []
        all_tables = []
        all_charts = []

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text() or ""
                all_text.append(f"--- Page {page_num} ---\n{text}")

                # Extract tables if requested
                if include_tables and self.extract_tables:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            # Filter out empty tables
                            if table and any(any(cell for cell in row if cell) for row in table):
                                # Convert to TableData
                                headers = [str(cell) if cell else "" for cell in table[0]]
                                rows = [
                                    [str(cell) if cell else "" for cell in row]
                                    for row in table[1:]
                                ]
                                all_tables.append(
                                    TableData(
                                        headers=headers,
                                        rows=rows,
                                        format=table_format,
                                        page_number=page_num
                                    )
                                )

                # Detect charts/images
                if self.extract_charts:
                    images = page.images
                    for image in images:
                        # Try to extract image metadata
                        chart_type = self._guess_chart_type(image)
                        all_charts.append(
                            ChartReference(
                                description=f"Image on page {page_num}",
                                page_number=page_num,
                                chart_type=chart_type
                            )
                        )

        return ParsedContent(
            text="\n\n".join(all_text),
            tables=all_tables,
            charts=all_charts,
            metadata={
                "source": str(path),
                "filename": path.name,
                "total_tables": len(all_tables),
                "total_charts": len(all_charts)
            }
        )

    def _parse_pdf_with_pypdf(
        self,
        path: Path,
        include_tables: bool,
        table_format: TableFormat
    ) -> ParsedContent:
        """Parse PDF using pypdf (fallback method)."""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        all_text = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            all_text.append(f"--- Page {page_num} ---\n{text}")

        # With pypdf, we can only do basic table detection via regex
        tables = []
        if include_tables and self.extract_tables:
            full_text = "\n".join(all_text)
            tables = self._extract_tables_from_text(full_text)

        return ParsedContent(
            text="\n\n".join(all_text),
            tables=tables,
            charts=[],
            metadata={
                "source": str(path),
                "filename": path.name,
                "total_pages": len(reader.pages),
                "total_tables": len(tables)
            }
        )

    def parse_text(
        self,
        text: str,
        include_tables: bool = True
    ) -> ParsedContent:
        """
        Parse text content and extract structured elements.

        Args:
            text: Text content to parse
            include_tables: Whether to extract tables

        Returns:
            ParsedContent with extracted elements
        """
        tables = []
        charts = []

        if include_tables and self.extract_tables:
            tables = self._extract_tables_from_text(text)

        if self.extract_charts:
            charts = self._detect_chart_references(text)

        return ParsedContent(
            text=text,
            tables=tables,
            charts=charts,
            metadata={
                "total_tables": len(tables),
                "total_charts": len(charts)
            }
        )

    def _extract_tables_from_text(self, text: str) -> List[TableData]:
        """
        Extract tables from plain text using heuristics.

        This method detects table-like structures in text by looking for:
        - Multiple lines with similar patterns
        - Separator lines (dashes, equals)
        - Consistent column alignment
        """
        tables = []
        lines = text.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Detect potential table by separator lines
            if re.match(r"^[\s\-\+|]+\s*$", line) and i > 0:
                # Look backwards for header
                header_line = lines[i - 1].strip()
                if self._looks_like_table_row(header_line):
                    # Look forwards for data rows
                    rows = []
                    j = i + 1
                    while j < len(lines) and self._looks_like_table_row(lines[j].strip()):
                        rows.append(self._parse_table_row(lines[j].strip()))
                        j += 1

                    if rows:
                        headers = self._parse_table_row(header_line)
                        tables.append(
                            TableData(
                                headers=headers,
                                rows=rows,
                                format=TableFormat.MARKDOWN
                            )
                        )
                        i = j - 1  # Skip processed rows
            i += 1

        return tables

    def _looks_like_table_row(self, line: str) -> bool:
        """Check if a line looks like a table row."""
        # Has separators or multiple columns
        return "|" in line or "\t" in line or ("  " in line and len(line.split()) >= 3)

    @staticmethod
    def _strip_boundary_empties(cells: List[str]) -> List[str]:
        """Drop empty cells from the start/end of a row only.

        Boundary empties arise from leading/trailing delimiters -- a markdown
        ``| a |`` row, an indented TSV ``\\ta\\tb\\t``, or a space-aligned
        ``"  a  b"`` -- and would otherwise create phantom cells that shift
        every later column under the wrong header (data corruption on the
        ingestion path). Empty cells in the MIDDLE are preserved: they carry
        column-alignment meaning for sparse rows (the old ``cells.index(c)``
        filter dropped those too).
        """
        while cells and cells[0] == "":
            cells.pop(0)
        while cells and cells[-1] == "":
            cells.pop()
        return cells

    def _parse_table_row(self, line: str) -> List[str]:
        """Parse a table row into cells."""
        # Every delimiter branch strips boundary empties so leading/trailing
        # delimiters don't create phantom cells; middle empties are preserved.
        if "|" in line:
            # Markdown table
            cells = [cell.strip() for cell in line.split("|")]
            return self._strip_boundary_empties(cells)
        elif "\t" in line:
            # TSV
            cells = [cell.strip() for cell in line.split("\t")]
            return self._strip_boundary_empties(cells)
        else:
            # Try multiple spaces
            cells = [cell.strip() for cell in re.split(r"\s{2,}", line)]
            return self._strip_boundary_empties(cells)

    def _detect_chart_references(self, text: str) -> List[ChartReference]:
        """
        Detect references to charts and figures in text.

        Looks for patterns like:
        - "Figure 1 shows..."
        - "As seen in Chart 2..."
        - "The graph below displays..."
        """
        charts = []
        lines = text.split("\n")

        chart_patterns = [
            r"(?:figure|chart|graph|plot)\s+(\d+)",
            r"(?:bar|line|pie|scatter)\s+(?:chart|graph|plot)",
            r"as\s+(?:shown|displayed|depicted)\s+in\s+(?:figure|chart)",
        ]

        for line_num, line in enumerate(lines):
            line_lower = line.lower()
            for pattern in chart_patterns:
                if re.search(pattern, line_lower):
                    # Determine chart type
                    chart_type = None
                    if "bar" in line_lower:
                        chart_type = "bar"
                    elif "line" in line_lower:
                        chart_type = "line"
                    elif "pie" in line_lower:
                        chart_type = "pie"

                    charts.append(
                        ChartReference(
                            description=line.strip(),
                            chart_type=chart_type
                        )
                    )
                    break

        return charts

    def _guess_chart_type(self, image_obj: Any) -> Optional[str]:
        """
        Guess chart type from image properties.

        This is a heuristic - actual analysis would require image processing.
        """
        # Check image dimensions as a hint
        if hasattr(image_obj, "width") and hasattr(image_obj, "height"):
            width = image_obj.width
            height = image_obj.height
            aspect_ratio = width / height if height > 0 else 1

            # Wide images might be bar charts or line charts
            if aspect_ratio > 1.5:
                return "bar"
            # Square images might be pie charts
            elif 0.8 < aspect_ratio < 1.2:
                return "pie"

        return "unknown"


class FormattingParser:
    """
    Parser for preserving document formatting.

    Handles:
    - Headers (markdown-style #, ##, ###)
    - Lists (bullet and numbered)
    - Code blocks
    - Bold/italic emphasis
    """

    @staticmethod
    def parse_structure(text: str) -> Dict[str, Any]:
        """
        Parse document structure and return metadata.

        Returns:
            Dict with headers, lists, code_blocks, etc.
        """
        structure = {
            "headers": [],
            "lists": [],
            "code_blocks": [],
            "links": []
        }

        lines = text.split("\n")
        in_code_block = False
        code_start = 0

        for i, line in enumerate(lines):
            # Headers
            if line.startswith("#"):
                level = len(re.match(r"^#+", line).group())
                title = line.lstrip("#").strip()
                structure["headers"].append({
                    "level": level,
                    "title": title,
                    "line": i
                })

            # Code blocks
            if line.strip().startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    code_start = i
                else:
                    structure["code_blocks"].append({
                        "start": code_start,
                        "end": i,
                        "language": lines[code_start].strip()[3:].strip()
                    })
                    in_code_block = False

            # Lists
            if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
                structure["lists"].append({
                    "line": i,
                    "content": line.strip()
                })

            # Links
            links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", line)
            for link_text, link_url in links:
                structure["links"].append({
                    "text": link_text,
                    "url": link_url,
                    "line": i
                })

        return structure

    @staticmethod
    def extract_sections(text: str) -> List[Dict[str, Any]]:
        """
        Extract document sections based on headers.

        Returns:
            List of sections with title and content
        """
        sections = []
        lines = text.split("\n")

        current_section = {"title": "Introduction", "content": [], "level": 0}

        for line in lines:
            if line.startswith("#"):
                # Save previous section
                if current_section["content"]:
                    sections.append({
                        **current_section,
                        "content": "\n".join(current_section["content"]).strip()
                    })

                # Start new section
                level = len(re.match(r"^#+", line).group())
                current_section = {
                    "title": line.lstrip("#").strip(),
                    "content": [],
                    "level": level
                }
            else:
                current_section["content"].append(line)

        # Add last section
        if current_section["content"]:
            sections.append({
                **current_section,
                "content": "\n".join(current_section["content"]).strip()
            })

        return sections
