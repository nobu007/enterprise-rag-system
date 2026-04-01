"""
Enhanced Document Loading and Processing

This module provides enhanced document loading with table and chart extraction.
For backward compatibility, the original document_loader.py is maintained.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import hashlib

from app.core.logging_config import get_logger
from app.services.parser import DocumentParser, TableFormat, ParsedContent


logger = get_logger(__name__)


# Re-export Document from base module
from app.services.document_loader import Document


class EnhancedDocumentLoader:
    """Enhanced document loader with parsing capabilities"""

    def __init__(self, enable_advanced_parsing: bool = True):
        """
        Initialize document loader.

        Args:
            enable_advanced_parsing: Whether to use advanced parsing for tables/charts
        """
        self.enable_advanced_parsing = enable_advanced_parsing
        self.parser = DocumentParser() if enable_advanced_parsing else None

    @staticmethod
    def load_text_file(file_path: str) -> Document:
        """Load a plain text file"""
        # Import from base module for code reuse
        from app.services.document_loader import DocumentLoader
        return DocumentLoader.load_text_file(file_path)

    def load_pdf(
        self,
        file_path: str,
        extract_tables: bool = True,
        table_format: TableFormat = TableFormat.MARKDOWN
    ) -> List[Document]:
        """
        Load a PDF file with optional table extraction.

        Args:
            file_path: Path to PDF file
            extract_tables: Whether to extract tables from PDF
            table_format: Format for extracted tables

        Returns:
            List of Document objects
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Use advanced parser if available
        if self.parser and extract_tables:
            try:
                parsed = self.parser.parse_pdf(
                    file_path,
                    include_tables=extract_tables,
                    table_format=table_format
                )

                # Create document with enhanced content
                metadata = {
                    'source': str(path),
                    'filename': path.name,
                    'file_type': 'pdf',
                    'tables_extracted': len(parsed.tables),
                    'charts_detected': len(parsed.charts),
                    'parsing_method': 'enhanced'
                }

                # Use full text with tables included
                content = parsed.get_full_text()

                return [Document(content=content, metadata=metadata)]

            except Exception as e:
                logger.warning(f"Advanced parsing failed, falling back to basic: {e}")

        # Fallback to basic PDF loading
        from app.services.document_loader import DocumentLoader
        return DocumentLoader.load_pdf(file_path)

    @staticmethod
    def load_markdown(file_path: str) -> Document:
        """Load a Markdown file"""
        from app.services.document_loader import DocumentLoader
        return DocumentLoader.load_markdown(file_path)

    def load_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        extract_tables: bool = True
    ) -> List[Document]:
        """
        Load all documents from a directory with enhanced parsing.

        Args:
            directory_path: Path to directory
            file_extensions: List of file extensions to load
            recursive: Whether to search recursively
            extract_tables: Whether to extract tables from documents

        Returns:
            List of Document objects
        """
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.pdf']

        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        documents = []

        # Get all files
        if recursive:
            files = [f for f in directory.rglob('*') if f.is_file()]
        else:
            files = [f for f in directory.glob('*') if f.is_file()]

        # Filter by extension
        files = [f for f in files if f.suffix.lower() in file_extensions]

        logger.info(f"Found {len(files)} files to process")

        # Load each file
        for file_path in files:
            try:
                ext = file_path.suffix.lower()

                if ext == '.pdf':
                    docs = self.load_pdf(str(file_path), extract_tables=extract_tables)
                    documents.extend(docs)
                elif ext == '.md':
                    doc = self.load_markdown(str(file_path))
                    # Parse for tables if enabled
                    if self.parser and extract_tables:
                        parsed = self.parser.parse_text(doc.content, include_tables=True)
                        if parsed.tables:
                            doc.content = parsed.get_full_text()
                            doc.metadata['tables_extracted'] = len(parsed.tables)
                    documents.append(doc)
                elif ext == '.txt':
                    doc = self.load_text_file(str(file_path))
                    documents.append(doc)

                logger.debug(f"Loaded: {file_path.name}")

            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents


# For backward compatibility, provide the same API as the original
class DocumentLoader(EnhancedDocumentLoader):
    """Enhanced DocumentLoader with backward compatibility"""

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load a PDF file (backward compatible - no table extraction)"""
        from app.services.document_loader import DocumentLoader as BaseLoader
        return BaseLoader.load_pdf(file_path)

    @staticmethod
    def load_directory(
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Document]:
        """Load directory (backward compatible - no table extraction)"""
        from app.services.document_loader import DocumentLoader as BaseLoader
        return BaseLoader.load_directory(directory_path, file_extensions, recursive)
