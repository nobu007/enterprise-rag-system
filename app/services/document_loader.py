"""
Document Loading and Processing

This module handles loading documents from various sources and formats.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import hashlib

from app.core.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class Document:
    """Document representation"""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate document ID if not provided"""
        if not self.doc_id:
            self.doc_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique document ID based on content"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        source = self.metadata.get('source', 'unknown')
        return f"{source}_{content_hash[:16]}"


class DocumentLoader:
    """Base class for document loaders"""
    
    @staticmethod
    def load_text_file(file_path: str) -> Document:
        """Load a plain text file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            'source': str(path),
            'filename': path.name,
            'file_type': 'txt',
            'size_bytes': path.stat().st_size
        }
        
        return Document(content=content, metadata=metadata)
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load a PDF file"""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf not installed. Run: pip install pypdf")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        reader = PdfReader(str(path))
        documents = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            
            if text.strip():  # Only include non-empty pages
                metadata = {
                    'source': str(path),
                    'filename': path.name,
                    'file_type': 'pdf',
                    'page': page_num + 1,
                    'total_pages': len(reader.pages)
                }
                
                documents.append(Document(content=text, metadata=metadata))
        
        return documents
    
    @staticmethod
    def load_markdown(file_path: str) -> Document:
        """Load a Markdown file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            'source': str(path),
            'filename': path.name,
            'file_type': 'markdown',
            'size_bytes': path.stat().st_size
        }
        
        return Document(content=content, metadata=metadata)
    
    @staticmethod
    def load_directory(
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Document]:
        """Load all documents from a directory"""
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
                    docs = DocumentLoader.load_pdf(str(file_path))
                    documents.extend(docs)
                elif ext == '.md':
                    doc = DocumentLoader.load_markdown(str(file_path))
                    documents.append(doc)
                elif ext == '.txt':
                    doc = DocumentLoader.load_text_file(str(file_path))
                    documents.append(doc)

                logger.debug(f"Loaded: {file_path.name}")

            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents


class TextSplitter:
    """Split documents into smaller chunks for embedding"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = []
        
        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""
                
                for part in parts:
                    # If adding this part exceeds chunk_size, save current chunk
                    if len(current_chunk) + len(part) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            # Start new chunk with overlap
                            overlap_text = current_chunk[-self.chunk_overlap:]
                            current_chunk = overlap_text + separator + part
                        else:
                            current_chunk = part
                    else:
                        if current_chunk:
                            current_chunk += separator + part
                        else:
                            current_chunk = part
                
                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                break
        
        # If no separator worked, use fixed-size chunking
        if not chunks and text:
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        chunked_documents = []
        
        for doc in documents:
            chunks = self.split_text(doc.content)
            
            for i, chunk in enumerate(chunks):
                # Create new document for each chunk
                chunk_metadata = doc.metadata.copy()
                chunk_metadata['chunk_index'] = i
                chunk_metadata['total_chunks'] = len(chunks)
                chunk_metadata['original_doc_id'] = doc.doc_id
                
                chunked_doc = Document(
                    content=chunk,
                    metadata=chunk_metadata
                )
                chunked_documents.append(chunked_doc)
        
        return chunked_documents
