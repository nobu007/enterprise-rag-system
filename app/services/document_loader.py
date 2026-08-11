"""
Document Loading and Processing

This module handles loading documents from various sources and formats.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import hashlib

from app.core.logging_config import get_logger
from app.core.encryption import DocumentEncryption, EncryptionError


logger = get_logger(__name__)


@dataclass
class Document:
    """Document representation"""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    encrypted_content: Optional[str] = None
    
    def __post_init__(self):
        """Generate document ID if not provided"""
        if not self.doc_id:
            self.doc_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique document ID based on content"""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        source = self.metadata.get('source', 'unknown')
        return f"{source}_{content_hash[:16]}"
    
    def encrypt_content(self, encryptor: DocumentEncryption) -> None:
        """
        Encrypt document content using provided encryptor.
        
        Args:
            encryptor: DocumentEncryption instance
        """
        try:
            result = encryptor.encrypt(self.content)
            self.encrypted_content = result.encrypted_data
            self.metadata['encrypted'] = True
            self.metadata['encryption_nonce'] = result.nonce
            self.metadata['encryption_salt'] = result.salt
            self.metadata['encryption_tag'] = result.tag
            logger.debug(f"Encrypted document {self.doc_id}")
        except EncryptionError as e:
            logger.error(f"Failed to encrypt document {self.doc_id}: {e}")
            raise
    
    def decrypt_content(self, encryptor: DocumentEncryption) -> str:
        """
        Decrypt document content using provided encryptor.
        
        Args:
            encryptor: DocumentEncryption instance
        
        Returns:
            Decrypted content string
        """
        if not self.encrypted_content:
            raise ValueError("Document has no encrypted content")
        
        try:
            nonce = self.metadata.get('encryption_nonce', '')
            salt = self.metadata.get('encryption_salt', '')
            tag = self.metadata.get('encryption_tag', '')
            
            decrypted = encryptor.decrypt(
                self.encrypted_content,
                nonce,
                salt,
                tag
            )
            logger.debug(f"Decrypted document {self.doc_id}")
            return decrypted
        except Exception as e:
            logger.error(f"Failed to decrypt document {self.doc_id}: {e}")
            raise
    
    def is_encrypted(self) -> bool:
        """Check if document content is encrypted."""
        return self.encrypted_content is not None


class DocumentLoader:
    """Base class for document loaders"""
    
    @staticmethod
    def load_text_file(file_path: str, encryptor: Optional[DocumentEncryption] = None) -> Document:
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
        
        doc = Document(content=content, metadata=metadata)
        
        # Encrypt if encryptor is provided
        if encryptor:
            doc.encrypt_content(encryptor)
        
        return doc
    
    @staticmethod
    def load_pdf(file_path: str, encryptor: Optional[DocumentEncryption] = None) -> List[Document]:
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
                
                doc = Document(content=text, metadata=metadata)
                
                # Encrypt if encryptor is provided
                if encryptor:
                    doc.encrypt_content(encryptor)
                
                documents.append(doc)
        
        return documents
    
    @staticmethod
    def load_markdown(file_path: str, encryptor: Optional[DocumentEncryption] = None) -> Document:
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
        
        doc = Document(content=content, metadata=metadata)
        
        # Encrypt if encryptor is provided
        if encryptor:
            doc.encrypt_content(encryptor)
        
        return doc
    
    @staticmethod
    def load_directory(
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        encryptor: Optional[DocumentEncryption] = None
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
                    docs = DocumentLoader.load_pdf(str(file_path), encryptor=encryptor)
                    documents.extend(docs)
                elif ext == '.md':
                    doc = DocumentLoader.load_markdown(str(file_path), encryptor=encryptor)
                    documents.append(doc)
                elif ext == '.txt':
                    doc = DocumentLoader.load_text_file(str(file_path), encryptor=encryptor)
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
        # chunk_overlap must be strictly less than chunk_size: the
        # fixed-size fallback advances by (chunk_size - chunk_overlap), and a
        # non-positive step either raises (==: "range() arg 3 must not be
        # zero") or silently yields zero chunks (>: empty range), dropping
        # every document — e.g. CJK text with no usable separator.
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size}); a non-negative chunking step is "
                "required or no chunks are produced."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = []
        
        # Try each separator in order
        for separator in self.separators:
            # Skip the empty separator: "" is always "in" text, but
            # text.split("") raises ValueError. The fixed-size fallback
            # below handles text with no usable separator (e.g. CJK).
            if separator and separator in text:
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
