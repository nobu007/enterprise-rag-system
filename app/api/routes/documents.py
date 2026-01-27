"""
Document Management API Routes

This module defines API endpoints for document ingestion and management.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import os

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion"""
    source_path: str = Field(..., description="Path to documents to ingest")
    collection: Optional[str] = Field(None, description="Collection name")
    chunk_size: int = Field(1000, description="Chunk size for splitting")
    chunk_overlap: int = Field(200, description="Chunk overlap")


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion"""
    success: bool
    documents_processed: int
    chunks_created: int
    collection: str
    message: str


class DocumentStats(BaseModel):
    """Document statistics"""
    total_documents: int
    total_chunks: int
    collections: List[str]


@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_documents(request: DocumentIngestRequest) -> DocumentIngestResponse:
    """
    Ingest documents from a directory
    
    Args:
        request: Ingestion request with source path and parameters
    
    Returns:
        DocumentIngestResponse with ingestion statistics
    """
    try:
        from app.services.document_loader import DocumentLoader, TextSplitter
        from app.core.embeddings import get_embedding_model
        from app.core.vectordb import get_vector_db
        from app.core.config import get_settings
        
        settings = get_settings()
        
        # Load documents
        print(f"📂 Loading documents from: {request.source_path}")
        documents = DocumentLoader.load_directory(request.source_path)
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents found in the specified path"
            )
        
        # Split documents into chunks
        splitter = TextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        
        # Generate embeddings
        print(f"🧠 Generating embeddings for {len(chunks)} chunks")
        embedding_model = get_embedding_model()
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_model.embed_texts(texts)
        
        # Store in vector database
        vector_db = get_vector_db(db_type="faiss", index_path="./data/faiss_index.bin")
        
        if vector_db.index is None:
            vector_db.create_index(dimension=embedding_model.dimension)
        
        ids = [chunk.doc_id for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        
        vector_db.upsert(vectors=embeddings, ids=ids, metadata=metadata)
        
        # Save index
        if hasattr(vector_db, 'save'):
            vector_db.save("./data/faiss_index.bin")
        
        return DocumentIngestResponse(
            success=True,
            documents_processed=len(documents),
            chunks_created=len(chunks),
            collection=request.collection or "default",
            message=f"Successfully ingested {len(documents)} documents"
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
) -> DocumentIngestResponse:
    """
    Upload and ingest a single document
    
    Args:
        file: Uploaded file
        collection: Collection name
        chunk_size: Chunk size for splitting
        chunk_overlap: Chunk overlap
    
    Returns:
        DocumentIngestResponse with ingestion statistics
    """
    try:
        from app.services.document_loader import DocumentLoader, TextSplitter
        from app.core.embeddings import get_embedding_model
        from app.core.vectordb import get_vector_db
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Load document
            file_ext = Path(file.filename).suffix.lower()
            
            if file_ext == '.pdf':
                documents = DocumentLoader.load_pdf(tmp_path)
            elif file_ext == '.md':
                documents = [DocumentLoader.load_markdown(tmp_path)]
            elif file_ext == '.txt':
                documents = [DocumentLoader.load_text_file(tmp_path)]
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file_ext}"
                )
            
            # Split and embed
            splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(documents)
            
            embedding_model = get_embedding_model()
            texts = [chunk.content for chunk in chunks]
            embeddings = embedding_model.embed_texts(texts)
            
            # Store in vector database
            vector_db = get_vector_db(db_type="faiss", index_path="./data/faiss_index.bin")
            
            if vector_db.index is None:
                vector_db.create_index(dimension=embedding_model.dimension)
            
            ids = [chunk.doc_id for chunk in chunks]
            metadata = [chunk.metadata for chunk in chunks]
            
            vector_db.upsert(vectors=embeddings, ids=ids, metadata=metadata)
            
            if hasattr(vector_db, 'save'):
                vector_db.save("./data/faiss_index.bin")
            
            return DocumentIngestResponse(
                success=True,
                documents_processed=len(documents),
                chunks_created=len(chunks),
                collection=collection or "default",
                message=f"Successfully uploaded and ingested {file.filename}"
            )
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/stats", response_model=DocumentStats)
async def get_stats() -> DocumentStats:
    """Get statistics about ingested documents"""
    try:
        from app.core.vectordb import get_vector_db
        
        vector_db = get_vector_db(db_type="faiss", index_path="./data/faiss_index.bin")
        vector_db.connect()
        
        stats = vector_db.get_stats()
        
        return DocumentStats(
            total_documents=stats.get('total_vectors', 0),
            total_chunks=stats.get('total_vectors', 0),
            collections=["default"]  # TODO: Implement multi-collection support
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )
