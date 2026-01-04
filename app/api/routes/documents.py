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

from app.core.logging_config import get_logger


logger = get_logger(__name__)
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


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    summary="Ingest Documents from Directory / ディレクトリからドキュメントをインジェスト",
    description="Load, process, and store documents from a directory into the vector database / ディレクトリからドキュメントを読み込み、処理してベクトルデータベースに保存します",
    response_description="Document ingestion statistics and status / ドキュメントインジェストの統計とステータス",
    responses={
        200: {"description": "Documents ingested successfully / ドキュメントインジェスト成功"},
        400: {"description": "No documents found or invalid parameters / ドキュメントが見つからないか不正なパラメータ"},
        404: {"description": "Directory not found / ディレクトリが見つからない"},
        500: {"description": "Ingestion failed / インジェスト失敗"}
    },
    tags=["Documents"]
)
async def ingest_documents(request: DocumentIngestRequest) -> DocumentIngestResponse:
    """
    Ingest documents from a directory / ディレクトリからドキュメントをインジェストします

    ## Supported Formats / 対応フォーマット

    - **PDF**: `.pdf` files using PyPDF2 / PyPDF2を使用したPDFファイル
    - **Markdown**: `.md` files / Markdownファイル
    - **Text**: `.txt` files / テキストファイル
    - **HTML**: `.html` files (with html2text) / HTMLファイル（html2text使用）

    ## Process / 処理フロー

    1. **Load**: Read documents from source path / ソースパスからドキュメントを読み込み
    2. **Split**: Chunk documents with overlap / ドキュメントをオーバーラップ付きでチャンク分割
    3. **Embed**: Generate vector embeddings / ベクトル埋め込みを生成
    4. **Store**: Save to vector database / ベクトルデータベースに保存

    ## Parameters / パラメータ

    - **source_path**: Path to directory containing documents / ドキュメントを含むディレクトリへのパス
    - **collection**: Collection name for organization / 整理用のコレクション名
    - **chunk_size**: Size of text chunks (100-4000) / テキストチャンクのサイズ (100-4000)
    - **chunk_overlap**: Overlap between chunks (0-500) / チャンク間のオーバーラップ (0-500)

    ## Example / 例

    ```json
    {
      "source_path": "./data/hr-policies",
      "collection": "hr-policies",
      "chunk_size": 1000,
      "chunk_overlap": 200
    }
    ```

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
        logger.info(f"Loading documents from: {request.source_path}")
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
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embedding_model = get_embedding_model()
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_model.embed_texts(texts)
        
        # Store in vector database
        vector_db = get_vector_db(db_type="faiss", index_path=settings.faiss_index_path)

        if vector_db.index is None:
            vector_db.create_index(dimension=embedding_model.dimension)

        ids = [chunk.doc_id for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]

        # Use collection from request
        collection = request.collection or "default"
        vector_db.upsert(vectors=embeddings, ids=ids, metadata=metadata, collection=collection)

        # Save index
        if hasattr(vector_db, 'save'):
            vector_db.save(settings.faiss_index_path)
        
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


@router.post(
    "/upload",
    response_model=DocumentIngestResponse,
    summary="Upload Single Document / 単一ドキュメントアップロード",
    description="Upload and ingest a single document file into the vector database / 単一のドキュメントファイルをアップロードしてベクトルデータベースにインジェストします",
    response_description="Document ingestion statistics and status / ドキュメントインジェストの統計とステータス",
    responses={
        200: {"description": "Document uploaded and ingested successfully / ドキュメントアップロードとインジェスト成功"},
        400: {"description": "Unsupported file type or invalid parameters / サポートされていないファイルタイプか不正なパラメータ"},
        500: {"description": "Upload failed / アップロード失敗"}
    },
    tags=["Documents"]
)
async def upload_document(
    file: UploadFile = File(...),
    collection: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
) -> DocumentIngestResponse:
    """
    Upload and ingest a single document / 単一のドキュメントをアップロードしてインジェストします

    ## Supported File Types / 対応ファイルタイプ

    - **PDF**: `.pdf` - PDF documents / PDFドキュメント
    - **Markdown**: `.md` - Markdown files / Markdownファイル
    - **Text**: `.txt` - Plain text files / テキストファイル

    ## Form Data / フォームデータ

    - **file**: The document file to upload (required) / アップロードするドキュメントファイル（必須）
    - **collection**: Collection name (optional, default: "default") / コレクション名（オプション、デフォルト: "default"）
    - **chunk_size**: Size of text chunks (optional, default: 1000) / テキストチャンクのサイズ（オプション、デフォルト: 1000）
    - **chunk_overlap**: Overlap between chunks (optional, default: 200) / チャンク間のオーバーラップ（オプション、デフォルト: 200）

    ## Example with curl / curl使用例

    ```bash
    curl -X POST "http://localhost:8000/api/v1/documents/upload" \
      -F "file=@document.pdf" \
      -F "collection=hr-policies" \
      -F "chunk_size=1000" \
      -F "chunk_overlap=200"
    ```

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
        from app.core.config import get_settings

        settings = get_settings()

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
            vector_db = get_vector_db(db_type="faiss", index_path=settings.faiss_index_path)

            if vector_db.index is None:
                vector_db.create_index(dimension=embedding_model.dimension)

            ids = [chunk.doc_id for chunk in chunks]
            metadata = [chunk.metadata for chunk in chunks]

            # Use collection from request
            collection_name = collection or "default"
            vector_db.upsert(vectors=embeddings, ids=ids, metadata=metadata, collection=collection_name)

            if hasattr(vector_db, 'save'):
                vector_db.save(settings.faiss_index_path)
            
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


@router.get(
    "/stats",
    response_model=DocumentStats,
    summary="Get Document Statistics / ドキュメント統計取得",
    description="Retrieve statistics about ingested documents and collections / インジェストされたドキュメントとコレクションに関する統計を取得します",
    response_description="Document statistics including counts and collections / ドキュメント数とコレクションを含む統計",
    responses={
        200: {"description": "Statistics retrieved successfully / 統計取得成功"},
        500: {"description": "Failed to retrieve statistics / 統計取得失敗"}
    },
    tags=["Documents"]
)
async def get_stats() -> DocumentStats:
    """Get statistics about ingested documents / インジェストされたドキュメントに関する統計を取得します

    ## Returns / 戻り値

    - **total_documents**: Total number of documents across all collections / すべてのコレクションのドキュメント総数
    - **total_chunks**: Total number of chunks across all collections / すべてのコレクションのチャンク総数
    - **collections**: List of collection names / コレクション名のリスト

    ## Example Response / レスポンス例

    ```json
    {
      "total_documents": 150,
      "total_chunks": 2250,
      "collections": ["default", "hr-policies", "tech-docs"]
    }
    ```
    """
    try:
        from app.core.vectordb import get_vector_db
        from app.core.config import get_settings

        settings = get_settings()

        vector_db = get_vector_db(db_type="faiss", index_path=settings.faiss_index_path)
        vector_db.connect()
        
        stats = vector_db.get_stats()

        # Extract collection names from stats
        collection_names = list(stats.get('collections', {}).keys())

        return DocumentStats(
            total_documents=stats.get('total_vectors', 0),
            total_chunks=stats.get('total_vectors', 0),
            collections=collection_names if collection_names else ["default"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )
