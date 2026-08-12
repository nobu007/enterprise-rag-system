"""
Document Management API Routes

This module defines API endpoints for document ingestion and management.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import os
import uuid

from app.core.logging_config import get_logger
from app.services.validator import DocumentValidator


logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# Initialize validator with default settings
validator = DocumentValidator()


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


# Batch processing models
class DocumentCreateRequest(BaseModel):
    """Request model for single document creation in batch"""
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class BatchIngestRequest(BaseModel):
    """Request model for batch document ingestion"""
    documents: List[DocumentCreateRequest] = Field(
        ...,
        description="List of documents to process (max 1000)",
        max_length=1000
    )
    collection: str = Field("default", description="Collection name")
    chunk_size: int = Field(1000, description="Chunk size for splitting", ge=100, le=4000)
    chunk_overlap: int = Field(200, description="Chunk overlap", ge=0, le=500)


class BatchIngestResponse(BaseModel):
    """Response model for batch ingestion initiation"""
    task_id: str = Field(..., description="Celery task ID for tracking")
    status: str = Field(..., description="Task status")
    total_documents: int = Field(..., description="Number of documents submitted")
    collection: str = Field(..., description="Collection name")


class BatchStatusResponse(BaseModel):
    """Response model for batch processing status"""
    task_id: str
    status: str = Field(..., description="Task state (PENDING/PROGRESS/SUCCESS/FAILURE)")
    result: Optional[Dict[str, Any]] = Field(None, description="Processing results if complete")
    error: Optional[str] = Field(None, description="Error message if failed")


# Versioning models
class DocumentVersionCreate(BaseModel):
    """Request model for creating a versioned document"""
    document_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    change_summary: str = Field("Initial version", description="Description of the change")
    created_by: str = Field("system", description="User or system creating the document")


class DocumentVersionUpdate(BaseModel):
    """Request model for updating a versioned document"""
    content: str = Field(..., description="New document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="New metadata (merged with existing)")
    change_summary: str = Field("Document updated", description="Description of the change")
    updated_by: str = Field("system", description="User or system updating the document")
    expected_version: Optional[int] = Field(None, description="Expected current version for optimistic locking")


class DocumentVersionRollback(BaseModel):
    """Request model for rolling back to a previous version"""
    target_version: int = Field(..., description="Version number to rollback to", ge=1)
    rolled_back_by: str = Field("system", description="User or system performing rollback")
    change_summary: Optional[str] = Field(None, description="Custom change summary (auto-generated if not provided)")


class DocumentVersionInfo(BaseModel):
    """Information about a document version"""
    version_number: int
    created_at: str
    created_by: str
    change_summary: str
    content_hash: str
    file_size_bytes: int
    content: Optional[str] = None
    metadata_preview: Optional[Dict[str, Any]] = None


class VersionHistoryResponse(BaseModel):
    """Response model for version history"""
    document_id: str
    current_version: int
    total_versions: int
    versions: List[DocumentVersionInfo]


class VersionComparisonResponse(BaseModel):
    """Response model for version comparison"""
    document_id: str
    version1: int
    version2: int
    content_same: bool
    size_difference_bytes: int
    version1_info: Dict[str, Any]
    version2_info: Dict[str, Any]


class VersioningStatsResponse(BaseModel):
    """Response model for versioning statistics"""
    total_documents: int
    total_versions: int
    total_storage_bytes: int
    total_storage_mb: float
    unique_contributors: int
    average_versions_per_document: float


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

        # Validate documents before processing
        logger.info(f"Validating {len(documents)} documents before ingestion")
        validation_results = validator.validate_batch(documents)

        # Separate valid and invalid documents
        valid_documents = []
        validation_errors = []

        for doc, result in zip(documents, validation_results):
            if result.is_valid:
                valid_documents.append(doc)
                # Log warnings if any
                if result.warnings:
                    logger.warning(
                        f"Document {doc.metadata.get('source', 'unknown')} "
                        f"has warnings: {result.warnings}"
                    )
            else:
                error_messages = [str(e) for e in result.errors]
                validation_errors.append({
                    'source': doc.metadata.get('source', 'unknown'),
                    'errors': error_messages
                })
                logger.warning(
                    f"Document {doc.metadata.get('source', 'unknown')} "
                    f"failed validation: {error_messages}"
                )

        # Update statistics to reflect only valid documents
        documents = valid_documents

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "No valid documents after validation",
                    "validation_errors": validation_errors
                }
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

        # Prepare message with validation info
        message = f"Successfully ingested {len(documents)} documents"
        if validation_errors:
            message += f" ({len(validation_errors)} documents failed validation)"

        return DocumentIngestResponse(
            success=True,
            documents_processed=len(documents),
            chunks_created=len(chunks),
            collection=request.collection or "default",
            message=message
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

            # Validate documents before processing
            validation_results = validator.validate_batch(documents)

            # Check if any document failed validation
            for doc, result in zip(documents, validation_results):
                if not result.is_valid:
                    error_messages = [str(e) for e in result.errors]
                    logger.warning(
                        f"Uploaded file {file.filename} failed validation: {error_messages}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "message": "Document validation failed",
                            "file": file.filename,
                            "errors": error_messages
                        }
                    )
                elif result.warnings:
                    logger.warning(
                        f"Uploaded file {file.filename} has warnings: {result.warnings}"
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


@router.post(
    "/batch",
    response_model=BatchIngestResponse,
    summary="Batch Ingest Documents / ドキュメント一括インジェスト",
    description="Submit multiple documents for asynchronous batch processing / 複数のドキュメントを非同期バッチ処理として送信します",
    response_description="Task ID for tracking progress / 進捗追跡用のタスクID",
    responses={
        202: {"description": "Task accepted and processing started / タスク受理、処理開始"},
        400: {"description": "Invalid request parameters / 不正なリクエストパラメータ"},
        500: {"description": "Failed to queue task / タスクキュー追加失敗"}
    },
    tags=["Documents"]
)
async def ingest_documents_batch(
    request: BatchIngestRequest,
    background_tasks: BackgroundTasks = None
) -> BatchIngestResponse:
    """
    Submit documents for batch processing / ドキュメントをバッチ処理に送信します

    ## Features / 機能

    - **Asynchronous Processing**: Tasks run in background using Celery / Celeryを使用した非同期処理
    - **Large Batches**: Process up to 1000 documents in one request / 1リクエストで最大1000ドキュメント処理
    - **Progress Tracking**: Monitor processing status with task ID / タスクIDで進捗をモニタリング
    - **Error Isolation**: Failed documents don't affect others / 失敗ドキュメントは他に影響しない

    ## Process / 処理フロー

    1. **Submit**: Send document list to API / APIにドキュメントリストを送信
    2. **Queue**: Task added to Celery queue / Celeryキューにタスク追加
    3. **Process**: Worker processes in background / ワーカーがバックグラウンドで処理
    4. **Track**: Check status with task_id / task_idでステータス確認

    ## Parameters / パラメータ

    - **documents**: List of documents (max 1000) / ドキュメントリスト（最大1000件）
      - **id**: Unique identifier / 一意識別子
      - **content**: Text content / テキスト内容
      - **metadata**: Optional metadata / オプションのメタデータ
    - **collection**: Collection name (default: "default") / コレクション名
    - **chunk_size**: Chunk size (100-4000, default: 1000) / チャンクサイズ
    - **chunk_overlap**: Chunk overlap (0-500, default: 200) / チャンクオーバーラップ

    ## Example Request / リクエスト例

    ```json
    {
      "documents": [
        {
          "id": "doc1",
          "content": "This is the first document...",
          "metadata": {"source": "hr-policies", "category": "benefits"}
        },
        {
          "id": "doc2",
          "content": "This is the second document...",
          "metadata": {"source": "hr-policies", "category": "leave"}
        }
      ],
      "collection": "hr-policies",
      "chunk_size": 1000,
      "chunk_overlap": 200
    }
    ```

    ## Example Response / レスポンス例

    ```json
    {
      "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "status": "PROCESSING",
      "total_documents": 2,
      "collection": "hr-policies"
    }
    ```

    ## Check Status / ステータス確認

    Use the returned `task_id` with GET `/documents/batch/{task_id}/status`

    Args:
        request: Batch ingestion request
        background_tasks: FastAPI background tasks (not used, kept for compatibility)

    Returns:
        BatchIngestResponse with task ID for tracking
    """
    try:
        from app.tasks.batch_tasks import process_document_batch

        # Validate request size
        if len(request.documents) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size exceeds maximum of 1000 documents"
            )

        # Validate document IDs are unique
        doc_ids = [doc.id for doc in request.documents]
        if len(doc_ids) != len(set(doc_ids)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document IDs must be unique"
            )

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Prepare documents for Celery (convert Pydantic models to dicts)
        documents_data = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            }
            for doc in request.documents
        ]

        # Submit task to Celery (AsyncResult intentionally unused: the
        # response tracks the task via the caller-generated task_id above)
        process_document_batch.apply_async(
            args=[documents_data, request.collection, request.chunk_size, request.chunk_overlap],
            task_id=task_id
        )

        logger.info(
            f"Submitted batch task {task_id}: "
            f"{len(request.documents)} documents to collection '{request.collection}'"
        )

        return BatchIngestResponse(
            task_id=task_id,
            status="PROCESSING",
            total_documents=len(request.documents),
            collection=request.collection
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit batch task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue batch task: {str(e)}"
        )


@router.get(
    "/batch/{task_id}/status",
    response_model=BatchStatusResponse,
    summary="Get Batch Processing Status / バッチ処理ステータス取得",
    description="Check the progress and results of a batch processing task / バッチ処理タスクの進捗と結果を確認します",
    response_description="Task status and results if complete / タスクステータスと完了時の結果",
    responses={
        200: {"description": "Status retrieved successfully / ステータス取得成功"},
        404: {"description": "Task not found / タスクが見つからない"},
        500: {"description": "Failed to retrieve status / ステータス取得失敗"}
    },
    tags=["Documents"]
)
async def get_batch_status(task_id: str) -> BatchStatusResponse:
    """
    Get batch processing status / バッチ処理のステータスを取得します

    ## Status Values / ステータス値

    - **PENDING**: Task waiting to be processed / 処理待ち
    - **PROGRESS**: Task currently processing / 処理中
    - **SUCCESS**: Task completed successfully / 処理成功
    - **FAILURE**: Task failed / 処理失敗

    ## Result Structure / 結果構造（成功時）

    ```json
    {
      "total": 100,
      "success": 98,
      "failed": 2,
      "errors": [
        {
          "doc_id": "doc45",
          "error": "Invalid content",
          "error_type": "ValueError"
        }
      ],
      "chunks_created": 1250
    }
    ```

    ## Example / 例

    ```bash
    # Check status
    curl "http://localhost:8000/documents/batch/a1b2c3d4-e5f6-7890-abcd-ef1234567890/status"
    ```

    Args:
        task_id: Celery task ID from batch submission

    Returns:
        BatchStatusResponse with current status and results
    """
    try:
        from app.tasks.batch_tasks import process_document_batch
        from celery.result import AsyncResult

        # Get task result
        task = AsyncResult(task_id, app=process_document_batch.app)

        response_data = {
            "task_id": task_id,
            "status": task.state,
            "result": None,
            "error": None
        }

        # Handle different task states
        if task.state == 'PENDING':
            response_data["status"] = "PENDING"
        elif task.state == 'PROGRESS':
            response_data["status"] = "PROGRESS"
            response_data["result"] = task.info
        elif task.state == 'SUCCESS':
            response_data["status"] = "SUCCESS"
            response_data["result"] = task.result
        elif task.state == 'FAILURE':
            response_data["status"] = "FAILURE"
            response_data["error"] = str(task.info)
        else:
            # Handle other Celery states
            response_data["status"] = task.state

        return BatchStatusResponse(**response_data)

    except Exception as e:
        logger.error(f"Failed to get batch task status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task status: {str(e)}"
        )


# ========== Versioning Endpoints ==========

def _get_versioning_service():
    """Get or create versioning service instance"""
    from app.services.versioning import DocumentVersioningService
    from app.core.config import get_settings

    settings = get_settings()
    storage_path = getattr(settings, 'versioning_storage_path', './data/versioning')
    return DocumentVersioningService(storage_path=storage_path)


@router.post(
    "/versioning",
    response_model=DocumentVersionInfo,
    summary="Create Versioned Document / バージョン管理ドキュメント作成",
    description="Create a new document with version tracking enabled / バージョン管理が有効な新しいドキュメントを作成します",
    response_description="Created document version information / 作成されたドキュメントバージョン情報",
    responses={
        200: {"description": "Document created successfully / ドキュメント作成成功"},
        400: {"description": "Document already exists or invalid parameters / ドキュメントが既に存在するか不正なパラメータ"},
        500: {"description": "Creation failed / 作成失敗"}
    },
    tags=["Documents", "Versioning"]
)
async def create_versioned_document(request: DocumentVersionCreate) -> DocumentVersionInfo:
    """
    Create a new document with version tracking / バージョン管理付きで新しいドキュメントを作成します

    ## Features / 機能

    - **Automatic Versioning**: Every document starts at version 1 / すべてのドキュメントはバージョン1から開始
    - **Content Hashing**: SHA-256 hash for integrity verification / 完全性検証のためのSHA-256ハッシュ
    - **Audit Trail**: Tracks who created the document and when / 作成者と作成日時の記録

    ## Example / 例

    ```json
    {
      "document_id": "doc-001",
      "content": "This is the initial document content...",
      "metadata": {"source": "policy.pdf", "category": "HR"},
      "change_summary": "Initial HR policy document",
      "created_by": "admin@company.com"
    }
    ```

    Args:
        request: Document creation request

    Returns:
        DocumentVersionInfo with version details
    """
    try:
        service = _get_versioning_service()

        version = service.create_document(
            document_id=request.document_id,
            content=request.content,
            metadata=request.metadata,
            created_by=request.created_by,
            change_summary=request.change_summary
        )

        return DocumentVersionInfo(
            version_number=version.version_number,
            created_at=version.created_at,
            created_by=version.created_by,
            change_summary=version.change_summary,
            content_hash=version.content_hash,
            file_size_bytes=version.file_size_bytes,
            content=version.content,
            metadata_preview=version.metadata
        )

    except Exception as e:
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document: {str(e)}"
        )


@router.put(
    "/versioning/{document_id}",
    response_model=DocumentVersionInfo,
    summary="Update Versioned Document / バージョン管理ドキュメント更新",
    description="Update a document and create a new version / ドキュメントを更新して新しいバージョンを作成します",
    response_description="New document version information / 新しいドキュメントバージョン情報",
    responses={
        200: {"description": "Document updated successfully / ドキュメント更新成功"},
        404: {"description": "Document not found / ドキュメントが見つからない"},
        409: {"description": "Version conflict / バージョン衝突"},
        500: {"description": "Update failed / 更新失敗"}
    },
    tags=["Documents", "Versioning"]
)
async def update_versioned_document(
    document_id: str,
    request: DocumentVersionUpdate
) -> DocumentVersionInfo:
    """
    Update a versioned document / バージョン管理ドキュメントを更新します

    ## Features / 機能

    - **Automatic Versioning**: Creates new version on each update / 更新ごとに新しいバージョンを作成
    - **Optimistic Locking**: Optional version conflict detection / オプションのバージョン衝突検出
    - **Metadata Merging**: New metadata merged with existing / 新しいメタデータを既存とマージ

    ## Example / 例

    ```json
    {
      "content": "Updated document content...",
      "metadata": {"category": "Updated HR", "reviewed": true},
      "change_summary": "Updated HR policy with new benefits section",
      "updated_by": "admin@company.com",
      "expected_version": 1
    }
    ```

    Args:
        document_id: Document identifier
        request: Update request

    Returns:
        DocumentVersionInfo for the new version
    """
    try:
        service = _get_versioning_service()

        version = service.update_document(
            document_id=document_id,
            content=request.content,
            metadata=request.metadata,
            updated_by=request.updated_by,
            change_summary=request.change_summary,
            expected_version=request.expected_version
        )

        return DocumentVersionInfo(
            version_number=version.version_number,
            created_at=version.created_at,
            created_by=version.created_by,
            change_summary=version.change_summary,
            content_hash=version.content_hash,
            file_size_bytes=version.file_size_bytes,
            content=version.content,
            metadata_preview=version.metadata
        )

    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        if "conflict" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}"
        )


@router.post(
    "/versioning/{document_id}/rollback",
    response_model=DocumentVersionInfo,
    summary="Rollback Document Version / ドキュメントバージョンロールバック",
    description="Rollback a document to a previous version / ドキュメントを以前のバージョンにロールバックします",
    response_description="New document version created from rollback / ロールバックで作成された新しいバージョン",
    responses={
        200: {"description": "Rollback successful / ロールバック成功"},
        404: {"description": "Document or version not found / ドキュメントまたはバージョンが見つからない"},
        500: {"description": "Rollback failed / ロールバック失敗"}
    },
    tags=["Documents", "Versioning"]
)
async def rollback_document_version(
    document_id: str,
    request: DocumentVersionRollback
) -> DocumentVersionInfo:
    """
    Rollback a document to a previous version / ドキュメントを以前のバージョンにロールバックします

    ## Features / 機能

    - **Safe Rollback**: Creates new version from old content / 古いコンテンツから新しいバージョンを作成
    - **Preserves History**: Original versions remain intact / 元のバージョンは保持
    - **Audit Trail**: Tracks rollback operations / ロールバック操作を記録

    ## Example / 例

    ```json
    {
      "target_version": 2,
      "rolled_back_by": "admin@company.com",
      "change_summary": "Reverting mistaken changes"
    }
    ```

    Args:
        document_id: Document identifier
        request: Rollback request

    Returns:
        DocumentVersionInfo for the rollback version
    """
    try:
        service = _get_versioning_service()

        version = service.rollback_to_version(
            document_id=document_id,
            target_version=request.target_version,
            rolled_back_by=request.rolled_back_by,
            change_summary=request.change_summary
        )

        return DocumentVersionInfo(
            version_number=version.version_number,
            created_at=version.created_at,
            created_by=version.created_by,
            change_summary=version.change_summary,
            content_hash=version.content_hash,
            file_size_bytes=version.file_size_bytes,
            content=version.content,
            metadata_preview=version.metadata
        )

    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rollback document: {str(e)}"
        )


@router.get(
    "/versioning/{document_id}/history",
    response_model=VersionHistoryResponse,
    summary="Get Document Version History / ドキュメントバージョン履歴取得",
    description="Retrieve complete version history for a document / ドキュメントの完全なバージョン履歴を取得します",
    response_description="Version history with all versions / すべてのバージョンを含む履歴",
    responses={
        200: {"description": "History retrieved successfully / 履歴取得成功"},
        404: {"description": "Document not found / ドキュメントが見つからない"},
        500: {"description": "Failed to retrieve history / 履歴取得失敗"}
    },
    tags=["Documents", "Versioning"]
)
async def get_document_version_history(
    document_id: str,
    include_content: bool = False
) -> VersionHistoryResponse:
    """
    Get document version history / ドキュメントのバージョン履歴を取得します

    ## Parameters / パラメータ

    - **include_content**: Include full content in response (default: false) / レスポンスに完全なコンテンツを含める（デフォルト: false）

    ## Example / 例

    ```bash
    # Get history without content
    curl "http://localhost:8000/documents/versioning/doc-001/history"

    # Get history with full content
    curl "http://localhost:8000/documents/versioning/doc-001/history?include_content=true"
    ```

    Args:
        document_id: Document identifier
        include_content: Whether to include full content

    Returns:
        VersionHistoryResponse with all versions
    """
    try:
        service = _get_versioning_service()

        history_data = service.get_version_history(
            document_id=document_id,
            include_content=include_content
        )

        if history_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found"
            )

        versions = [
            DocumentVersionInfo(**v) for v in history_data['versions']
        ]

        return VersionHistoryResponse(
            document_id=history_data['document_id'],
            current_version=history_data['current_version'],
            total_versions=history_data['total_versions'],
            versions=versions
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get(
    "/versioning/{document_id}/versions/{version_number}",
    response_model=DocumentVersionInfo,
    summary="Get Specific Document Version / 特定バージョンのドキュメント取得",
    description="Retrieve a specific version of a document / ドキュメントの特定のバージョンを取得します",
    response_description="Document version information / ドキュメントバージョン情報",
    responses={
        200: {"description": "Version retrieved successfully / バージョン取得成功"},
        404: {"description": "Document or version not found / ドキュメントまたはバージョンが見つからない"},
        500: {"description": "Failed to retrieve version / バージョン取得失敗"}
    },
    tags=["Documents", "Versioning"]
)
async def get_document_version(
    document_id: str,
    version_number: int
) -> DocumentVersionInfo:
    """
    Get a specific document version / 特定のドキュメントバージョンを取得します

    ## Example / 例

    ```bash
    # Get version 2 of document
    curl "http://localhost:8000/documents/versioning/doc-001/versions/2"
    ```

    Args:
        document_id: Document identifier
        version_number: Version number to retrieve

    Returns:
        DocumentVersionInfo with full content
    """
    try:
        service = _get_versioning_service()

        version = service.get_version(
            document_id=document_id,
            version_number=version_number
        )

        if version is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' or version {version_number} not found"
            )

        return DocumentVersionInfo(
            version_number=version.version_number,
            created_at=version.created_at,
            created_by=version.created_by,
            change_summary=version.change_summary,
            content_hash=version.content_hash,
            file_size_bytes=version.file_size_bytes,
            content=version.content,
            metadata_preview=version.metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve version: {str(e)}"
        )


@router.get(
    "/versioning/{document_id}/compare",
    response_model=VersionComparisonResponse,
    summary="Compare Document Versions / ドキュメントバージョン比較",
    description="Compare two versions of a document / ドキュメントの2つのバージョンを比較します",
    response_description="Comparison results / 比較結果",
    responses={
        200: {"description": "Comparison successful / 比較成功"},
        404: {"description": "Document or versions not found / ドキュメントまたはバージョンが見つからない"},
        400: {"description": "Invalid version parameters / 不正なバージョンパラメータ"},
        500: {"description": "Comparison failed / 比較失敗"}
    },
    tags=["Documents", "Versioning"]
)
async def compare_document_versions(
    document_id: str,
    version1: int,
    version2: int
) -> VersionComparisonResponse:
    """
    Compare two document versions / 2つのドキュメントバージョンを比較します

    ## Example / 例

    ```bash
    # Compare version 1 and 3
    curl "http://localhost:8000/documents/versioning/doc-001/compare?version1=1&version2=3"
    ```

    Args:
        document_id: Document identifier
        version1: First version number
        version2: Second version number

    Returns:
        VersionComparisonResponse with comparison details
    """
    try:
        if version1 < 1 or version2 < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Version numbers must be >= 1"
            )

        service = _get_versioning_service()

        comparison = service.compare_versions(
            document_id=document_id,
            version1=version1,
            version2=version2
        )

        if comparison is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' or specified versions not found"
            )

        return VersionComparisonResponse(**comparison)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare versions: {str(e)}"
        )


@router.delete(
    "/versioning/{document_id}",
    summary="Delete Versioned Document / バージョン管理ドキュメント削除",
    description="Delete a document and all its versions / ドキュメントとすべてのバージョンを削除します",
    response_description="Deletion confirmation / 削除確認",
    responses={
        200: {"description": "Document deleted successfully / ドキュメント削除成功"},
        404: {"description": "Document not found / ドキュメントが見つからない"},
        500: {"description": "Deletion failed / 削除失敗"}
    },
    tags=["Documents", "Versioning"]
)
async def delete_versioned_document(document_id: str) -> Dict[str, Any]:
    """
    Delete a versioned document / バージョン管理ドキュメントを削除します

    ## Warning / 注意

    **This action is irreversible!** All versions of the document will be permanently deleted.
    **この操作は元に戻せません！** ドキュメントのすべてのバージョンが永久に削除されます。

    ## Example / 例

    ```bash
    curl -X DELETE "http://localhost:8000/documents/versioning/doc-001"
    ```

    Args:
        document_id: Document identifier

    Returns:
        Deletion confirmation
    """
    try:
        service = _get_versioning_service()

        service.delete_document(document_id=document_id)

        logger.info(f"Deleted versioned document '{document_id}'")

        return {
            "success": True,
            "message": f"Document '{document_id}' and all versions deleted successfully"
        }

    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.get(
    "/versioning",
    summary="List All Versioned Documents / すべてのバージョン管理ドキュメント一覧",
    description="List all documents with versioning information / バージョン管理情報付きですべてのドキュメントを一覧表示します",
    response_description="List of documents with version info / バージョン情報付きのドキュメントリスト",
    responses={
        200: {"description": "Documents listed successfully / ドキュメント一覧取得成功"},
        500: {"description": "Failed to list documents / ドキュメント一覧取得失敗"}
    },
    tags=["Documents", "Versioning"]
)
async def list_versioned_documents() -> Dict[str, Any]:
    """
    List all versioned documents / すべてのバージョン管理ドキュメントを一覧表示します

    ## Example / 例

    ```bash
    curl "http://localhost:8000/documents/versioning"
    ```

    Returns:
        List of document summaries
    """
    try:
        service = _get_versioning_service()

        documents = service.list_documents()

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get(
    "/versioning/stats",
    response_model=VersioningStatsResponse,
    summary="Get Versioning Statistics / バージョン管理統計取得",
    description="Retrieve versioning system statistics / バージョン管理システムの統計を取得します",
    response_description="Versioning statistics / バージョン管理統計",
    responses={
        200: {"description": "Statistics retrieved successfully / 統計取得成功"},
        500: {"description": "Failed to retrieve statistics / 統計取得失敗"}
    },
    tags=["Documents", "Versioning"]
)
async def get_versioning_statistics() -> VersioningStatsResponse:
    """
    Get versioning system statistics / バージョン管理システムの統計を取得します

    ## Example / 例

    ```bash
    curl "http://localhost:8000/documents/versioning/stats"
    ```

    Returns:
        VersioningStatsResponse with system metrics
    """
    try:
        service = _get_versioning_service()

        stats = service.get_statistics()

        return VersioningStatsResponse(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )
