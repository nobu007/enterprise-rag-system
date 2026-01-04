"""
Celery tasks for batch document processing

This module defines asynchronous tasks for processing large batches of documents
using Celery for distributed task queue management.
"""

from celery import Celery
from typing import List, Dict, Any
from app.core.config import settings
from app.core.logging_config import get_logger


logger = get_logger(__name__)

# Create Celery app instance with configuration
celery_app = Celery("rag_tasks")
celery_app.config_from_object('celeryconfig')


@celery_app.task(bind=True, name='tasks.process_document_batch')
def process_document_batch(
    self,
    documents: List[Dict[str, Any]],
    collection: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Dict[str, Any]:
    """
    Process a batch of documents asynchronously

    This task handles:
    1. Document loading and validation
    2. Text splitting into chunks
    3. Vector embedding generation
    4. Vector database upsert
    5. Error handling and reporting

    Args:
        self: Celery task instance (bind=True)
        documents: List of document dictionaries with keys:
            - id: str - Unique document identifier
            - content: str - Document text content
            - metadata: Dict - Optional metadata
        collection: Collection name in vector database
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks

    Returns:
        Dictionary with processing results:
            - total: Total number of documents
            - success: Number of successfully processed documents
            - failed: Number of failed documents
            - errors: List of error details
            - chunks_created: Total number of chunks created

    Example:
        >>> result = process_document_batch.delay(
        ...     documents=[
        ...         {"id": "doc1", "content": "Text 1", "metadata": {}},
        ...         {"id": "doc2", "content": "Text 2", "metadata": {}}
        ...     ],
        ...     collection="hr-policies"
        ... )
        >>> result.status
        'PENDING'
    """
    try:
        from app.services.document_loader import Document, TextSplitter
        from app.core.embeddings import get_embedding_model
        from app.core.vectordb import get_vector_db

        # Handle empty document list early
        if not documents:
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "errors": [],
                "chunks_created": 0
            }

        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': len(documents), 'status': 'Starting...'}
        )

        # Initialize services
        splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        embedding_model = get_embedding_model()
        vector_db = get_vector_db(
            db_type="faiss",
            index_path=settings.faiss_index_path
        )

        # Ensure index exists
        if vector_db.index is None:
            vector_db.create_index(dimension=embedding_model.dimension)

        # Process documents
        results = {
            "total": len(documents),
            "success": 0,
            "failed": 0,
            "errors": [],
            "chunks_created": 0
        }

        all_chunks = []

        for idx, doc_data in enumerate(documents):
            try:
                # Create Document object
                doc = Document(
                    content=doc_data["content"],
                    metadata=doc_data.get("metadata", {}),
                    doc_id=doc_data["id"]
                )

                # Split document into chunks
                chunks = splitter.split_documents([doc])
                all_chunks.extend(chunks)

                # Update progress
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': idx + 1,
                        'total': len(documents),
                        'status': f'Processed {doc.doc_id}'
                    }
                )

                results["success"] += 1
                logger.info(f"Successfully processed document {doc.doc_id}")

            except Exception as e:
                results["failed"] += 1
                error_detail = {
                    "doc_id": doc_data.get("id", "unknown"),
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                results["errors"].append(error_detail)
                logger.error(f"Failed to process document {doc_data.get('id')}: {e}")

        # Batch embedding generation
        if all_chunks:
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
            texts = [chunk.content for chunk in all_chunks]
            embeddings = embedding_model.embed_texts(texts)

            # Prepare vector data
            ids = [chunk.doc_id for chunk in all_chunks]
            metadata = [chunk.metadata for chunk in all_chunks]

            # Batch upsert to vector database
            vector_db.upsert(
                vectors=embeddings,
                ids=ids,
                metadata=metadata,
                collection=collection
            )

            # Save index
            if hasattr(vector_db, 'save'):
                vector_db.save(settings.faiss_index_path)

            results["chunks_created"] = len(all_chunks)

        logger.info(
            f"Batch processing completed: "
            f"{results['success']}/{results['total']} success, "
            f"{results['failed']} failed, "
            f"{results['chunks_created']} chunks created"
        )

        return results

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        # Re-raise to trigger Celery retry mechanism
        raise


@celery_app.task(bind=True, name='tasks.process_document_batch_retry')
def process_document_batch_retry(
    self,
    task_id: str,
    failed_doc_ids: List[str],
    collection: str = "default"
) -> Dict[str, Any]:
    """
    Retry processing of failed documents from a batch

    Args:
        self: Celery task instance
        task_id: Original task ID
        failed_doc_ids: List of document IDs that failed
        collection: Collection name

    Returns:
        Retry results dictionary
    """
    # This would typically load failed documents from storage
    # and retry their processing
    # For now, it's a placeholder for future implementation
    logger.info(f"Retry task {task_id} for {len(failed_doc_ids)} documents")

    return {
        "retry_task_id": task_id,
        "retry_count": len(failed_doc_ids),
        "status": "retry_completed"
    }
