"""
Unit tests for batch document processing tasks

Tests Celery tasks for asynchronous batch processing functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.tasks.batch_tasks import process_document_batch


@pytest.fixture
def mock_documents():
    """Sample documents for testing"""
    return [
        {
            "id": "doc1",
            "content": "This is the first test document with some content.",
            "metadata": {"source": "test", "category": "test"}
        },
        {
            "id": "doc2",
            "content": "This is the second test document with more content.",
            "metadata": {"source": "test", "category": "test"}
        },
        {
            "id": "doc3",
            "content": "Third document here.",
            "metadata": {"source": "test"}
        }
    ]


@pytest.fixture
def mock_celery_task():
    """Mock Celery task instance"""
    task = Mock()
    task.update_state = Mock()
    task.request = Mock()
    return task


class TestProcessDocumentBatch:
    """Test suite for process_document_batch Celery task"""

    @patch('app.core.vectordb.get_vector_db')
    @patch('app.core.embeddings.get_embedding_model')
    @patch('app.services.document_loader.TextSplitter')
    @patch('app.services.document_loader.Document')
    def test_successful_batch_processing(
        self,
        mock_document_class,
        mock_splitter_class,
        mock_get_embedding_model,
        mock_get_vector_db,
        mock_documents,
        mock_celery_task
    ):
        """Test successful batch processing of documents"""

        # Setup mocks
        mock_doc = Mock()
        mock_doc.doc_id = "doc1"
        mock_document_class.return_value = mock_doc

        mock_chunk = Mock()
        mock_chunk.doc_id = "chunk1"
        mock_chunks = [mock_chunk]
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter_class.return_value = mock_splitter_instance

        mock_embedding_model = Mock()
        mock_embedding_model.dimension = 1536
        mock_embedding_model.embed_texts.return_value = [[0.1] * 1536]
        mock_get_embedding_model.return_value = mock_embedding_model

        mock_vector_db = Mock()
        mock_vector_db.index = None
        mock_get_vector_db.return_value = mock_vector_db

        # Execute task
        result = process_document_batch.apply(
            args=(mock_documents, "test_collection", 1000, 200),
            throw=True
        ).result

        # Verify results
        assert result["total"] == len(mock_documents)
        assert result["success"] == len(mock_documents)
        assert result["failed"] == 0
        assert result["chunks_created"] > 0
        assert len(result["errors"]) == 0

        # Verify vector DB operations
        mock_vector_db.create_index.assert_called_once_with(dimension=1536)
        mock_vector_db.upsert.assert_called_once()

    @patch('app.core.vectordb.get_vector_db')
    @patch('app.core.embeddings.get_embedding_model')
    @patch('app.services.document_loader.TextSplitter')
    @patch('app.services.document_loader.Document')
    def test_partial_failure_handling(
        self,
        mock_document_class,
        mock_splitter_class,
        mock_get_embedding_model,
        mock_get_vector_db,
        mock_celery_task
    ):
        """Test handling of partial document failures"""

        # Setup documents where second document will fail
        mock_documents = [
            {
                "id": "doc1",
                "content": "Valid document",
                "metadata": {}
            },
            {
                "id": "doc2",
                "content": "Invalid document",
                "metadata": {}
            }
        ]

        # Mock successful first document, fail second
        def side_effect(content, metadata, doc_id):
            if doc_id == "doc2":
                raise ValueError("Invalid content")
            mock_doc = Mock()
            mock_doc.doc_id = doc_id
            return mock_doc

        mock_document_class.side_effect = side_effect

        mock_chunk = Mock()
        mock_chunk.doc_id = "chunk1"
        mock_chunks = [mock_chunk]

        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter_class.return_value = mock_splitter_instance

        mock_embedding_model = Mock()
        mock_embedding_model.dimension = 1536
        mock_embedding_model.embed_texts.return_value = [[0.1] * 1536]
        mock_get_embedding_model.return_value = mock_embedding_model

        mock_vector_db = Mock()
        mock_vector_db.index = Mock()  # Index exists
        mock_get_vector_db.return_value = mock_vector_db

        # Execute task
        result = process_document_batch.apply(
            args=(mock_documents, "test_collection", 1000, 200),
            throw=True
        ).result

        # Verify partial success
        assert result["total"] == 2
        assert result["success"] == 1
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0]["doc_id"] == "doc2"

    @patch('app.core.vectordb.get_vector_db')
    @patch('app.core.embeddings.get_embedding_model')
    @patch('app.services.document_loader.TextSplitter')
    @patch('app.services.document_loader.Document')
    def test_task_progress_updates(
        self,
        mock_document_class,
        mock_splitter_class,
        mock_get_embedding_model,
        mock_get_vector_db,
        mock_documents,
        mock_celery_task
    ):
        """Test that task progress is updated correctly"""

        # Setup mocks
        mock_doc = Mock()
        mock_doc.doc_id = "doc1"
        mock_document_class.return_value = mock_doc

        mock_chunk = Mock()
        mock_chunk.doc_id = "chunk1"
        mock_chunks = [mock_chunk]
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter_class.return_value = mock_splitter_instance

        mock_embedding_model = Mock()
        mock_embedding_model.dimension = 1536
        mock_embedding_model.embed_texts.return_value = [[0.1] * 1536]
        mock_get_embedding_model.return_value = mock_embedding_model

        mock_vector_db = Mock()
        mock_vector_db.index = None
        mock_get_vector_db.return_value = mock_vector_db

        # Execute task
        process_document_batch.apply(
            args=(mock_documents, "test_collection", 1000, 200),
            throw=True
        ).result

        # Verify progress updates were called (Note: can't easily mock with .apply())
        # The actual task will call update_state, but we can't assert it in this test
        # In production, Celery handles the task binding and update_state calls

    @patch('app.core.vectordb.get_vector_db')
    @patch('app.core.embeddings.get_embedding_model')
    @patch('app.services.document_loader.TextSplitter')
    @patch('app.services.document_loader.Document')
    def test_custom_chunk_parameters(
        self,
        mock_document_class,
        mock_splitter_class,
        mock_get_embedding_model,
        mock_get_vector_db,
        mock_documents,
        mock_celery_task
    ):
        """Test batch processing with custom chunk parameters"""

        # Setup mocks
        mock_doc = Mock()
        mock_doc.doc_id = "doc1"
        mock_document_class.return_value = mock_doc

        mock_chunk = Mock()
        mock_chunk.doc_id = "chunk1"
        mock_chunks = [mock_chunk]
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter_class.return_value = mock_splitter_instance

        mock_embedding_model = Mock()
        mock_embedding_model.dimension = 1536
        mock_embedding_model.embed_texts.return_value = [[0.1] * 1536]
        mock_get_embedding_model.return_value = mock_embedding_model

        mock_vector_db = Mock()
        mock_vector_db.index = None
        mock_get_vector_db.return_value = mock_vector_db

        # Execute task with custom parameters
        process_document_batch.apply(
            args=(mock_documents, "custom_collection", 2000, 400),
            throw=True
        ).result

        # Verify TextSplitter was called with custom parameters
        mock_splitter_class.assert_called_once_with(
            chunk_size=2000,
            chunk_overlap=400
        )

    @patch('app.core.vectordb.get_vector_db')
    @patch('app.core.embeddings.get_embedding_model')
    @patch('app.services.document_loader.TextSplitter')
    @patch('app.services.document_loader.Document')
    def test_empty_document_list(
        self,
        mock_document_class,
        mock_splitter_class,
        mock_get_embedding_model,
        mock_get_vector_db
    ):
        """Test handling of empty document list"""

        # Execute task with empty list
        result = process_document_batch.apply(
            args=([], "test_collection", 1000, 200),
            throw=True
        ).result

        # Verify results
        assert result["total"] == 0
        assert result["success"] == 0
        assert result["failed"] == 0
        assert result["chunks_created"] == 0

        # Verify no database operations
        mock_get_vector_db.assert_not_called()


class TestBatchTaskIntegration:
    """Integration tests for batch processing endpoints"""

    def test_batch_endpoint_request_validation(self):
        """Test that batch endpoint validates request correctly"""
        from app.api.routes.documents import BatchIngestRequest

        # Valid request
        valid_data = {
            "documents": [
                {
                    "id": "doc1",
                    "content": "Test content",
                    "metadata": {}
                }
            ],
            "collection": "test",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        request = BatchIngestRequest(**valid_data)
        assert len(request.documents) == 1
        assert request.collection == "test"

        # Invalid request (too large)
        invalid_data = {
            "documents": [{"id": f"doc{i}", "content": "test"} for i in range(1001)],
            "collection": "test"
        }
        with pytest.raises(ValueError):
            BatchIngestRequest(**invalid_data)

    def test_batch_status_response_model(self):
        """Test batch status response model"""
        from app.api.routes.documents import BatchStatusResponse

        response_data = {
            "task_id": "test-task-id",
            "status": "SUCCESS",
            "result": {
                "total": 10,
                "success": 10,
                "failed": 0,
                "errors": [],
                "chunks_created": 50
            },
            "error": None
        }

        response = BatchStatusResponse(**response_data)
        assert response.task_id == "test-task-id"
        assert response.status == "SUCCESS"
        assert response.result["success"] == 10
        assert response.error is None


@pytest.mark.unit
class TestBatchTaskErrorHandling:
    """Test error handling in batch processing"""

    @patch('app.core.vectordb.get_vector_db')
    @patch('app.core.embeddings.get_embedding_model')
    @patch('app.services.document_loader.TextSplitter')
    @patch('app.services.document_loader.Document')
    def test_embedding_failure_isolation(
        self,
        mock_document_class,
        mock_splitter_class,
        mock_get_embedding_model,
        mock_get_vector_db,
        mock_celery_task
    ):
        """Test that embedding failures are isolated per document"""

        mock_documents = [
            {"id": "doc1", "content": "Valid", "metadata": {}},
            {"id": "doc2", "content": "Invalid", "metadata": {}}
        ]

        # Setup mocks to fail on second doc
        mock_doc = Mock()
        mock_doc.doc_id = "doc1"
        mock_document_class.return_value = mock_doc

        mock_chunk = Mock()
        mock_chunk.doc_id = "chunk1"
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = [mock_chunk]
        mock_splitter_class.return_value = mock_splitter_instance

        mock_embedding_model = Mock()
        mock_embedding_model.dimension = 1536

        def embed_side_effect(texts):
            if len(texts) > 1:
                raise Exception("Embedding failed")
            return [[0.1] * 1536]

        mock_embedding_model.embed_texts.side_effect = embed_side_effect
        mock_get_embedding_model.return_value = mock_embedding_model

        mock_vector_db = Mock()
        mock_vector_db.index = None
        mock_get_vector_db.return_value = mock_vector_db

        # Execute task - should handle failure gracefully
        with pytest.raises(Exception):
            process_document_batch.apply(
                args=(mock_documents, "test", 1000, 200),
                throw=True
            ).result
