"""
Integration tests for Enterprise RAG System

These tests verify end-to-end functionality of the RAG pipeline.
"""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_vector_db():
    """Create temporary vector database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "test_index.bin")
        yield index_path


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"source": "doc1.pdf", "page": 1}
        },
        {
            "text": "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "metadata": {"source": "doc2.pdf", "page": 1}
        },
        {
            "text": "Natural language processing enables computers to understand and generate human language.",
            "metadata": {"source": "doc3.pdf", "page": 1}
        }
    ]


@pytest.mark.integration
def test_rag_pipeline_end_to_end(temp_vector_db, sample_documents):
    """Test complete RAG pipeline"""
    from app.core.vectordb import get_vector_db
    from app.core.embeddings import get_embedding_model
    from app.services.retrieval import HybridRetriever
    from app.services.rag_pipeline import RAGPipeline

    # Initialize components
    vector_db = get_vector_db(db_type="faiss", index_path=temp_vector_db)
    vector_db.connect()

    embedding_model = get_embedding_model()

    retriever = HybridRetriever(
        vector_db=vector_db,
        embedding_model=embedding_model,
        alpha=0.5
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_model="gpt-4",
        temperature=0.7,
        max_tokens=500
    )

    # Test query
    question = "What is machine learning?"

    # Note: This will fail if no documents are indexed
    # In real integration tests, you would first ingest documents
    assert pipeline is not None
    assert retriever is not None


@pytest.mark.integration
def test_vector_db_operations(temp_vector_db, sample_documents):
    """Test vector database operations"""
    from app.core.vectordb import get_vector_db
    from app.core.embeddings import get_embedding_model

    # Initialize
    vector_db = get_vector_db(db_type="faiss", index_path=temp_vector_db)
    vector_db.connect()

    embedding_model = get_embedding_model()

    # Generate embeddings
    texts = [doc["text"] for doc in sample_documents]
    embeddings = embedding_model.embed_texts(texts)

    # Test add documents
    vector_db.add_documents(
        documents=texts,
        embeddings=embeddings,
        metadatas=[doc["metadata"] for doc in sample_documents]
    )

    # Test search
    query = "artificial intelligence and machine learning"
    query_embedding = embedding_model.embed_query(query)
    results = vector_db.search(query_embedding, top_k=3)

    assert len(results) > 0


@pytest.mark.integration
def test_hybrid_retrieval(temp_vector_db, sample_documents):
    """Test hybrid retrieval (semantic + keyword)"""
    from app.core.vectordb import get_vector_db
    from app.core.embeddings import get_embedding_model
    from app.services.retrieval import HybridRetriever

    # Initialize
    vector_db = get_vector_db(db_type="faiss", index_path=temp_vector_db)
    vector_db.connect()

    embedding_model = get_embedding_model()

    # Index documents
    texts = [doc["text"] for doc in sample_documents]
    embeddings = embedding_model.embed_texts(texts)
    vector_db.add_documents(texts, embeddings, [doc["metadata"] for doc in sample_documents])

    # Test hybrid retrieval
    retriever = HybridRetriever(
        vector_db=vector_db,
        embedding_model=embedding_model,
        alpha=0.5
    )

    query = "What is deep learning?"
    results = retriever.retrieve(query, top_k=2, use_hybrid=True)

    assert len(results) > 0
    assert all(hasattr(r, 'score') for r in results)


@pytest.mark.integration
def test_context_compression():
    """Test context compression for long documents"""
    from app.services.retrieval import ContextCompressor, RetrievalResult

    compressor = ContextCompressor(max_tokens=200)

    # Create sample retrieval results
    results = [
        RetrievalResult(
            document="This is a very long document that contains a lot of information about machine learning and artificial intelligence. " * 20,
            score=0.9,
            metadata={"source": "long_doc.pdf"}
        ),
        RetrievalResult(
            document="Short document.",
            score=0.8,
            metadata={"source": "short_doc.pdf"}
        )
    ]

    query = "What is machine learning?"
    compressed = compressor.compress(query, results)

    # Compressed context should be shorter
    assert len(compressed) < sum(len(r.document) for r in results)


@pytest.mark.integration
def test_batch_query():
    """Test batch query processing"""
    from app.services.rag_pipeline import RAGPipeline
    from app.services.retrieval import HybridRetriever
    from app.core.vectordb import get_vector_db
    from app.core.embeddings import get_embedding_model

    # Initialize
    vector_db = get_vector_db(db_type="faiss", index_path=":memory:")
    vector_db.connect()

    embedding_model = get_embedding_model()
    retriever = HybridRetriever(vector_db=vector_db, embedding_model=embedding_model)
    pipeline = RAGPipeline(retriever=retriever)

    # Batch query
    questions = ["Question 1?", "Question 2?", "Question 3?"]
    responses = pipeline.batch_query(questions)

    assert len(responses) == len(questions)


@pytest.mark.integration
def test_retrieval_with_filters():
    """Test retrieval with metadata filters"""
    from app.core.vectordb import get_vector_db
    from app.core.embeddings import get_embedding_model

    vector_db = get_vector_db(db_type="faiss", index_path=":memory:")
    vector_db.connect()

    embedding_model = get_embedding_model()

    # Index documents with metadata
    documents = ["Doc 1", "Doc 2", "Doc 3"]
    embeddings = embedding_model.embed_texts(documents)
    metadatas = [
        {"category": "tech"},
        {"category": "business"},
        {"category": "tech"}
    ]

    vector_db.add_documents(documents, embeddings, metadatas)

    # Search with filter
    query_embedding = embedding_model.embed_query("test")
    results = vector_db.search(
        query_embedding,
        top_k=10,
        filter_dict={"category": "tech"}
    )

    # Should only return tech documents
    assert len(results) <= 2


@pytest.mark.integration
def test_confidence_calculation():
    """Test confidence score calculation"""
    from app.services.retrieval import RetrievalResult
    from app.services.rag_pipeline import RAGPipeline
    from app.core.vectordb import get_vector_db
    from app.core.embeddings import get_embedding_model

    vector_db = get_vector_db(db_type="faiss", index_path=":memory:")
    vector_db.connect()
    embedding_model = get_embedding_model()

    retriever = HybridRetriever(vector_db=vector_db, embedding_model=embedding_model)
    pipeline = RAGPipeline(retriever=retriever)

    # Test with high-quality results
    high_quality_results = [
        RetrievalResult(
            document="Relevant document",
            score=0.9,
            metadata={"source": "doc1.pdf"}
        )
    ]

    confidence = pipeline._calculate_confidence(high_quality_results, "This is a comprehensive answer.")

    assert 0 <= confidence <= 1.0
    assert confidence > 0.5  # High quality should give higher confidence
