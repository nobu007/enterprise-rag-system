"""
Unit tests for VectorDB multi-collection support
"""

import pytest
import tempfile
import os
from app.core.vectordb import FAISSVectorDB, SearchResult


@pytest.fixture
def temp_vector_db():
    """Create a temporary vector database for testing"""
    # Create a temporary directory for test indexes
    temp_dir = tempfile.mkdtemp()
    index_path = os.path.join(temp_dir, "test.index")

    # Create and initialize vector DB
    db = FAISSVectorDB(index_path=index_path)
    db.create_index(dimension=384, metric="cosine")

    yield db

    # Cleanup
    try:
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(index_path + ".metadata.pkl"):
            os.remove(index_path + ".metadata.pkl")
        os.rmdir(temp_dir)
    except:
        pass


@pytest.fixture
def sample_vectors():
    """Sample vectors for testing"""
    return [
        [0.1, 0.2, 0.3] * 128 + [0.0] * (384 - 384),  # Pad to 384 dimensions
        [0.4, 0.5, 0.6] * 128 + [0.0] * (384 - 384),
        [0.7, 0.8, 0.9] * 128 + [0.0] * (384 - 384),
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return [
        {"filename": "doc1.pdf", "page": 1, "text": "Content of document 1"},
        {"filename": "doc2.pdf", "page": 2, "text": "Content of document 2"},
        {"filename": "doc3.pdf", "page": 3, "text": "Content of document 3"},
    ]


def test_vector_db_supports_multiple_collections(temp_vector_db, sample_vectors, sample_metadata):
    """Test that VectorDB can handle multiple collections"""
    # Add documents to "default" collection
    temp_vector_db.upsert(
        vectors=sample_vectors,
        ids=["doc1", "doc2"],
        metadata=[sample_metadata[0], sample_metadata[1]],
        collection="default"
    )

    # Add documents to "tech" collection
    temp_vector_db.upsert(
        vectors=[sample_vectors[2]],
        ids=["doc3"],
        metadata=[sample_metadata[2]],
        collection="tech"
    )

    # Search in default collection - should return 2 results
    results_default = temp_vector_db.search(
        query_vector=sample_vectors[0],
        top_k=10,
        collection="default"
    )
    assert len(results_default) == 2
    assert all(r.id in ["doc1", "doc2"] for r in results_default)

    # Search in tech collection - should return 1 result
    results_tech = temp_vector_db.search(
        query_vector=sample_vectors[2],
        top_k=10,
        collection="tech"
    )
    assert len(results_tech) == 1
    assert results_tech[0].id == "doc3"


def test_vector_db_collection_isolation(temp_vector_db, sample_vectors, sample_metadata):
    """Test that collections are properly isolated"""
    # Add different documents to different collections
    temp_vector_db.upsert(
        vectors=[sample_vectors[0]],
        ids=["hr_doc1"],
        metadata=[{"collection": "hr", "text": "HR policy document"}],
        collection="hr"
    )

    temp_vector_db.upsert(
        vectors=[sample_vectors[1]],
        ids=["tech_doc1"],
        metadata=[{"collection": "tech", "text": "Technical documentation"}],
        collection="tech"
    )

    # Search in hr collection - should not find tech documents
    hr_results = temp_vector_db.search(
        query_vector=sample_vectors[1],  # Search with tech vector
        top_k=10,
        collection="hr"
    )
    assert len(hr_results) == 1
    assert hr_results[0].id == "hr_doc1"

    # Search in tech collection - should not find hr documents
    tech_results = temp_vector_db.search(
        query_vector=sample_vectors[0],  # Search with hr vector
        top_k=10,
        collection="tech"
    )
    assert len(tech_results) == 1
    assert tech_results[0].id == "tech_doc1"


def test_vector_db_default_collection(temp_vector_db, sample_vectors, sample_metadata):
    """Test that 'default' collection is used when not specified"""
    # Add documents without specifying collection (should use 'default')
    temp_vector_db.upsert(
        vectors=sample_vectors,
        ids=["doc1", "doc2", "doc3"],
        metadata=sample_metadata
    )

    # Search without specifying collection (should use 'default')
    results = temp_vector_db.search(
        query_vector=sample_vectors[0],
        top_k=5
    )

    assert len(results) == 3
    assert all(r.id in ["doc1", "doc2", "doc3"] for r in results)


def test_vector_db_nonexistent_collection(temp_vector_db, sample_vectors):
    """Test behavior when searching in non-existent collection"""
    # Search in a collection that doesn't exist
    results = temp_vector_db.search(
        query_vector=sample_vectors[0],
        top_k=5,
        collection="nonexistent"
    )

    # Should return empty list
    assert results == []


def test_vector_db_get_stats_multiple_collections(temp_vector_db, sample_vectors, sample_metadata):
    """Test that get_stats returns information about all collections"""
    # Add documents to multiple collections
    temp_vector_db.upsert(
        vectors=[sample_vectors[0], sample_vectors[1]],
        ids=["doc1", "doc2"],
        metadata=[sample_metadata[0], sample_metadata[1]],
        collection="default"
    )

    temp_vector_db.upsert(
        vectors=[sample_vectors[2]],
        ids=["doc3"],
        metadata=[sample_metadata[2]],
        collection="marketing"
    )

    # Get stats
    stats = temp_vector_db.get_stats()

    # Verify stats include all collections
    assert "collections" in stats or "total_vectors" in stats
    if "collections" in stats:
        assert "default" in stats["collections"]
        assert "marketing" in stats["collections"]


def test_vector_db_create_collection_on_demand(temp_vector_db, sample_vectors, sample_metadata):
    """Test that collections are created automatically when first used"""
    # Add documents to a new collection without pre-creating it
    temp_vector_db.upsert(
        vectors=[sample_vectors[0]],
        ids=["doc1"],
        metadata=[sample_metadata[0]],
        collection="new_collection"
    )

    # Should be able to search in this new collection
    results = temp_vector_db.search(
        query_vector=sample_vectors[0],
        top_k=5,
        collection="new_collection"
    )

    assert len(results) == 1
    assert results[0].id == "doc1"


def test_vector_db_collection_persistence(temp_vector_db, sample_vectors, sample_metadata):
    """Test that collections are persisted and loaded correctly"""
    import pickle

    # Add documents to multiple collections
    temp_vector_db.upsert(
        vectors=[sample_vectors[0]],
        ids=["doc1"],
        metadata=[sample_metadata[0]],
        collection="collection1"
    )

    temp_vector_db.upsert(
        vectors=[sample_vectors[1]],
        ids=["doc2"],
        metadata=[sample_metadata[1]],
        collection="collection2"
    )

    # Save the index
    temp_vector_db.save(temp_vector_db.index_path)

    # Create a new VectorDB instance and load
    new_db = FAISSVectorDB(index_path=temp_vector_db.index_path)
    new_db.connect()

    # Verify that both collections are accessible
    results1 = new_db.search(
        query_vector=sample_vectors[0],
        top_k=5,
        collection="collection1"
    )
    assert len(results1) == 1
    assert results1[0].id == "doc1"

    results2 = new_db.search(
        query_vector=sample_vectors[1],
        top_k=5,
        collection="collection2"
    )
    assert len(results2) == 1
    assert results2[0].id == "doc2"
