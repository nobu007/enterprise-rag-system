"""
Unit tests for VectorDB multi-collection support
"""

import pytest
import tempfile
import os
import json
import logging
import pickle
from unittest.mock import Mock, call
from app.core.vectordb import FAISSVectorDB, PineconeVectorDB


@pytest.mark.parametrize(
    "kwargs, namespace",
    [({}, ""), ({"collection": "default"}, ""), ({"collection": "hr"}, "hr")],
)
def test_pinecone_collection_namespace_round_trip(kwargs, namespace):
    """Every batch, search and delete must target the same collection."""
    db = PineconeVectorDB("test-key", "test-environment", "test-index")
    db.index = Mock()
    vectors = [[0.1, 0.2, 0.3]] * 101
    ids = [f"doc-{i}" for i in range(101)]
    metadata = [{"text": f"Content {i}"} for i in range(101)]
    items = list(zip(ids, vectors, metadata))

    db.upsert(vectors, ids, metadata, **kwargs)

    assert db.index.upsert.call_args_list == [
        call(vectors=items[:100], namespace=namespace),
        call(vectors=items[100:], namespace=namespace),
    ]
    db.index.query.return_value.matches = [
        Mock(id=ids[0], score=0.9, metadata=metadata[0])
    ]
    results = db.search(vectors[0], filter_dict={"kind": "text"}, **kwargs)
    db.index.query.assert_called_once_with(
        vector=vectors[0], top_k=5, filter={"kind": "text"},
        include_metadata=True, namespace=namespace,
    )
    assert results[0].id == ids[0]
    assert results[0].text == metadata[0]["text"]

    db.delete(ids, **kwargs)
    db.index.delete.assert_called_once_with(ids=ids, namespace=namespace)


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
        # save() writes JSON since the JSON metadata format landed; without
        # this the rmdir above silently no-ops and leaks the temp dir.
        if os.path.exists(index_path + ".metadata.json"):
            os.remove(index_path + ".metadata.json")
        os.rmdir(temp_dir)
    except Exception:
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


class TestCollectionLogInjectionVectorDB:
    """CWE-117: a client-controlled ``collection`` must not forge log lines.

    ``collection`` is the ``QueryRequest.collection`` body field; the query
    route passes it straight through ``RAGPipeline.query`` ->
    ``HybridRetriever.semantic_search`` -> ``FAISSVectorDB.search``. When the
    collection does not exist, ``search`` logs the name at WARNING. A CR/LF in
    the name terminates the real log line and starts a forged one -- the same
    log-injection class as the ``collection`` field in rag_pipeline (91d9640)
    and the ``query`` body field (43166fa). 91d9640 sanitised the field only at
    the rag_pipeline retrieval log; this is the live FAISS query-path site that
    that sweep missed. The validation middleware scans body values for
    XSS/SQL/path/command but NOT control chars, so CR/LF reaches this logger
    unfiltered.
    """

    def test_collection_crlf_neutralised_in_not_found_log(
        self, temp_vector_db, sample_vectors
    ):
        # Capture the vectordb logger's records directly (robust to root-handler
        # / level config, unlike caplog propagation).
        vectordb_logger = logging.getLogger("app.core.vectordb")
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = _Capture(logging.DEBUG)
        vectordb_logger.addHandler(handler)
        vectordb_logger.setLevel(logging.DEBUG)
        try:
            results = temp_vector_db.search(
                query_vector=sample_vectors[0],
                top_k=5,
                collection="x\nFAKE LOG line\r",
            )
        finally:
            vectordb_logger.removeHandler(handler)

        # Non-existent collection still returns no results (behavior unchanged).
        assert results == []

        not_found_msgs = [
            r.getMessage()
            for r in captured
            if "not found" in r.getMessage()
        ]
        assert not_found_msgs, "'Collection ... not found' warning was not emitted"
        msg = not_found_msgs[0]
        # No raw CR/LF survives -> the forged "FAKE LOG line" cannot start a
        # new log line. Pre-fix the raw chr(10)/chr(13) were present.
        assert "\n" not in msg
        assert "\r" not in msg
        # Escaped form is present -> value preserved, only log rep changed.
        assert "x\\nFAKE LOG line\\r" in msg



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


def test_vector_db_default_collection_persistence(temp_vector_db, sample_vectors, sample_metadata):
    """The default collection must survive a save -> connect round trip.

    connect() loads the JSON metadata into the per-collection stores, not
    just the legacy attributes. Otherwise a follow-up upsert into "default"
    raises KeyError ("default" is already in self.indices so the on-demand
    bootstrap in _get_or_create_collection is skipped) and search silently
    returns no results.
    """

    temp_vector_db.upsert(
        vectors=[sample_vectors[0]],
        ids=["doc1"],
        metadata=[sample_metadata[0]],
        collection="default"
    )

    # Save the index (writes the JSON metadata format)
    temp_vector_db.save(temp_vector_db.index_path)

    # Create a new VectorDB instance and load from disk
    new_db = FAISSVectorDB(index_path=temp_vector_db.index_path)
    new_db.connect()

    # Follow-up upsert into the loaded default collection must not KeyError
    new_db.upsert(
        vectors=[sample_vectors[1]],
        ids=["doc2"],
        metadata=[sample_metadata[1]],
        collection="default"
    )

    # Both documents must be searchable with their metadata intact
    results = new_db.search(
        query_vector=sample_vectors[0],
        top_k=5,
        collection="default"
    )
    assert {r.id for r in results} == {"doc1", "doc2"}
    by_id = {r.id: r for r in results}
    assert by_id["doc1"].text == sample_metadata[0]["text"]
    assert by_id["doc2"].text == sample_metadata[1]["text"]


def test_vector_db_default_collection_search_after_connect(temp_vector_db, sample_vectors, sample_metadata):
    """connect() alone must make saved default-collection rows searchable.

    Pins the silent-drop half of the Issue 9 fix independently of upsert:
    search resolves hits through the per-collection idx_to_id/metadata
    stores, so a fresh instance whose connect() left them empty returned []
    for every query instead of raising. With the stores populated, doc1
    must come back with its metadata even though this instance never
    upserts anything.
    """

    temp_vector_db.upsert(
        vectors=[sample_vectors[0]],
        ids=["doc1"],
        metadata=[sample_metadata[0]],
        collection="default"
    )
    temp_vector_db.save(temp_vector_db.index_path)

    new_db = FAISSVectorDB(index_path=temp_vector_db.index_path)
    new_db.connect()

    # Search-only: no upsert primes the stores on this instance.
    results = new_db.search(
        query_vector=sample_vectors[0],
        top_k=5,
        collection="default"
    )
    assert [r.id for r in results] == ["doc1"]
    assert results[0].text == sample_metadata[0]["text"]
    assert results[0].metadata["filename"] == sample_metadata[0]["filename"]


def test_vector_db_legacy_pickle_metadata_migration(temp_vector_db, sample_vectors, sample_metadata):
    """Legacy pickle metadata must stay searchable and migrate to JSON on save.

    Covers the sibling branch of the Issue 9 fix: the legacy pickle path
    must also feed the per-collection stores (search keeps working), and
    the next save() re-serialises them as JSON so a later instance reads
    the safe format again.
    """

    temp_vector_db.upsert(
        vectors=[sample_vectors[0]],
        ids=["doc1"],
        metadata=[sample_metadata[0]],
        collection="default"
    )
    temp_vector_db.save(temp_vector_db.index_path)

    # Rewind the on-disk metadata to the legacy pickle layout (int keys,
    # as pickle.dump preserved them before the JSON format existed).
    json_path = temp_vector_db.index_path + ".metadata.json"
    pkl_path = temp_vector_db.index_path + ".metadata.pkl"
    with open(json_path, encoding="utf-8") as f:
        legacy = json.load(f)
    legacy["idx_to_id"] = {int(k): v for k, v in legacy["idx_to_id"].items()}
    os.remove(json_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(legacy, f)

    legacy_db = FAISSVectorDB(index_path=temp_vector_db.index_path)
    legacy_db.connect()

    results = legacy_db.search(
        query_vector=sample_vectors[0],
        top_k=5,
        collection="default"
    )
    assert [r.id for r in results] == ["doc1"]

    # The next save() migrates the metadata to the JSON format and a third
    # instance reads doc1 back from it.
    legacy_db.save(legacy_db.index_path)
    assert os.path.exists(json_path)

    migrated_db = FAISSVectorDB(index_path=temp_vector_db.index_path)
    migrated_db.connect()
    results = migrated_db.search(
        query_vector=sample_vectors[0],
        top_k=5,
        collection="default"
    )
    assert [r.id for r in results] == ["doc1"]
    assert results[0].text == sample_metadata[0]["text"]


def test_upsert_same_id_updates_instead_of_duplicating(temp_vector_db, sample_vectors, sample_metadata):
    """Re-upserting a content-hash id must replace, not append.

    Document ids are content hashes, so re-running an ingest re-upserts the
    same ids; the previous add-only behavior duplicated every vector and
    search returned the same document once per stale copy.
    """
    temp_vector_db.upsert(
        vectors=sample_vectors[:2],
        ids=["doc1", "doc2"],
        metadata=[sample_metadata[0], sample_metadata[1]],
        collection="default",
    )

    updated_meta = {"filename": "doc1.pdf", "page": 1, "text": "Content of document 1 v2"}
    temp_vector_db.upsert(
        vectors=[sample_vectors[0]],
        ids=["doc1"],
        metadata=[updated_meta],
        collection="default",
    )

    assert temp_vector_db.index.ntotal == 2
    results = temp_vector_db.search(
        query_vector=sample_vectors[0], top_k=10, collection="default"
    )
    assert sorted(r.id for r in results) == ["doc1", "doc2"]
    assert [r for r in results if r.id == "doc1"][0].text == "Content of document 1 v2"
    assert temp_vector_db.get_stats()["collections"]["default"]["total_vectors"] == 2


def test_upsert_mixed_batch_keeps_new_and_updated_ids(temp_vector_db, sample_vectors, sample_metadata):
    """A batch may contain both new and already-ingested ids."""
    temp_vector_db.upsert(
        vectors=[sample_vectors[0]],
        ids=["doc1"],
        metadata=[sample_metadata[0]],
        collection="default",
    )

    temp_vector_db.upsert(
        vectors=sample_vectors[:2],
        ids=["doc1", "doc3"],
        metadata=[sample_metadata[0], sample_metadata[2]],
        collection="default",
    )

    assert temp_vector_db.index.ntotal == 2
    results = temp_vector_db.search(
        query_vector=sample_vectors[2], top_k=1, collection="default"
    )
    assert [r.id for r in results] == ["doc3"]


def test_upsert_update_survives_save_and_reload(temp_vector_db, sample_vectors, sample_metadata):
    """The rebuilt mappings persist consistently through the JSON metadata."""
    temp_vector_db.upsert(
        vectors=sample_vectors[:2],
        ids=["doc1", "doc2"],
        metadata=[sample_metadata[0], sample_metadata[1]],
        collection="default",
    )
    temp_vector_db.upsert(
        vectors=[sample_vectors[0]],
        ids=["doc1"],
        metadata=[{"text": "updated"}],
        collection="default",
    )
    temp_vector_db.save(temp_vector_db.index_path)

    reloaded = FAISSVectorDB(index_path=temp_vector_db.index_path)
    reloaded.connect()
    results = reloaded.search(
        query_vector=sample_vectors[0], top_k=10, collection="default"
    )
    assert sorted(r.id for r in results) == ["doc1", "doc2"]
    assert [r for r in results if r.id == "doc1"][0].text == "updated"


def test_upsert_update_preserves_euclidean_metric(temp_vector_db):
    """A rebuilt L2 collection must stay L2, not silently become cosine."""
    temp_vector_db.create_index(dimension=2, metric="euclidean", collection="l2c")
    temp_vector_db.upsert(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        ids=["a", "b"],
        metadata=[{"text": "a"}, {"text": "b"}],
        collection="l2c",
    )
    temp_vector_db.upsert(
        vectors=[[1.0, 0.0]],
        ids=["a"],
        metadata=[{"text": "a2"}],
        collection="l2c",
    )

    import faiss
    assert temp_vector_db.indices["l2c"].metric_type == faiss.METRIC_L2
    assert temp_vector_db.indices["l2c"].ntotal == 2
    results = temp_vector_db.search([1.0, 0.0], top_k=10, collection="l2c")
    assert sorted(r.id for r in results) == ["a", "b"]


def test_upsert_full_reingest_replaces_every_id(temp_vector_db, sample_vectors, sample_metadata):
    """Re-running a full ingest (every id already known) must stay idempotent.

    The mixed-batch pin exercises the rebuild branch that keeps nothing only
    incidentally; this pins it directly: both ids are dropped, the rebuilt
    index starts empty, and both re-added docs carry the fresh metadata.
    """
    temp_vector_db.upsert(
        vectors=sample_vectors[:2],
        ids=["doc1", "doc2"],
        metadata=[sample_metadata[0], sample_metadata[1]],
        collection="default",
    )
    temp_vector_db.upsert(
        vectors=sample_vectors[:2],
        ids=["doc1", "doc2"],
        metadata=[
            {"filename": "doc1.pdf", "page": 1, "text": "Content of document 1 v2"},
            {"filename": "doc2.pdf", "page": 2, "text": "Content of document 2 v2"},
        ],
        collection="default",
    )

    assert temp_vector_db.index.ntotal == 2
    results = temp_vector_db.search(sample_vectors[0], top_k=10, collection="default")
    assert sorted(r.id for r in results) == ["doc1", "doc2"]
    assert [r for r in results if r.id == "doc1"][0].text == "Content of document 1 v2"
    assert [r for r in results if r.id == "doc2"][0].text == "Content of document 2 v2"
    assert temp_vector_db.get_stats()["collections"]["default"]["total_vectors"] == 2


def test_upsert_rebuild_in_named_collection_leaves_default_intact(temp_vector_db, sample_vectors, sample_metadata):
    """A rebuild in one collection must not disturb siblings or the default alias."""
    temp_vector_db.upsert(
        vectors=sample_vectors[:2],
        ids=["doc1", "doc2"],
        metadata=[sample_metadata[0], sample_metadata[1]],
        collection="default",
    )
    temp_vector_db.create_index(dimension=2, metric="euclidean", collection="l2c")
    temp_vector_db.upsert(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        ids=["a", "b"],
        metadata=[{"text": "a"}, {"text": "b"}],
        collection="l2c",
    )

    # Triggers the rebuild path inside "l2c" only.
    temp_vector_db.upsert(
        vectors=[[1.0, 0.0]],
        ids=["a"],
        metadata=[{"text": "a2"}],
        collection="l2c",
    )

    import faiss
    assert temp_vector_db.indices["l2c"].metric_type == faiss.METRIC_L2
    assert temp_vector_db.indices["l2c"].ntotal == 2
    assert [r.id for r in temp_vector_db.search([1.0, 0.0], top_k=10, collection="l2c")] == ["a", "b"]

    # "default" keeps its index, alias, mappings and metadata untouched.
    assert temp_vector_db.index is temp_vector_db.indices["default"]
    assert temp_vector_db.indices["default"].ntotal == 2
    results = temp_vector_db.search(sample_vectors[0], top_k=10, collection="default")
    assert sorted(r.id for r in results) == ["doc1", "doc2"]
    assert [r for r in results if r.id == "doc1"][0].text == sample_metadata[0]["text"]


def test_upsert_duplicate_ids_within_one_batch_stay_unique(temp_vector_db, sample_vectors):
    """Desired contract: the last occurrence of an in-batch duplicate id wins."""
    temp_vector_db.upsert(
        vectors=[sample_vectors[0], sample_vectors[1]],
        ids=["x", "x"],
        metadata=[{"text": "v1"}, {"text": "v2"}],
        collection="default",
    )

    assert temp_vector_db.index.ntotal == 1
    results = temp_vector_db.search(sample_vectors[1], top_k=10, collection="default")
    assert [r.id for r in results] == ["x"]
    assert results[0].text == "v2"


def test_delete_drops_id_and_keeps_collection_searchable(temp_vector_db, sample_vectors, sample_metadata):
    """delete() must rebuild without the id instead of warning and keeping it.

    The removed id must vanish from the index, the id<->idx mappings and the
    metadata store; unknown ids and unknown collections are no-ops.
    """
    temp_vector_db.upsert(
        vectors=sample_vectors[:2],
        ids=["doc1", "doc2"],
        metadata=[sample_metadata[0], sample_metadata[1]],
        collection="default",
    )

    temp_vector_db.delete(["doc1"], collection="default")

    assert temp_vector_db.index.ntotal == 1
    assert temp_vector_db.id_to_idx_mappings["default"] == {"doc2": 0}
    assert temp_vector_db.idx_to_id_mappings["default"] == {0: "doc2"}
    assert "doc1" not in temp_vector_db.metadata_stores["default"]
    results = temp_vector_db.search(sample_vectors[1], top_k=10, collection="default")
    assert [r.id for r in results] == ["doc2"]

    # Unknown ids and unknown collections must not raise or disturb the index.
    temp_vector_db.delete(["ghost"], collection="default")
    temp_vector_db.delete(["doc2"], collection="missing")
    assert temp_vector_db.index.ntotal == 1
    assert [r.id for r in temp_vector_db.search(sample_vectors[1], top_k=10, collection="default")] == ["doc2"]
