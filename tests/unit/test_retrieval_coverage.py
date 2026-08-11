"""Coverage-focused tests for app/services/retrieval.py.

Targets branches the existing test_retrieval.py does not reach:
HybridRetriever.build_bm25_index ImportError fallback (rank_bm25
absent), the keyword-results RRF branch of hybrid_search (both the
overlap and keyword-only sub-branches), and the ContextCompressor
unknown-method fallback to truncation.

semantic_search returns vectordb SearchResult objects (.id/.text/
.metadata); keyword_search returns dicts with a 'document' (Document
with .doc_id/.content/.metadata). Those are mocked directly so no real
vector store or BM25 index is needed.
"""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from app.services.document_loader import Document
from app.services.retrieval import (
    ContextCompressor,
    HybridRetriever,
    RetrievalResult,
)


def _retriever():
    return HybridRetriever(vector_db=Mock(), embedding_model=Mock())


class TestBuildBm25ImportError:
    def test_missing_rank_bm25_disables_keyword_search(self, monkeypatch):
        # None sentinel -> the `from rank_bm25 import` raises ImportError.
        monkeypatch.setitem(sys.modules, "rank_bm25", None)
        retriever = _retriever()
        retriever.build_bm25_index(
            [Document(content="hello world", metadata={})]
        )
        # ImportError swallowed; index stays None (keyword search disabled).
        assert retriever.bm25_index is None


class TestHybridSearchKeywordBranch:
    def test_combines_semantic_and_keyword_via_rrf(self):
        retriever = _retriever()
        retriever.bm25_index = object()  # truthy -> keyword_search is called

        sem = [
            SimpleNamespace(
                id="d1", text="sem text", metadata={"source": "s1"},
                score=0.9,
            )
        ]
        # d1 overlaps a semantic result (L152-154); d2 keyword-only (L155-161).
        kw = [
            {
                "document": SimpleNamespace(
                    doc_id="d1", content="kw1", metadata={"source": "k1"}
                ),
                "score": 3.0,
                "index": 0,
            },
            {
                "document": SimpleNamespace(
                    doc_id="d2", content="kw2", metadata={"source": "k2"}
                ),
                "score": 2.0,
                "index": 1,
            },
        ]
        retriever.semantic_search = Mock(return_value=sem)
        retriever.keyword_search = Mock(return_value=kw)

        results = retriever.hybrid_search("q", top_k=5)

        assert len(results) == 2
        # Overlap doc (d1) got semantic+keyword; keyword-only (d2) scores less.
        assert results[0].source == "s1"
        assert results[1].source == "k2"
        assert results[0].score > results[1].score > 0


class TestContextCompressorFallback:
    def test_unknown_method_falls_back_to_truncate(self):
        comp = ContextCompressor()
        results = [
            RetrievalResult(
                document="the quick brown fox",
                score=0.9,
                metadata={"source": "s"},
                source="s",
            )
        ]
        out = comp.compress("q", results, method="bogus")
        assert "the quick brown fox" in out
