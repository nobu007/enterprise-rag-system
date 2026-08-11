"""Coverage-focused tests for app/services/ranking.py.

Targets previously-uncovered branches of QueryResultRanker: the
``last_updated`` / ``created_at`` date-parsing variants and their
error handlers, the per-result exception handler in rank_results,
the conversion/reconstruction error paths in rank_rag_results, and
the freshness/popularity branches of update_weights.

ranking.py is a P0 (MS-001) deliverable with no external-service
dependencies, so every path below is exercised as a plain unit test.
"""
import sys

from app.services.ranking import QueryResultRanker


class TestWeightNormalization:
    """Cover the weight-normalization branch of __init__ (L58-66)."""

    def test_weights_normalized_when_not_summing_to_one(self):
        """Weights summing outside ~1.0 are normalized in place."""
        ranker = QueryResultRanker(
            semantic_weight=0.1,
            keyword_weight=0.1,
            freshness_weight=0.1,
            popularity_weight=0.1,
        )
        weights = ranker.get_weights()
        # Each original 0.1 / 0.4 total == 0.25
        assert weights == {
            'semantic': 0.25,
            'keyword': 0.25,
            'freshness': 0.25,
            'popularity': 0.25,
        }

    def test_all_zero_weights_do_not_crash(self):
        """All-zero weights must not raise ZeroDivisionError.

        Regression: the config-driven ``get_ranker()`` sources all four
        weights from env-settable ``settings.ranking_*_weight``; if an
        operator sets every weight to 0, ``total_weight == 0`` entered the
        normalization branch and divided by zero, crashing construction.
        The guard now skips normalization and degrades to 0.0 scores.
        """
        ranker = QueryResultRanker(
            semantic_weight=0.0,
            keyword_weight=0.0,
            freshness_weight=0.0,
            popularity_weight=0.0,
        )
        # No crash; weights preserved as-is (all zero) instead of normalized.
        assert ranker.get_weights() == {
            'semantic': 0.0,
            'keyword': 0.0,
            'freshness': 0.0,
            'popularity': 0.0,
        }
        # Score degrades to 0.0 rather than raising.
        assert ranker.calculate_ranking_score({'semantic_score': 0.9}) == 0.0

    def test_net_negative_weights_do_not_invert(self):
        """A net-negative weight total must not flip signs via division."""
        ranker = QueryResultRanker(
            semantic_weight=1.0,
            keyword_weight=1.0,
            freshness_weight=-1.5,
            popularity_weight=-1.5,
        )  # total == -1.0
        # Guard skips normalization, so the passed-in weights are preserved
        # verbatim (no division by a negative that would invert them).
        assert ranker.get_weights() == {
            'semantic': 1.0,
            'keyword': 1.0,
            'freshness': -1.5,
            'popularity': -1.5,
        }


class TestFreshnessLastUpdated:
    """Cover the ``last_updated`` freshness branch (L118-141)."""

    def test_last_updated_iso_with_z_suffix(self):
        """tz-aware ('Z') ISO date decays correctly via fromisoformat (L126).

        Regression: a trailing-'Z' / '+00:00' date parsed to a tz-aware
        datetime used to crash on ``datetime.now() - aware`` (TypeError),
        which the ValueError/TypeError handler swallowed by falling back to
        the 0.5 default -- defeating freshness decay for the most common
        ISO shape. 2024-01-15 is ~900+ days old, so exp(-days/90) ~= 0.0,
        nowhere near the 0.5 fallback. The old range-only assert hid this.
        """
        ranker = QueryResultRanker()
        features = ranker.extract_features(
            {
                'score': 0.5,
                'metadata': {'last_updated': '2024-01-15T00:00:00Z'},
            },
            query_length=2,
        )
        assert 0.0 <= features['freshness'] < 0.05

    def test_last_updated_dateutil_fallback(self):
        """Non-ISO date falls back to dateutil.parser (L127-130)."""
        ranker = QueryResultRanker()
        features = ranker.extract_features(
            {'score': 0.5, 'metadata': {'last_updated': 'Jan 15, 2024'}},
            query_length=2,
        )
        assert 0.0 <= features['freshness'] <= 1.0

    def test_last_updated_unparseable_uses_metadata_freshness(self):
        """Non-ISO and dateutil-unparseable -> ValueError handler."""
        ranker = QueryResultRanker()
        features = ranker.extract_features(
            {
                'score': 0.5,
                'metadata': {
                    'last_updated': 'not-a-date',
                    'freshness': 0.42,
                },
            },
            query_length=2,
        )
        assert features['freshness'] == 0.42

    def test_last_updated_dateutil_import_error(self, monkeypatch):
        """dateutil unavailable -> ImportError handler (L136-138)."""
        monkeypatch.setitem(sys.modules, 'dateutil', None)
        ranker = QueryResultRanker()
        features = ranker.extract_features(
            {
                'score': 0.5,
                'metadata': {
                    'last_updated': 'Jan 15, 2024',
                    'freshness': 0.6,
                },
            },
            query_length=2,
        )
        assert features['freshness'] == 0.6


class TestFreshnessCreatedAtErrors:
    """Cover created_at error paths (L150-155, L160-162)."""

    def test_created_at_iso_with_z_suffix(self):
        """tz-aware ('Z') created_at decays correctly (L149/L158).

        Same naive-vs-aware regression as last_updated: the Z-suffix parses
        to a tz-aware datetime, and the old ``datetime.now() - created_at``
        raised TypeError that the outer handler swallowed into the 0.5
        fallback. ~900+ days old -> exp(-days/90) ~= 0.0, not 0.5.
        """
        ranker = QueryResultRanker()
        features = ranker.extract_features(
            {
                'score': 0.5,
                'metadata': {'created_at': '2024-01-15T00:00:00Z'},
            },
            query_length=2,
        )
        assert 0.0 <= features['freshness'] < 0.05

    def test_created_at_dateutil_fallback(self):
        """Non-ISO created_at falls back to dateutil (L150-155)."""
        ranker = QueryResultRanker()
        features = ranker.extract_features(
            {'score': 0.5, 'metadata': {'created_at': 'Feb 1, 2024'}},
            query_length=2,
        )
        assert 0.0 <= features['freshness'] <= 1.0

    def test_created_at_unparseable_uses_metadata_freshness(self):
        """Unparseable created_at -> outer Exception handler."""
        ranker = QueryResultRanker()
        features = ranker.extract_features(
            {
                'score': 0.5,
                'metadata': {
                    'created_at': 'not-a-date',
                    'freshness': 0.33,
                },
            },
            query_length=2,
        )
        assert features['freshness'] == 0.33

    def test_created_at_dateutil_import_error(self, monkeypatch):
        """dateutil unavailable -> ImportError sets created_at=None."""
        monkeypatch.setitem(sys.modules, 'dateutil', None)
        ranker = QueryResultRanker()
        features = ranker.extract_features(
            {'score': 0.5, 'metadata': {'created_at': 'Jan 15, 2024'}},
            query_length=2,
        )
        # created_at becomes None -> freshness stays at the 0.5 default
        assert features['freshness'] == 0.5


class TestRankResultsErrorPath:
    """Cover the per-result exception handler in rank_results (L254-260)."""

    def test_failing_result_kept_at_zero_score(self):
        """A result that raises during extraction is kept at score 0."""
        ranker = QueryResultRanker()
        # A non-numeric doc_length makes `doc_length > 0` raise
        # TypeError inside extract_features, exercising the handler.
        results = [
            {'score': 0.9, 'document': 'good doc', 'metadata': {}},
            {
                'score': 0.5,
                'document': 'bad',
                'metadata': {},
                'doc_length': 'oops',
            },
        ]
        ranked = ranker.rank_results(
            query='machine learning', results=results
        )

        assert len(ranked) == 2
        bad = next(r for r in ranked if r['document'] == 'bad')
        assert bad['ranking_score'] == 0.0
        good = next(r for r in ranked if r['document'] == 'good doc')
        assert good['ranking_score'] > 0.0


class TestRankRagResultsErrors:
    """Cover conversion/reconstruction error paths in rank_rag_results."""

    def test_conversion_error_is_skipped(self):
        """A result whose document len() fails is skipped (L312-314)."""
        ranker = QueryResultRanker()

        class _GoodResult:
            # __init__ mirrors the rebuild signature
            # ResultClass(document=, score=, metadata=, source=).
            def __init__(
                self, document='', score=0.0,
                metadata=None, source='',
            ):
                self.document = document
                self.score = score
                self.metadata = metadata or {}
                self.source = source
                self.keyword_score = 0.0

        class _BadResult:
            # document is an int -> len() raises TypeError on conversion
            document = 12345
            score = 0.5
            metadata = {}
            source = ''
            keyword_score = 0.0

        good = _GoodResult(document='good doc', score=0.9)
        ranked = ranker.rank_rag_results('query', [good, _BadResult()])
        assert len(ranked) == 1
        assert ranked[0].document == 'good doc'

    def test_reconstruction_error_is_skipped(self):
        """A result whose ResultClass rebuild fails is skipped."""
        ranker = QueryResultRanker()

        class _Result:
            def __init__(
                self, document='', score=0.0,
                metadata=None, source='',
            ):
                if document == 'FAIL':
                    raise ValueError('cannot rebuild')
                self.document = document
                self.score = score
                self.metadata = metadata or {}
                self.source = source

        ok = _Result(document='ok', score=0.9)
        # Build the failing instance without invoking __init__ so it
        # can be passed in; the rebuild path calls
        # _Result(document='FAIL', ...) and raises, exercising the skip.
        fail = object.__new__(_Result)
        fail.document = 'FAIL'
        fail.score = 0.5
        fail.metadata = {}
        fail.source = ''
        ranked = ranker.rank_rag_results('query', [ok, fail])
        assert len(ranked) == 1
        assert ranked[0].document == 'ok'


class TestUpdateWeightsAllKeys:
    """Cover freshness & popularity branches of update_weights (L357-360)."""

    def test_update_weights_sets_all_four(self):
        ranker = QueryResultRanker()
        ranker.update_weights({
            'semantic': 0.35,
            'keyword': 0.25,
            'freshness': 0.15,
            'popularity': 0.25,
        })
        assert ranker.get_weights() == {
            'semantic': 0.35,
            'keyword': 0.25,
            'freshness': 0.15,
            'popularity': 0.25,
        }
