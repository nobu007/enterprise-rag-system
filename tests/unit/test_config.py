"""
Unit tests for Settings environment-variable binding.

These guard the pydantic-settings v2 migration that removes the redundant
``env=`` kwarg from each ``Field(...)`` in ``app/core/config.py``. Under
``case_sensitive=False`` a field reads its value from an env var matching
the field name case-insensitively, so an explicit ``env="FIELD"`` that
equals ``field.upper()`` is redundant and can be dropped without changing
which env vars are read. The case-insensitivity tests below are the
invariant that makes that removal safe.
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _settings(monkeypatch, **env):
    """Build a fresh ``Settings`` with the given UPPER-case env vars set.

    ``OPENAI_API_KEY`` is always provided because ``openai_api_key`` is the
    one required field.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "required-key")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    return Settings()


class TestEnvVarCaseInsensitive:
    """A field must read its env var regardless of case spelling.

    This is the invariant the ``Field(env=)`` removal relies on: with
    ``case_sensitive=False``, field ``openai_api_key`` matches
    ``OPENAI_API_KEY``, ``openai_api_key`` and ``Openai_Api_Key`` alike.
    """

    @pytest.mark.parametrize(
        "env_name", ["OPENAI_API_KEY", "openai_api_key", "Openai_Api_Key"]
    )
    def test_str_field_read_regardless_of_env_case(self, monkeypatch, env_name):
        monkeypatch.setenv(env_name, "case-binding-value")
        assert Settings().openai_api_key == "case-binding-value"

    def test_int_field_read_from_upper_env(self, monkeypatch):
        result = _settings(monkeypatch, REDIS_PORT="7777")
        assert result.redis_port == 7777
        assert isinstance(result.redis_port, int)

    def test_bool_field_read_from_upper_env(self, monkeypatch):
        assert _settings(monkeypatch, DEBUG="false").debug is False
        assert _settings(monkeypatch, DEBUG="true").debug is True

    def test_float_field_read_from_upper_env(self, monkeypatch):
        assert _settings(monkeypatch, HYBRID_SEARCH_ALPHA="0.25").hybrid_search_alpha == 0.25

    @pytest.mark.parametrize("bad_alpha", ["1.5", "-0.5", "2", "-1"])
    def test_hybrid_search_alpha_rejected_out_of_range(self, monkeypatch, bad_alpha):
        # hybrid_search_alpha is a convex-combination weight in HybridRetriever
        # RRF fusion (alpha*semantic + (1-alpha)*keyword); outside [0, 1] one
        # signal inverts (alpha=1.5 -> keyword term = -0.5, keyword matches
        # reduce the score). The Field must reject it at Settings load rather
        # than let inverted-signal retrieval ranking through.
        with pytest.raises(ValidationError):
            _settings(monkeypatch, HYBRID_SEARCH_ALPHA=bad_alpha)

    @pytest.mark.parametrize("good_alpha", ["0", "0.0", "1", "1.0", "0.25"])
    def test_hybrid_search_alpha_accepted_in_range(self, monkeypatch, good_alpha):
        assert _settings(monkeypatch, HYBRID_SEARCH_ALPHA=good_alpha).hybrid_search_alpha == float(good_alpha)

    @pytest.mark.parametrize("field", ["RANKING_SEMANTIC_WEIGHT", "RANKING_KEYWORD_WEIGHT", "RANKING_FRESHNESS_WEIGHT", "RANKING_POPULARITY_WEIGHT"])
    def test_ranking_weight_rejected_negative(self, monkeypatch, field):
        # ranking_*_weight feed QueryResultRanker straight from get_ranker().
        # The ranker's normalization only guards a *total* weight <= 0; a single
        # negative weight whose sum stays positive slips through: e.g.
        # RANKING_KEYWORD_WEIGHT=-0.5 with the others at 2.0 leaves total=1.5,
        # normalization turns keyword_weight negative, and a strong-keyword doc
        # then scores LOWER than a no-keyword doc (signal inversion — same
        # class as hybrid_search_alpha). ge=0.0 must reject it at Settings load.
        with pytest.raises(ValidationError):
            _settings(monkeypatch, **{field: "-0.5"})

    @pytest.mark.parametrize("good", ["0", "0.0", "0.4", "2"])
    def test_ranking_weight_accepted_non_negative(self, monkeypatch, good):
        # 0 and large positive are valid: 0 drops a feature (legitimate),
        # large positive just dominates after normalization.
        result = _settings(monkeypatch, RANKING_KEYWORD_WEIGHT=good)
        assert result.ranking_keyword_weight == float(good)

    @pytest.mark.parametrize("bad_concurrency", ["0", "-1", "-5"])
    def test_max_concurrent_requests_rejected_below_one(self, monkeypatch, bad_concurrency):
        # max_concurrent_requests feeds ConcurrencyLimiter, whose constructor
        # rejects max_concurrent < 1 with ValueError (concurrency.py). The
        # Field must enforce the same lower bound at Settings load so a
        # misconfigured MAX_CONCURRENT_REQUESTS (0 / negative) fails fast
        # with a clear ValidationError instead of crashing lifespan init.
        with pytest.raises(ValidationError):
            _settings(monkeypatch, MAX_CONCURRENT_REQUESTS=bad_concurrency)

    @pytest.mark.parametrize("good_concurrency", ["1", "10", "100"])
    def test_max_concurrent_requests_accepted_at_least_one(self, monkeypatch, good_concurrency):
        assert _settings(monkeypatch, MAX_CONCURRENT_REQUESTS=good_concurrency).max_concurrent_requests == int(good_concurrency)

    @pytest.mark.parametrize("bad_size", ["0", "-1", "-5"])
    def test_max_request_size_rejected_below_one(self, monkeypatch, bad_size):
        # max_request_size feeds ValidationMiddleware (main.py wiring), whose
        # _validate_content_length rejects any body whose Content-Length exceeds
        # it with HTTP 413 (validation.py). That guard runs on every
        # POST/PUT/PATCH -- every /query, /batch/query, /ingest and /documents
        # body. A value of 0 (or negative) therefore 413s every body-bearing
        # request: the app boots and /health stays 200, but the whole query+
        # ingest surface is bricked (a silently broken deployment). The Field
        # must enforce the same lower bound at Settings load so a misconfigured
        # MAX_REQUEST_SIZE (0 / negative) fails fast with a clear ValidationError
        # -- same fail-fast class as max_concurrent_requests ge=1 above.
        with pytest.raises(ValidationError):
            _settings(monkeypatch, MAX_REQUEST_SIZE=bad_size)

    @pytest.mark.parametrize("good_size", ["1", "1048576", "10485760"])
    def test_max_request_size_accepted_at_least_one(self, monkeypatch, good_size):
        # 1 byte is the smallest useful cap (a tiny JSON body); 1 MiB and the
        # 10 MiB default are normal deployments. The upper bound is
        # deployment-specific (memory budget) and intentionally unbounded.
        assert _settings(monkeypatch, MAX_REQUEST_SIZE=good_size).max_request_size == int(good_size)

    @pytest.mark.parametrize("bad_temp", ["-0.1", "-0.5", "-1", "-2.0"])
    def test_llm_temperature_rejected_negative(self, monkeypatch, bad_temp):
        # llm_temperature is wired straight into the OpenAI chat completion
        # call: main.py passes settings.llm_temperature to RAGPipeline, which
        # stores it and sends it as ``temperature=`` at rag_pipeline.py L116 /
        # L422 and streaming.py L149. Every LLM provider defines a non-negative
        # sampling temperature (0 = deterministic); a negative value is
        # meaningless and the provider rejects it per-request. Without a
        # Settings bound, a misconfigured LLM_TEMPERATURE=-0.5 lets the app
        # boot and then 500 every query (a silently broken deployment). The
        # Field must reject it at Settings load -- same fail-fast class as
        # max_concurrent_requests ge=1 above. Only the universal lower bound is
        # enforced; the upper bound is provider-specific (OpenAI <= 2, Anthropic
        # <= 1) and intentionally left unbounded (see the accepted test below).
        with pytest.raises(ValidationError):
            _settings(monkeypatch, LLM_TEMPERATURE=bad_temp)

    @pytest.mark.parametrize("good_temp", ["0", "0.0", "0.7", "1", "2.0"])
    def test_llm_temperature_accepted_non_negative(self, monkeypatch, good_temp):
        # 0 is valid (deterministic). Positive values are accepted up to each
        # provider's own maximum: 2.0 is included here to lock in that the
        # Field enforces ONLY the lower bound -- the upper bound (deferred,
        # provider-specific) must stay unbounded so a valid OpenAI temperature
        # of 2.0 is not rejected at Settings load.
        assert _settings(monkeypatch, LLM_TEMPERATURE=good_temp).llm_temperature == float(good_temp)

    @pytest.mark.parametrize("bad_tokens", ["0", "-1", "-5", "-100"])
    def test_llm_max_tokens_rejected_below_one(self, monkeypatch, bad_tokens):
        # llm_max_tokens is wired straight into the OpenAI chat completion
        # call: main.py passes settings.llm_max_tokens to RAGPipeline
        # (max_tokens=), which sends it as ``max_tokens=`` at
        # rag_pipeline.py L117 (the non-streaming /query and /batch paths).
        # Every LLM provider treats max_tokens as a positive integer; OpenAI
        # rejects ``0`` / negatives per-request ("0 is less than the minimum
        # of 1"). That BadRequestError is wrapped as RuntimeError in
        # _call_llm, and the LLM circuit breaker (expected_exception=
        # RuntimeError) counts it -- after failure_threshold=5 such failures
        # the breaker opens and EVERY subsequent query 500s (a silently
        # broken deployment). This is the same fail-open-to-brick mechanism
        # as a negative llm_temperature. ge=1 enforces the universal lower
        # bound so a misconfigured LLM_MAX_TOKENS=0 fails fast at Settings
        # load instead of bricking the deployment one query at a time. Only
        # the lower bound is enforced; the upper bound is model/context-window
        # specific (deferred) and intentionally left unbounded (see below).
        with pytest.raises(ValidationError):
            _settings(monkeypatch, LLM_MAX_TOKENS=bad_tokens)

    @pytest.mark.parametrize("good_tokens", ["1", "2", "2048", "100000"])
    def test_llm_max_tokens_accepted_at_least_one(self, monkeypatch, good_tokens):
        # 1 is the minimum useful cap. 100000 is included to lock in that the
        # Field enforces ONLY the lower bound -- the upper bound (deferred,
        # model/context-window-specific) must stay unbounded so a valid large
        # max_tokens for a long-context model is not rejected at Settings load.
        assert _settings(monkeypatch, LLM_MAX_TOKENS=good_tokens).llm_max_tokens == int(good_tokens)


class TestEnvBindingAcrossFieldGroups:
    """Env binding must cover every field group and type, so dropping
    ``env=`` cannot silently drop a field's env source."""

    def test_optional_str_field(self, monkeypatch):
        assert _settings(monkeypatch, ANTHROPIC_API_KEY="ant-secret").anthropic_api_key == "ant-secret"

    def test_unset_optional_defaults_to_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert Settings().anthropic_api_key is None

    def test_multiple_fields_different_types(self, monkeypatch):
        result = _settings(
            monkeypatch,
            REDIS_HOST="redis.prod.example",
            POSTGRES_PORT="6543",
            RATE_LIMIT_ENABLED="false",
            RANKING_ENABLED="true",
            EMBEDDING_DIMENSION="3072",
        )
        assert result.redis_host == "redis.prod.example"
        assert result.postgres_port == 6543
        assert result.rate_limit_enabled is False
        assert result.ranking_enabled is True
        assert result.embedding_dimension == 3072


class TestDefaults:
    """Default values apply when the env var is absent."""

    def test_int_default_when_unset(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.delenv("REDIS_PORT", raising=False)
        assert Settings().redis_port == 6379

    def test_str_default_when_unset(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
        assert Settings().pinecone_index_name == "enterprise-rag"


class TestDerivedProperties:
    """Comma-separated string fields parse into list properties."""

    def test_allowed_origins_parsed(self, monkeypatch):
        result = _settings(monkeypatch, ALLOWED_ORIGINS="https://a.com, https://b.com")
        assert result.ALLOWED_ORIGINS == ["https://a.com", "https://b.com"]

    def test_allowed_headers_list_parsed(self, monkeypatch):
        result = _settings(monkeypatch, ALLOWED_HEADERS="Content-Type,X-Custom")
        assert result.ALLOWED_HEADERS_LIST == ["Content-Type", "X-Custom"]

    def test_non_env_fields_keep_literal_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        result = Settings()
        assert result.app_name == "Enterprise RAG System"
        assert result.app_version == "0.2.0"
