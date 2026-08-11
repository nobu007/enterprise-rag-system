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
