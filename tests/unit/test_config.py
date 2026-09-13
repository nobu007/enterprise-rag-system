"""
Unit tests for Settings environment-variable binding.

These guard the pydantic-settings v2 configuration in ``app/core/config.py``:
under ``case_sensitive=False`` a field reads its value from an env var
matching the field name case-insensitively. The case-insensitivity tests
below pin that invariant.
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

    With ``case_sensitive=False``, field ``openai_api_key`` matches
    ``OPENAI_API_KEY``, ``openai_api_key`` and ``Openai_Api_Key`` alike.
    """

    @pytest.mark.parametrize(
        "env_name", ["OPENAI_API_KEY", "openai_api_key", "Openai_Api_Key"]
    )
    def test_str_field_read_regardless_of_env_case(self, monkeypatch, env_name):
        monkeypatch.setenv(env_name, "case-binding-value")
        assert Settings().openai_api_key == "case-binding-value"

    def test_int_field_read_from_upper_env(self, monkeypatch):
        result = _settings(monkeypatch, SERVER_PORT="9000")
        assert result.server_port == 9000
        assert isinstance(result.server_port, int)

    def test_bool_field_read_from_upper_env(self, monkeypatch):
        assert _settings(monkeypatch, DEBUG="false").debug is False
        assert _settings(monkeypatch, DEBUG="true").debug is True


class TestEnvBindingAcrossFieldGroups:
    """Env binding must cover every field group and type."""

    def test_optional_str_field(self, monkeypatch):
        assert _settings(monkeypatch, COHERE_API_KEY="co-secret").cohere_api_key == "co-secret"

    def test_unset_optional_defaults_to_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        assert Settings().cohere_api_key is None

    def test_multiple_fields_different_types(self, monkeypatch):
        result = _settings(
            monkeypatch,
            PINECONE_ENVIRONMENT="us-east1-gcp",
            SERVER_PORT="6543",
            EMBEDDING_MODEL="text-embedding-3-large",
        )
        assert result.pinecone_environment == "us-east1-gcp"
        assert result.server_port == 6543
        assert result.embedding_model == "text-embedding-3-large"


class TestDefaults:
    """Default values apply when the env var is absent."""

    def test_int_default_when_unset(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.delenv("SERVER_PORT", raising=False)
        assert Settings().server_port == 8000

    def test_str_default_when_unset(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
        assert Settings().pinecone_index_name == "enterprise-rag"

    def test_faiss_index_path_default_when_unset(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.delenv("FAISS_INDEX_PATH", raising=False)
        assert Settings().faiss_index_path == "./data/faiss_index.bin"


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
        assert result.app_version == "0.3.0"


class TestRequiredFields:
    """The one required field must fail fast when absent."""

    def test_missing_openai_api_key_rejected(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError):
            Settings()
