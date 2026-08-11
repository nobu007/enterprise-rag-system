"""Coverage-focused tests for app/core/embeddings.py.

Targets the previously 0%-covered CohereEmbeddings provider (init +
embed_texts/embed_query happy/error paths + dimension), the cohere
branch of get_embedding_model, and the EmbeddingModel base-class async
defaults (aembed_texts/aembed_query). OpenAIEmbeddings and the OpenAI
factory branches are already covered by test_embeddings.py.

``cohere`` is not installed in this environment, so a fake ``cohere``
module is injected via monkeypatch: ``import cohere`` resolves and
``cohere.Client(...)`` returns a controllable mock client.
"""
import sys
from unittest.mock import Mock

import pytest

from app.core.embeddings import (
    CohereEmbeddings,
    EmbeddingModel,
    get_embedding_model,
)


def _fake_cohere(monkeypatch, client=None):
    """Inject a fake ``cohere`` module; Client(...) returns ``client``."""
    if client is None:
        client = Mock()
    module = Mock()
    module.Client = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "cohere", module)
    return client


class TestCohereEmbeddingsInit:
    """Cover CohereEmbeddings.__init__ (L115-126)."""

    def test_init_constructs_client(self, monkeypatch):
        client = _fake_cohere(monkeypatch)
        model = CohereEmbeddings(api_key="my-key")

        assert model.model == "embed-english-v3.0"
        assert model.api_key == "my-key"
        assert model.client is client

    def test_init_missing_api_key_raises_value_error(self, monkeypatch):
        _fake_cohere(monkeypatch)
        monkeypatch.setattr(
            "app.core.embeddings.settings.cohere_api_key", None
        )
        with pytest.raises(ValueError, match="Cohere API key"):
            CohereEmbeddings(api_key=None)

    def test_init_import_error_when_cohere_absent(self, monkeypatch):
        # None sentinel -> `import cohere` raises ImportError.
        monkeypatch.setitem(sys.modules, "cohere", None)
        with pytest.raises(ImportError, match="cohere not installed"):
            CohereEmbeddings(api_key="key")


class TestCohereEmbeddingsEmbed:
    """Cover embed_texts/embed_query happy + error paths (L128-152)."""

    def _model(self, monkeypatch, client):
        _fake_cohere(monkeypatch, client=client)
        return CohereEmbeddings(api_key="key")

    def test_embed_texts_returns_embeddings(self, monkeypatch):
        client = Mock()
        client.embed.return_value = Mock(embeddings=[[0.1], [0.2]])
        model = self._model(monkeypatch, client)

        assert model.embed_texts(["a", "b"]) == [[0.1], [0.2]]
        client.embed.assert_called_once()

    def test_embed_texts_wraps_errors(self, monkeypatch):
        client = Mock()
        client.embed.side_effect = Exception("cohere down")
        model = self._model(monkeypatch, client)

        with pytest.raises(RuntimeError, match="Failed to generate"):
            model.embed_texts(["a"])

    def test_embed_query_returns_first_embedding(self, monkeypatch):
        client = Mock()
        client.embed.return_value = Mock(embeddings=[[0.1, 0.2]])
        model = self._model(monkeypatch, client)

        assert model.embed_query("q") == [0.1, 0.2]

    def test_embed_query_wraps_errors(self, monkeypatch):
        client = Mock()
        client.embed.side_effect = Exception("cohere down")
        model = self._model(monkeypatch, client)

        with pytest.raises(RuntimeError, match="Failed to generate"):
            model.embed_query("q")

    def test_dimension_is_1024(self, monkeypatch):
        _fake_cohere(monkeypatch)
        assert CohereEmbeddings(api_key="key").dimension == 1024


class TestGetEmbeddingModelCohere:
    """Cover the cohere branch of get_embedding_model (L167-168)."""

    def test_factory_returns_cohere_for_cohere_model(self, monkeypatch):
        _fake_cohere(monkeypatch)
        monkeypatch.setattr(
            "app.core.embeddings.settings.cohere_api_key", "key"
        )
        model = get_embedding_model("cohere-embed-english-v3.0")
        assert isinstance(model, CohereEmbeddings)


class TestEmbeddingModelBaseDefaults:
    """Cover the ABC async defaults aembed_texts/aembed_query (L34, L38)."""

    class _Minimal(EmbeddingModel):
        def embed_texts(self, texts):
            return [[0.1]]

        def embed_query(self, text):
            return [0.1]

        @property
        def dimension(self):
            return 1

    @pytest.mark.asyncio
    async def test_base_aembed_texts_delegates_to_sync(self):
        result = await self._Minimal().aembed_texts(["x"])
        assert result == [[0.1]]

    @pytest.mark.asyncio
    async def test_base_aembed_query_delegates_to_sync(self):
        result = await self._Minimal().aembed_query("x")
        assert result == [0.1]
