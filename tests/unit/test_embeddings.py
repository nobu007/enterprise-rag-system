"""
Unit tests for Embedding Generation Module
"""

import pytest
from unittest.mock import Mock, AsyncMock
from app.core.embeddings import (
    OpenAIEmbeddings,
    get_embedding_model,
    EmbeddingModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI embeddings response"""
    item1 = Mock()
    item1.embedding = [0.1, 0.2, 0.3]
    item2 = Mock()
    item2.embedding = [0.4, 0.5, 0.6]
    response = Mock()
    response.data = [item1, item2]
    return response


@pytest.fixture
def mock_single_response():
    """Mock OpenAI single embedding response"""
    item = Mock()
    item.embedding = [0.1, 0.2, 0.3]
    response = Mock()
    response.data = [item]
    return response


@pytest.fixture
def mock_empty_response():
    """Mock OpenAI embeddings response with an empty data list.

    The SDK types ``CreateEmbeddingResponse.data`` as ``List[Embedding]``
    (required=True, but a List may be empty). A degenerate OpenAI-compatible
    provider can return a 200 with ``data: []`` -- this exercises the
    embed_query/aembed_query empty-data branch.
    """
    response = Mock()
    response.data = []
    return response


# ---------------------------------------------------------------------------
# OpenAIEmbeddings Tests
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddings:
    """Tests for OpenAIEmbeddings"""

    def test_initialization_defaults(self):
        """Test default initialization uses settings."""
        model = OpenAIEmbeddings(api_key="test-key")
        assert model.model == "text-embedding-ada-002"
        assert model.dimension == 1536

    def test_dimension_mapping(self):
        """Test dimension lookup for different models."""
        assert OpenAIEmbeddings(model="text-embedding-ada-002", api_key="k").dimension == 1536
        assert OpenAIEmbeddings(model="text-embedding-3-large", api_key="k").dimension == 3072
        # Unknown model falls back to 1536
        assert OpenAIEmbeddings(model="unknown-model", api_key="k").dimension == 1536

    def test_embed_texts_sync(self, mock_openai_response):
        """Test synchronous embed_texts."""
        model = OpenAIEmbeddings(api_key="test-key")
        model._sync_client = Mock()
        model._sync_client.embeddings.create.return_value = mock_openai_response

        result = model.embed_texts(["hello", "world"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        model._sync_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=["hello", "world"],
        )

    def test_embed_query_sync(self, mock_single_response):
        """Test synchronous embed_query."""
        model = OpenAIEmbeddings(api_key="test-key")
        model._sync_client = Mock()
        model._sync_client.embeddings.create.return_value = mock_single_response

        result = model.embed_query("hello")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aembed_texts(self, mock_openai_response):
        """Test async embed_texts."""
        model = OpenAIEmbeddings(api_key="test-key")
        model._async_client = Mock()
        model._async_client.embeddings.create = AsyncMock(return_value=mock_openai_response)

        result = await model.aembed_texts(["hello", "world"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @pytest.mark.asyncio
    async def test_aembed_query(self, mock_single_response):
        """Test async embed_query."""
        model = OpenAIEmbeddings(api_key="test-key")
        model._async_client = Mock()
        model._async_client.embeddings.create = AsyncMock(return_value=mock_single_response)

        result = await model.aembed_query("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_texts_error_handling(self):
        """Test sync embed_texts raises RuntimeError on API failure."""
        model = OpenAIEmbeddings(api_key="test-key")
        model._sync_client = Mock()
        model._sync_client.embeddings.create.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            model.embed_texts(["test"])

    @pytest.mark.asyncio
    async def test_aembed_texts_error_handling(self):
        """Test async embed_texts raises RuntimeError on API failure."""
        model = OpenAIEmbeddings(api_key="test-key")
        model._async_client = Mock()
        model._async_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))

        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            await model.aembed_texts(["test"])

    def test_embed_query_empty_data_raises_runtime_error(self, mock_empty_response):
        """Empty ``data`` (degenerate provider) must raise RuntimeError, not IndexError.

        ``embed_query`` indexes ``embed_texts([text])[0]``; an empty ``data``
        list used to leak a raw IndexError -> 500 on /query. The module's
        contract is that every embedding failure raises RuntimeError (mirrored
        by CohereEmbeddings.embed_query and embed_texts' API-error wrapping).
        """
        model = OpenAIEmbeddings(api_key="test-key")
        model._sync_client = Mock()
        model._sync_client.embeddings.create.return_value = mock_empty_response

        with pytest.raises(RuntimeError, match="Failed to generate embedding"):
            model.embed_query("test")

    @pytest.mark.asyncio
    async def test_aembed_query_empty_data_raises_runtime_error(self, mock_empty_response):
        """Async empty ``data`` must raise RuntimeError, not IndexError (sibling of sync)."""
        model = OpenAIEmbeddings(api_key="test-key")
        model._async_client = Mock()
        model._async_client.embeddings.create = AsyncMock(return_value=mock_empty_response)

        with pytest.raises(RuntimeError, match="Failed to generate embedding"):
            await model.aembed_query("test")

    def test_no_global_api_key_mutation(self):
        """Test that creating OpenAIEmbeddings does not mutate global openai.api_key."""
        import openai
        original = getattr(openai, "api_key", None)
        OpenAIEmbeddings(api_key="should-not-leak")
        assert getattr(openai, "api_key", None) == original


# ---------------------------------------------------------------------------
# get_embedding_model Factory Tests
# ---------------------------------------------------------------------------


class TestGetEmbeddingModel:
    """Tests for factory function"""

    def test_returns_openai_for_ada(self):
        """Test factory returns OpenAIEmbeddings for ada model."""
        model = get_embedding_model("text-embedding-ada-002")
        assert isinstance(model, OpenAIEmbeddings)

    def test_returns_openai_for_default(self):
        """Test factory returns OpenAIEmbeddings for unknown model name."""
        model = get_embedding_model("some-custom-model")
        assert isinstance(model, OpenAIEmbeddings)

    def test_abstract_base_class(self):
        """Test EmbeddingModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingModel()
