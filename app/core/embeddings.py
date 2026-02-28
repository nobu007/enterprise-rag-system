"""
Embedding Generation Module

This module handles text embedding generation using various models.
Supports both synchronous and asynchronous OpenAI API calls.
"""

from typing import List, Optional
from abc import ABC, abstractmethod
from openai import AsyncOpenAI, OpenAI
from app.core.config import get_settings
from app.core.logging_config import get_logger


settings = get_settings()
logger = get_logger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts (sync)"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query (sync)"""
        pass

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts (async). Override in subclasses."""
        return self.embed_texts(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query (async). Override in subclasses."""
        return self.embed_query(text)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension"""
        pass


class OpenAIEmbeddings(EmbeddingModel):
    """OpenAI embedding model implementation with sync and async support.

    Uses explicit client instances instead of global openai.api_key to avoid
    global state mutation and enable proper dependency injection.
    """

    # Model dimension mapping
    _DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        async_client: Optional[AsyncOpenAI] = None,
    ):
        self.model = model
        resolved_key = api_key or settings.openai_api_key
        # Explicit client instances – no global state mutation
        self._sync_client = OpenAI(api_key=resolved_key)
        self._async_client = async_client or AsyncOpenAI(api_key=resolved_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (sync)"""
        try:
            response = self._sync_client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Sync embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query (sync)"""
        return self.embed_texts([text])[0]

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (async, non-blocking)"""
        try:
            response = await self._async_client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Async embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")

    async def aembed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query (async, non-blocking)"""
        embeddings = await self.aembed_texts([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self._DIMENSIONS.get(self.model, 1536)


class CohereEmbeddings(EmbeddingModel):
    """Cohere embedding model implementation"""

    def __init__(self, model: str = "embed-english-v3.0", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or settings.cohere_api_key

        if not self.api_key:
            raise ValueError("Cohere API key not provided")

        try:
            import cohere
            self.client = cohere.Client(self.api_key)
        except ImportError:
            raise ImportError("cohere not installed. Run: pip install cohere")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"
            )
            return response.embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query"
            )
            return response.embeddings[0]

        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        # Cohere embed-english-v3.0 has 1024 dimensions
        return 1024


def get_embedding_model(model_name: Optional[str] = None) -> EmbeddingModel:
    """Factory function to get embedding model instance"""
    model_name = model_name or settings.embedding_model

    if "ada" in model_name.lower() or "openai" in model_name.lower():
        return OpenAIEmbeddings(model=model_name)
    elif "cohere" in model_name.lower():
        return CohereEmbeddings(model=model_name)
    else:
        # Default to OpenAI
        return OpenAIEmbeddings(model=model_name)
