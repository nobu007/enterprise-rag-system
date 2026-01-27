"""
Embedding Generation Module

This module handles text embedding generation using various models.
"""

from typing import List, Optional
from abc import ABC, abstractmethod
import openai
from app.core.config import get_settings


settings = get_settings()


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension"""
        pass


class OpenAIEmbeddings(EmbeddingModel):
    """OpenAI embedding model implementation"""
    
    def __init__(self, model: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or settings.openai_api_key
        openai.api_key = self.api_key
        
        # Model dimensions
        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
        
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        embeddings = self.embed_texts([text])
        return embeddings[0]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimensions.get(self.model, 1536)


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
