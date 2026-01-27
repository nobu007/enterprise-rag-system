"""
Configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API Keys
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # Vector Database
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-west1-gcp"
    PINECONE_INDEX_NAME: str = "enterprise-rag"

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = 1536

    # Search
    HYBRID_SEARCH_ALPHA: float = 0.5
    TOP_K_RESULTS: int = 5
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    ENABLE_CACHING: bool = True
    CACHE_TTL_SECONDS: int = 3600

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
