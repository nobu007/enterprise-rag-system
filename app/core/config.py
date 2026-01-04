"""
Configuration management for Enterprise RAG System

This module handles all configuration settings using Pydantic for validation.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")

    # Vector Database
    pinecone_api_key: Optional[str] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: str = Field("us-west1-gcp", env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field("enterprise-rag", env="PINECONE_INDEX_NAME")

    # File Paths (Security: no hardcoded paths)
    faiss_index_path: str = Field("./data/faiss_index.bin", env="FAISS_INDEX_PATH")
    chroma_persist_dir: str = Field("./data/chroma", env="CHROMA_PERSIST_DIR")

    # CORS (Security: controlled origins)
    allowed_origins: str = Field(
        "http://localhost:8000,http://localhost:3000",
        env="ALLOWED_ORIGINS"
    )
    
    # Embedding Configuration
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(1536, env="EMBEDDING_DIMENSION")
    
    # Search Configuration
    hybrid_search_alpha: float = Field(0.5, env="HYBRID_SEARCH_ALPHA")
    top_k_results: int = Field(5, env="TOP_K_RESULTS")
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        env="RERANKER_MODEL"
    )
    
    # LLM Configuration
    llm_model: str = Field("gpt-4-turbo-preview", env="LLM_MODEL")
    llm_temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(2048, env="LLM_MAX_TOKENS")
    
    # Performance
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    max_workers: int = Field(4, env="MAX_WORKERS")
    
    # Monitoring
    langsmith_api_key: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field("enterprise-rag", env="LANGSMITH_PROJECT")
    arize_api_key: Optional[str] = Field(None, env="ARIZE_API_KEY")
    
    # Application
    app_name: str = "Enterprise RAG System"
    app_version: str = "0.1.0"
    debug: bool = Field(False, env="DEBUG")

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Parse comma-separated origins into a list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
