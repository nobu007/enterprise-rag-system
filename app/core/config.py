"""
Configuration management for Enterprise RAG System

This module handles all configuration settings using Pydantic for validation.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional, List


class Settings(BaseSettings):
    """Application settings with environment variable support.

    Each field is bound to its environment variable by field name
    (case-insensitively, via ``case_sensitive=False`` in ``model_config``).
    The field name upper-cased equals the env var a deployment sets, e.g.
    field ``openai_api_key`` reads ``OPENAI_API_KEY``.
    """

    # API Keys
    openai_api_key: str = Field(...)
    cohere_api_key: Optional[str] = Field(None)

    # Vector Database
    pinecone_api_key: Optional[str] = Field(None)
    pinecone_environment: str = Field("us-west1-gcp")
    pinecone_index_name: str = Field("enterprise-rag")

    # File Paths (Security: no hardcoded paths)
    faiss_index_path: str = Field("./data/faiss_index.bin")

    # CORS (Security: controlled origins)
    allowed_origins: str = Field(
        "http://localhost:8000,http://localhost:3000",
    )

    # Embedding Configuration
    embedding_model: str = Field("text-embedding-ada-002")

    # Application
    app_name: str = "Enterprise RAG System"
    app_version: str = "0.3.0"
    debug: bool = Field(False)

    # Server
    server_host: str = Field("0.0.0.0")
    server_port: int = Field(8000)

    # CORS Headers (security: restrict allowed headers)
    allowed_headers: str = Field(
        "Content-Type,Authorization,X-API-Key,X-Request-ID",
    )

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Parse comma-separated origins into a list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def ALLOWED_HEADERS_LIST(self) -> List[str]:
        """Parse comma-separated headers into a list"""
        return [h.strip() for h in self.allowed_headers.split(",")]

    # Pydantic-settings v2: SettingsConfigDict replaces the deprecated
    # class-based ``Config``. case_sensitive=False binds each field to its
    # env var by field name case-insensitively.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
