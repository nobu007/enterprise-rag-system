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
    field ``openai_api_key`` reads ``OPENAI_API_KEY``. Explicit ``env=``
    aliases were therefore redundant and have been removed.
    """

    # API Keys
    openai_api_key: str = Field(...)
    anthropic_api_key: Optional[str] = Field(None)
    cohere_api_key: Optional[str] = Field(None)

    # Vector Database
    pinecone_api_key: Optional[str] = Field(None)
    pinecone_environment: str = Field("us-west1-gcp")
    pinecone_index_name: str = Field("enterprise-rag")

    # File Paths (Security: no hardcoded paths)
    faiss_index_path: str = Field("./data/faiss_index.bin")
    chroma_persist_dir: str = Field("./data/chroma")

    # CORS (Security: controlled origins)
    allowed_origins: str = Field(
        "http://localhost:8000,http://localhost:3000",
    )

    # Embedding Configuration
    embedding_model: str = Field("text-embedding-ada-002")
    embedding_dimension: int = Field(1536)

    # Search Configuration
    # ``hybrid_search_alpha`` is the convex-combination weight HybridRetriever
    # applies in its RRF fusion: ``alpha * semantic + (1 - alpha) * keyword``
    # (retrieval.py HybridRetriever.__init__). The retriever docstring bounds it
    # to ``[0, 1]`` ("0=keyword only, 1=semantic only"). Outside that range one
    # signal inverts -- e.g. alpha=1.5 makes the keyword term (1-1.5=-0.5)
    # *subtract*, so a keyword-matching doc scores LOWER and retrieval ranking
    # is silently corrupted. Bound it here so a misconfigured
    # HYBRID_SEARCH_ALPHA fails fast at Settings load instead of producing
    # inverted-signal retrieval.
    hybrid_search_alpha: float = Field(0.5, ge=0.0, le=1.0)
    top_k_results: int = Field(5)
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
    )

    # Feature-Based Ranking Configuration.
    # Each weight is a non-negative importance coefficient fed straight into
    # QueryResultRanker by get_ranker(). The ranker's normalization only guards
    # a *total* weight <= 0 (divide-by-zero / net sign-flip); it does NOT catch
    # an *individual* negative weight whose sum stays positive. E.g.
    # RANKING_KEYWORD_WEIGHT=-0.5 with the others positive leaves total > 0, so
    # normalization proceeds and keyword_weight goes negative — then a doc with
    # a STRONG keyword match (feature clamped to 1.0) gets a *negative*
    # contribution and ranks BELOW a no-match doc: the signal inverts. Bound
    # each weight to >= 0 so a misconfigured env var fails fast at Settings load
    # (same class as hybrid_search_alpha above; negative importance is
    # meaningless, large positive just dominates after normalization).
    ranking_semantic_weight: float = Field(0.4, ge=0.0)
    ranking_keyword_weight: float = Field(0.3, ge=0.0)
    ranking_freshness_weight: float = Field(0.1, ge=0.0)
    ranking_popularity_weight: float = Field(0.2, ge=0.0)
    ranking_enabled: bool = Field(False)

    # LLM Configuration
    llm_model: str = Field("gpt-4-turbo-preview")
    # ``llm_temperature`` is wired straight into the OpenAI chat completion
    # call: main.py passes settings.llm_temperature to RAGPipeline, which sends
    # it as ``temperature=`` at rag_pipeline.py L116/L422 and streaming.py
    # L149. Every LLM provider defines a non-negative sampling temperature
    # (0 = deterministic); a negative value is meaningless and the provider
    # rejects it per-request. Without a Settings bound a misconfigured
    # LLM_TEMPERATURE=-0.5 lets the app boot and then 500 every query (a
    # silently broken deployment). Enforce the universal lower bound ge=0.0 so
    # it fails fast at Settings load (same class as max_concurrent_requests
    # ge=1). The UPPER bound is provider-specific (OpenAI <= 2, Anthropic <= 1)
    # and intentionally left unbounded here.
    llm_temperature: float = Field(0.7, ge=0.0)
    # ``llm_max_tokens`` is wired straight into the OpenAI chat completion
    # call: main.py passes settings.llm_max_tokens to RAGPipeline (max_tokens=),
    # which sends it as ``max_tokens=`` at rag_pipeline.py L117 on the
    # non-streaming /query and /batch paths. Every LLM provider treats
    # max_tokens as a positive integer; OpenAI rejects ``0`` / negatives
    # per-request ("0 is less than the minimum of 1"). That BadRequestError is
    # wrapped as RuntimeError in _call_llm, and the LLM circuit breaker
    # (expected_exception=RuntimeError) counts it -- after failure_threshold=5
    # such failures the breaker opens and EVERY subsequent query 500s (a
    # silently broken deployment), the same brick mechanism as a negative
    # llm_temperature. Enforce the universal lower bound ge=1 so a misconfigured
    # LLM_MAX_TOKENS=0 fails fast at Settings load (same fail-fast class as
    # llm_temperature ge=0.0 / max_concurrent_requests ge=1). The UPPER bound
    # is model / context-window-specific (deferred, provider-dependent) and
    # intentionally left unbounded here.
    llm_max_tokens: int = Field(2048, ge=1)

    # Performance
    enable_caching: bool = Field(True)
    cache_ttl_seconds: int = Field(3600)
    max_workers: int = Field(4)

    # Monitoring
    langsmith_api_key: Optional[str] = Field(None)
    langsmith_project: str = Field("enterprise-rag")
    arize_api_key: Optional[str] = Field(None)
    enable_metrics: bool = Field(True)

    # Application
    app_name: str = "Enterprise RAG System"
    app_version: str = "0.2.0"
    debug: bool = Field(False)

    # Server
    server_host: str = Field("0.0.0.0")
    server_port: int = Field(8000)

    # CORS Headers (security: restrict allowed headers)
    allowed_headers: str = Field(
        "Content-Type,Authorization,X-API-Key,X-Request-ID",
    )

    # Request size limit (bytes)
    max_request_size: int = Field(10 * 1024 * 1024)

    # Rate Limiting
    rate_limit_enabled: bool = Field(True)
    rate_limit_per_minute: int = Field(60)
    rate_limit_per_hour: int = Field(1000)
    rate_limit_burst: int = Field(10)

    # Concurrency Control
    # ``max_concurrent_requests`` feeds ``ConcurrencyLimiter`` (main.py wiring),
    # whose constructor rejects ``max_concurrent < 1`` with ValueError
    # (concurrency.py). Bound it here so a misconfigured MAX_CONCURRENT_REQUESTS
    # (0 / negative) fails fast at Settings load with a clear ValidationError
    # instead of crashing deep in lifespan init.
    max_concurrent_requests: int = Field(10, ge=1)

    # Redis Cache Configuration
    redis_host: str = Field("localhost")
    redis_port: int = Field(6379)
    redis_db: int = Field(0)
    redis_password: Optional[str] = Field(None)
    cache_enabled: bool = Field(True)
    cache_ttl_seconds: int = Field(3600)

    # Celery Configuration
    celery_broker_url: str = Field("redis://localhost:6379/1")
    celery_result_backend: str = Field("redis://localhost:6379/2")

    # PostgreSQL Database Configuration
    postgres_host: str = Field("localhost")
    postgres_port: int = Field(5432)
    postgres_database: str = Field("enterprise_rag")
    postgres_user: str = Field("postgres")
    postgres_password: str = Field("")
    postgres_pool_min_size: int = Field(10)
    postgres_pool_max_size: int = Field(50)
    postgres_command_timeout: int = Field(60)

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
    # env var by field name case-insensitively, so the explicit env aliases
    # above were redundant and have been removed.
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
