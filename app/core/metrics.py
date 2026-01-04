"""
Prometheus Metrics Definition

This module defines all Prometheus metrics for monitoring the Enterprise RAG System.
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# =============================================================================
# HTTP Request Metrics
# =============================================================================

request_counter = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

response_time = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['endpoint']
)

# =============================================================================
# RAG Query Metrics
# =============================================================================

rag_query_counter = Counter(
    'rag_queries_total',
    'Total RAG queries',
    ['collection', 'rerank_enabled']
)

rag_query_latency = Histogram(
    'rag_query_duration_seconds',
    'RAG query latency',
    ['collection']
)

# =============================================================================
# Cache Metrics
# =============================================================================

cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['collection']
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['collection']
)

# =============================================================================
# LLM Call Metrics
# =============================================================================

llm_calls = Counter(
    'llm_calls_total',
    'Total LLM API calls',
    ['model', 'operation']
)

llm_tokens = Counter(
    'llm_tokens_total',
    'Total LLM tokens processed',
    ['model', 'type']  # type: input/output
)

llm_latency = Histogram(
    'llm_call_duration_seconds',
    'LLM call latency',
    ['model']
)

# =============================================================================
# Vector Database Metrics
# =============================================================================

documents_total = Gauge(
    'documents_total',
    'Total number of documents',
    ['collection']
)

vector_db_size = Gauge(
    'vector_db_size_bytes',
    'Vector database size in bytes',
    ['collection']
)

# =============================================================================
# Retrieval Metrics
# =============================================================================

retrieval_latency = Histogram(
    'retrieval_duration_seconds',
    'Document retrieval latency',
    ['collection', 'search_type']  # search_type: hybrid/vector/bm25
)

# =============================================================================
# Application Info
# =============================================================================

app_info = Info(
    'application',
    'Application information'
)

# =============================================================================
# Circuit Breaker Metrics
# =============================================================================

circuit_breaker_state_change = Counter(
    'circuit_breaker_state_changes_total',
    'Total circuit breaker state transitions',
    ['name', 'from_state', 'to_state']
)

circuit_breaker_rejected = Counter(
    'circuit_breaker_requests_rejected_total',
    'Total requests rejected by open circuit',
    ['name']
)

circuit_breaker_failure = Counter(
    'circuit_breaker_failures_total',
    'Total failures through circuit breaker',
    ['name', 'exception_type']
)
