"""
Tests for Prometheus Metrics

This module tests all Prometheus metrics definitions and endpoints.
"""

import pytest
from prometheus_client import REGISTRY
from fastapi.testclient import TestClient

from app.main import app
from app.core import metrics


class TestMetricsDefinitions:
    """Test metrics are properly defined"""

    def test_http_metrics_exist(self):
        """Test HTTP metrics are defined"""
        # Check that metrics are in the registry
        metric_names = {metric.name for metric in REGISTRY.collect()}

        # Note: prometheus_client strips _total suffix from counters in registry
        assert 'http_requests' in metric_names
        assert 'http_request_duration_seconds' in metric_names

    def test_rag_metrics_exist(self):
        """Test RAG query metrics are defined"""
        metric_names = {metric.name for metric in REGISTRY.collect()}

        # Note: prometheus_client strips _total suffix from counters in registry
        assert 'rag_queries' in metric_names
        assert 'rag_query_duration_seconds' in metric_names

    def test_cache_metrics_exist(self):
        """Test cache metrics are defined"""
        metric_names = {metric.name for metric in REGISTRY.collect()}

        # Note: prometheus_client strips _total suffix from counters in registry
        assert 'cache_hits' in metric_names
        assert 'cache_misses' in metric_names

    def test_llm_metrics_exist(self):
        """Test LLM metrics are defined"""
        metric_names = {metric.name for metric in REGISTRY.collect()}

        # Note: prometheus_client strips _total suffix from counters in registry
        assert 'llm_calls' in metric_names
        assert 'llm_tokens' in metric_names
        assert 'llm_call_duration_seconds' in metric_names

    def test_vectordb_metrics_exist(self):
        """Test VectorDB metrics are defined"""
        metric_names = {metric.name for metric in REGISTRY.collect()}

        assert 'documents_total' in metric_names
        assert 'vector_db_size_bytes' in metric_names

    def test_retrieval_metrics_exist(self):
        """Test retrieval metrics are defined"""
        metric_names = {metric.name for metric in REGISTRY.collect()}

        assert 'retrieval_duration_seconds' in metric_names

    def test_application_info_exists(self):
        """Test application info metric is defined"""
        metric_names = {metric.name for metric in REGISTRY.collect()}

        # Application info is just 'application' not 'application_info'
        assert 'application' in metric_names


class TestMetricsIncrement:
    """Test metrics can be incremented"""

    def test_query_counter_increment(self):
        """Test RAG query counter can be incremented"""
        # Increment counter
        metrics.rag_query_counter.labels(
            collection='test',
            rerank_enabled=True
        ).inc()

        # Get new value
        found = False
        for metric in REGISTRY.collect():
            if metric.name == 'rag_queries':
                for sample in metric.samples:
                    if sample.labels.get('collection') == 'test' and \
                       sample.labels.get('rerank_enabled') == 'True':
                        found = True
                        assert sample.value >= 1
                        break
        assert found, "Metric sample not found"

    def test_cache_hit_increment(self):
        """Test cache hit counter can be incremented"""
        metrics.cache_hits.labels(collection='test').inc()

        metric = next(metric for metric in REGISTRY.collect()
                     if metric.name == 'cache_hits')
        sample = next(sample for sample in metric.samples
                     if sample.labels.get('collection') == 'test')

        assert sample.value >= 1

    def test_cache_miss_increment(self):
        """Test cache miss counter can be incremented"""
        metrics.cache_misses.labels(collection='test').inc()

        metric = next(metric for metric in REGISTRY.collect()
                     if metric.name == 'cache_misses')
        sample = next(sample for sample in metric.samples
                     if sample.labels.get('collection') == 'test')

        assert sample.value >= 1

    def test_llm_call_increment(self):
        """Test LLM call counter can be incremented"""
        metrics.llm_calls.labels(
            model='gpt-3.5-turbo',
            operation='generate'
        ).inc()

        metric = next(metric for metric in REGISTRY.collect()
                     if metric.name == 'llm_calls')
        sample = next(sample for sample in metric.samples
                     if sample.labels.get('model') == 'gpt-3.5-turbo' and
                        sample.labels.get('operation') == 'generate')

        assert sample.value >= 1

    def test_llm_token_increment(self):
        """Test LLM token counter can be incremented"""
        metrics.llm_tokens.labels(
            model='gpt-3.5-turbo',
            type='input'
        ).inc(100)

        metric = next(metric for metric in REGISTRY.collect()
                     if metric.name == 'llm_tokens')
        sample = next(sample for sample in metric.samples
                     if sample.labels.get('model') == 'gpt-3.5-turbo' and
                        sample.labels.get('type') == 'input')

        assert sample.value >= 100

    def test_documents_gauge_set(self):
        """Test documents gauge can be set"""
        metrics.documents_total.labels(collection='test').set(42)

        metric = next(metric for metric in REGISTRY.collect()
                     if metric.name == 'documents_total')
        sample = next(sample for sample in metric.samples
                     if sample.labels.get('collection') == 'test')

        assert sample.value == 42

    def test_vector_db_size_gauge_set(self):
        """Test vector DB size gauge can be set"""
        test_size = 1024 * 1024  # 1MB
        metrics.vector_db_size.labels(collection='test').set(test_size)

        metric = next(metric for metric in REGISTRY.collect()
                     if metric.name == 'vector_db_size_bytes')
        sample = next(sample for sample in metric.samples
                     if sample.labels.get('collection') == 'test')

        assert sample.value == test_size


class TestMetricsEndpoint:
    """Test /metrics endpoint"""

    def test_metrics_endpoint_exists(self, client):
        """Test /metrics endpoint is accessible"""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_content_type(self, client):
        """Test /metrics endpoint returns correct content type"""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_endpoint_contains_http_metrics(self, client):
        """Test /metrics endpoint contains HTTP metrics"""
        response = client.get("/metrics")
        content = response.text

        assert 'http_requests_total' in content
        assert 'http_request_duration_seconds' in content

    def test_metrics_endpoint_contains_rag_metrics(self, client):
        """Test /metrics endpoint contains RAG metrics"""
        response = client.get("/metrics")
        content = response.text

        assert 'rag_queries_total' in content
        assert 'rag_query_duration_seconds' in content

    def test_metrics_endpoint_contains_cache_metrics(self, client):
        """Test /metrics endpoint contains cache metrics"""
        response = client.get("/metrics")
        content = response.text

        assert 'cache_hits_total' in content
        assert 'cache_misses_total' in content

    def test_metrics_endpoint_contains_llm_metrics(self, client):
        """Test /metrics endpoint contains LLM metrics"""
        response = client.get("/metrics")
        content = response.text

        assert 'llm_calls_total' in content
        assert 'llm_tokens_total' in content
        assert 'llm_call_duration_seconds' in content

    def test_metrics_endpoint_contains_vectordb_metrics(self, client):
        """Test /metrics endpoint contains VectorDB metrics"""
        response = client.get("/metrics")
        content = response.text

        assert 'documents_total' in content
        assert 'vector_db_size_bytes' in content

    def test_metrics_endpoint_format(self, client):
        """Test /metrics endpoint returns Prometheus format"""
        response = client.get("/metrics")
        content = response.text

        # Check for Prometheus format indicators
        assert '# HELP' in content or '# TYPE' in content
        assert '{}' in content or '{' in content  # Labels


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)
