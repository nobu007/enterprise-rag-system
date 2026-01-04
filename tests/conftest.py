"""
Test configuration and fixtures for pytest

This module provides common fixtures and configuration for all tests.
"""

import os
import sys
import pytest


# Set test environment variables before importing app modules
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")
os.environ.setdefault("LLM_MODEL", "gpt-4")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-ada-002")
os.environ.setdefault("DEBUG", "true")


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings"""
    return {
        "openai_api_key": "test-key-for-testing",
        "llm_model": "gpt-4",
        "embedding_model": "text-embedding-ada-002",
        "debug": True
    }


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings before each test"""
    # This ensures clean state between tests
    yield
    # Cleanup after test if needed


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, may require external services)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "api: API endpoint tests"
    )
