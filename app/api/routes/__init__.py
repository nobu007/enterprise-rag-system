"""
API Routes Module
"""

# NOTE: relationships module is imported separately to avoid circular dependency issues
# when tests run without all heavy dependencies (openai, numpy, etc.)
__all__ = ['query', 'documents', 'health', 'ingest', 'relationships']
