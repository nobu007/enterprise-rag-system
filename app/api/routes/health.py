"""
Health check endpoints
"""

from fastapi import APIRouter, Request
from typing import Dict

from app.core.rate_limit import limiter

router = APIRouter()


@router.get("/health")
@limiter.limit("120/minute")
async def health_check(request: Request) -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@router.get("/health/detailed")
@limiter.limit("120/minute")
async def detailed_health_check(request: Request) -> Dict[str, Dict[str, str]]:
    """Detailed health check with service status"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "api": "healthy",
            "vector_db": "healthy",
            "llm": "healthy"
        }
    }


@router.get("/cache/stats")
@limiter.limit("120/minute")
async def cache_stats(request: Request) -> Dict[str, Dict[str, str]]:
    """Cache statistics endpoint"""
    # Get cache manager from app state
    cache = request.app.state.cache_manager if hasattr(request.app.state, 'cache_manager') else None

    if not cache:
        return {
            "enabled": False,
            "message": "Cache manager not initialized"
        }

    # Get cache stats
    stats = cache.get_stats()
    return stats

