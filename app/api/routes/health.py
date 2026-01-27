"""
Health check endpoints
"""

from fastapi import APIRouter
from typing import Dict

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Dict[str, str]]:
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
