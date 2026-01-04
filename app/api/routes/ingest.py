"""
Document ingestion endpoints
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Dict
from datetime import datetime

from app.core.rate_limit import limiter
from app.api.dependencies import get_api_key

router = APIRouter()


@router.post("/ingest")
@limiter.limit("20/minute")
async def ingest_documents(
    request: Request,
    source_path: str,
    collection: str = "default",
    api_key=Depends(get_api_key),  # Require authentication
) -> Dict:
    """
    Ingest documents from a source path

    Args:
        request: FastAPI Request object
        source_path: Path to documents
        collection: Collection name
        api_key: Authenticated API key

    Returns:
        Ingestion status
    """
    try:
        # Mock implementation
        return {
            "status": "success",
            "message": "Documents ingested successfully",
            "documents_processed": 10,
            "chunks_created": 50,
            "collection": collection,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingest/status/{task_id}")
async def get_ingestion_status(
    task_id: str,
    api_key=Depends(get_api_key),  # Require authentication
) -> Dict:
    """Get ingestion task status"""
    return {
        "task_id": task_id,
        "status": "completed",
        "progress": 100
    }
