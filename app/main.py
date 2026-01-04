"""
FastAPI Application Entry Point

This is the main application file that sets up the FastAPI server.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import uuid

from app.core.config import get_settings
from app.core.vectordb import get_vector_db
from app.core.embeddings import get_embedding_model
from app.core.logging_config import setup_logging, get_logger
from app.services.retrieval import HybridRetriever
from app.services.rag_pipeline import RAGPipeline
from app.api.routes import query, health, ingest


# Setup logging first
setup_logging()
logger = get_logger(__name__)

settings = get_settings()


# Global instances (initialized at startup)
_rag_pipeline: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting Enterprise RAG System...")

    global _rag_pipeline

    try:
        # Initialize components
        logger.info("Initializing vector database...")
        vector_db = get_vector_db(
            db_type="faiss",
            index_path="./data/faiss_index.bin"
        )
        vector_db.connect()

        logger.info("Initializing embedding model...")
        embedding_model = get_embedding_model()

        logger.info("Initializing retriever...")
        retriever = HybridRetriever(
            vector_db=vector_db,
            embedding_model=embedding_model,
            alpha=settings.hybrid_search_alpha
        )

        logger.info("Initializing RAG pipeline...")
        _rag_pipeline = RAGPipeline(
            retriever=retriever,
            llm_model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )

        logger.info("Enterprise RAG System ready!")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        logger.warning("Some features may not work correctly")

    yield

    # Shutdown
    logger.info("Shutting down Enterprise RAG System...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade RAG pipeline for enterprise knowledge bases",
    lifespan=lifespan
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """
    Middleware to add unique request ID to each request for tracing.
    """
    # Generate or use existing request ID from header
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Store in request state for access in endpoints
    request.state.request_id = request_id

    # Add request ID to log context
    import logging
    from app.core.logging_config import RequestIDFilter

    # Create a custom filter for this request
    class RequestContextFilter(logging.Filter):
        def __init__(self, req_id):
            super().__init__()
            self.req_id = req_id

        def filter(self, record):
            record.request_id = self.req_id
            return True

    # Add filter to all loggers for this request
    context_filter = RequestContextFilter(request_id)
    root_logger = logging.getLogger()
    root_logger.addFilter(context_filter)

    try:
        response = await call_next(request)
        # Add request ID to response header
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        # Clean up filter after request
        root_logger.removeFilter(context_filter)


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.app_version
    }


def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAG pipeline instance"""
    if _rag_pipeline is None:
        raise RuntimeError("RAG pipeline not initialized")
    return _rag_pipeline


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
