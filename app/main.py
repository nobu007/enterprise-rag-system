"""
FastAPI Application Entry Point

This is the main application file that sets up the FastAPI server.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import uuid

from app.core.config import get_settings
from app.core.vectordb import get_vector_db
from app.core.embeddings import get_embedding_model
from app.core.logging_config import setup_logging, get_logger
from app.core.rate_limit import limiter
from app.services.retrieval import HybridRetriever
from app.services.rag_pipeline import RAGPipeline
from app.api.routes import query, health, ingest
from openai import AsyncOpenAI
from slowapi.errors import RateLimitExceeded


# Setup logging first
setup_logging()
logger = get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting Enterprise RAG System...")

    try:
        # Initialize OpenAI client
        logger.info("Initializing OpenAI async client...")
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        app.state.openai_client = openai_client

        # Initialize components
        logger.info("Initializing vector database...")
        vector_db = get_vector_db(
            db_type="faiss",
            index_path=settings.faiss_index_path
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
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=openai_client,
            llm_model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )
        app.state.rag_pipeline = rag_pipeline

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


async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom error handler for rate limit exceeded.

    Args:
        request: FastAPI request object
        exc: RateLimitExceeded exception

    Returns:
        JSONResponse with 429 status code
    """
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": str(exc.retry_after) if hasattr(exc, 'retry_after') else None
        },
        headers={"Retry-After": str(exc.retry_after)} if hasattr(exc, 'retry_after') else {}
    )


# Configure rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Use configured origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
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
@limiter.limit("120/minute")
async def root(request: Request):
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
@limiter.limit("120/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.app_version
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
