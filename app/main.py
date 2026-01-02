"""
FastAPI Application Entry Point

This is the main application file that sets up the FastAPI server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.core.config import get_settings
from app.core.vectordb import get_vector_db
from app.core.embeddings import get_embedding_model
from app.services.retrieval import HybridRetriever
from app.services.rag_pipeline import RAGPipeline
from app.api.routes import query, documents


settings = get_settings()


# Global instances (initialized at startup)
_rag_pipeline: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    print("🚀 Starting Enterprise RAG System...")
    
    global _rag_pipeline
    
    try:
        # Initialize components
        print("📊 Initializing vector database...")
        vector_db = get_vector_db(
            db_type="faiss",
            index_path="./data/faiss_index.bin"
        )
        vector_db.connect()
        
        print("🧠 Initializing embedding model...")
        embedding_model = get_embedding_model()
        
        print("🔍 Initializing retriever...")
        retriever = HybridRetriever(
            vector_db=vector_db,
            embedding_model=embedding_model,
            alpha=settings.hybrid_search_alpha
        )
        
        print("🤖 Initializing RAG pipeline...")
        _rag_pipeline = RAGPipeline(
            retriever=retriever,
            llm_model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )
        
        print("✅ Enterprise RAG System ready!")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("⚠️  Some features may not work correctly")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Enterprise RAG System...")


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


# Include routers
app.include_router(query.router, prefix="/api")
app.include_router(documents.router, prefix="/api")


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
