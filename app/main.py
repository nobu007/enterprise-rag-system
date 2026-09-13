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
from app.core.logging_config import setup_logging, get_logger
from app.api.routes import health, documents


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
        logger.info("Initializing vector database...")
        vector_db = get_vector_db(
            db_type="faiss",
            index_path=settings.faiss_index_path
        )
        vector_db.connect()
        app.state.vector_db = vector_db

        logger.info("Initializing embedding model...")
        app.state.embedding_model = get_embedding_model()

        logger.info("Enterprise RAG System ready!")

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        raise RuntimeError(f"Enterprise RAG System initialization failed: {e}") from e

    yield

    # Shutdown
    logger.info("Shutting down Enterprise RAG System...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
Enterprise RAG System API / 企業向けRAGシステム API

## Features / 機能

* **Document Ingestion**: Load, validate, chunk, embed, and store documents / ドキュメントの読み込み・検証・チャンク分割・埋め込み・保存
* **Multi-collection**: Logical separation and management of document collections / ドキュメントコレクションの論理的分離と管理
* **Validation**: PII / XSS / SQL injection / content-quality checks before ingestion / 取り込み前のPII・XSS・SQLインジェクション・品質チェック

## Documentation / ドキュメント

* **Swagger UI**: Interactive API documentation at `/docs` `/docs`でのインタラクティブなAPIドキュメント
* **ReDoc**: Alternative documentation at `/redoc` `/redoc`での代替ドキュメント
* **OpenAPI JSON**: Schema export at `/openapi.json` `/openapi.json`でのスキーマエクスポート
    """,
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_tags=[
        {
            "name": "Documents",
            "description": "Document registration and management / ドキュメント登録と管理"
        },
        {
            "name": "Health",
            "description": "Health checks and system information / ヘルスチェックとシステム情報"
        }
    ],
    contact={
        "name": "API Support",
        "email": "support@example.com",
        "url": "https://dev.azure.com/jinno0/enterprise-rag-system/_git/enterprise-rag-sys"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Use configured origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=settings.ALLOWED_HEADERS_LIST,  # Security: restrict allowed headers
)


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug
    )
