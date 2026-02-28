#!/usr/bin/env python3
"""
Document Ingestion Script

This script ingests documents into the RAG system's vector database.

Usage:
    python scripts/ingest.py --source ./data/documents
    python scripts/ingest.py --source ./data/documents --collection my-docs
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.document_loader import DocumentLoader, TextSplitter
from app.core.embeddings import get_embedding_model
from app.core.vectordb import get_vector_db
from app.core.config import get_settings
from app.core.logging_config import setup_logging, get_logger

# Setup logging for CLI script
setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to directory containing documents"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="Collection name for the documents"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for text splitting"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories"
    )
    parser.add_argument(
        "--db-type",
        type=str,
        choices=["faiss", "pinecone"],
        default="faiss",
        help="Vector database type"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="./data/faiss_index.bin",
        help="Path to FAISS index file (only for FAISS)"
    )

    args = parser.parse_args()

logger.info("Enterprise RAG System - Document Ingestion starting")

    try:
        # Step 1: Load documents
        logger.info(f"Loading documents from: {args.source} (recursive={args.recursive})")

        documents = DocumentLoader.load_directory(
            directory_path=args.source,
            recursive=args.recursive
        )

        if not documents:
            logger.error("No documents found!")
            sys.exit(1)

        logger.info(f"Loaded {len(documents)} documents")

        # Step 2: Split documents into chunks
        logger.info(f"Splitting documents (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})")
        splitter = TextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        chunks = splitter.split_documents(documents)

        logger.info(f"Created {len(chunks)} chunks")

        # Step 3: Generate embeddings
        logger.info("Generating embeddings...")

        settings = get_settings()
        logger.info(f"Embedding model: {settings.embedding_model}")

        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]
            embeddings = embedding_model.embed_texts(texts)
            all_embeddings.extend(embeddings)

logger.info(f"Embedding progress: {min(i + batch_size, len(chunks))}/{len(chunks)}")

        logger.info(f"Generated {len(all_embeddings)} embeddings")

        # Step 4: Store in vector database
        logger.info(f"Storing in vector database (type={args.db_type})")

        if args.db_type == "faiss":
            vector_db = get_vector_db(db_type="faiss", index_path=args.index_path)
            vector_db.connect()

            # Create index if it doesn't exist
            if vector_db.index is None:
                logger.info(f"Creating new FAISS index (dimension: {embedding_model.dimension})")
                vector_db.create_index(dimension=embedding_model.dimension)

        elif args.db_type == "pinecone":
            vector_db = get_vector_db(
                db_type="pinecone",
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment,
                index_name=settings.pinecone_index_name
            )
            vector_db.connect()

        # Prepare data
        ids = [chunk.doc_id for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]

        # Add collection to metadata
        for meta in metadata:
            meta['collection'] = args.collection

        # Upsert vectors
        vector_db.upsert(
            vectors=all_embeddings,
            ids=ids,
            metadata=metadata
        )

        # Save FAISS index
        if args.db_type == "faiss" and hasattr(vector_db, 'save'):
            logger.info(f"Saving index to: {args.index_path}")
            vector_db.save(args.index_path)

        # Step 5: Summary
        logger.info(
            f"INGESTION COMPLETE - Documents: {len(documents)}, "
            f"Chunks: {len(chunks)}, Embeddings: {len(all_embeddings)}, "
            f"Collection: {args.collection}, Database: {args.db_type}"
        )

        # Get stats
        stats = vector_db.get_stats()
        logger.info(
            f"Database stats - Total vectors: {stats.get('total_vectors', 'N/A')}, "
            f"Dimension: {stats.get('dimension', 'N/A')}"
        )

        logger.info("Ready to query! Start the API server with: uvicorn app.main:app --reload")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
