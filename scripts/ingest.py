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
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.document_loader import DocumentLoader, TextSplitter
from app.core.embeddings import get_embedding_model
from app.core.vectordb import get_vector_db
from app.core.config import get_settings


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
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Enterprise RAG System - Document Ingestion           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Step 1: Load documents
        print(f"📂 Loading documents from: {args.source}")
        print(f"   Recursive: {args.recursive}")
        
        documents = DocumentLoader.load_directory(
            directory_path=args.source,
            recursive=args.recursive
        )
        
        if not documents:
            print("❌ No documents found!")
            sys.exit(1)
        
        print(f"✅ Loaded {len(documents)} documents")
        
        # Step 2: Split documents into chunks
        print(f"\n📝 Splitting documents into chunks...")
        print(f"   Chunk size: {args.chunk_size}")
        print(f"   Chunk overlap: {args.chunk_overlap}")
        
        splitter = TextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        print(f"\n🧠 Generating embeddings...")
        
        settings = get_settings()
        print(f"   Model: {settings.embedding_model}")
        
        embedding_model = get_embedding_model()
        
        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]
            embeddings = embedding_model.embed_texts(texts)
            all_embeddings.extend(embeddings)
            
            print(f"   Progress: {min(i + batch_size, len(chunks))}/{len(chunks)}")
        
        print(f"✅ Generated {len(all_embeddings)} embeddings")
        
        # Step 4: Store in vector database
        print(f"\n💾 Storing in vector database...")
        print(f"   Database type: {args.db_type}")
        
        if args.db_type == "faiss":
            vector_db = get_vector_db(db_type="faiss", index_path=args.index_path)
            vector_db.connect()
            
            # Create index if it doesn't exist
            if vector_db.index is None:
                print(f"   Creating new FAISS index (dimension: {embedding_model.dimension})")
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
            print(f"   Saving index to: {args.index_path}")
            vector_db.save(args.index_path)
        
        # Step 5: Summary
        print("\n" + "="*60)
        print("✅ INGESTION COMPLETE!")
        print("="*60)
        print(f"Documents processed: {len(documents)}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Embeddings generated: {len(all_embeddings)}")
        print(f"Collection: {args.collection}")
        print(f"Database: {args.db_type}")
        print("="*60)
        
        # Get stats
        stats = vector_db.get_stats()
        print(f"\nDatabase stats:")
        print(f"  Total vectors: {stats.get('total_vectors', 'N/A')}")
        print(f"  Dimension: {stats.get('dimension', 'N/A')}")
        
        print("\n🚀 Ready to query! Start the API server with:")
        print("   uvicorn app.main:app --reload")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
