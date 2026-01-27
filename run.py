#!/usr/bin/env python3
"""
CLI Tool for Enterprise RAG System

Usage:
    python run.py query "Your question here"
    python run.py ingest path/to/documents
    python run.py interactive
    python run.py server
    python run.py stats
"""

import argparse
import sys
from pathlib import Path

from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval import HybridRetriever
from app.core.vectordb import get_vector_db
from app.core.embeddings import get_embedding_model
from app.services.document_loader import DocumentLoader


def query_command(args):
    """Query the RAG system"""
    print("🔧 Initializing RAG pipeline...")

    # Initialize components
    vector_db = get_vector_db(
        db_type=args.db_type,
        index_path=args.index_path
    )
    vector_db.connect()

    embedding_model = get_embedding_model()
    retriever = HybridRetriever(
        vector_db=vector_db,
        embedding_model=embedding_model,
        alpha=args.alpha
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    print(f"\n❓ Question: {args.question}")
    print("\n⏳ Processing...")

    # Query
    response = pipeline.query(
        question=args.question,
        top_k=args.top_k,
        use_hybrid=not args.keyword_only
    )

    # Print result
    print("\n" + "=" * 60)
    print("✅ Answer")
    print("=" * 60)
    print(f"\n{response.answer}")

    print(f"\n📊 Metrics:")
    print(f"   Confidence: {response.confidence:.2f}")
    print(f"   Latency: {response.latency_ms}ms")
    print(f"   Tokens: {response.tokens_used}")

    if response.sources:
        print(f"\n📚 Sources ({len(response.sources)}):")
        for i, source in enumerate(response.sources, 1):
            doc = source.get('document', 'unknown')
            page = source.get('page', 'N/A')
            score = source.get('relevance_score', 0)
            print(f"   {i}. {doc} (page {page}) [relevance: {score:.3f}]")


def ingest_command(args):
    """Ingest documents into the knowledge base"""
    print("🔧 Initializing document loader...")

    # Initialize components
    vector_db = get_vector_db(
        db_type=args.db_type,
        index_path=args.index_path
    )
    vector_db.connect()

    embedding_model = get_embedding_model()
    loader = DocumentLoader(embedding_model=embedding_model)

    # Load documents
    source_path = Path(args.source)
    print(f"\n📂 Loading documents from: {source_path}")

    if source_path.is_file():
        documents = loader.load_file(str(source_path))
    elif source_path.is_dir():
        documents = loader.load_directory(str(source_path))
    else:
        print(f"❌ Error: Path not found: {source_path}")
        sys.exit(1)

    print(f"✅ Loaded {len(documents)} documents")

    # Chunk documents
    print("\n✂️  Chunking documents...")
    chunks = loader.chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    print(f"✅ Created {len(chunks)} chunks")

    # Generate embeddings and index
    print("\n🧠 Generating embeddings...")
    embeddings = embedding_model.embed_texts([chunk.text for chunk in chunks])

    print("💾 Indexing in vector database...")
    vector_db.add_documents(
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks]
    )

    print(f"✅ Successfully indexed {len(chunks)} chunks")


def interactive_command(args):
    """Interactive query mode"""
    print("🔧 Initializing RAG pipeline...")

    vector_db = get_vector_db(
        db_type=args.db_type,
        index_path=args.index_path
    )
    vector_db.connect()

    embedding_model = get_embedding_model()
    retriever = HybridRetriever(
        vector_db=vector_db,
        embedding_model=embedding_model
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_model=args.model
    )

    print("\n" + "=" * 60)
    print("🔄 Interactive Mode")
    print("=" * 60)
    print("Commands: query, help, quit\n")

    while True:
        try:
            user_input = input("📝 Enter question (or 'quit'): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            if user_input == 'help':
                print("\nAvailable commands:")
                print("  query     - Ask a question")
                print("  quit/exit - Exit")
                continue

            # Query
            print("\n⏳ Processing...")
            response = pipeline.query(user_input, top_k=5)

            print("\n" + "=" * 60)
            print("✅ Answer")
            print("=" * 60)
            print(f"\n{response.answer}")

            if response.sources:
                print(f"\n📚 {len(response.sources)} sources")
                for source in response.sources[:3]:
                    doc = source.get('document', 'unknown')[:40]
                    print(f"   • {doc}...")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def server_command(args):
    """Start API server"""
    import uvicorn

    print(f"\n🚀 Starting RAG API server")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def stats_command(args):
    """Show system statistics"""
    print("📊 System Statistics")
    print("=" * 60)

    # Check vector database
    index_path = Path(args.index_path)
    if index_path.exists():
        size_mb = index_path.stat().st_size / (1024 * 1024)
        print(f"\n💾 Vector Database:")
        print(f"   Path: {args.index_path}")
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"\n💾 Vector Database: Not initialized")
        print(f"   Run: python run.py ingest <documents>")

    # Model info
    print(f"\n🤖 Model Configuration:")
    print(f"   LLM: {args.model}")
    print(f"   Embeddings: {args.embedding_model or 'default'}")

    # Settings
    print(f"\n⚙️  Settings:")
    print(f"   Temperature: {args.temperature}")
    print(f"   Max Tokens: {args.max_tokens}")
    print(f"   Top K: {args.top_k}")
    print(f"   Alpha: {args.alpha}")


def batch_command(args):
    """Batch query from file"""
    print("🔧 Initializing RAG pipeline...")

    vector_db = get_vector_db(
        db_type=args.db_type,
        index_path=args.index_path
    )
    vector_db.connect()

    embedding_model = get_embedding_model()
    retriever = HybridRetriever(
        vector_db=vector_db,
        embedding_model=embedding_model,
        alpha=args.alpha
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_model=args.model,
        temperature=args.temperature
    )

    # Load questions
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: File '{args.file}' not found")
        sys.exit(1)

    print(f"\n📄 Processing {len(questions)} questions...")

    # Batch query
    responses = pipeline.batch_query(questions, top_k=args.top_k)

    # Results
    print("\n" + "=" * 60)
    print("📊 Batch Results")
    print("=" * 60)

    for i, (question, response) in enumerate(zip(questions, responses), 1):
        print(f"\n{i}. {question}")
        print(f"   {response.answer[:100]}...")
        print(f"   Confidence: {response.confidence:.2f} | Latency: {response.latency_ms}ms")

    # Summary
    avg_confidence = sum(r.confidence for r in responses) / len(responses)
    avg_latency = sum(r.latency_ms for r in responses) / len(responses)

    print("\n" + "-" * 60)
    print(f"Average Confidence: {avg_confidence:.2f}")
    print(f"Average Latency: {avg_latency:.0f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Enterprise RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py query "What is the refund policy?"
  python run.py ingest ./documents
  python run.py batch questions.txt
  python run.py interactive
  python run.py server
  python run.py stats
        """
    )

    # Global options
    parser.add_argument('--db-type', default='faiss', help='Vector database type')
    parser.add_argument('--index-path', default='./data/faiss_index.bin',
                       help='Vector database index path')
    parser.add_argument('--model', default='gpt-4', help='LLM model')
    parser.add_argument('--embedding-model', help='Embedding model name')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM temperature')
    parser.add_argument('--max-tokens', type=int, default=2048, help='Max tokens')
    parser.add_argument('--alpha', type=float, default=0.5, help='Hybrid search alpha')
    parser.add_argument('--top-k', type=int, default=5, help='Top K results')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--keyword-only', action='store_true',
                            help='Use keyword search only')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('source', help='File or directory to ingest')
    ingest_parser.add_argument('--chunk-size', type=int, default=500,
                             help='Chunk size for documents')
    ingest_parser.add_argument('--overlap', type=int, default=50,
                             help='Chunk overlap')

    # Interactive command
    subparsers.add_parser('interactive', help='Interactive query mode')

    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host')
    server_parser.add_argument('--port', type=int, default=8000, help='Port')
    server_parser.add_argument('--reload', action='store_true', help='Enable reload')

    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch query from file')
    batch_parser.add_argument('file', help='File with questions (one per line)')

    args = parser.parse_args()

    if args.command == 'query':
        query_command(args)
    elif args.command == 'ingest':
        ingest_command(args)
    elif args.command == 'interactive':
        interactive_command(args)
    elif args.command == 'server':
        server_command(args)
    elif args.command == 'stats':
        stats_command(args)
    elif args.command == 'batch':
        batch_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
