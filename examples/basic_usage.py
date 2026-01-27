"""
Basic Usage Example for Enterprise RAG System

This example demonstrates how to:
1. Initialize the RAG pipeline
2. Ingest documents
3. Query the system
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval import HybridRetriever
from app.core.vectordb import get_vector_db
from app.core.embeddings import get_embedding_model
from app.services.document_loader import DocumentLoader


def example_basic_query():
    """Basic query example"""
    print("=" * 60)
    print("Enterprise RAG System - Basic Usage Example")
    print("=" * 60)

    # Initialize components
    print("\n📊 Initializing components...")
    vector_db = get_vector_db(
        db_type="faiss",
        index_path="./data/faiss_index.bin"
    )
    vector_db.connect()

    embedding_model = get_embedding_model()

    retriever = HybridRetriever(
        vector_db=vector_db,
        embedding_model=embedding_model,
        alpha=0.5  # Balance between semantic and keyword search
    )

    # Initialize RAG pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_model="gpt-4",
        temperature=0.7,
        max_tokens=2048
    )

    # Example queries
    questions = [
        "What is the company's remote work policy?",
        "How do I request time off?",
        "What benefits are available?"
    ]

    print("\n" + "=" * 60)
    print("Running Example Queries")
    print("=" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-" * 60)

        try:
            response = pipeline.query(
                question=question,
                top_k=5,
                use_hybrid=True
            )

            print(f"✅ Answer: {response.answer}")
            print(f"\n📊 Confidence: {response.confidence}")
            print(f"⏱️  Latency: {response.latency_ms}ms")
            print(f"🔤 Tokens: {response.tokens_used}")

            if response.sources:
                print(f"\n📚 Sources:")
                for source in response.sources:
                    print(f"  - {source['document']} (page {source.get('page', 'N/A')}) "
                          f"[relevance: {source['relevance_score']}]")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


def example_batch_query():
    """Batch query example"""
    print("\n" + "=" * 60)
    print("Batch Query Example")
    print("=" * 60)

    # Initialize pipeline (same as above)
    vector_db = get_vector_db(db_type="faiss", index_path="./data/faiss_index.bin")
    vector_db.connect()
    embedding_model = get_embedding_model()
    retriever = HybridRetriever(vector_db=vector_db, embedding_model=embedding_model)
    pipeline = RAGPipeline(retriever=retriever)

    # Batch query
    questions = [
        "What is the refund policy?",
        "How do I contact support?",
        "What are the pricing plans?"
    ]

    print(f"\nProcessing {len(questions)} questions...")

    responses = pipeline.batch_query(questions, top_k=3)

    for i, (question, response) in enumerate(zip(questions, responses), 1):
        print(f"\n{i}. {question}")
        print(f"   {response.answer[:100]}...")


def example_streaming():
    """Streaming query example"""
    print("\n" + "=" * 60)
    print("Streaming Query Example")
    print("=" * 60)

    # Initialize pipeline
    vector_db = get_vector_db(db_type="faiss", index_path="./data/faiss_index.bin")
    vector_db.connect()
    embedding_model = get_embedding_model()
    retriever = HybridRetriever(vector_db=vector_db, embedding_model=embedding_model)

    # Create streaming pipeline
    from app.services.rag_pipeline import StreamingRAGPipeline
    pipeline = StreamingRAGPipeline(retriever=retriever)

    question = "Explain the company culture in detail"

    print(f"\n❓ Question: {question}")
    print("\n🔄 Streaming response:")

    for chunk in pipeline.stream_query(question, top_k=5):
        if chunk['type'] == 'sources':
            print("\n📚 Sources retrieved:")
            for source in chunk['content']:
                print(f"  - {source['document']} [score: {source['score']}]")

        elif chunk['type'] == 'answer':
            print(chunk['content'], end='', flush=True)

        elif chunk['type'] == 'error':
            print(f"\n❌ Error: {chunk['content']}")

        elif chunk.get('done'):
            print(f"\n\n⏱️  Total latency: {chunk.get('latency_ms', 'N/A')}ms")


if __name__ == "__main__":
    # Run examples
    try:
        # Basic query
        example_basic_query()

        # Uncomment to run batch example:
        # example_batch_query()

        # Uncomment to run streaming example:
        # example_streaming()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
