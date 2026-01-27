"""
Unit tests for RAG Pipeline
"""

import pytest
from unittest.mock import Mock, patch
from app.services.rag_pipeline import RAGPipeline, RAGResponse
from app.services.retrieval import RetrievalResult


@pytest.fixture
def mock_retriever():
    """Mock retriever"""
    retriever = Mock()
    return retriever


@pytest.fixture
def mock_llm_response():
    """Mock LLM response"""
    return {
        'answer': 'This is a test answer based on the context.',
        'tokens_used': 100,
        'finish_reason': 'stop'
    }


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results"""
    return [
        RetrievalResult(
            document='Sample document text 1',
            score=0.85,
            metadata={'filename': 'test1.pdf', 'page': 1}
        ),
        RetrievalResult(
            document='Sample document text 2',
            score=0.75,
            metadata={'filename': 'test2.pdf', 'page': 2}
        )
    ]


def test_rag_pipeline_initialization(mock_retriever):
    """Test RAG pipeline initialization"""
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_model='gpt-4',
        temperature=0.7,
        max_tokens=2048
    )

    assert pipeline.retriever == mock_retriever
    assert pipeline.llm_model == 'gpt-4'
    assert pipeline.temperature == 0.7
    assert pipeline.max_tokens == 2048


@patch('app.services.rag_pipeline.openai')
def test_rag_pipeline_query(mock_openai, mock_retriever, sample_retrieval_results, mock_llm_response):
    """Test RAG pipeline query"""
    # Setup mocks
    mock_retriever.retrieve.return_value = sample_retrieval_results
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content=mock_llm_response['answer']))
    ]
    mock_openai.chat.completions.create.return_value.usage.total_tokens = mock_llm_response['tokens_used']

    # Create pipeline
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_model='gpt-4'
    )

    # Run query
    response = pipeline.query("What is the test question?")

    # Assertions
    assert isinstance(response, RAGResponse)
    assert response.answer == mock_llm_response['answer']
    assert response.confidence > 0
    assert len(response.sources) == 2
    assert response.latency_ms > 0
    assert response.tokens_used == mock_llm_response['tokens_used']


@patch('app.services.rag_pipeline.openai')
def test_rag_pipeline_no_results(mock_openai, mock_retriever):
    """Test RAG pipeline with no retrieval results"""
    mock_retriever.retrieve.return_value = []

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_model='gpt-4'
    )

    response = pipeline.query("What is the test question?")

    assert isinstance(response, RAGResponse)
    assert "couldn't find any relevant information" in response.answer.lower()
    assert response.confidence == 0.0
    assert len(response.sources) == 0


def test_rag_pipeline_batch_query(mock_retriever, sample_retrieval_results):
    """Test batch query processing"""
    mock_retriever.retrieve.return_value = sample_retrieval_results

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_model='gpt-4'
    )

    questions = ["Question 1?", "Question 2?", "Question 3?"]
    responses = pipeline.batch_query(questions)

    assert len(responses) == 3


def test_confidence_calculation(mock_retriever, sample_retrieval_results):
    """Test confidence score calculation"""
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_model='gpt-4'
    )

    confidence = pipeline._calculate_confidence(
        sample_retrieval_results,
        "This is a reasonable test answer with enough length to be valid."
    )

    assert 0 <= confidence <= 1.0


def test_prompt_building():
    """Test prompt building"""
    pipeline = RAGPipeline(
        retriever=Mock(),
        llm_model='gpt-4'
    )

    prompt = pipeline._build_prompt(
        "What is AI?",
        "AI stands for Artificial Intelligence."
    )

    assert "What is AI?" in prompt
    assert "AI stands for Artificial Intelligence." in prompt
    assert "Answer the question" in prompt
