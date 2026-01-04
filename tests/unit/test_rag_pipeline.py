"""
Unit tests for RAG Pipeline
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.rag_pipeline import RAGPipeline, RAGResponse
from app.services.retrieval import RetrievalResult


@pytest.fixture
def mock_retriever():
    """Mock retriever"""
    retriever = Mock()
    return retriever


@pytest.fixture
def mock_openai_client():
    """Mock async OpenAI client"""
    client = Mock()
    return client


@pytest.fixture
def mock_llm_response():
    """Mock LLM response"""
    return {
        'answer': 'This is a test answer based on the context.',
        'tokens_used': 100,
        'finish_reason': 'stop'
    }


@pytest.fixture
def mock_chat_completion():
    """Mock chat completion response"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = 'This is a test answer based on the context.'
    mock_response.choices[0].finish_reason = 'stop'
    mock_response.usage.total_tokens = 100
    return mock_response


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results"""
    return [
        RetrievalResult(
            document='Sample document text 1',
            score=0.85,
            metadata={'filename': 'test1.pdf', 'page': 1},
            source='test1.pdf'
        ),
        RetrievalResult(
            document='Sample document text 2',
            score=0.75,
            metadata={'filename': 'test2.pdf', 'page': 2},
            source='test2.pdf'
        )
    ]


def test_rag_pipeline_initialization(mock_retriever, mock_openai_client):
    """Test RAG pipeline initialization"""
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_client=mock_openai_client,
        llm_model='gpt-4',
        temperature=0.7,
        max_tokens=2048
    )

    assert pipeline.retriever == mock_retriever
    assert pipeline.llm_client == mock_openai_client
    assert pipeline.llm_model == 'gpt-4'
    assert pipeline.temperature == 0.7
    assert pipeline.max_tokens == 2048


@pytest.mark.asyncio
async def test_rag_pipeline_query(mock_openai_client, mock_retriever, sample_retrieval_results, mock_chat_completion):
    """Test RAG pipeline query"""
    # Setup mocks
    mock_retriever.retrieve.return_value = sample_retrieval_results
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)

    # Create pipeline
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_client=mock_openai_client,
        llm_model='gpt-4'
    )

    # Run query
    response = await pipeline.query("What is the test question?")

    # Assertions
    assert isinstance(response, RAGResponse)
    assert response.answer == mock_chat_completion.choices[0].message.content
    assert response.confidence > 0
    assert len(response.sources) == 2
    assert response.latency_ms >= 0  # Can be 0 in fast tests
    assert response.tokens_used == mock_chat_completion.usage.total_tokens


@pytest.mark.asyncio
async def test_rag_pipeline_no_results(mock_openai_client, mock_retriever):
    """Test RAG pipeline with no retrieval results"""
    mock_retriever.retrieve.return_value = []

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_client=mock_openai_client,
        llm_model='gpt-4'
    )

    response = await pipeline.query("What is the test question?")

    assert isinstance(response, RAGResponse)
    assert "couldn't find any relevant information" in response.answer.lower()
    assert response.confidence == 0.0
    assert len(response.sources) == 0


@pytest.mark.asyncio
async def test_rag_pipeline_batch_query(mock_openai_client, mock_retriever, sample_retrieval_results, mock_chat_completion):
    """Test batch query processing"""
    mock_retriever.retrieve.return_value = sample_retrieval_results
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_client=mock_openai_client,
        llm_model='gpt-4'
    )

    questions = ["Question 1?", "Question 2?", "Question 3?"]
    responses = await pipeline.batch_query(questions)

    assert len(responses) == 3


def test_confidence_calculation(mock_retriever, mock_openai_client, sample_retrieval_results):
    """Test confidence score calculation"""
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_client=mock_openai_client,
        llm_model='gpt-4'
    )

    confidence = pipeline._calculate_confidence(
        sample_retrieval_results,
        "This is a reasonable test answer with enough length to be valid."
    )

    assert 0 <= confidence <= 1.0


def test_prompt_building(mock_openai_client):
    """Test prompt building"""
    pipeline = RAGPipeline(
        retriever=Mock(),
        llm_client=mock_openai_client,
        llm_model='gpt-4'
    )

    prompt = pipeline._build_prompt(
        "What is AI?",
        "AI stands for Artificial Intelligence."
    )

    assert "What is AI?" in prompt
    assert "AI stands for Artificial Intelligence." in prompt
    assert "Answer the question" in prompt
