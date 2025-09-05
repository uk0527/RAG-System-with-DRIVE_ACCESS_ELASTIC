"""
Tests for the generation module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from app.core.generate import (
    is_unsafe_query, calculate_grounding_score, is_grounded,
    create_prompt_template, format_context_for_prompt,
    create_huggingface_llm, generate_answer, generate_extractive_answer,
    format_citations
)

# =========================
# Test Fixtures
# =========================

@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results for testing."""
    return [
        {
            "_id": "1",
            "_score": 0.95,
            "_source": {
                "content": "The company policy allows remote work for up to 3 days per week.",
                "metadata": {
                    "filename": "employee_handbook.pdf",
                    "drive_url": "https://drive.google.com/file/1",
                    "chunk_id": "handbook_0"
                }
            }
        },
        {
            "_id": "2",
            "_score": 0.85,
            "_source": {
                "content": "Remote work requires manager approval and must be requested in advance.",
                "metadata": {
                    "filename": "remote_work_policy.pdf",
                    "drive_url": "https://drive.google.com/file/2",
                    "chunk_id": "policy_0"
                }
            }
        }
    ]

@pytest.fixture
def mock_llm_response():
    """Mock LLM response."""
    return "The company allows remote work for up to 3 days per week with manager approval."

# =========================
# Safety Tests
# =========================

def test_is_unsafe_query():
    """Test unsafe query detection."""
    # Safe queries
    assert not is_unsafe_query("What is the company policy?")
    assert not is_unsafe_query("How do I submit a vacation request?")
    assert not is_unsafe_query("")
    
    # Unsafe queries
    assert is_unsafe_query("How to make a bomb")
    assert is_unsafe_query("Where can I buy drugs")
    assert is_unsafe_query("How to commit suicide")
    assert is_unsafe_query("How to create ransomware")
    
    # Case insensitive
    assert is_unsafe_query("HOW TO MAKE A BOMB")
    assert is_unsafe_query("How To Make A Bomb")

def test_is_unsafe_query_edge_cases():
    """Test unsafe query detection edge cases."""
    # Partial matches
    assert is_unsafe_query("bombing the competition")
    assert is_unsafe_query("drug testing policy")
    
    # False positives (should be safe)
    assert not is_unsafe_query("bombastic presentation")
    assert not is_unsafe_query("drug store location")

# =========================
# Grounding Tests
# =========================

def test_calculate_grounding_score():
    """Test grounding score calculation."""
    answer = "The company allows remote work for 3 days per week"
    context = "The company policy allows remote work for up to 3 days per week with manager approval"
    
    score = calculate_grounding_score(answer, context)
    
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Should be well grounded

def test_calculate_grounding_score_no_overlap():
    """Test grounding score with no overlap."""
    answer = "The weather is sunny today"
    context = "The company policy allows remote work for up to 3 days per week"
    
    score = calculate_grounding_score(answer, context)
    
    assert score == 0.0

def test_calculate_grounding_score_empty_inputs():
    """Test grounding score with empty inputs."""
    assert calculate_grounding_score("", "context") == 0.0
    assert calculate_grounding_score("answer", "") == 0.0
    assert calculate_grounding_score("", "") == 0.0

def test_is_grounded():
    """Test grounding check."""
    answer = "The company allows remote work for 3 days per week"
    context = "The company policy allows remote work for up to 3 days per week with manager approval"
    
    assert is_grounded(answer, context, min_overlap=0.1)
    assert not is_grounded(answer, context, min_overlap=0.9)

# =========================
# Prompt Template Tests
# =========================

def test_create_prompt_template():
    """Test prompt template creation."""
    template = create_prompt_template()
    
    assert template is not None
    assert hasattr(template, 'format')
    
    # Test formatting
    formatted = template.format(context="Test context", question="Test question")
    assert "Test context" in formatted
    assert "Test question" in formatted
    assert "Answer:" in formatted

def test_format_context_for_prompt(sample_retrieval_results):
    """Test context formatting for prompts."""
    context = format_context_for_prompt(sample_retrieval_results, max_tokens=1000)
    
    assert "[1]" in context
    assert "[2]" in context
    assert "employee_handbook.pdf" in context
    assert "remote_work_policy.pdf" in context
    assert "remote work" in context

def test_format_context_for_prompt_token_limit(sample_retrieval_results):
    """Test context formatting with token limit."""
    # Very low token limit
    context = format_context_for_prompt(sample_retrieval_results, max_tokens=50)
    
    # Should be truncated
    assert len(context) < len(format_context_for_prompt(sample_retrieval_results, max_tokens=1000))

# =========================
# Hugging Face LLM Tests
# =========================

@patch('app.core.generate.HuggingFaceEndpoint')
def test_create_huggingface_llm(mock_endpoint_class):
    """Test Hugging Face LLM creation."""
    mock_endpoint = Mock()
    mock_endpoint_class.return_value = mock_endpoint
    
    llm = create_huggingface_llm()
    
    assert llm == mock_endpoint
    mock_endpoint_class.assert_called_once()

# =========================
# Main Generation Tests
# =========================

@patch('app.core.generate.create_huggingface_llm')
@patch('app.core.generate.create_prompt_template')
def test_generate_answer_success(mock_template, mock_llm_factory, sample_retrieval_results, mock_llm_response):
    """Test successful answer generation."""
    # Mock LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value = mock_llm_response
    mock_llm_factory.return_value = mock_llm
    
    # Mock template
    mock_template.return_value.format.return_value = "formatted prompt"
    
    result = generate_answer("What is the remote work policy?", sample_retrieval_results)
    
    assert result["answer"] == mock_llm_response
    assert len(result["citations"]) == 2
    assert result["grounding_score"] > 0
    assert result["is_grounded"] is True
    assert result["latency_ms"] > 0

@patch('app.core.generate.create_huggingface_llm')
def test_generate_answer_unsafe_query(mock_llm_factory, sample_retrieval_results):
    """Test answer generation with unsafe query."""
    result = generate_answer("How to make a bomb", sample_retrieval_results)
    
    assert result["answer"] == "I cannot help with that request."
    assert result["citations"] == []
    assert result["score"] == 0.0
    assert result["grounding_score"] == 0.0
    assert result["is_grounded"] is False
    assert result["latency_ms"] == 0

def test_generate_answer_empty_results():
    """Test answer generation with empty results."""
    result = generate_answer("What is the policy?", [])
    
    assert result["answer"] == "I don't know."
    assert result["citations"] == []
    assert result["score"] == 0.0
    assert result["grounding_score"] == 0.0
    assert result["is_grounded"] is False
    assert result["latency_ms"] == 0

@patch('app.core.generate.create_huggingface_llm')
def test_generate_answer_llm_error(mock_llm_factory, sample_retrieval_results):
    """Test answer generation with LLM error."""
    # Mock LLM that raises exception
    mock_llm = Mock()
    mock_llm.invoke.side_effect = Exception("LLM error")
    mock_llm_factory.return_value = mock_llm
    
    result = generate_answer("What is the policy?", sample_retrieval_results)
    
    # Should fall back to extractive answer
    assert "remote work" in result["answer"].lower()
    assert result["citations"] == []
    assert result["latency_ms"] > 0

@patch('app.core.generate.create_huggingface_llm')
def test_generate_answer_ungrounded_response(mock_llm_factory, sample_retrieval_results):
    """Test answer generation with ungrounded LLM response."""
    # Mock LLM that returns ungrounded response
    mock_llm = Mock()
    mock_llm.invoke.return_value = "The weather is sunny today"
    mock_llm_factory.return_value = mock_llm
    
    result = generate_answer("What is the policy?", sample_retrieval_results)
    
    # Should fall back to extractive answer due to low grounding
    assert "remote work" in result["answer"].lower()
    assert result["grounding_score"] > 0

# =========================
# Extractive Answer Tests
# =========================

def test_generate_extractive_answer(sample_retrieval_results):
    """Test extractive answer generation."""
    result = generate_extractive_answer(
        "What is the remote work policy?",
        sample_retrieval_results,
        0.0  # start_time
    )
    
    assert "remote work" in result["answer"].lower()
    assert len(result["citations"]) == 2
    assert result["score"] > 0
    assert result["grounding_score"] > 0
    assert result["is_grounded"] is True
    assert result["latency_ms"] >= 0

def test_generate_extractive_answer_no_match():
    """Test extractive answer with no matching content."""
    results = [{
        "_id": "1",
        "_score": 0.9,
        "_source": {
            "content": "This is completely unrelated content about cooking recipes.",
            "metadata": {
                "filename": "cookbook.pdf",
                "drive_url": "https://drive.google.com/file/1",
                "chunk_id": "cookbook_0"
            }
        }
    }]
    
    result = generate_extractive_answer(
        "What is the remote work policy?",
        results,
        0.0
    )
    
    assert result["answer"] == "I don't know."
    assert result["score"] == 0.0
    assert result["grounding_score"] == 0.0
    assert result["is_grounded"] is False

# =========================
# Citation Formatting Tests
# =========================

def test_format_citations(sample_retrieval_results):
    """Test citation formatting."""
    citations = format_citations(sample_retrieval_results, max_citations=2)
    
    assert len(citations) == 2
    assert citations[0]["id"] == 1
    assert citations[0]["title"] == "employee_handbook.pdf"
    assert citations[0]["link"] == "https://drive.google.com/file/1"
    assert "remote work" in citations[0]["snippet"]
    assert citations[0]["score"] == 0.95
    assert citations[0]["chunk_id"] == "handbook_0"

def test_format_citations_max_limit(sample_retrieval_results):
    """Test citation formatting with max limit."""
    citations = format_citations(sample_retrieval_results, max_citations=1)
    
    assert len(citations) == 1
    assert citations[0]["id"] == 1

def test_format_citations_empty_results():
    """Test citation formatting with empty results."""
    citations = format_citations([])
    
    assert len(citations) == 0

def test_format_citations_snippet_truncation():
    """Test citation snippet truncation."""
    long_content = "This is a very long content that should be truncated. " * 20
    results = [{
        "_id": "1",
        "_score": 0.9,
        "_source": {
            "content": long_content,
            "metadata": {
                "filename": "long_doc.pdf",
                "drive_url": "https://drive.google.com/file/1",
                "chunk_id": "long_doc_0"
            }
        }
    }]
    
    citations = format_citations(results)
    
    assert len(citations[0]["snippet"]) <= 280

# =========================
# Error Handling Tests
# =========================

@patch('app.core.generate.create_huggingface_llm')
def test_generate_answer_connection_error(mock_llm_factory, sample_retrieval_results):
    """Test answer generation with connection error."""
    # Mock LLM that raises connection error
    mock_llm = Mock()
    mock_llm.invoke.side_effect = ConnectionError("Connection failed")
    mock_llm_factory.return_value = mock_llm
    
    result = generate_answer("What is the policy?", sample_retrieval_results)
    
    # Should fall back to extractive answer
    assert "remote work" in result["answer"].lower()
    assert result["latency_ms"] > 0

def test_generate_answer_malformed_results():
    """Test answer generation with malformed results."""
    malformed_results = [
        {
            "_id": "1",
            "_score": 0.9,
            "_source": {
                # Missing content
                "metadata": {
                    "filename": "test.pdf",
                    "drive_url": "https://drive.google.com/file/1",
                    "chunk_id": "test_0"
                }
            }
        }
    ]
    
    result = generate_answer("What is the policy?", malformed_results)
    
    # Should handle gracefully
    assert result["answer"] in ["I don't know.", ""]
    assert result["citations"] == []
