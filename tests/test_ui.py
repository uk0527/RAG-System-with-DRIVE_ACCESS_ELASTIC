"""
Tests for the UI module.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO

# Mock streamlit components
sys.modules['streamlit'] = Mock()
sys.modules['streamlit.components'] = Mock()
sys.modules['streamlit.components.v1'] = Mock()

from app.ui.app import (
    render_sidebar, render_main_content, format_citation,
    validate_user_input, process_query, display_results
)

# =========================
# Test Fixtures
# =========================

@pytest.fixture
def mock_streamlit():
    """Mock streamlit components."""
    mock_st = Mock()
    mock_st.title = Mock()
    mock_st.header = Mock()
    mock_st.subheader = Mock()
    mock_st.text = Mock()
    mock_st.write = Mock()
    mock_st.markdown = Mock()
    mock_st.text_input = Mock()
    mock_st.selectbox = Mock()
    mock_st.slider = Mock()
    mock_st.button = Mock()
    mock_st.sidebar = Mock()
    mock_st.error = Mock()
    mock_st.success = Mock()
    mock_st.warning = Mock()
    mock_st.info = Mock()
    mock_st.expander = Mock()
    mock_st.columns = Mock()
    mock_st.container = Mock()
    mock_st.empty = Mock()
    mock_st.session_state = {}
    return mock_st

@pytest.fixture
def sample_query_response():
    """Sample query response."""
    return {
        "answer": "The company allows remote work for up to 3 days per week.",
        "citations": [
            {
                "id": 1,
                "title": "Employee Handbook",
                "link": "https://drive.google.com/file/1",
                "snippet": "The company allows remote work for up to 3 days per week with manager approval.",
                "score": 0.95,
                "chunk_id": "handbook_0"
            }
        ],
        "score": 0.95,
        "grounding_score": 0.8,
        "is_grounded": True,
        "latency_ms": 150
    }

# =========================
# Sidebar Tests
# =========================

@patch('app.ui.app.st')
def test_render_sidebar(mock_st, mock_streamlit):
    """Test sidebar rendering."""
    # Mock sidebar components
    mock_st.sidebar.title.return_value = None
    mock_st.sidebar.selectbox.return_value = "hybrid"
    mock_st.sidebar.slider.return_value = 5
    mock_st.sidebar.slider.return_value = 0.2
    
    # Mock session state
    mock_st.session_state = {}
    
    # Test sidebar rendering
    result = render_sidebar()
    
    # Verify sidebar components were called
    mock_st.sidebar.title.assert_called()
    mock_st.sidebar.selectbox.assert_called()
    mock_st.sidebar.slider.assert_called()

@patch('app.ui.app.st')
def test_render_sidebar_default_values(mock_st, mock_streamlit):
    """Test sidebar with default values."""
    mock_st.sidebar.selectbox.return_value = "hybrid"
    mock_st.sidebar.slider.return_value = 5
    mock_st.sidebar.slider.return_value = 0.2
    mock_st.session_state = {}
    
    result = render_sidebar()
    
    # Verify default values are set
    assert result['mode'] == "hybrid"
    assert result['top_k'] == 5
    assert result['min_score'] == 0.2

# =========================
# Main Content Tests
# =========================

@patch('app.ui.app.st')
def test_render_main_content(mock_st, mock_streamlit):
    """Test main content rendering."""
    mock_st.title.return_value = None
    mock_st.text_input.return_value = "test query"
    mock_st.button.return_value = True
    mock_st.session_state = {}
    
    result = render_main_content()
    
    # Verify main components were called
    mock_st.title.assert_called()
    mock_st.text_input.assert_called()
    mock_st.button.assert_called()

@patch('app.ui.app.st')
def test_render_main_content_with_query(mock_st, mock_streamlit):
    """Test main content with user query."""
    mock_st.text_input.return_value = "What is the company policy?"
    mock_st.button.return_value = True
    mock_st.session_state = {}
    
    result = render_main_content()
    
    assert result['query'] == "What is the company policy?"
    assert result['submit_clicked'] is True

# =========================
# Citation Formatting Tests
# =========================

def test_format_citation():
    """Test citation formatting."""
    citation = {
        "id": 1,
        "title": "Employee Handbook",
        "link": "https://drive.google.com/file/1",
        "snippet": "The company allows remote work for up to 3 days per week.",
        "score": 0.95,
        "chunk_id": "handbook_0"
    }
    
    formatted = format_citation(citation)
    
    assert "[1]" in formatted
    assert "Employee Handbook" in formatted
    assert "https://drive.google.com/file/1" in formatted
    assert "remote work" in formatted
    assert "0.95" in formatted

def test_format_citation_minimal():
    """Test citation formatting with minimal data."""
    citation = {
        "id": 1,
        "title": "Test Document",
        "link": "https://example.com",
        "snippet": "Test content",
        "score": 0.8,
        "chunk_id": "test_0"
    }
    
    formatted = format_citation(citation)
    
    assert "[1]" in formatted
    assert "Test Document" in formatted
    assert "https://example.com" in formatted

def test_format_citation_long_snippet():
    """Test citation formatting with long snippet."""
    long_snippet = "This is a very long snippet that should be truncated. " * 20
    citation = {
        "id": 1,
        "title": "Long Document",
        "link": "https://example.com",
        "snippet": long_snippet,
        "score": 0.9,
        "chunk_id": "long_0"
    }
    
    formatted = format_citation(citation)
    
    # Should be truncated
    assert len(formatted) < len(long_snippet) + 100

# =========================
# Input Validation Tests
# =========================

def test_validate_user_input_valid():
    """Test user input validation with valid input."""
    result = validate_user_input("What is the company policy?")
    
    assert result['valid'] is True
    assert len(result['errors']) == 0

def test_validate_user_input_empty():
    """Test user input validation with empty input."""
    result = validate_user_input("")
    
    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert any('empty' in error.lower() for error in result['errors'])

def test_validate_user_input_whitespace():
    """Test user input validation with whitespace-only input."""
    result = validate_user_input("   \n\t   ")
    
    assert result['valid'] is False
    assert len(result['errors']) > 0

def test_validate_user_input_too_long():
    """Test user input validation with too long input."""
    long_input = "This is a very long query. " * 100
    result = validate_user_input(long_input)
    
    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert any('length' in error.lower() for error in result['errors'])

def test_validate_user_input_unsafe():
    """Test user input validation with unsafe input."""
    result = validate_user_input("How to make a bomb")
    
    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert any('unsafe' in error.lower() for error in result['errors'])

# =========================
# Query Processing Tests
# =========================

@patch('app.ui.app.requests.post')
def test_process_query_success(mock_post, sample_query_response):
    """Test successful query processing."""
    # Mock API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_query_response
    mock_post.return_value = mock_response
    
    result = process_query(
        query="What is the company policy?",
        mode="hybrid",
        top_k=5,
        min_score=0.2
    )
    
    assert result['success'] is True
    assert result['response'] == sample_query_response
    assert result['error'] is None

@patch('app.ui.app.requests.post')
def test_process_query_api_error(mock_post):
    """Test query processing with API error."""
    # Mock API error response
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"error": "API error"}
    mock_post.return_value = mock_response
    
    result = process_query(
        query="What is the company policy?",
        mode="hybrid",
        top_k=5,
        min_score=0.2
    )
    
    assert result['success'] is False
    assert result['response'] is None
    assert "API error" in result['error']

@patch('app.ui.app.requests.post')
def test_process_query_connection_error(mock_post):
    """Test query processing with connection error."""
    mock_post.side_effect = Exception("Connection error")
    
    result = process_query(
        query="What is the company policy?",
        mode="hybrid",
        top_k=5,
        min_score=0.2
    )
    
    assert result['success'] is False
    assert result['response'] is None
    assert "Connection error" in result['error']

@patch('app.ui.app.requests.post')
def test_process_query_timeout(mock_post):
    """Test query processing with timeout."""
    mock_post.side_effect = Exception("Request timeout")
    
    result = process_query(
        query="What is the company policy?",
        mode="hybrid",
        top_k=5,
        min_score=0.2
    )
    
    assert result['success'] is False
    assert result['response'] is None
    assert "timeout" in result['error'].lower()

# =========================
# Results Display Tests
# =========================

@patch('app.ui.app.st')
def test_display_results_success(mock_st, sample_query_response):
    """Test successful results display."""
    mock_st.subheader.return_value = None
    mock_st.write.return_value = None
    mock_st.markdown.return_value = None
    mock_st.expander.return_value = None
    
    display_results(sample_query_response)
    
    # Verify display components were called
    mock_st.subheader.assert_called()
    mock_st.write.assert_called()
    mock_st.markdown.assert_called()

@patch('app.ui.app.st')
def test_display_results_with_citations(mock_st, sample_query_response):
    """Test results display with citations."""
    mock_st.subheader.return_value = None
    mock_st.write.return_value = None
    mock_st.markdown.return_value = None
    mock_st.expander.return_value = None
    
    display_results(sample_query_response)
    
    # Should display citations
    assert mock_st.expander.called

@patch('app.ui.app.st')
def test_display_results_no_citations(mock_st):
    """Test results display without citations."""
    response_no_citations = {
        "answer": "I don't know.",
        "citations": [],
        "score": 0.0,
        "grounding_score": 0.0,
        "is_grounded": False,
        "latency_ms": 0
    }
    
    mock_st.subheader.return_value = None
    mock_st.write.return_value = None
    mock_st.markdown.return_value = None
    
    display_results(response_no_citations)
    
    # Should not display citations expander
    mock_st.expander.assert_not_called()

@patch('app.ui.app.st')
def test_display_results_ungrounded(mock_st):
    """Test results display with ungrounded response."""
    ungrounded_response = {
        "answer": "The weather is sunny today.",
        "citations": [],
        "score": 0.0,
        "grounding_score": 0.1,
        "is_grounded": False,
        "latency_ms": 100
    }
    
    mock_st.subheader.return_value = None
    mock_st.write.return_value = None
    mock_st.markdown.return_value = None
    mock_st.warning.return_value = None
    
    display_results(ungrounded_response)
    
    # Should display warning for ungrounded response
    mock_st.warning.assert_called()

# =========================
# Error Handling Tests
# =========================

@patch('app.ui.app.st')
def test_display_results_malformed_response(mock_st):
    """Test results display with malformed response."""
    malformed_response = {
        "answer": "Test answer"
        # Missing other required fields
    }
    
    mock_st.subheader.return_value = None
    mock_st.write.return_value = None
    mock_st.error.return_value = None
    
    display_results(malformed_response)
    
    # Should handle gracefully and show error
    mock_st.error.assert_called()

@patch('app.ui.app.st')
def test_display_results_none_response(mock_st):
    """Test results display with None response."""
    mock_st.error.return_value = None
    
    display_results(None)
    
    # Should handle gracefully and show error
    mock_st.error.assert_called()

# =========================
# Integration Tests
# =========================

@patch('app.ui.app.st')
@patch('app.ui.app.requests.post')
def test_full_ui_flow(mock_post, mock_st, sample_query_response):
    """Test full UI flow from input to display."""
    # Mock streamlit components
    mock_st.sidebar.selectbox.return_value = "hybrid"
    mock_st.sidebar.slider.return_value = 5
    mock_st.sidebar.slider.return_value = 0.2
    mock_st.text_input.return_value = "What is the company policy?"
    mock_st.button.return_value = True
    mock_st.session_state = {}
    
    # Mock API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_query_response
    mock_post.return_value = mock_response
    
    # Mock display components
    mock_st.subheader.return_value = None
    mock_st.write.return_value = None
    mock_st.markdown.return_value = None
    mock_st.expander.return_value = None
    
    # Test the full flow
    sidebar_config = render_sidebar()
    main_content = render_main_content()
    
    if main_content['submit_clicked'] and main_content['query']:
        validation = validate_user_input(main_content['query'])
        if validation['valid']:
            result = process_query(
                query=main_content['query'],
                mode=sidebar_config['mode'],
                top_k=sidebar_config['top_k'],
                min_score=sidebar_config['min_score']
            )
            if result['success']:
                display_results(result['response'])
    
    # Verify the flow completed successfully
    assert sidebar_config['mode'] == "hybrid"
    assert main_content['query'] == "What is the company policy?"
    assert validation['valid'] is True
    assert result['success'] is True
