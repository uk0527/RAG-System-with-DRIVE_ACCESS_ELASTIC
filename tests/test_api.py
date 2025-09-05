"""
Tests for the API module.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.api.main import app
from app.api.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse
from app.api.guardrails import (
    check_safety, check_grounding, validate_query, 
    extract_citations, format_response
)

# =========================
# Test Fixtures
# =========================

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def sample_query_request():
    """Sample query request."""
    return {
        "query": "What is the company policy on remote work?",
        "mode": "hybrid",
        "top_k": 5,
        "min_score": 0.2
    }

@pytest.fixture
def sample_ingest_request():
    """Sample ingest request."""
    return {
        "folder_id": "test_folder_id",
        "force": False,
        "max_files": 10
    }

@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results."""
    return [
        {
            "_id": "1",
            "_score": 0.95,
            "_source": {
                "content": "The company allows remote work for up to 3 days per week.",
                "metadata": {
                    "filename": "employee_handbook.pdf",
                    "drive_url": "https://drive.google.com/file/1",
                    "chunk_id": "handbook_0"
                }
            }
        }
    ]

# =========================
# Schema Tests
# =========================

def test_query_request_schema(sample_query_request):
    """Test QueryRequest schema validation."""
    request = QueryRequest(**sample_query_request)
    
    assert request.query == "What is the company policy on remote work?"
    assert request.mode == "hybrid"
    assert request.top_k == 5
    assert request.min_score == 0.2

def test_query_request_defaults():
    """Test QueryRequest with default values."""
    request = QueryRequest(query="test query")
    
    assert request.query == "test query"
    assert request.mode == "hybrid"
    assert request.top_k == 5
    assert request.min_score == 0.2

def test_query_request_invalid_mode():
    """Test QueryRequest with invalid mode."""
    with pytest.raises(ValueError):
        QueryRequest(query="test", mode="invalid_mode")

def test_query_request_invalid_scores():
    """Test QueryRequest with invalid score values."""
    with pytest.raises(ValueError):
        QueryRequest(query="test", min_score=-0.1)
    
    with pytest.raises(ValueError):
        QueryRequest(query="test", min_score=1.1)

def test_ingest_request_schema(sample_ingest_request):
    """Test IngestRequest schema validation."""
    request = IngestRequest(**sample_ingest_request)
    
    assert request.folder_id == "test_folder_id"
    assert request.force is False
    assert request.max_files == 10

def test_ingest_request_defaults():
    """Test IngestRequest with default values."""
    request = IngestRequest(folder_id="test_folder")
    
    assert request.folder_id == "test_folder"
    assert request.force is False
    assert request.max_files == 100

def test_query_response_schema():
    """Test QueryResponse schema."""
    response = QueryResponse(
        answer="The company allows remote work for up to 3 days per week.",
        citations=[
            {
                "id": 1,
                "title": "Employee Handbook",
                "link": "https://drive.google.com/file/1",
                "snippet": "The company allows remote work...",
                "score": 0.95,
                "chunk_id": "handbook_0"
            }
        ],
        score=0.95,
        grounding_score=0.8,
        is_grounded=True,
        latency_ms=150
    )
    
    assert response.answer == "The company allows remote work for up to 3 days per week."
    assert len(response.citations) == 1
    assert response.score == 0.95
    assert response.grounding_score == 0.8
    assert response.is_grounded is True
    assert response.latency_ms == 150

def test_ingest_response_schema():
    """Test IngestResponse schema."""
    response = IngestResponse(
        files_seen=5,
        chunks_indexed=25,
        duration_seconds=10.5,
        success=True
    )
    
    assert response.files_seen == 5
    assert response.chunks_indexed == 25
    assert response.duration_seconds == 10.5
    assert response.success is True

# =========================
# Guardrails Tests
# =========================

def test_check_safety_safe_queries():
    """Test safety check with safe queries."""
    safe_queries = [
        "What is the company policy?",
        "How do I submit a vacation request?",
        "What are the benefits?",
        ""
    ]
    
    for query in safe_queries:
        assert check_safety(query) is True

def test_check_safety_unsafe_queries():
    """Test safety check with unsafe queries."""
    unsafe_queries = [
        "How to make a bomb",
        "Where can I buy drugs",
        "How to commit suicide",
        "How to create ransomware"
    ]
    
    for query in unsafe_queries:
        assert check_safety(query) is False

def test_check_grounding_grounded():
    """Test grounding check with grounded content."""
    answer = "The company allows remote work for 3 days per week"
    context = "The company policy allows remote work for up to 3 days per week with manager approval"
    
    assert check_grounding(answer, context, min_overlap=0.1) is True

def test_check_grounding_ungrounded():
    """Test grounding check with ungrounded content."""
    answer = "The weather is sunny today"
    context = "The company policy allows remote work for up to 3 days per week"
    
    assert check_grounding(answer, context, min_overlap=0.1) is False

def test_validate_query_valid():
    """Test query validation with valid query."""
    result = validate_query("What is the company policy?")
    
    assert result['valid'] is True
    assert len(result['errors']) == 0

def test_validate_query_empty():
    """Test query validation with empty query."""
    result = validate_query("")
    
    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert any('empty' in error.lower() for error in result['errors'])

def test_validate_query_too_long():
    """Test query validation with too long query."""
    long_query = "This is a very long query. " * 100
    result = validate_query(long_query)
    
    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert any('length' in error.lower() for error in result['errors'])

def test_validate_query_unsafe():
    """Test query validation with unsafe query."""
    result = validate_query("How to make a bomb")
    
    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert any('unsafe' in error.lower() for error in result['errors'])

def test_extract_citations(sample_retrieval_results):
    """Test citation extraction."""
    citations = extract_citations(sample_retrieval_results, max_citations=5)
    
    assert len(citations) == 1
    assert citations[0]['id'] == 1
    assert citations[0]['title'] == "employee_handbook.pdf"
    assert citations[0]['link'] == "https://drive.google.com/file/1"
    assert "remote work" in citations[0]['snippet']
    assert citations[0]['score'] == 0.95
    assert citations[0]['chunk_id'] == "handbook_0"

def test_extract_citations_max_limit(sample_retrieval_results):
    """Test citation extraction with max limit."""
    citations = extract_citations(sample_retrieval_results, max_citations=0)
    
    assert len(citations) == 0

def test_format_response():
    """Test response formatting."""
    response_data = {
        "answer": "Test answer",
        "citations": [{"id": 1, "title": "Test"}],
        "score": 0.9,
        "grounding_score": 0.8,
        "is_grounded": True,
        "latency_ms": 100
    }
    
    response = format_response(response_data)
    
    assert isinstance(response, QueryResponse)
    assert response.answer == "Test answer"
    assert len(response.citations) == 1
    assert response.score == 0.9

# =========================
# API Endpoint Tests
# =========================

@patch('app.api.main.retrieve_documents')
@patch('app.api.main.generate_answer')
def test_query_endpoint_success(mock_generate, mock_retrieve, client, sample_query_request, sample_retrieval_results):
    """Test successful query endpoint."""
    # Mock retrieval
    mock_retrieve.return_value = sample_retrieval_results
    
    # Mock generation
    mock_generate.return_value = {
        "answer": "The company allows remote work for up to 3 days per week.",
        "citations": [{"id": 1, "title": "Employee Handbook", "link": "https://drive.google.com/file/1", "snippet": "remote work", "score": 0.95, "chunk_id": "handbook_0"}],
        "score": 0.95,
        "grounding_score": 0.8,
        "is_grounded": True,
        "latency_ms": 150
    }
    
    response = client.post("/query", json=sample_query_request)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "score" in data
    assert "grounding_score" in data
    assert "is_grounded" in data
    assert "latency_ms" in data

@patch('app.api.main.retrieve_documents')
def test_query_endpoint_no_results(mock_retrieve, client, sample_query_request):
    """Test query endpoint with no results."""
    mock_retrieve.return_value = []
    
    response = client.post("/query", json=sample_query_request)
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I don't know."
    assert data["citations"] == []
    assert data["score"] == 0.0

def test_query_endpoint_invalid_request(client):
    """Test query endpoint with invalid request."""
    invalid_request = {
        "query": "",  # Empty query
        "mode": "invalid_mode"
    }
    
    response = client.post("/query", json=invalid_request)
    
    assert response.status_code == 422  # Validation error

def test_query_endpoint_unsafe_query(client):
    """Test query endpoint with unsafe query."""
    unsafe_request = {
        "query": "How to make a bomb",
        "mode": "hybrid"
    }
    
    response = client.post("/query", json=unsafe_request)
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I cannot help with that request."
    assert data["citations"] == []

@patch('app.api.main.ingest_documents')
def test_ingest_endpoint_success(mock_ingest, client, sample_ingest_request):
    """Test successful ingest endpoint."""
    mock_ingest.return_value = {
        "files_seen": 5,
        "chunks_indexed": 25,
        "duration_seconds": 10.5
    }
    
    response = client.post("/ingest", json=sample_ingest_request)
    
    assert response.status_code == 200
    data = response.json()
    assert data["files_seen"] == 5
    assert data["chunks_indexed"] == 25
    assert data["duration_seconds"] == 10.5
    assert data["success"] is True

@patch('app.api.main.ingest_documents')
def test_ingest_endpoint_error(mock_ingest, client, sample_ingest_request):
    """Test ingest endpoint with error."""
    mock_ingest.side_effect = Exception("Ingestion failed")
    
    response = client.post("/ingest", json=sample_ingest_request)
    
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert "Ingestion failed" in data["error"]

def test_ingest_endpoint_missing_folder_id(client):
    """Test ingest endpoint with missing folder ID."""
    invalid_request = {
        "folder_id": None,
        "force": False
    }
    
    response = client.post("/ingest", json=invalid_request)
    
    assert response.status_code == 422  # Validation error

def test_healthz_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/healthz")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

# =========================
# Error Handling Tests
# =========================

@patch('app.api.main.retrieve_documents')
def test_query_endpoint_retrieval_error(mock_retrieve, client, sample_query_request):
    """Test query endpoint with retrieval error."""
    mock_retrieve.side_effect = Exception("Retrieval failed")
    
    response = client.post("/query", json=sample_query_request)
    
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert "Retrieval failed" in data["error"]

@patch('app.api.main.retrieve_documents')
@patch('app.api.main.generate_answer')
def test_query_endpoint_generation_error(mock_generate, mock_retrieve, client, sample_query_request, sample_retrieval_results):
    """Test query endpoint with generation error."""
    mock_retrieve.return_value = sample_retrieval_results
    mock_generate.side_effect = Exception("Generation failed")
    
    response = client.post("/query", json=sample_query_request)
    
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert "Generation failed" in data["error"]

def test_query_endpoint_malformed_json(client):
    """Test query endpoint with malformed JSON."""
    response = client.post("/query", data="invalid json")
    
    assert response.status_code == 422

def test_ingest_endpoint_malformed_json(client):
    """Test ingest endpoint with malformed JSON."""
    response = client.post("/ingest", data="invalid json")
    
    assert response.status_code == 422

# =========================
# Integration Tests
# =========================

@patch('app.api.main.retrieve_documents')
@patch('app.api.main.generate_answer')
def test_full_query_flow(mock_generate, mock_retrieve, client, sample_query_request, sample_retrieval_results):
    """Test full query flow from request to response."""
    # Mock retrieval
    mock_retrieve.return_value = sample_retrieval_results
    
    # Mock generation
    mock_generate.return_value = {
        "answer": "The company allows remote work for up to 3 days per week.",
        "citations": [{"id": 1, "title": "Employee Handbook", "link": "https://drive.google.com/file/1", "snippet": "remote work", "score": 0.95, "chunk_id": "handbook_0"}],
        "score": 0.95,
        "grounding_score": 0.8,
        "is_grounded": True,
        "latency_ms": 150
    }
    
    # Test the full flow
    response = client.post("/query", json=sample_query_request)
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify all expected fields are present
    expected_fields = ["answer", "citations", "score", "grounding_score", "is_grounded", "latency_ms"]
    for field in expected_fields:
        assert field in data
    
    # Verify the response structure
    assert isinstance(data["citations"], list)
    if data["citations"]:
        citation = data["citations"][0]
        citation_fields = ["id", "title", "link", "snippet", "score", "chunk_id"]
        for field in citation_fields:
            assert field in citation
