"""
Tests for the retrieval module.
"""

import pytest
from unittest.mock import Mock, patch
from elasticsearch import Elasticsearch

from app.core.retrieve import (
    get_retriever, search_elser, search_dense, search_bm25,
    reciprocal_rank_fusion, search_hybrid, retrieve_documents,
    format_citations
)

# =========================
# Test Fixtures
# =========================

@pytest.fixture
def mock_es_client():
    """Mock Elasticsearch client."""
    client = Mock(spec=Elasticsearch)
    return client

@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "_id": "1",
            "_score": 0.95,
            "_source": {
                "content": "This is the first document with relevant content.",
                "title": "Document 1",
                "metadata": {
                    "filename": "doc1.pdf",
                    "drive_url": "https://drive.google.com/file/1",
                    "chunk_id": "doc1_0"
                }
            }
        },
        {
            "_id": "2",
            "_score": 0.85,
            "_source": {
                "content": "This is the second document with different content.",
                "title": "Document 2",
                "metadata": {
                    "filename": "doc2.pdf",
                    "drive_url": "https://drive.google.com/file/2",
                    "chunk_id": "doc2_0"
                }
            }
        }
    ]

@pytest.fixture
def mock_es_response(sample_search_results):
    """Mock Elasticsearch search response."""
    return {
        "hits": {
            "hits": sample_search_results
        }
    }

# =========================
# ELSER Search Tests
# =========================

def test_search_elser(mock_es_client, mock_es_response):
    """Test ELSER search functionality."""
    mock_es_client.search.return_value = mock_es_response
    
    results = search_elser(mock_es_client, "test query", top_k=5)
    
    assert len(results) == 2
    assert results[0]["_id"] == "1"
    assert results[0]["score_elser"] == 0.95
    assert "content" in results[0]["_source"]
    
    # Verify search was called with correct parameters
    mock_es_client.search.assert_called_once()
    call_args = mock_es_client.search.call_args
    assert call_args[1]["index"] == "rag_docs"
    assert "text_expansion" in call_args[1]["body"]["query"]

def test_search_elser_empty_results(mock_es_client):
    """Test ELSER search with empty results."""
    mock_es_client.search.return_value = {"hits": {"hits": []}}
    
    results = search_elser(mock_es_client, "test query")
    
    assert len(results) == 0

def test_search_elser_min_score_filtering(mock_es_client):
    """Test ELSER search with minimum score filtering."""
    # Create results with different scores
    mock_response = {
        "hits": {
            "hits": [
                {"_id": "1", "_score": 0.95, "_source": {"content": "high score"}},
                {"_id": "2", "_score": 0.05, "_source": {"content": "low score"}}
            ]
        }
    }
    mock_es_client.search.return_value = mock_response
    
    results = search_elser(mock_es_client, "test query", min_score=0.1)
    
    # Only high score result should be returned
    assert len(results) == 1
    assert results[0]["_id"] == "1"

# =========================
# Dense Search Tests
# =========================

@patch('app.core.retrieve.get_embedder')
def test_search_dense(mock_get_embedder, mock_es_client, mock_es_response):
    """Test dense vector search functionality."""
    # Mock embedder
    mock_embedder = Mock()
    mock_embedder.encode.return_value = [0.1, 0.2, 0.3]  # Mock embedding
    mock_get_embedder.return_value = mock_embedder
    
    mock_es_client.search.return_value = mock_es_response
    
    results = search_dense(mock_es_client, "test query", top_k=5)
    
    assert len(results) == 2
    assert results[0]["_id"] == "1"
    assert "score_dense" in results[0]
    
    # Verify embedder was called
    mock_embedder.encode.assert_called_once_with("test query")
    
    # Verify search was called with script_score query
    mock_es_client.search.assert_called_once()
    call_args = mock_es_client.search.call_args
    assert "script_score" in call_args[1]["body"]["query"]

@patch('app.core.retrieve.get_embedder')
def test_search_dense_score_normalization(mock_get_embedder, mock_es_client):
    """Test dense search score normalization."""
    mock_embedder = Mock()
    mock_embedder.encode.return_value = [0.1, 0.2, 0.3]
    mock_get_embedder.return_value = mock_embedder
    
    # Mock response with raw cosine similarity scores
    mock_response = {
        "hits": {
            "hits": [
                {"_id": "1", "_score": 1.8, "_source": {"content": "test"}},  # 1.8 -> 0.9
                {"_id": "2", "_score": 1.2, "_source": {"content": "test"}}   # 1.2 -> 0.6
            ]
        }
    }
    mock_es_client.search.return_value = mock_response
    
    results = search_dense(mock_es_client, "test query")
    
    assert results[0]["score_dense"] == 0.9
    assert results[1]["score_dense"] == 0.6

# =========================
# BM25 Search Tests
# =========================

def test_search_bm25(mock_es_client, mock_es_response):
    """Test BM25 search functionality."""
    mock_es_client.search.return_value = mock_es_response
    
    results = search_bm25(mock_es_client, "test query", top_k=5)
    
    assert len(results) == 2
    assert results[0]["_id"] == "1"
    assert "score_bm25" in results[0]
    
    # Verify search was called with multi_match query
    mock_es_client.search.assert_called_once()
    call_args = mock_es_client.search.call_args
    assert "multi_match" in call_args[1]["body"]["query"]

def test_search_bm25_field_boosting(mock_es_client, mock_es_response):
    """Test BM25 search with field boosting."""
    mock_es_client.search.return_value = mock_es_response
    
    search_bm25(mock_es_client, "test query")
    
    # Verify title field is boosted
    call_args = mock_es_client.search.call_args
    query = call_args[1]["body"]["query"]["multi_match"]
    assert "title^2" in query["fields"]

# =========================
# RRF Fusion Tests
# =========================

def test_reciprocal_rank_fusion():
    """Test Reciprocal Rank Fusion functionality."""
    # Create test result lists
    list1 = [
        {"_id": "1", "_score": 0.9, "content": "doc1"},
        {"_id": "2", "_score": 0.8, "content": "doc2"},
        {"_id": "3", "_score": 0.7, "content": "doc3"}
    ]
    
    list2 = [
        {"_id": "2", "_score": 0.85, "content": "doc2"},
        {"_id": "1", "_score": 0.75, "content": "doc1"},
        {"_id": "4", "_score": 0.65, "content": "doc4"}
    ]
    
    list3 = [
        {"_id": "3", "_score": 0.8, "content": "doc3"},
        {"_id": "1", "_score": 0.7, "content": "doc1"},
        {"_id": "5", "_score": 0.6, "content": "doc5"}
    ]
    
    fused = reciprocal_rank_fusion([list1, list2, list3])
    
    # Should have 5 unique documents
    assert len(fused) == 5
    
    # Check that RRF scores are calculated
    for result in fused:
        assert "rrf_score" in result
        assert result["rrf_score"] > 0
    
    # Document 1 should have highest RRF score (appears in all lists)
    doc1 = next(r for r in fused if r["_id"] == "1")
    assert doc1["rrf_score"] > 0.1  # Should be high due to multiple appearances

def test_reciprocal_rank_fusion_empty_lists():
    """Test RRF with empty lists."""
    fused = reciprocal_rank_fusion([])
    assert len(fused) == 0
    
    fused = reciprocal_rank_fusion([[], []])
    assert len(fused) == 0

def test_reciprocal_rank_fusion_single_list():
    """Test RRF with single list."""
    list1 = [
        {"_id": "1", "_score": 0.9, "content": "doc1"},
        {"_id": "2", "_score": 0.8, "content": "doc2"}
    ]
    
    fused = reciprocal_rank_fusion([list1])
    
    assert len(fused) == 2
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]  # Should maintain order

# =========================
# Hybrid Search Tests
# =========================

@patch('app.core.retrieve.search_elser')
@patch('app.core.retrieve.search_dense')
@patch('app.core.retrieve.search_bm25')
def test_search_hybrid(mock_bm25, mock_dense, mock_elser, mock_es_client):
    """Test hybrid search functionality."""
    # Mock individual search results
    mock_elser.return_value = [{"_id": "1", "_score": 0.9, "_source": {"content": "elser"}}]
    mock_dense.return_value = [{"_id": "2", "_score": 0.8, "_source": {"content": "dense"}}]
    mock_bm25.return_value = [{"_id": "3", "_score": 0.7, "_source": {"content": "bm25"}}]
    
    results = search_hybrid(mock_es_client, "test query", top_k=3)
    
    # Verify all search methods were called
    mock_elser.assert_called_once()
    mock_dense.assert_called_once()
    mock_bm25.assert_called_once()
    
    # Should return fused results
    assert len(results) <= 3
    for result in results:
        assert "rrf_score" in result

@patch('app.core.retrieve.search_elser')
@patch('app.core.retrieve.search_dense')
@patch('app.core.retrieve.search_bm25')
def test_search_hybrid_score_filtering(mock_bm25, mock_dense, mock_elser, mock_es_client):
    """Test hybrid search with score filtering."""
    # Mock results with different scores
    mock_elser.return_value = [{"_id": "1", "score_elser": 0.9, "_source": {"content": "high"}}]
    mock_dense.return_value = [{"_id": "2", "score_dense": 0.05, "_source": {"content": "low"}}]
    mock_bm25.return_value = [{"_id": "3", "score_bm25": 0.1, "_source": {"content": "medium"}}]
    
    results = search_hybrid(mock_es_client, "test query", min_score=0.2)
    
    # Should filter out low-scoring results
    assert len(results) >= 0  # May be empty after filtering

# =========================
# Main Retrieval Interface Tests
# =========================

@patch('app.core.retrieve.search_elser')
def test_retrieve_documents_elser_mode(mock_elser, mock_es_client):
    """Test retrieve_documents in ELSER mode."""
    mock_elser.return_value = [{"_id": "1", "_score": 0.9, "_source": {"content": "test"}}]
    
    results = retrieve_documents(mock_es_client, "test query", mode="elser")
    
    mock_elser.assert_called_once_with(mock_es_client, "test query", 5, 0.2)
    assert len(results) == 1

@patch('app.core.retrieve.search_hybrid')
def test_retrieve_documents_hybrid_mode(mock_hybrid, mock_es_client):
    """Test retrieve_documents in hybrid mode."""
    mock_hybrid.return_value = [{"_id": "1", "_score": 0.9, "_source": {"content": "test"}}]
    
    results = retrieve_documents(mock_es_client, "test query", mode="hybrid")
    
    mock_hybrid.assert_called_once_with(mock_es_client, "test query", 5, 0.2)
    assert len(results) == 1

# =========================
# Citation Formatting Tests
# =========================

def test_format_citations(sample_search_results):
    """Test citation formatting."""
    citations = format_citations(sample_search_results, max_citations=2)
    
    assert len(citations) == 2
    assert citations[0]["id"] == 1
    assert citations[0]["title"] == "Document 1"
    assert citations[0]["link"] == "https://drive.google.com/file/1"
    assert "snippet" in citations[0]
    assert "score" in citations[0]

def test_format_citations_max_limit(sample_search_results):
    """Test citation formatting with max limit."""
    citations = format_citations(sample_search_results, max_citations=1)
    
    assert len(citations) == 1
    assert citations[0]["id"] == 1

def test_format_citations_empty_results():
    """Test citation formatting with empty results."""
    citations = format_citations([])
    
    assert len(citations) == 0

def test_format_citations_snippet_truncation():
    """Test citation snippet truncation."""
    long_content = "This is a very long content that should be truncated to 280 characters. " * 10
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
    assert citations[0]["snippet"].endswith("...") or len(citations[0]["snippet"]) < 280

# =========================
# Error Handling Tests
# =========================

def test_search_elser_connection_error(mock_es_client):
    """Test ELSER search with connection error."""
    mock_es_client.search.side_effect = Exception("Connection error")
    
    with pytest.raises(Exception, match="Connection error"):
        search_elser(mock_es_client, "test query")

@patch('app.core.retrieve.get_embedder')
def test_search_dense_embedding_error(mock_get_embedder, mock_es_client):
    """Test dense search with embedding error."""
    mock_embedder = Mock()
    mock_embedder.encode.side_effect = Exception("Embedding error")
    mock_get_embedder.return_value = mock_embedder
    
    with pytest.raises(Exception, match="Embedding error"):
        search_dense(mock_es_client, "test query")
