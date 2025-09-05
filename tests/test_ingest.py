"""
Tests for the ingestion module.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from app.core.ingest import (
    _clean_text, _count_tokens, _enrich_metadata, create_text_splitter,
    chunk_documents, load_from_google_drive, create_elasticsearch_store,
    create_elasticsearch_mapping, create_elser_pipeline, ingest_documents
)

# =========================
# Test Fixtures
# =========================

@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return Document(
        page_content="This is a sample document with some content for testing purposes.",
        metadata={"file_info": {"name": "test.pdf", "id": "123", "webViewLink": "https://drive.google.com/file/123"}}
    )

@pytest.fixture
def sample_documents():
    """Multiple sample documents for testing."""
    return [
        Document(
            page_content="First document content with important information.",
            metadata={"file_info": {"name": "doc1.pdf", "id": "1", "webViewLink": "https://drive.google.com/file/1"}}
        ),
        Document(
            page_content="Second document content with different information.",
            metadata={"file_info": {"name": "doc2.pdf", "id": "2", "webViewLink": "https://drive.google.com/file/2"}}
        )
    ]

# =========================
# Text Processing Tests
# =========================

def test_clean_text():
    """Test text cleaning functionality."""
    # Test with soft hyphens
    text_with_hyphens = "This is a test\u00ad with soft hyphens."
    cleaned = _clean_text(text_with_hyphens)
    assert "\u00ad" not in cleaned
    assert "soft hyphens" in cleaned
    
    # Test with multiple spaces
    text_with_spaces = "This   has    multiple    spaces."
    cleaned = _clean_text(text_with_spaces)
    assert "  " not in cleaned
    assert " " in cleaned
    
    # Test with multiple newlines
    text_with_newlines = "This\n\n\nhas\n\nmultiple\n\nnewlines."
    cleaned = _clean_text(text_with_newlines)
    assert "\n\n\n" not in cleaned
    assert "\n\n" not in cleaned
    assert "\n" in cleaned
    
    # Test empty text
    assert _clean_text("") == ""
    assert _clean_text(None) == ""

def test_count_tokens():
    """Test token counting functionality."""
    text = "This is a test sentence with some words."
    token_count = _count_tokens(text)
    assert isinstance(token_count, int)
    assert token_count > 0
    
    # Test with empty text
    assert _count_tokens("") == 0
    
    # Test with longer text
    long_text = " ".join(["word"] * 100)
    long_count = _count_tokens(long_text)
    assert long_count > token_count

def test_enrich_metadata(sample_document):
    """Test metadata enrichment."""
    file_info = {
        "name": "test.pdf",
        "id": "123",
        "webViewLink": "https://drive.google.com/file/123",
        "modifiedTime": "2024-01-01T00:00:00Z"
    }
    
    enriched_doc = _enrich_metadata(sample_document, file_info)
    
    assert "filename" in enriched_doc.metadata
    assert "drive_url" in enriched_doc.metadata
    assert "file_id" in enriched_doc.metadata
    assert "ingestion_time" in enriched_doc.metadata
    
    assert enriched_doc.metadata["filename"] == "test.pdf"
    assert enriched_doc.metadata["drive_url"] == "https://drive.google.com/file/123"
    assert enriched_doc.metadata["file_id"] == "123"

# =========================
# Chunking Tests
# =========================

def test_create_text_splitter():
    """Test text splitter creation."""
    splitter = create_text_splitter()
    assert splitter is not None
    assert hasattr(splitter, 'split_documents')

def test_chunk_documents(sample_documents):
    """Test document chunking."""
    chunks = chunk_documents(sample_documents)
    
    assert len(chunks) > len(sample_documents)  # Should create multiple chunks
    
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "total_chunks" in chunk.metadata
        assert isinstance(chunk.metadata["chunk_index"], int)
        assert isinstance(chunk.metadata["total_chunks"], int)

def test_chunk_documents_single_doc(sample_document):
    """Test chunking with single document."""
    chunks = chunk_documents([sample_document])
    
    assert len(chunks) >= 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["total_chunks"] == len(chunks)

# =========================
# Google Drive Tests
# =========================

@patch('app.core.ingest.GoogleDriveLoader')
def test_load_from_google_drive(mock_loader_class, tmp_path):
    """Test Google Drive document loading."""
    # Create a temporary service account file
    service_account_file = tmp_path / "service_account.json"
    service_account_file.write_text('{"type": "service_account"}')
    
    # Mock the loader
    mock_loader = Mock()
    mock_loader.load.return_value = [
        Document(
            page_content="Test content",
            metadata={"file_info": {"name": "test.pdf", "id": "123", "webViewLink": "https://drive.google.com/file/123"}}
        )
    ]
    mock_loader_class.return_value = mock_loader
    
    # Test loading
    docs = load_from_google_drive(
        folder_id="test_folder",
        service_account_path=str(service_account_file)
    )
    
    assert len(docs) == 1
    assert docs[0].page_content == "Test content"
    assert "filename" in docs[0].metadata
    
    # Verify loader was called correctly
    mock_loader_class.assert_called_once()
    mock_loader.load.assert_called_once()

@patch('app.core.ingest.GoogleDriveLoader')
def test_load_from_google_drive_empty(mock_loader_class, tmp_path):
    """Test Google Drive loading with empty results."""
    service_account_file = tmp_path / "service_account.json"
    service_account_file.write_text('{"type": "service_account"}')
    
    mock_loader = Mock()
    mock_loader.load.return_value = []
    mock_loader_class.return_value = mock_loader
    
    docs = load_from_google_drive(
        folder_id="test_folder",
        service_account_path=str(service_account_file)
    )
    
    assert len(docs) == 0

def test_load_from_google_drive_missing_file():
    """Test Google Drive loading with missing service account file."""
    with pytest.raises(FileNotFoundError):
        load_from_google_drive(
            folder_id="test_folder",
            service_account_path="nonexistent.json"
        )

# =========================
# Elasticsearch Tests
# =========================

def test_create_elasticsearch_mapping():
    """Test Elasticsearch mapping creation."""
    mapping = create_elasticsearch_mapping()
    
    assert "mappings" in mapping
    assert "properties" in mapping["mappings"]
    
    properties = mapping["mappings"]["properties"]
    assert "content" in properties
    assert "title" in properties
    assert "vector" in properties
    assert "ml.tokens" in properties
    assert "metadata" in properties
    
    # Check vector field configuration
    assert properties["vector"]["type"] == "dense_vector"
    assert properties["vector"]["dims"] == 384
    assert properties["vector"]["similarity"] == "cosine"

def test_create_elser_pipeline():
    """Test ELSER pipeline creation."""
    pipeline = create_elser_pipeline()
    
    assert "description" in pipeline
    assert "processors" in pipeline
    assert len(pipeline["processors"]) == 1
    
    processor = pipeline["processors"][0]
    assert "inference" in processor
    assert processor["inference"]["model_id"] == ".elser_model_2_linux-x86_64_ingest"

@patch('app.core.ingest.get_langchain_embedder')
def test_create_elasticsearch_store(mock_get_embedder):
    """Test Elasticsearch store creation."""
    mock_embedder = Mock()
    mock_get_embedder.return_value = mock_embedder
    
    mock_es_client = Mock()
    
    store = create_elasticsearch_store(mock_es_client, "test_index")
    
    assert store is not None
    mock_get_embedder.assert_called_once()

# =========================
# Integration Tests
# =========================

@patch('app.core.ingest.load_from_google_drive')
@patch('app.core.ingest.create_elasticsearch_store')
def test_ingest_documents(mock_create_store, mock_load_drive):
    """Test complete ingestion pipeline."""
    # Mock Google Drive loading
    mock_docs = [
        Document(
            page_content="Test content for ingestion",
            metadata={"file_info": {"name": "test.pdf", "id": "123", "webViewLink": "https://drive.google.com/file/123"}}
        )
    ]
    mock_load_drive.return_value = mock_docs
    
    # Mock Elasticsearch store
    mock_store = Mock()
    mock_store.add_documents.return_value = None
    mock_create_store.return_value = mock_store
    
    # Mock Elasticsearch client
    mock_es_client = Mock()
    mock_es_client.indices.exists.return_value = False
    mock_es_client.ingest.get_pipeline.side_effect = Exception("Pipeline not found")
    
    # Test ingestion
    result = ingest_documents(
        es_client=mock_es_client,
        folder_id="test_folder",
        force=True,
        max_files=10,
        verbose=False
    )
    
    assert "files_seen" in result
    assert "chunks_indexed" in result
    assert "duration_seconds" in result
    assert result["files_seen"] == 1
    assert result["chunks_indexed"] > 0

@patch('app.core.ingest.load_from_google_drive')
def test_ingest_documents_empty(mock_load_drive):
    """Test ingestion with no documents."""
    mock_load_drive.return_value = []
    
    mock_es_client = Mock()
    
    result = ingest_documents(
        es_client=mock_es_client,
        folder_id="test_folder",
        verbose=False
    )
    
    assert result["files_seen"] == 0
    assert result["chunks_indexed"] == 0
    assert result["duration_seconds"] == 0

# =========================
# Error Handling Tests
# =========================

def test_ingest_documents_missing_folder_id():
    """Test ingestion with missing folder ID."""
    mock_es_client = Mock()
    
    with pytest.raises(ValueError, match="No Google Drive folder ID provided"):
        ingest_documents(
            es_client=mock_es_client,
            folder_id=None,
            verbose=False
        )

@patch('app.core.ingest.load_from_google_drive')
def test_ingest_documents_error_handling(mock_load_drive):
    """Test ingestion error handling."""
    mock_load_drive.side_effect = Exception("Drive error")
    
    mock_es_client = Mock()
    
    with pytest.raises(Exception, match="Drive error"):
        ingest_documents(
            es_client=mock_es_client,
            folder_id="test_folder",
            verbose=False
        )
