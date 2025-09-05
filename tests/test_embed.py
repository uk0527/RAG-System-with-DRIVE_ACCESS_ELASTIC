"""
Tests for the embedding module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.core.embed import (
    get_langchain_embedder, get_embedder, encode_text, 
    get_embedding_dimensions, validate_embeddings
)

# =========================
# Test Fixtures
# =========================

@pytest.fixture
def mock_embedding():
    """Mock embedding vector."""
    return np.random.rand(384).astype(np.float32)

@pytest.fixture
def sample_texts():
    """Sample texts for embedding."""
    return [
        "This is a test document about machine learning.",
        "Another document about natural language processing.",
        "A third document about artificial intelligence."
    ]

# =========================
# LangChain Embedder Tests
# =========================

@patch('app.core.embed.HuggingFaceEmbeddings')
def test_get_langchain_embedder(mock_embeddings_class):
    """Test LangChain embedder creation."""
    mock_embedder = Mock()
    mock_embeddings_class.return_value = mock_embedder
    
    embedder = get_langchain_embedder()
    
    assert embedder == mock_embedder
    mock_embeddings_class.assert_called_once_with(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@patch('app.core.embed.HuggingFaceEmbeddings')
def test_get_langchain_embedder_custom_model(mock_embeddings_class):
    """Test LangChain embedder creation with custom model."""
    mock_embedder = Mock()
    mock_embeddings_class.return_value = mock_embedder
    
    embedder = get_langchain_embedder(model_name="custom-model")
    
    mock_embeddings_class.assert_called_once_with(
        model_name="custom-model",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# =========================
# Direct Embedder Tests
# =========================

@patch('app.core.embed.SentenceTransformer')
def test_get_embedder(mock_sentence_transformer):
    """Test direct embedder creation."""
    mock_model = Mock()
    mock_sentence_transformer.return_value = mock_model
    
    embedder = get_embedder()
    
    assert embedder == mock_model
    mock_sentence_transformer.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")

@patch('app.core.embed.SentenceTransformer')
def test_get_embedder_custom_model(mock_sentence_transformer):
    """Test direct embedder creation with custom model."""
    mock_model = Mock()
    mock_sentence_transformer.return_value = mock_model
    
    embedder = get_embedder(model_name="custom-model")
    
    mock_sentence_transformer.assert_called_once_with("custom-model")

# =========================
# Text Encoding Tests
# =========================

@patch('app.core.embed.get_embedder')
def test_encode_text_single(mock_get_embedder, mock_embedding):
    """Test encoding single text."""
    mock_embedder = Mock()
    mock_embedder.encode.return_value = mock_embedding
    mock_get_embedder.return_value = mock_embedder
    
    text = "This is a test sentence."
    result = encode_text(text)
    
    assert np.array_equal(result, mock_embedding)
    mock_embedder.encode.assert_called_once_with(text)

@patch('app.core.embed.get_embedder')
def test_encode_text_batch(mock_get_embedder, sample_texts, mock_embedding):
    """Test encoding multiple texts."""
    mock_embedder = Mock()
    mock_embedder.encode.return_value = np.array([mock_embedding] * len(sample_texts))
    mock_get_embedder.return_value = mock_embedder
    
    results = encode_text(sample_texts)
    
    assert len(results) == len(sample_texts)
    assert all(np.array_equal(result, mock_embedding) for result in results)
    mock_embedder.encode.assert_called_once_with(sample_texts)

@patch('app.core.embed.get_embedder')
def test_encode_text_empty(mock_get_embedder):
    """Test encoding empty text."""
    mock_embedder = Mock()
    mock_embedder.encode.return_value = np.zeros(384, dtype=np.float32)
    mock_get_embedder.return_value = mock_embedder
    
    result = encode_text("")
    
    assert len(result) == 384
    assert np.allclose(result, 0)

@patch('app.core.embed.get_embedder')
def test_encode_text_whitespace(mock_get_embedder):
    """Test encoding whitespace-only text."""
    mock_embedder = Mock()
    mock_embedder.encode.return_value = np.zeros(384, dtype=np.float32)
    mock_get_embedder.return_value = mock_embedder
    
    result = encode_text("   \n\t   ")
    
    assert len(result) == 384
    mock_embedder.encode.assert_called_once_with("   \n\t   ")

# =========================
# Embedding Dimensions Tests
# =========================

def test_get_embedding_dimensions():
    """Test getting embedding dimensions."""
    dims = get_embedding_dimensions()
    assert dims == 384

@patch('app.core.embed.get_embedder')
def test_get_embedding_dimensions_custom_model(mock_get_embedder):
    """Test getting embedding dimensions for custom model."""
    mock_embedder = Mock()
    mock_embedder.get_sentence_embedding_dimension.return_value = 512
    mock_get_embedder.return_value = mock_embedder
    
    dims = get_embedding_dimensions(model_name="custom-model")
    
    assert dims == 512
    mock_embedder.get_sentence_embedding_dimension.assert_called_once()

# =========================
# Embedding Validation Tests
# =========================

def test_validate_embeddings_valid():
    """Test embedding validation with valid embeddings."""
    embeddings = [
        np.random.rand(384).astype(np.float32),
        np.random.rand(384).astype(np.float32),
        np.random.rand(384).astype(np.float32)
    ]
    
    result = validate_embeddings(embeddings)
    
    assert result['valid'] is True
    assert result['count'] == 3
    assert result['dimensions'] == 384
    assert len(result['errors']) == 0

def test_validate_embeddings_wrong_dimensions():
    """Test embedding validation with wrong dimensions."""
    embeddings = [
        np.random.rand(512).astype(np.float32),  # Wrong dimension
        np.random.rand(384).astype(np.float32)
    ]
    
    result = validate_embeddings(embeddings)
    
    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert any('dimension' in error.lower() for error in result['errors'])

def test_validate_embeddings_empty():
    """Test embedding validation with empty list."""
    result = validate_embeddings([])
    
    assert result['valid'] is False
    assert result['count'] == 0
    assert len(result['errors']) > 0

def test_validate_embeddings_none_values():
    """Test embedding validation with None values."""
    embeddings = [
        np.random.rand(384).astype(np.float32),
        None,
        np.random.rand(384).astype(np.float32)
    ]
    
    result = validate_embeddings(embeddings)
    
    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert any('none' in error.lower() for error in result['errors'])

def test_validate_embeddings_wrong_type():
    """Test embedding validation with wrong data type."""
    embeddings = [
        "not an array",
        np.random.rand(384).astype(np.float32)
    ]
    
    result = validate_embeddings(embeddings)
    
    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert any('type' in error.lower() for error in result['errors'])

# =========================
# Error Handling Tests
# =========================

@patch('app.core.embed.get_embedder')
def test_encode_text_embedding_error(mock_get_embedder):
    """Test encoding text with embedding error."""
    mock_embedder = Mock()
    mock_embedder.encode.side_effect = Exception("Embedding error")
    mock_get_embedder.return_value = mock_embedder
    
    with pytest.raises(Exception, match="Embedding error"):
        encode_text("test text")

@patch('app.core.embed.HuggingFaceEmbeddings')
def test_get_langchain_embedder_error(mock_embeddings_class):
    """Test LangChain embedder creation with error."""
    mock_embeddings_class.side_effect = Exception("Model loading error")
    
    with pytest.raises(Exception, match="Model loading error"):
        get_langchain_embedder()

@patch('app.core.embed.SentenceTransformer')
def test_get_embedder_error(mock_sentence_transformer):
    """Test direct embedder creation with error."""
    mock_sentence_transformer.side_effect = Exception("Model loading error")
    
    with pytest.raises(Exception, match="Model loading error"):
        get_embedder()

# =========================
# Performance Tests
# =========================

@patch('app.core.embed.get_embedder')
def test_encode_text_large_batch(mock_get_embedder, mock_embedding):
    """Test encoding large batch of texts."""
    mock_embedder = Mock()
    large_batch = ["text"] * 1000
    mock_embedder.encode.return_value = np.array([mock_embedding] * 1000)
    mock_get_embedder.return_value = mock_embedder
    
    results = encode_text(large_batch)
    
    assert len(results) == 1000
    mock_embedder.encode.assert_called_once_with(large_batch)

@patch('app.core.embed.get_embedder')
def test_encode_text_very_long_text(mock_get_embedder, mock_embedding):
    """Test encoding very long text."""
    mock_embedder = Mock()
    mock_embedder.encode.return_value = mock_embedding
    mock_get_embedder.return_value = mock_embedder
    
    long_text = "This is a very long text. " * 1000
    result = encode_text(long_text)
    
    assert np.array_equal(result, mock_embedding)
    mock_embedder.encode.assert_called_once_with(long_text)
