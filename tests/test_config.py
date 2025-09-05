"""
Tests for the configuration module.
"""

import pytest
import os
from unittest.mock import patch, mock_open

from app.core.config import (
    get_elasticsearch_config, get_huggingface_config, get_google_drive_config,
    get_ocr_config, get_rag_config, validate_config
)

# =========================
# Test Fixtures
# =========================

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    return {
        'ES_HOST': 'localhost',
        'ES_PORT': '9200',
        'ES_USERNAME': 'elastic',
        'ES_PASSWORD': 'password',
        'ES_INDEX': 'test_index',
        'HF_API_KEY': 'test_api_key',
        'HF_MODEL_ID': 'test_model',
        'HF_ENDPOINT_URL': 'https://api.huggingface.co',
        'GOOGLE_DRIVE_FOLDER_ID': 'test_folder_id',
        'GOOGLE_SERVICE_ACCOUNT_PATH': '/path/to/service.json',
        'TESSERACT_PATH': '/usr/bin/tesseract',
        'POPPLER_PATH': '/usr/bin/poppler',
        'CHUNK_SIZE': '300',
        'CHUNK_OVERLAP': '50',
        'MAX_CONTEXT_TOKENS': '4000',
        'MAX_ANSWER_TOKENS': '500',
        'MIN_GROUNDING_OVERLAP': '0.3',
        'MIN_ANSWER_LEN': '10'
    }

# =========================
# Elasticsearch Config Tests
# =========================

@patch.dict(os.environ, {
    'ES_HOST': 'localhost',
    'ES_PORT': '9200',
    'ES_USERNAME': 'elastic',
    'ES_PASSWORD': 'password',
    'ES_INDEX': 'test_index'
})
def test_get_elasticsearch_config():
    """Test Elasticsearch configuration retrieval."""
    config = get_elasticsearch_config()
    
    assert config['host'] == 'localhost'
    assert config['port'] == 9200
    assert config['username'] == 'elastic'
    assert config['password'] == 'password'
    assert config['index'] == 'test_index'
    assert config['use_ssl'] is False
    assert config['verify_certs'] is False

@patch.dict(os.environ, {
    'ES_HOST': 'es.example.com',
    'ES_PORT': '443',
    'ES_USE_SSL': 'true',
    'ES_VERIFY_CERTS': 'true'
})
def test_get_elasticsearch_config_ssl():
    """Test Elasticsearch configuration with SSL."""
    config = get_elasticsearch_config()
    
    assert config['host'] == 'es.example.com'
    assert config['port'] == 443
    assert config['use_ssl'] is True
    assert config['verify_certs'] is True

def test_get_elasticsearch_config_defaults():
    """Test Elasticsearch configuration with defaults."""
    with patch.dict(os.environ, {}, clear=True):
        config = get_elasticsearch_config()
        
        assert config['host'] == 'localhost'
        assert config['port'] == 9200
        assert config['username'] is None
        assert config['password'] is None
        assert config['index'] == 'rag_docs'
        assert config['use_ssl'] is False
        assert config['verify_certs'] is False

# =========================
# Hugging Face Config Tests
# =========================

@patch.dict(os.environ, {
    'HF_API_KEY': 'test_api_key',
    'HF_MODEL_ID': 'meta-llama/Llama-3-8B-Instruct',
    'HF_ENDPOINT_URL': 'https://api.huggingface.co'
})
def test_get_huggingface_config():
    """Test Hugging Face configuration retrieval."""
    config = get_huggingface_config()
    
    assert config['api_key'] == 'test_api_key'
    assert config['model_id'] == 'meta-llama/Llama-3-8B-Instruct'
    assert config['endpoint_url'] == 'https://api.huggingface.co'

def test_get_huggingface_config_defaults():
    """Test Hugging Face configuration with defaults."""
    with patch.dict(os.environ, {}, clear=True):
        config = get_huggingface_config()
        
        assert config['api_key'] is None
        assert config['model_id'] == 'meta-llama/Meta-Llama-3-8B-Instruct'
        assert config['endpoint_url'] == 'https://api-inference.huggingface.co'

# =========================
# Google Drive Config Tests
# =========================

@patch.dict(os.environ, {
    'GOOGLE_DRIVE_FOLDER_ID': 'test_folder_id',
    'GOOGLE_SERVICE_ACCOUNT_PATH': '/path/to/service.json'
})
def test_get_google_drive_config():
    """Test Google Drive configuration retrieval."""
    config = get_google_drive_config()
    
    assert config['folder_id'] == 'test_folder_id'
    assert config['service_account_path'] == '/path/to/service.json'

def test_get_google_drive_config_defaults():
    """Test Google Drive configuration with defaults."""
    with patch.dict(os.environ, {}, clear=True):
        config = get_google_drive_config()
        
        assert config['folder_id'] is None
        assert config['service_account_path'] is None

# =========================
# OCR Config Tests
# =========================

@patch.dict(os.environ, {
    'TESSERACT_PATH': '/usr/bin/tesseract',
    'POPPLER_PATH': '/usr/bin/poppler'
})
def test_get_ocr_config():
    """Test OCR configuration retrieval."""
    config = get_ocr_config()
    
    assert config['tesseract_path'] == '/usr/bin/tesseract'
    assert config['poppler_path'] == '/usr/bin/poppler'

def test_get_ocr_config_defaults():
    """Test OCR configuration with defaults."""
    with patch.dict(os.environ, {}, clear=True):
        config = get_ocr_config()
        
        assert config['tesseract_path'] == 'tesseract'
        assert config['poppler_path'] == 'pdftoppm'

# =========================
# RAG Config Tests
# =========================

@patch.dict(os.environ, {
    'CHUNK_SIZE': '500',
    'CHUNK_OVERLAP': '100',
    'MAX_CONTEXT_TOKENS': '6000',
    'MAX_ANSWER_TOKENS': '800',
    'MIN_GROUNDING_OVERLAP': '0.4',
    'MIN_ANSWER_LEN': '20'
})
def test_get_rag_config():
    """Test RAG configuration retrieval."""
    config = get_rag_config()
    
    assert config['chunk_size'] == 500
    assert config['chunk_overlap'] == 100
    assert config['max_context_tokens'] == 6000
    assert config['max_answer_tokens'] == 800
    assert config['min_grounding_overlap'] == 0.4
    assert config['min_answer_len'] == 20

def test_get_rag_config_defaults():
    """Test RAG configuration with defaults."""
    with patch.dict(os.environ, {}, clear=True):
        config = get_rag_config()
        
        assert config['chunk_size'] == 300
        assert config['chunk_overlap'] == 50
        assert config['max_context_tokens'] == 4000
        assert config['max_answer_tokens'] == 500
        assert config['min_grounding_overlap'] == 0.3
        assert config['min_answer_len'] == 10

# =========================
# Config Validation Tests
# =========================

@patch.dict(os.environ, {
    'ES_HOST': 'localhost',
    'ES_PORT': '9200',
    'HF_API_KEY': 'test_key',
    'GOOGLE_DRIVE_FOLDER_ID': 'test_folder',
    'GOOGLE_SERVICE_ACCOUNT_PATH': '/path/to/service.json'
})
def test_validate_config_valid():
    """Test configuration validation with valid config."""
    # Mock file existence
    with patch('os.path.exists', return_value=True):
        result = validate_config()
        assert result['valid'] is True
        assert len(result['errors']) == 0

@patch.dict(os.environ, {
    'ES_HOST': 'localhost',
    'ES_PORT': '9200',
    'HF_API_KEY': 'test_key',
    'GOOGLE_DRIVE_FOLDER_ID': 'test_folder',
    'GOOGLE_SERVICE_ACCOUNT_PATH': '/path/to/service.json'
})
def test_validate_config_missing_file():
    """Test configuration validation with missing service account file."""
    with patch('os.path.exists', return_value=False):
        result = validate_config()
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('service account file' in error.lower() for error in result['errors'])

def test_validate_config_missing_required():
    """Test configuration validation with missing required fields."""
    with patch.dict(os.environ, {}, clear=True):
        result = validate_config()
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('required' in error.lower() for error in result['errors'])

@patch.dict(os.environ, {
    'ES_PORT': 'invalid_port',
    'CHUNK_SIZE': 'invalid_size',
    'MIN_GROUNDING_OVERLAP': 'invalid_overlap'
})
def test_validate_config_invalid_types():
    """Test configuration validation with invalid data types."""
    result = validate_config()
    assert result['valid'] is False
    assert len(result['errors']) > 0

# =========================
# Edge Cases and Error Handling
# =========================

@patch.dict(os.environ, {
    'ES_PORT': '99999'  # Invalid port
})
def test_get_elasticsearch_config_invalid_port():
    """Test Elasticsearch config with invalid port."""
    with pytest.raises(ValueError):
        get_elasticsearch_config()

@patch.dict(os.environ, {
    'CHUNK_SIZE': '-100'  # Negative chunk size
})
def test_get_rag_config_invalid_values():
    """Test RAG config with invalid values."""
    with pytest.raises(ValueError):
        get_rag_config()

@patch.dict(os.environ, {
    'MIN_GROUNDING_OVERLAP': '1.5'  # Over 1.0
})
def test_get_rag_config_overlap_over_one():
    """Test RAG config with overlap over 1.0."""
    with pytest.raises(ValueError):
        get_rag_config()
