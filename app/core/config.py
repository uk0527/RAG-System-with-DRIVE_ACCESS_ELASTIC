"""Configuration for environment variables and system settings."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load from project-level .env if present
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

# Elasticsearch
ELASTIC_URL: str = os.getenv("ELASTIC_URL", "http://localhost:9200")
ELASTIC_API_KEY_ID: Optional[str] = os.getenv("ELASTIC_API_KEY_ID")
ELASTIC_API_KEY: Optional[str] = os.getenv("ELASTIC_API_KEY")
INDEX_NAME: str = os.getenv("INDEX_NAME", "rag_docs")
ELSER_MODEL_ID: str = os.getenv("ELSER_MODEL_ID", ".elser_model_2_linux-x86_64")

# Hugging Face
HF_API_KEY: str = os.getenv("HF_API_KEY", "")
HF_MODEL_ID: str = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_ENDPOINT_URL: str = os.getenv("HF_ENDPOINT_URL", "https://api-inference.huggingface.co/models")

# Embeddings
EMBEDDER_NAME: str = os.getenv("EMBEDDER_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

# Google Drive
DRIVE_FOLDER_ID: Optional[str] = os.getenv("DRIVE_FOLDER_ID")
GOOGLE_SERVICE_ACCOUNT_PATH: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH", "service_account.json")

# Chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "1600"))

# Retrieval
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
MIN_SCORE_THRESHOLD: float = float(os.getenv("MIN_SCORE_THRESHOLD", "0.2"))
RRF_RANK_CONSTANT: int = int(os.getenv("RRF_RANK_CONSTANT", "10"))

# Guardrails
MIN_GROUNDING_OVERLAP: float = float(os.getenv("MIN_GROUNDING_OVERLAP", "0.10"))
MIN_ANSWER_LEN: int = int(os.getenv("MIN_ANSWER_LEN", "50"))
MAX_ANSWER_TOKENS: int = int(os.getenv("MAX_ANSWER_TOKENS", "1200"))

# Modern PDF Processing
# OCR settings
PADDLEOCR_LANG: str = os.getenv("PADDLEOCR_LANG", "en")  # PaddleOCR language
EASYOCR_LANG: str = os.getenv("EASYOCR_LANG", "en")      # EasyOCR language
OCR_QUALITY_THRESHOLD: float = float(os.getenv("OCR_QUALITY_THRESHOLD", "0.3"))  # Minimum quality to use OCR

# PDF processing
PDF_DPI: int = int(os.getenv("PDF_DPI", "300"))  # DPI for PDF to image conversion
EXTRACT_IMAGES: bool = os.getenv("EXTRACT_IMAGES", "true").lower() == "true"  # Extract images from PDFs
EXTRACT_TABLES: bool = os.getenv("EXTRACT_TABLES", "true").lower() == "true"  # Extract tables from PDFs

# API
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
RAG_API_URL: str = os.getenv("RAG_API_URL", f"http://{API_HOST}:{API_PORT}")

# Validation
def validate_config() -> None:
    """Validate that required configuration is present."""
    required_vars = [
        ("ELASTIC_URL", ELASTIC_URL),
        ("HF_API_KEY", HF_API_KEY),
    ]
    
    missing = []
    for name, value in required_vars:
        if not value:
            missing.append(name)
    
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    
    # Warn if running without API key
    if not ELASTIC_API_KEY:
        print("Warning: ELASTIC_API_KEY not set. Using unauthenticated connection.")

# Utility
def get_elasticsearch_config() -> dict:
    """Get Elasticsearch connection configuration."""
    config = {
        "hosts": [ELASTIC_URL],
        "request_timeout": 300,  # Increased to 5 minutes for large batches
        "max_retries": 3,
        "retry_on_timeout": True,
        "retry_on_status": [502, 503, 504],  # Retry on server errors
    }
    
    if ELASTIC_API_KEY:
        if ELASTIC_API_KEY_ID:
            config["api_key"] = (ELASTIC_API_KEY_ID, ELASTIC_API_KEY)
        else:
            config["api_key"] = ELASTIC_API_KEY
    
    return config

def get_huggingface_config() -> dict:
    """Get Hugging Face API configuration."""
    return {
        "api_key": HF_API_KEY,
        "model_id": HF_MODEL_ID,
        "endpoint_url": HF_ENDPOINT_URL,
    }
