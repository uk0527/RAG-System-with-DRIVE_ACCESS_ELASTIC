"""
Core module for the RAG system.
"""

from .config import validate_config, get_elasticsearch_config, get_huggingface_config
from .embed import get_embedder, get_langchain_embedder, encode_texts
# Lazy import to avoid memory issues with unstructured library
# from .ingest import ingest_documents, create_elasticsearch_mapping, create_elser_pipeline
from .retrieve import retrieve_documents, search_hybrid, search_elser
from .generate import generate_answer, create_huggingface_llm

__all__ = [
    "validate_config",
    "get_elasticsearch_config",
    "get_huggingface_config",
    "get_embedder",
    "get_langchain_embedder", 
    "encode_texts",
    "ingest_documents",
    "create_elasticsearch_mapping",
    "create_elser_pipeline",
    "retrieve_documents",
    "search_hybrid",
    "search_elser",
    "generate_answer",
    "create_huggingface_llm"
]
