"""Embedding utilities and cached model access."""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import EMBEDDER_NAME, EMBEDDING_DIMENSIONS

# Cached global instances
_embedder: Optional[SentenceTransformer] = None
_langchain_embedder: Optional[HuggingFaceEmbeddings] = None

def get_embedder() -> SentenceTransformer:
    """Return a cached sentence-transformers model."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDER_NAME)
    return _embedder

def get_langchain_embedder() -> HuggingFaceEmbeddings:
    """Return a cached LangChain HuggingFace embeddings wrapper."""
    global _langchain_embedder
    if _langchain_embedder is None:
        _langchain_embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDER_NAME,
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
        )
    return _langchain_embedder

def encode_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Encode multiple texts to embedding vectors."""
    embedder = get_embedder()
    embeddings = embedder.encode(
        texts, 
        show_progress_bar=len(texts) > 100,
        batch_size=batch_size,
        convert_to_tensor=False
    )
    return embeddings.tolist()

def encode_single_text(text: str) -> List[float]:
    """Encode a single text to an embedding vector."""
    return encode_texts([text])[0]

def get_embedding_dimensions() -> int:
    """Expected embedding dimensions for the configured model."""
    return EMBEDDING_DIMENSIONS

def validate_embedding(embedding: List[float]) -> bool:
    """Return True if the vector length matches expected dimensions."""
    return len(embedding) == EMBEDDING_DIMENSIONS
