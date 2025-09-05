"""
Pydantic models for the RAG system API.
Defines request/response schemas for all endpoints.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime

# =========================
# Request Schemas
# =========================

class QueryRequest(BaseModel):
    """Request schema for the /query endpoint."""
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    mode: Literal["elser", "hybrid", "bm25"] = Field("hybrid", description="Retrieval mode")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to consider")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class IngestRequest(BaseModel):
    """Request schema for the /ingest endpoint."""
    folder_id: Optional[str] = Field(None, description="Google Drive folder ID")
    pipeline_name: str = Field("elser_v2_pipeline", description="Elasticsearch ingest pipeline name")
    embedder_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Embedding model name")
    force: bool = Field(False, description="Force re-ingestion of all files")
    ocr_mode: Literal["auto", "off", "force"] = Field("auto", description="OCR processing mode")
    max_files: int = Field(1000, ge=1, le=10000, description="Maximum number of files to process")
    max_pages: int = Field(200, ge=1, le=2000, description="Maximum pages per file")
    batch_size: int = Field(64, ge=1, le=256, description="Batch size for processing")
    verbose: bool = Field(True, description="Enable verbose logging")

# =========================
# Response Schemas
# =========================

class Citation(BaseModel):
    """Citation schema for source documents."""
    id: int = Field(..., description="Citation ID")
    title: str = Field(..., description="Document title")
    link: str = Field(..., description="Document URL")
    snippet: str = Field(..., description="Content snippet")
    score: Optional[float] = Field(None, description="Relevance score")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")

class GuardrailsResult(BaseModel):
    """Guardrails validation results."""
    safe: bool = Field(..., description="Whether content is safe")
    grounded: bool = Field(..., description="Whether answer is grounded in context")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Content quality score")
    grounding_score: float = Field(..., ge=0.0, le=1.0, description="Grounding score")
    notes: List[str] = Field(default_factory=list, description="Additional notes")

class QueryResponse(BaseModel):
    """Response schema for the /query endpoint."""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    used_mode: str = Field(..., description="Retrieval mode used")
    latency_ms: int = Field(..., description="Response latency in milliseconds")
    guardrails: GuardrailsResult = Field(..., description="Guardrails validation results")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class IngestResponse(BaseModel):
    """Response schema for the /ingest endpoint."""
    indexed: int = Field(..., description="Number of chunks indexed")
    timings: Dict[str, Any] = Field(..., description="Processing timings")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HealthResponse(BaseModel):
    """Response schema for the /healthz endpoint."""
    status: str = Field(..., description="Overall system status")
    elastic: str = Field(..., description="Elasticsearch status")
    models: str = Field(..., description="Models status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# =========================
# Error Schemas
# =========================

class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# =========================
# Debug Schemas
# =========================

class DebugRetrieveResponse(BaseModel):
    """Response schema for the /debug/retrieve endpoint."""
    results: List[Dict[str, Any]] = Field(..., description="Retrieval results")
    query: str = Field(..., description="Original query")
    mode: str = Field(..., description="Retrieval mode used")
    top_k: int = Field(..., description="Number of results requested")
    total_hits: int = Field(..., description="Total number of hits found")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# =========================
# Configuration Schemas
# =========================

class SystemConfig(BaseModel):
    """System configuration schema."""
    elasticsearch_url: str = Field(..., description="Elasticsearch URL")
    index_name: str = Field(..., description="Elasticsearch index name")
    embedding_model: str = Field(..., description="Embedding model name")
    llm_model: str = Field(..., description="LLM model name")
    chunk_size: int = Field(..., description="Chunk size in tokens")
    chunk_overlap: int = Field(..., description="Chunk overlap in tokens")
    max_context_tokens: int = Field(..., description="Maximum context tokens")
    default_top_k: int = Field(..., description="Default top-k value")
    min_score_threshold: float = Field(..., description="Minimum score threshold")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# =========================
# Validation Helpers
# =========================

def validate_query_request(request: QueryRequest) -> QueryRequest:
    """Validate and clean query request."""
    # Additional validation logic can be added here
    return request

def validate_ingest_request(request: IngestRequest) -> IngestRequest:
    """Validate and clean ingest request."""
    # Additional validation logic can be added here
    return request

# =========================
# Example Data
# =========================

class ExampleData:
    """Example data for API documentation."""
    
    @staticmethod
    def query_request() -> Dict[str, Any]:
        return {
            "question": "What is the company's policy on remote work?",
            "mode": "hybrid",
            "top_k": 5
        }
    
    @staticmethod
    def query_response() -> Dict[str, Any]:
        return {
            "answer": "The company allows remote work for up to 3 days per week for eligible employees. Full remote work is available for senior positions with manager approval.",
            "citations": [
                {
                    "id": 1,
                    "title": "Employee Handbook 2024",
                    "link": "https://drive.google.com/file/d/123/view",
                    "snippet": "Remote work policy: Employees may work remotely up to 3 days per week...",
                    "score": 0.95,
                    "chunk_id": "handbook_2024_1"
                }
            ],
            "used_mode": "hybrid",
            "latency_ms": 1250,
            "guardrails": {
                "safe": True,
                "grounded": True,
                "quality_score": 0.9,
                "grounding_score": 0.85,
                "notes": []
            }
        }
    
    @staticmethod
    def ingest_request() -> Dict[str, Any]:
        return {
            "folder_id": "1ABC123DEF456GHI789",
            "pipeline_name": "elser_v2_pipeline",
            "embedder_name": "sentence-transformers/all-MiniLM-L6-v2",
            "force": False,
            "ocr_mode": "auto",
            "max_files": 100,
            "max_pages": 50,
            "batch_size": 32,
            "verbose": True
        }
    
    @staticmethod
    def ingest_response() -> Dict[str, Any]:
        return {
            "indexed": 1250,
            "timings": {
                "total_duration_seconds": 45.2,
                "download_time_seconds": 12.1,
                "processing_time_seconds": 28.5,
                "indexing_time_seconds": 4.6
            }
        }
    
    @staticmethod
    def health_response() -> Dict[str, Any]:
        return {
            "status": "ok",
            "elastic": "ok",
            "models": "ok",
            "timestamp": "2024-01-15T10:30:00Z"
        }
