"""
FastAPI main application for the RAG system.
Provides endpoints for querying, ingestion, and health checks.
"""

import time
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

from app.core.config import (
    get_elasticsearch_config, validate_config, INDEX_NAME,
    ELSER_MODEL_ID, API_HOST, API_PORT
)
from app.core.retrieve import retrieve_documents, format_citations
from app.core.generate import generate_answer
from app.core.ingest import ingest_documents, create_elasticsearch_mapping, create_elser_pipeline
from app.api.schemas import (
    QueryRequest, QueryResponse, IngestRequest, IngestResponse,
    HealthResponse, ErrorResponse, Citation, GuardrailsResult
)
from app.api.guardrails import apply_guardrails, should_reject_answer

# =========================
# Application Setup
# =========================

# Validate configuration
try:
    validate_config()
except RuntimeError as e:
    print(f"Configuration error: {e}")
    exit(1)

# Create FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation system with Elasticsearch and Hugging Face",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Elasticsearch Client
# =========================

def get_elasticsearch_client() -> Elasticsearch:
    """Get configured Elasticsearch client."""
    config = get_elasticsearch_config()
    return Elasticsearch(**config)

# =========================
# Health Check
# =========================

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns system status including Elasticsearch and model availability.
    """
    try:
        es = get_elasticsearch_client()
        
        # Check Elasticsearch connection
        es_status = "ok"
        try:
            es.ping()
            # Check if index exists
            if not es.indices.exists(index=INDEX_NAME):
                es_status = "warning: index not found"
        except ConnectionError:
            es_status = "error: connection failed"
        except Exception as e:
            es_status = f"error: {str(e)}"
        
        # Check models (simplified check)
        models_status = "ok"
        try:
            # This would check if models are loaded/available
            # For now, we'll assume they're ok if we can import them
            from app.core.embed import get_embedder
            from app.core.generate import create_huggingface_llm
            get_embedder()
            # Don't actually create LLM to avoid API calls
        except Exception as e:
            models_status = f"error: {str(e)}"
        
        # Overall status
        overall_status = "ok"
        if "error" in es_status or "error" in models_status:
            overall_status = "error"
        elif "warning" in es_status:
            overall_status = "warning"
        
        return HealthResponse(
            status=overall_status,
            elastic=es_status,
            models=models_status
        )
        
    except Exception as e:
        return HealthResponse(
            status="error",
            elastic="error: client creation failed",
            models="error: client creation failed"
        )

# =========================
# Query Endpoint
# =========================

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest = Body(...)):
    """
    Query documents using RAG pipeline.
    
    Args:
        request: Query request with question, mode, and top_k
        
    Returns:
        Query response with answer, citations, and metadata
    """
    start_time = time.time()
    
    try:
        # Get Elasticsearch client
        es = get_elasticsearch_client()
        
        # Retrieve documents
        retrieval_results = retrieve_documents(
            es_client=es,
            query=request.question,
            mode=request.mode,
            top_k=request.top_k
        )
        
        # Generate answer
        generation_result = generate_answer(
            question=request.question,
            retrieval_results=retrieval_results
        )
        
        # Apply guardrails
        context = " ".join([r.get("_source", {}).get("content", "") for r in retrieval_results])
        guardrails_result = apply_guardrails(
            query=request.question,
            answer=generation_result["answer"],
            context=context,
            citations=generation_result["citations"]
        )
        
        # Check if answer should be rejected
        if should_reject_answer(guardrails_result):
            generation_result["answer"] = (
                "Sorry, I'm not able to discuss that. I only have access to and can answer based on "
                "your internal documents that have been indexed. Please ask about topics contained in the "
                "uploaded content."
            )
            generation_result["citations"] = []
            guardrails_result["grounded"] = False
            guardrails_result["grounding_score"] = 0.0
        
        # Format citations
        citations = [
            Citation(
                id=c["id"],
                title=c["title"],
                link=c["link"],
                snippet=c["snippet"],
                score=c.get("score"),
                chunk_id=c.get("chunk_id")
            )
            for c in generation_result["citations"]
        ]
        
        # Calculate total latency
        total_latency = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            answer=generation_result["answer"],
            citations=citations,
            used_mode=request.mode,
            latency_ms=total_latency,
            guardrails=GuardrailsResult(
                safe=guardrails_result["safe"],
                grounded=guardrails_result["grounded"],
                quality_score=guardrails_result["quality_score"],
                grounding_score=guardrails_result["grounding_score"],
                notes=guardrails_result["notes"]
            )
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

# =========================
# Ingest Endpoint
# =========================

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents_endpoint(request: IngestRequest = Body(...)):
    """
    Ingest documents from Google Drive.
    
    Args:
        request: Ingest request with folder_id and processing options
        
    Returns:
        Ingest response with indexing results and timings
    """
    try:
        # Get Elasticsearch client
        es = get_elasticsearch_client()
        
        # Ensure index exists with proper mapping
        if not es.indices.exists(index=INDEX_NAME):
            mapping = create_elasticsearch_mapping()
            es.indices.create(index=INDEX_NAME, body=mapping)
        
        # Ensure ELSER pipeline exists
        try:
            es.ingest.get_pipeline(id="elser_v2_pipeline")
        except NotFoundError:
            pipeline = create_elser_pipeline()
            es.ingest.put_pipeline(id="elser_v2_pipeline", body=pipeline)
        
        # Perform ingestion
        result = ingest_documents(
            es_client=es,
            folder_id=request.folder_id,
            index_name=INDEX_NAME,
            force=request.force,
            max_files=request.max_files,
            batch_size=request.batch_size,
            verbose=request.verbose
        )
        
        return IngestResponse(
            indexed=result["chunks_indexed"],
            timings={
                "total_duration_seconds": result["duration_seconds"],
                "files_processed": result["files_seen"],
                "chunks_per_second": result["chunks_indexed"] / max(1, result["duration_seconds"])
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

# =========================
# Debug Endpoints
# =========================

@app.get("/debug/retrieve")
async def debug_retrieve(
    q: str = Query(..., min_length=1, description="Query string"),
    mode: str = Query("hybrid", description="Retrieval mode"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results")
):
    """
    Debug endpoint for testing retrieval without generation.
    
    Args:
        q: Query string
        mode: Retrieval mode (elser or hybrid)
        top_k: Number of results to return
        
    Returns:
        Raw retrieval results
    """
    try:
        es = get_elasticsearch_client()
        
        # Retrieve documents
        results = retrieve_documents(
            es_client=es,
            query=q,
            mode=mode,
            top_k=top_k
        )
        
        # Format results for debugging
        debug_results = []
        for result in results:
            source = result.get("_source", {})
            debug_results.append({
                "title": source.get("metadata", {}).get("filename", "Untitled"),
                "snippet": source.get("content", "")[:220],
                "drive_url": source.get("metadata", {}).get("drive_url", ""),
                "score": result.get("_score", 0.0),
                "chunk_id": source.get("metadata", {}).get("chunk_id", "")
            })
        
        return {
            "results": debug_results,
            "query": q,
            "mode": mode,
            "top_k": top_k,
            "total_hits": len(results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Debug retrieval failed: {str(e)}"
        )

# =========================
# Error Handlers
# =========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with custom error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"Status code: {exc.status_code}"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    import json
    from datetime import datetime
    
    # Convert datetime objects to strings for JSON serialization
    def json_serial(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    try:
        error_detail = str(exc)
        # Remove any datetime objects from the error string
        error_detail = error_detail.replace("datetime.datetime", "datetime")
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=error_detail
            ).dict()
        )
    except Exception as json_error:
        # Fallback if JSON serialization still fails
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An error occurred while processing your request"
            }
        )

# =========================
# Application Startup
# =========================

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("Starting RAG System API...")
    print(f"Elasticsearch: {get_elasticsearch_config()['hosts'][0]}")
    print(f"Index: {INDEX_NAME}")
    print(f"ELSER Model: {ELSER_MODEL_ID}")
    print(f"API URL: http://{API_HOST}:{API_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    print("Shutting down RAG System API...")

# =========================
# Main Application
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
