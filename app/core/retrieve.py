"""Document retrieval (ELSER, dense, BM25) and hybrid RRF fusion."""

from typing import List, Dict, Any, Optional, Literal
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchRetriever
from langchain.schema import Document

from app.core.config import (
    INDEX_NAME, ELSER_MODEL_ID, DEFAULT_TOP_K, MIN_SCORE_THRESHOLD,
    RRF_RANK_CONSTANT, EMBEDDING_DIMENSIONS
)
from app.core.embed import get_embedder

# Retrieval modes

RetrievalMode = Literal["elser", "hybrid", "bm25"]

def get_retriever(
    es_client: Elasticsearch,
    mode: RetrievalMode = "hybrid",
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_SCORE_THRESHOLD
) -> ElasticsearchRetriever:
    """
    Get a configured ElasticsearchRetriever based on the specified mode.
    
    Args:
        es_client: Elasticsearch client instance
        mode: Retrieval mode ("elser" or "hybrid")
        top_k: Number of results to return
        min_score: Minimum score threshold
        
    Returns:
        Configured ElasticsearchRetriever
    """
    if mode == "elser":
        return _get_elser_retriever(es_client, top_k, min_score)
    else:
        return _get_hybrid_retriever(es_client, top_k, min_score)

def _get_elser_retriever(
    es_client: Elasticsearch,
    top_k: int,
    min_score: float
) -> ElasticsearchRetriever:
    """Get ELSER-only retriever."""
    return ElasticsearchRetriever(
        es_connection=es_client,
        index_name=INDEX_NAME,
        top_k=top_k,
        min_score=min_score,
        query_field="content",
        vector_query_field="ml.tokens",
        vector_query_type="text_expansion",
        vector_query_model_id=ELSER_MODEL_ID,
    )

def _get_hybrid_retriever(
    es_client: Elasticsearch,
    top_k: int,
    min_score: float
) -> ElasticsearchRetriever:
    """Get hybrid retriever with custom DSL."""
    # This will be implemented with custom DSL queries
    # For now, return a basic retriever
    return ElasticsearchRetriever(
        es_connection=es_client,
        index_name=INDEX_NAME,
        top_k=top_k,
        min_score=min_score,
        query_field="content",
    )

# Custom retrieval

def search_elser(
    es_client: Elasticsearch,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_SCORE_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Perform ELSER-only sparse retrieval.
    
    Args:
        es_client: Elasticsearch client instance
        query: Search query
        top_k: Number of results to return
        min_score: Minimum score threshold
        
    Returns:
        List of search results with scores
    """
    # Try ELSER first, fall back to BM25 if it fails
    try:
        search_body = {
            "size": top_k,
            "_source": ["content", "text", "title", "metadata"],
            "query": {
                "text_expansion": {
                    "ml.tokens": {
                        "model_id": ELSER_MODEL_ID,
                        "model_text": query
                    }
                }
            },
            "min_score": min_score
        }
        
        response = es_client.search(index=INDEX_NAME, body=search_body)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "_id": hit["_id"],
                "_score": hit["_score"],
                "_source": hit["_source"],
                "score_elser": hit["_score"]
            })
        
        # If ELSER returns results, use them
        if results:
            return results
        else:
            raise Exception("ELSER returned no results")
            
    except Exception as e:
        print(f"ELSER search error: {e}")
        print("Falling back to BM25 search...")
        # Fallback to BM25 when ELSER is not available
        return search_bm25(es_client, query, top_k, min_score)

def search_dense(
    es_client: Elasticsearch,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_SCORE_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Perform dense vector retrieval.
    
    Args:
        es_client: Elasticsearch client instance
        query: Search query
        top_k: Number of results to return
        min_score: Minimum score threshold
        
    Returns:
        List of search results with scores
    """
    # Get query embedding
    embedder = get_embedder()
    query_vector = embedder.encode(query).tolist()
    
    search_body = {
        "size": top_k,
        "_source": ["content", "text", "title", "metadata"],
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'metadata.vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        },
        "min_score": min_score + 1.0  # Adjust for cosine similarity + 1.0
    }
    
    response = es_client.search(index=INDEX_NAME, body=search_body)
    
    results = []
    for hit in response["hits"]["hits"]:
        # Normalize score back to [0, 1] range
        raw_score = hit["_score"]
        normalized_score = max(0.0, min(1.0, (raw_score - 1.0) * 0.5 + 0.5))
        
        results.append({
            "_id": hit["_id"],
            "_score": normalized_score,
            "_source": hit["_source"],
            "score_dense": normalized_score
        })
    
    return results

def search_bm25(
    es_client: Elasticsearch,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_SCORE_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Perform BM25 keyword retrieval.
    
    Args:
        es_client: Elasticsearch client instance
        query: Search query
        top_k: Number of results to return
        min_score: Minimum score threshold
        
    Returns:
        List of search results with scores
    """
    search_body = {
        "size": top_k,
        "_source": ["content", "text", "title", "metadata"],
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content", "text"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        },
        "min_score": min_score
    }
    
    response = es_client.search(index=INDEX_NAME, body=search_body)
    
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "_id": hit["_id"],
            "_score": hit["_score"],
            "_source": hit["_source"],
            "score_bm25": hit["_score"]
        })
    
    return results

# RRF fusion

def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    rank_constant: int = RRF_RANK_CONSTANT
) -> List[Dict[str, Any]]:
    """
    Perform Reciprocal Rank Fusion on multiple result lists.
    
    Args:
        result_lists: List of result lists from different retrievers
        rank_constant: RRF rank constant (k)
        
    Returns:
        Fused and ranked results
    """
    doc_scores = {}
    doc_metadata = {}
    
    for result_list in result_lists:
        for rank, result in enumerate(result_list, 1):
            doc_id = result["_id"]
            
            # Store metadata from first occurrence
            if doc_id not in doc_metadata:
                doc_metadata[doc_id] = result
            
            # Calculate RRF score
            rrf_score = 1.0 / (rank_constant + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
    
    # Create fused results
    fused_results = []
    for doc_id, rrf_score in doc_scores.items():
        result = doc_metadata[doc_id].copy()
        result["rrf_score"] = rrf_score
        result["_score"] = rrf_score
        fused_results.append(result)
    
    # Sort by RRF score
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    return fused_results

def search_hybrid(
    es_client: Elasticsearch,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_SCORE_THRESHOLD,
    candidate_multiplier: int = 20
) -> List[Dict[str, Any]]:
    """
    Perform hybrid retrieval: ELSER + dense + BM25 with RRF fusion.
    
    Args:
        es_client: Elasticsearch client instance
        query: Search query
        top_k: Number of final results to return
        min_score: Minimum score threshold
        candidate_multiplier: Multiplier for candidate pool size
        
    Returns:
        List of fused search results
    """
    # Get larger candidate pools
    candidate_k = max(100, top_k * candidate_multiplier)
    
    # Perform parallel searches with error handling
    try:
        elser_results = search_elser(es_client, query, candidate_k, 0.0)
        elser_filtered = [r for r in elser_results if r.get("score_elser", 0.0) >= min_score]
    except Exception as e:
        print(f"ELSER failed in hybrid search: {e}")
        elser_filtered = []
    
    try:
        dense_results = search_dense(es_client, query, candidate_k, 0.0)
        dense_filtered = [r for r in dense_results if r.get("score_dense", 0.0) >= min_score]
    except Exception as e:
        print(f"Dense search failed in hybrid search: {e}")
        dense_filtered = []
    
    try:
        bm25_results = search_bm25(es_client, query, candidate_k, 0.0)
        bm25_filtered = [r for r in bm25_results if r.get("score_bm25", 0.0) >= min_score]
    except Exception as e:
        print(f"BM25 failed in hybrid search: {e}")
        bm25_filtered = []
    
    # Only use available result lists
    result_lists = [lst for lst in [elser_filtered, dense_filtered, bm25_filtered] if lst]
    
    # If no results from any method, return empty
    if not result_lists:
        return []
    
    # Fuse results with RRF
    fused_results = reciprocal_rank_fusion(result_lists)
    
    # Apply final score threshold and return top_k
    # Use much lower threshold for RRF scores since they're different scale (typically 0.01-0.05)
    rrf_min_score = max(0.001, min_score * 0.01)  # RRF scores are typically much lower
    final_results = [r for r in fused_results if r.get("rrf_score", 0.0) >= rrf_min_score]
    
    return final_results[:top_k]

# Public API

def is_relevant_query(query: str, results: List[Dict[str, Any]]) -> bool:
    """
    Check if the query is relevant to the retrieved documents.
    
    Args:
        query: The search query
        results: Retrieved documents
        
    Returns:
        True if query appears relevant to the results
    """
    if not results:
        return False
    
    # Extract query terms (remove common words)
    query_terms = set(query.lower().split())
    common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "about", "what", "how", "when", "where", "why", "who", "which"}
    query_terms = query_terms - common_words
    
    if not query_terms:
        return False
    
    # Check if any query terms appear in the top results (content, text, title, filename)
    for result in results[:3]:  # Check top 3 results
        source = result.get("_source", {})
        metadata = source.get("metadata", {})
        content = (
            (source.get("content", "") or source.get("text", ""))
            + " " + source.get("title", "")
            + " " + metadata.get("filename", "")
        ).lower()
        if any(term in content for term in query_terms):
            return True
    
    return False

def retrieve_documents(
    es_client: Elasticsearch,
    query: str,
    mode: RetrievalMode = "hybrid",
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_SCORE_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Main retrieval interface with intelligent fallback and relevance checking.
    
    Args:
        es_client: Elasticsearch client instance
        query: Search query
        mode: Retrieval mode ("elser", "hybrid", or "bm25")
        top_k: Number of results to return
        min_score: Minimum score threshold
        
    Returns:
        List of retrieved documents with scores
    """
    try:
        if mode == "elser":
            results = search_elser(es_client, query, top_k, min_score)
        elif mode == "bm25":
            results = search_bm25(es_client, query, top_k, min_score)
        else:
            results = search_hybrid(es_client, query, top_k, min_score)
        
        # If none of the top results contain key query terms, treat as no relevant docs
        if not is_relevant_query(query, results):
            print(f"No relevant hits for query '{query}' in top results; returning empty set")
            return []
        
        return results
        
    except Exception as e:
        print(f"Advanced search ({mode}) failed: {e}")
        print("Falling back to basic BM25 search...")
        results = search_bm25(es_client, query, top_k, min_score)
        
        # Check relevance for fallback results (disabled for now - too aggressive)
        # if not is_relevant_query(query, results):
        #     print(f"Fallback results for '{query}' appear irrelevant to available documents")
        #     return []
        
        return results

def format_citations(results: List[Dict[str, Any]], max_citations: int = 5) -> List[Dict[str, Any]]:
    """
    Format retrieval results as citations.
    
    Args:
        results: List of retrieval results
        max_citations: Maximum number of citations to return
        
    Returns:
        List of formatted citations
    """
    citations = []
    
    for i, result in enumerate(results[:max_citations], 1):
        source = result.get("_source", {})
        metadata = source.get("metadata", {})
        
        citation = {
            "id": i,
            "title": metadata.get("filename", "Untitled"),
            "link": metadata.get("drive_url", ""),
            "snippet": (source.get("content", "") or source.get("text", ""))[:280],
            "score": result.get("_score", 0.0),
            "chunk_id": metadata.get("chunk_id", ""),
        }
        
        citations.append(citation)
    
    return citations
