"""Safety filters, grounding checks, and citation auditing."""

import re
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import MIN_GROUNDING_OVERLAP, MIN_ANSWER_LEN, MAX_ANSWER_TOKENS

# Safety configuration

SAFETY_BLOCKLIST = [
    # Violence and weapons
    "bomb", "explosive", "weapon", "gun", "knife", "violence", "attack", "kill", "murder",
    
    # Self-harm and suicide
    "suicide", "self-harm", "self-injury", "cutting", "overdose",
    
    # Illegal activities
    "drug", "cocaine", "heroin", "meth", "illegal", "fraud", "scam", "hack", "exploit",
    
    # Hate and discrimination
    "hate", "racist", "discrimination", "prejudice", "bigotry",
    
    # Malware and cyber threats
    "ransomware", "malware", "virus", "trojan", "phishing", "spam",
    
    # Dangerous substances
    "poison", "toxic", "chemical weapon", "bioweapon", "anthrax",
    
    # Other harmful content
    "grooming", "exploitation", "abuse", "harassment", "stalking"
]

CONTENT_QUALITY_PATTERNS = [
    r"^[^a-zA-Z]*$",  # No letters
    r"^.{1,10}$",     # Too short
    r"^[A-Z\s]{20,}$", # All caps
    r"^[a-z\s]{20,}$", # All lowercase
    r"^[0-9\s]{10,}$", # All numbers
]

# Safety functions

def is_unsafe_query(query: str) -> bool:
    """
    Check if a query contains unsafe content.
    
    Args:
        query: User query to check
        
    Returns:
        True if query is unsafe, False otherwise
    """
    if not query:
        return False
    
    query_lower = query.lower()
    
    # Check against blocklist
    for term in SAFETY_BLOCKLIST:
        if term in query_lower:
            return True
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r"how to (make|create|build).*(bomb|weapon|drug)",
        r"where to (buy|get).*(drug|weapon)",
        r"how to (kill|harm|hurt).*",
        r"suicide.*(method|way|how)",
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return False

def is_unsafe_answer(answer: str) -> bool:
    """
    Check if an answer contains unsafe content.
    
    Args:
        answer: Generated answer to check
        
    Returns:
        True if answer is unsafe, False otherwise
    """
    if not answer:
        return False
    
    answer_lower = answer.lower()
    
    # Check against blocklist
    for term in SAFETY_BLOCKLIST:
        if term in answer_lower:
            return True
    
    # Check for dangerous instructions
    dangerous_patterns = [
        r"step.*by.*step.*(how to|instructions)",
        r"first.*then.*(mix|combine|add)",
        r"you need.*(ingredients|materials|tools)",
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, answer_lower):
            return True
    
    return False

# Content quality

def is_low_quality_content(text: str) -> bool:
    """
    Check if text is low quality (gibberish, too short, etc.).
    
    Args:
        text: Text to check
        
    Returns:
        True if text is low quality, False otherwise
    """
    if not text or len(text.strip()) < MIN_ANSWER_LEN:
        return True
    
    # Check against quality patterns
    for pattern in CONTENT_QUALITY_PATTERNS:
        if re.match(pattern, text.strip()):
            return True
    
    # Check character diversity
    if len(set(text.lower())) < 5:  # Too few unique characters
        return True
    
    # Check for excessive repetition
    words = text.split()
    if len(words) > 10:
        unique_words = set(word.lower() for word in words)
        if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
            return True
    
    return False

def calculate_grounding_score(answer: str, context: str) -> float:
    """
    Calculate how well the answer is grounded in the context.
    
    Args:
        answer: Generated answer
        context: Retrieved context
        
    Returns:
        Grounding score between 0 and 1
    """
    if not answer or not context:
        return 0.0
    
    # Extract meaningful words (exclude stop words)
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "should", "could", "can", "may", "might",
        "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
    }
    
    answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
    context_words = set(re.findall(r'\b\w+\b', context.lower()))
    
    # Remove stop words
    answer_words = answer_words - stop_words
    context_words = context_words - stop_words
    
    if not answer_words:
        return 0.0
    
    # Calculate overlap
    overlap = len(answer_words & context_words)
    return overlap / len(answer_words)

def is_grounded(answer: str, context: str, min_overlap: float = MIN_GROUNDING_OVERLAP) -> bool:
    """
    Check if answer is sufficiently grounded in context.
    
    Args:
        answer: Generated answer
        context: Retrieved context
        min_overlap: Minimum overlap threshold
        
    Returns:
        True if answer is grounded, False otherwise
    """
    return calculate_grounding_score(answer, context) >= min_overlap

# Citation utilities

def validate_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and clean citations.
    
    Args:
        citations: List of citations to validate
        
    Returns:
        List of validated citations
    """
    validated = []
    seen_urls = set()
    
    for citation in citations:
        # Check required fields
        if not all(key in citation for key in ["title", "link", "snippet"]):
            continue
        
        # Check for duplicate URLs
        url = citation.get("link", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        
        # Validate snippet length
        snippet = citation.get("snippet", "")
        if len(snippet) < 10 or len(snippet) > 500:
            continue
        
        # Clean title
        title = citation.get("title", "").strip()
        if not title or len(title) < 3:
            title = "Untitled"
        
        validated.append({
            "title": title,
            "link": url,
            "snippet": snippet[:280],  # Limit snippet length
            "id": len(validated) + 1
        })
    
    return validated

def audit_citations(answer: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Audit citations for accuracy and relevance.
    
    Args:
        answer: Generated answer
        citations: List of citations
        
    Returns:
        Audit results
    """
    audit_results = {
        "total_citations": len(citations),
        "valid_citations": 0,
        "relevant_citations": 0,
        "issues": []
    }
    
    if not citations:
        audit_results["issues"].append("No citations provided")
        return audit_results
    
    # Check citation relevance
    answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
    
    for i, citation in enumerate(citations):
        snippet = citation.get("snippet", "").lower()
        snippet_words = set(re.findall(r'\b\w+\b', snippet))
        
        # Check if citation is relevant
        overlap = len(answer_words & snippet_words)
        if overlap > 0:
            audit_results["relevant_citations"] += 1
        
        # Check citation quality
        if len(snippet) < 20:
            audit_results["issues"].append(f"Citation {i+1} has very short snippet")
        
        if not citation.get("link"):
            audit_results["issues"].append(f"Citation {i+1} missing link")
        
        audit_results["valid_citations"] += 1
    
    return audit_results

# Public interface

def apply_guardrails(
    query: str,
    answer: str,
    context: str,
    citations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Apply all guardrails to a query-answer pair.
    
    Args:
        query: User query
        answer: Generated answer
        context: Retrieved context
        citations: List of citations
        
    Returns:
        Guardrails results
    """
    results = {
        "safe": True,
        "grounded": True,
        "quality_score": 0.0,
        "grounding_score": 0.0,
        "notes": []
    }
    
    # Safety checks
    if is_unsafe_query(query):
        results["safe"] = False
        results["notes"].append("Query contains unsafe content")
    
    if is_unsafe_answer(answer):
        results["safe"] = False
        results["notes"].append("Answer contains unsafe content")
    
    # Quality checks
    if is_low_quality_content(answer):
        results["quality_score"] = 0.0
        results["notes"].append("Answer is low quality")
    else:
        results["quality_score"] = 1.0
    
    # Grounding checks
    grounding_score = calculate_grounding_score(answer, context)
    results["grounding_score"] = grounding_score
    
    if not is_grounded(answer, context):
        results["grounded"] = False
        results["notes"].append("Answer not sufficiently grounded in context")
    
    # Citation validation
    validated_citations = validate_citations(citations)
    if len(validated_citations) != len(citations):
        results["notes"].append("Some citations were invalid and removed")
    
    # Citation audit
    audit_results = audit_citations(answer, validated_citations)
    if audit_results["issues"]:
        results["notes"].extend(audit_results["issues"])
    
    return results

def should_reject_answer(guardrails_results: Dict[str, Any]) -> bool:
    """
    Determine if an answer should be rejected based on guardrails.
    
    Args:
        guardrails_results: Results from apply_guardrails
        
    Returns:
        True if answer should be rejected, False otherwise
    """
    # Reject if unsafe
    if not guardrails_results["safe"]:
        return True
    
    # Reject if very low quality
    if guardrails_results["quality_score"] < 0.1:
        return True
    
    # Reject if not grounded and low quality
    if not guardrails_results["grounded"] and guardrails_results["quality_score"] < 0.5:
        return True
    
    return False
