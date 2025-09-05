"""Answer generation using Hugging Face and custom prompting."""

import re
import time
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import BaseRetriever
from langchain.prompts import PromptTemplate

from app.core.config import (
    HF_API_KEY, HF_MODEL_ID, HF_ENDPOINT_URL,
    MAX_CONTEXT_TOKENS, MAX_ANSWER_TOKENS,
    MIN_GROUNDING_OVERLAP, MIN_ANSWER_LEN
)
import tiktoken

def _count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        # Use cl100k_base encoding (compatible with most models)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimation (4 characters per token)
        return len(text) // 4

# Safety and grounding

SAFETY_BLOCKLIST = [
    "ransomware", "exploit", "bomb", "bioweapon", "self-harm",
    "suicide", "violence", "hate", "discrimination", "illegal"
]

GREETING_PATTERNS = [
    r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
    r'\b(how are you|how do you do|what\'s up|how\'s it going)\b',
    r'\b(thanks|thank you|thank you very much)\b',
    r'\b(bye|goodbye|see you|farewell)\b',
    r'\b(help|can you help|what can you do)\b',
    r'\b(introduce|who are you|what are you)\b'
]

FRIENDLY_RESPONSES = {
    "greeting": [
        "Hello! I'm your AI RAG Assistant. I'm here to help you learn from your documents. What would you like to know?",
        "Hi there! I'm ready to help you explore your knowledge base. Ask me anything about your documents!",
        "Hello! I'm your document assistant. Feel free to ask me questions about the content in your knowledge base."
    ],
    "how_are_you": [
        "I'm doing great, thank you for asking! I'm ready to help you learn from your documents. What would you like to explore?",
        "I'm excellent! I'm here and ready to assist you with questions about your knowledge base. What can I help you with?",
        "I'm doing well! I'm excited to help you discover insights from your documents. What would you like to know?"
    ],
    "thanks": [
        "You're very welcome! I'm happy to help. Feel free to ask me anything else about your documents.",
        "My pleasure! I'm here whenever you need help with your studies. What else would you like to explore?",
        "You're welcome! I enjoy helping you learn. Is there anything else you'd like to know about your documents?"
    ],
    "goodbye": [
        "Goodbye! It was great helping you today. Come back anytime you need assistance with your documents!",
        "See you later! I'm always here when you need help Analysing. Take care!",
        "Farewell! I hope I was helpful. Feel free to return anytime for more learning assistance!"
    ],
    "help": [
        "I'm your AI RAG Assistant! I can help you learn from your documents by answering questions about their content. Try asking me about specific topics, concepts, or information you're looking for.",
        "I'm here to help you Analyse! I can answer questions about the documents in your knowledge base. Just ask me about any topic you're curious about.",
        "I'm your document assistant! I can help you understand and explore the content in your knowledge base. What would you like to learn about?"
    ],
    "introduce": [
        "I'm your AI RAG Assistant! I'm designed to help you learn from your documents by answering questions about their content. I can search through your knowledge base and provide detailed, cited answers to help you analyse effectively.",
        "Hello! I'm an AI assistant specialized in helping you analyse from your documents. I can answer questions, explain concepts, and provide insights from your knowledge base to support your learning journey.",
        "I'm your personal RAG assistant! I help you explore and understand the content in your documents. I can answer questions, provide explanations, and help you find specific information you need for your studies."
    ]
}

def is_unsafe_query(query: str) -> bool:
    """Check if query contains unsafe content."""
    query_lower = query.lower()
    return any(term in query_lower for term in SAFETY_BLOCKLIST)

def is_greeting_query(query: str) -> tuple[bool, str]:
    """
    Check if query is a greeting or casual conversation.
    
    Returns:
        Tuple of (is_greeting, greeting_type)
    """
    query_lower = query.lower().strip()
    
    # Check for greeting patterns
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, query_lower):
            if re.search(r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b', query_lower):
                return True, "greeting"
            elif re.search(r'\b(how are you|how do you do|what\'s up|how\'s it going)\b', query_lower):
                return True, "how_are_you"
            elif re.search(r'\b(thanks|thank you|thank you very much)\b', query_lower):
                return True, "thanks"
            elif re.search(r'\b(bye|goodbye|see you|farewell)\b', query_lower):
                return True, "goodbye"
            elif re.search(r'\b(help|can you help|what can you do)\b', query_lower):
                return True, "help"
            elif re.search(r'\b(introduce|who are you|what are you)\b', query_lower):
                return True, "introduce"
    
    return False, ""

def get_friendly_response(greeting_type: str) -> str:
    """Get a random friendly response for the greeting type."""
    import random
    responses = FRIENDLY_RESPONSES.get(greeting_type, FRIENDLY_RESPONSES["greeting"])
    return random.choice(responses)

def calculate_grounding_score(answer: str, context: str) -> float:
    """
    Calculate grounding score based on lexical overlap.
    
    Args:
        answer: Generated answer
        context: Retrieved context
        
    Returns:
        Grounding score between 0 and 1
    """
    if not answer or not context:
        return 0.0
    
    # Simple word overlap calculation
    answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
    context_words = set(re.findall(r'\b\w+\b', context.lower()))
    
    if not answer_words:
        return 0.0
    
    overlap = len(answer_words & context_words)
    return overlap / len(answer_words)

def is_grounded(answer: str, context: str, min_overlap: float = MIN_GROUNDING_OVERLAP) -> bool:
    """Check if answer is sufficiently grounded in context."""
    return calculate_grounding_score(answer, context) >= min_overlap

# Prompt templates

SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive, detailed explanations based on the provided context.

MANDATORY REQUIREMENTS:
1. You MUST provide a detailed answer of at least 6-8 sentences that thoroughly explains the topic
2. COMPLETELY IGNORE all PDF formatting artifacts: page numbers, symbols like --`,`,`,`,,`,-`-, copyright notices, headers, footers
3. IGNORE document metadata phrases like "This third edition cancels and replaces" - these are NOT the answer
4. EXTRACT and SYNTHESIZE the actual meaningful content about the topic from across all sources
5. START with a clear definition, then explain the purpose, benefits, and key concepts
6. Use educational, professional language as if teaching someone about the topic
7. Make your answer comprehensive and informative - provide context and details

EXAMPLE TASK: If asked about ISO 27001, ignore metadata like "third edition" and focus on content about "information security management systems", "organizational processes", "security controls", etc.

YOU MUST WRITE AT LEAST 6-8 COMPLETE SENTENCES. Short answers are not acceptable.

Context (extract meaningful content, ignore all formatting artifacts):
{context}

Question: {question}

Comprehensive Answer (minimum 6-8 sentences, focus on educational content):"""

def create_prompt_template() -> PromptTemplate:
    """Create the prompt template for the RAG system."""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT
    )

def format_context_for_prompt(results: List[Dict[str, Any]], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """
    Format retrieval results as context for the prompt.
    
    Args:
        results: List of retrieval results
        max_tokens: Maximum tokens for context
        
    Returns:
        Formatted context string
    """
    context_parts = []
    current_tokens = 0
    
    for i, result in enumerate(results, 1):
        source = result.get("_source", {})
        raw_content = source.get("content", "") or source.get("text", "")
        title = source.get("metadata", {}).get("filename", f"Document {i}")
        
        # Use raw content but let the LLM handle artifact filtering
        content = raw_content
        
        # Only skip completely empty content
        if len(content.strip()) < 10:
            continue
        
        # Estimate tokens for this chunk
        chunk_text = f"[{i}] {title}: {content}\n\n"
        chunk_tokens = _count_tokens(chunk_text)
        
        if current_tokens + chunk_tokens > max_tokens:
            break
        
        context_parts.append(chunk_text)
        current_tokens += chunk_tokens
    
    return "".join(context_parts)

# Hugging Face integration

def create_huggingface_llm() -> HuggingFaceEndpoint:
    """Create Hugging Face endpoint for LLM inference."""
    return HuggingFaceEndpoint(
        endpoint_url=f"{HF_ENDPOINT_URL}/{HF_MODEL_ID}",
        huggingfacehub_api_token=HF_API_KEY,
        task="text-generation",
        model_kwargs={
            "temperature": 0.2,
            "max_new_tokens": MAX_ANSWER_TOKENS,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }
    )

# RetrievalQA chain

class CustomRetrievalQA(RetrievalQA):
    """Custom RetrievalQA chain with enhanced prompting and grounding."""
    
    def __init__(self, llm, retriever, prompt_template, **kwargs):
        super().__init__(
            llm=llm,
            retriever=retriever,
            prompt=prompt_template,
            return_source_documents=True,
            **kwargs
        )
    
    def _call(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Override _call to add custom processing."""
        question = inputs[self.input_key]
        
        # Check for greetings and casual conversation
        is_greeting, greeting_type = is_greeting_query(question)
        if is_greeting:
            friendly_response = get_friendly_response(greeting_type)
            return {
                "result": friendly_response,
                "source_documents": [],
                "grounding_score": 1.0,
                "is_grounded": True
            }
        
        # Get retrieved documents
        docs = self.retriever.get_relevant_documents(question)
        
        if not docs:
            return {
                "result": (
                    "Sorry, I'm not able to discuss that. I only have access to and can answer based on "
                    "your internal documents that have been indexed. Please ask about topics contained in the "
                    "uploaded content."
                ),
                "source_documents": [],
                "grounding_score": 0.0,
                "is_grounded": False
            }
        
        # Format context
        context = format_context_for_prompt(docs)
        
        # Generate answer
        prompt = self.prompt.format(context=context, question=question)
        answer = self.llm.invoke(prompt)
        
        # Calculate grounding
        full_context = " ".join([doc.page_content for doc in docs])
        grounding_score = calculate_grounding_score(answer, full_context)
        is_grounded_result = is_grounded(answer, full_context)
        
        # Apply grounding check
        if not is_grounded_result and grounding_score < MIN_GROUNDING_OVERLAP:
            answer = (
                "Sorry, I'm not able to discuss that. I only have access to and can answer based on "
                "your internal documents that have been indexed. Please ask about topics contained in the "
                "uploaded content."
            )
        
        return {
            "result": answer,
            "source_documents": docs,
            "grounding_score": grounding_score,
            "is_grounded": is_grounded_result
        }

def create_retrieval_qa_chain(retriever: BaseRetriever) -> CustomRetrievalQA:
    """
    Create a RetrievalQA chain with Hugging Face LLM.
    
    Args:
        retriever: Document retriever
        
    Returns:
        Configured RetrievalQA chain
    """
    llm = create_huggingface_llm()
    prompt_template = create_prompt_template()
    
    return CustomRetrievalQA(
        llm=llm,
        retriever=retriever,
        prompt_template=prompt_template
    )

# Public API

def generate_answer(
    question: str,
    retrieval_results: List[Dict[str, Any]],
    retriever: Optional[BaseRetriever] = None
) -> Dict[str, Any]:
    """
    Generate an answer using retrieval results.
    
    Args:
        question: User question
        retrieval_results: List of retrieval results
        retriever: Optional retriever for chain-based generation
        
    Returns:
        Dictionary with answer, citations, and metadata
    """
    start_time = time.time()
    
    # Safety check
    if is_unsafe_query(question):
        return {
            "answer": "I cannot help with that request.",
            "citations": [],
            "score": 0.0,
            "grounding_score": 0.0,
            "is_grounded": False,
            "latency_ms": 0
        }
    
    # Check for greetings and casual conversation
    is_greeting, greeting_type = is_greeting_query(question)
    if is_greeting:
        friendly_response = get_friendly_response(greeting_type)
        return {
            "answer": friendly_response,
            "citations": [],
            "score": 1.0,
            "grounding_score": 1.0,
            "is_grounded": True,
            "latency_ms": int((time.time() - start_time) * 1000)
        }
    
    # Handle empty results
    if not retrieval_results:
        return {
            "answer": (
                "Sorry, I'm not able to discuss that. I only have access to and can answer based on "
                "your internal documents that have been indexed. Please ask about topics contained in the "
                "uploaded content."
            ),
            "citations": [],
            "score": 0.0,
            "grounding_score": 0.0,
            "is_grounded": False,
            "latency_ms": 0
        }
    
    # Format context
    context = format_context_for_prompt(retrieval_results)
    
    # Generate answer using Hugging Face API
    try:
        llm = create_huggingface_llm()
        prompt = create_prompt_template().format(context=context, question=question)
        
        # Generate response
        response = llm.invoke(prompt)
        
        # Clean up response
        answer = response.strip()
        
        # Remove any prompt artifacts
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Clean up PDF artifacts from the answer
        answer = _clean_answer_text(answer)
        
        # Enhance with visual formatting hints
        answer = enhance_answer_with_visuals(answer, question, context)
        
        # Calculate grounding
        full_context = " ".join([r.get("_source", {}).get("content", "") or r.get("_source", {}).get("text", "") for r in retrieval_results])
        grounding_score = calculate_grounding_score(answer, full_context)
        is_grounded_result = is_grounded(answer, full_context)
        
        # Apply grounding check
        if not is_grounded_result and grounding_score < MIN_GROUNDING_OVERLAP:
            answer = (
                "Sorry, I'm not able to discuss that. I only have access to and can answer based on "
                "your internal documents that have been indexed. Please ask about topics contained in the "
                "uploaded content."
            )
            grounding_score = 0.0
            is_grounded_result = False
        
        # Format citations
        citations = format_citations(retrieval_results)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "answer": answer,
            "citations": citations,
            "score": grounding_score,
            "grounding_score": grounding_score,
            "is_grounded": is_grounded_result,
            "latency_ms": latency_ms
        }
        
    except Exception as e:
        # Fallback to extractive answer
        return generate_extractive_answer(question, retrieval_results, start_time)

def generate_extractive_answer(
    question: str,
    retrieval_results: List[Dict[str, Any]],
    start_time: float
) -> Dict[str, Any]:
    """
    Generate an extractive answer as fallback.
    
    Args:
        question: User question
        retrieval_results: List of retrieval results
        start_time: Start time for latency calculation
        
    Returns:
        Dictionary with extractive answer
    """
    # Find best matching sentence
    question_words = set(re.findall(r'\b\w+\b', question.lower()))
    best_sentence = ""
    best_score = 0.0
    
    for result in retrieval_results:
        source = result.get("_source", {})
        content = source.get("content", "") or source.get("text", "")
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(question_words & sentence_words)
            
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
    
    answer = best_sentence if best_sentence else (
        "Sorry, I'm not able to discuss that. I only have access to and can answer based on "
        "your internal documents that have been indexed. Please ask about topics contained in the "
        "uploaded content."
    )
    citations = format_citations(retrieval_results)
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "answer": answer,
        "citations": citations,
        "score": best_score / max(1, len(question_words)),
        "grounding_score": best_score / max(1, len(question_words)),
        "is_grounded": best_score > 0,
        "latency_ms": latency_ms
    }

def should_format_as_table(question: str, context: str) -> bool:
    """Determine if the answer should be formatted as a table."""
    table_keywords = [
        "table", "list", "compare", "comparison", "steps", "requirements", 
        "clauses", "structure", "breakdown", "categories", "types"
    ]
    return any(keyword in question.lower() for keyword in table_keywords)

def should_create_diagram(question: str, context: str) -> bool:
    """Determine if the answer should include a diagram."""
    diagram_keywords = [
        "flow", "process", "workflow", "architecture", "structure", 
        "relationship", "hierarchy", "diagram", "chart", "model"
    ]
    return any(keyword in question.lower() for keyword in diagram_keywords)

def _clean_and_filter_content(text: str) -> str:
    """Enhanced content cleaning and filtering for better context quality."""
    if not text:
        return ""
    
    # Remove PDF artifacts more aggressively
    text = re.sub(r'--- Page \d+ ---', '', text)
    text = re.sub(r'Page\d+\|\d+', '', text)
    text = re.sub(r'\d+\s*--`,`,.*?---', '', text)
    text = re.sub(r'--[`,\-\s]{3,}', ' ', text)
    text = re.sub(r'[,]{3,}', '', text)
    text = re.sub(r'[`]{2,}', '', text)
    text = re.sub(r'[-]{3,}', '', text)
    
    # Remove headers/footers
    text = re.sub(r'DZONE\.COM.*?REFCARDZ.*?FOR.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'© DZONE.*?\d{4}.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'BROUGHT TO YOU.*?WITH.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'© ISO/IEC \d{4}.*?\n', '', text, flags=re.IGNORECASE)
    
    # Remove standalone numbers and artifacts
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip artifact lines
        if re.match(r'^[\d\s\-`,]+$', line) and len(line) < 30:
            continue
        if len(line) > 10:  # Keep substantial lines
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Fix spacing and punctuation
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def _is_content_meaningful(text: str) -> bool:
    """Check if content is meaningful enough to use in context (less aggressive)."""
    if len(text.strip()) < 20:
        return False
    
    words = text.split()
    if len(words) < 5:
        return False
    
    # Only filter out the most obvious metadata
    if "third edition cancels and replaces" in text.lower():
        return False
    
    return True

def _clean_answer_text(text: str) -> str:
    """Clean PDF artifacts from generated answers."""
    if not text:
        return ""
    
    # Remove common PDF artifacts that appear in answers
    text = re.sub(r"[`]{2,}", "", text)  # Remove multiple backticks
    text = re.sub(r"[-—]{3,}", "", text)  # Remove long dashes
    text = re.sub(r"[,]{3,}", "", text)  # Remove multiple commas
    text = re.sub(r"[.]{3,}", "...", text)  # Normalize ellipsis
    text = re.sub(r"v\s*©", "©", text)  # Fix copyright formatting
    text = re.sub(r"--[`,\-\s]+", " ", text)  # Remove dash artifacts
    
    # Remove page markers and artifacts
    text = re.sub(r"--- Page \d+ ---", "", text)  # Remove page markers
    text = re.sub(r"Page\d+\|\d+", "", text)  # Remove page references
    text = re.sub(r"\d+\s*--`,`,.*?---", "", text)  # Remove complex artifacts
    
    # Remove standalone copyright lines that are formatting artifacts
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip lines that are just formatting artifacts
        if (len(line) < 10 and 
            any(char in line for char in ['`', '---', ',,,']) and 
            'ISO' not in line and 'IEC' not in line):
            continue
        # Skip lines that are mostly artifacts
        if re.match(r'^[\d\s\-`,]+$', line):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

def enhance_answer_with_visuals(answer: str, question: str, context: str) -> str:
    """Enhance answer with tables or diagrams if appropriate."""
    enhanced_answer = answer
    
    # Add table formatting hint if needed
    if should_format_as_table(question, context) and "|" not in answer:
        enhanced_answer += "\n\n*Note: This information could be displayed in a structured table format.*"
    
    # Add diagram hint if needed  
    if should_create_diagram(question, context):
        enhanced_answer += "\n\n*Note: This process could be visualized as a flowchart or diagram.*"
    
    return enhanced_answer

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
