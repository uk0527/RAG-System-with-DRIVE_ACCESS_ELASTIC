#!/usr/bin/env python3
"""Streamlit UI for the RAG assistant."""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import uuid

# Page config

st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chat session state
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = "Chat 1"
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1

# Styles

st.markdown("""
<style>
    /* Global reset and font */
    * {
        box-sizing: border-box;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Theme variables */
    :root {
        --bg-color: #ffffff;
        --text-color: #1f2937;
        --bubble-user-bg: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        --bubble-ai-bg: #f8fafc;
        --bubble-ai-border: #e5e7eb;
        --accent-color: #8b5cf6;
        --secondary-bg: #f3f4f6;
    }

    /* Dark mode */
    [data-theme="dark"] {
        --bg-color: #1f2937;
        --text-color: #f3f4f6;
        --bubble-user-bg: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
        --bubble-ai-bg: #374151;
        --bubble-ai-border: #4b5563;
        --secondary-bg: #374151;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, .stDeployButton {display: none;}
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1.5rem 1.5rem 160px 1.5rem;
        background: var(--bg-color);
        min-height: calc(100vh - 160px);
        color: var(--text-color);
        transition: background 0.3s ease;
        overflow-y: auto;
    }
    
    /* Header */
    .chat-header {
        text-align: center;
        padding: 2.5rem 0;
        border-bottom: 1px solid var(--bubble-ai-border);
        margin-bottom: 2rem;
    }
    
    .chat-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: var(--text-color);
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .chat-subtitle {
        color: #6b7280;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Message bubbles */
    .message-container {
        margin: 1.25rem 0;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }
    
    .user-message {
        flex-direction: row-reverse;
    }
    
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        font-weight: 600;
        flex-shrink: 0;
        color: white;
        border: 2px solid var(--bubble-ai-bg);
    }
    
    .user-avatar {
        background: var(--bubble-user-bg);
    }
    
    .ai-avatar {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .message-bubble {
        max-width: 75%;
        padding: 1rem 1.25rem;
        border-radius: 1.25rem;
        line-height: 1.6;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    
    .user-bubble {
        background: var(--bubble-user-bg);
        color: white;
        border-bottom-right-radius: 0.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .ai-bubble {
        background: var(--bubble-ai-bg);
        color: var(--text-color);
        border: 1px solid var(--bubble-ai-border);
        border-bottom-left-radius: 0.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Typing cursor animation */
    .cursor {
        animation: blink 0.8s infinite;
        color: var(--accent-color);
        font-weight: bold;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* Citations */
    .chat-citation {
        background: var(--secondary-bg);
        border: 1px solid var(--bubble-ai-border);
        border-radius: 0.75rem;
        padding: 0.75rem;
        margin: 0.75rem 0;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    
    .chat-citation:hover {
        border-color: var(--accent-color);
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.1);
    }
    
    /* Single citation styling */
    .single-citation {
        border-left: 3px solid var(--accent-color);
        background: linear-gradient(135deg, var(--secondary-bg) 0%, rgba(139, 92, 246, 0.05) 100%);
    }
    
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid var(--bubble-ai-border);
    }
    
    .citation-score {
        background: var(--accent-color);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .citation-snippet {
        margin: 0.5rem 0;
        line-height: 1.4;
        color: var(--text-color);
        opacity: 0.8;
    }
    
    .citation-link {
        margin-top: 0.5rem;
    }
    
    /* Multiple citations styling */
    .citations-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 0.5rem;
        margin: 0.75rem 0;
    }
    
    .multi-citation {
        padding: 0.5rem;
        margin: 0;
        border-radius: 0.5rem;
        font-size: 0.8rem;
    }
    
    .citation-mini-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.3rem;
        font-size: 0.8rem;
    }
    
    .citation-mini-snippet {
        margin: 0.3rem 0;
        font-size: 0.75rem;
        opacity: 0.7;
        line-height: 1.3;
    }
    
    .citation-mini-link {
        margin-top: 0.3rem;
    }
    
    .chat-citation a {
        color: var(--accent-color);
        text-decoration: none;
        font-weight: 500;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        transition: all 0.2s ease;
    }
    
    .chat-citation a:hover {
        background: var(--accent-color);
        color: white;
        text-decoration: none;
    }
    
    /* Input area - Fixed at bottom */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--bg-color);
        border-top: 1px solid var(--bubble-ai-border);
        padding: 1rem;
        z-index: 99999;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .chat-input-wrapper {
        max-width: 900px;
        margin: 0 auto;
        display: flex;
        gap: 0.75rem;
        align-items: center;
    }
    
    /* Input field */
    .stTextInput input {
        border-radius: 0.75rem !important;
        border: 1px solid var(--bubble-ai-border) !important;
        padding: 0.75rem !important;
        background: var(--bubble-ai-bg) !important;
        color: var(--text-color) !important;
        font-size: 0.95rem !important;
    }
    
    /* Metadata badges */
    .metadata-badges {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
        flex-wrap: wrap;
    }
    
    .badge {
        background: var(--secondary-bg);
        color: var(--text-color);
        padding: 0.3rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .badge-success {
        background: #dcfce7;
        color: #166534;
    }
    
    .badge-info {
        background: #dbeafe;
        color: #1e40af;
    }
    
    /* Scrollable chat area */
    .main .block-container { padding-bottom: 12rem; }
    
    /* Button styling */
    .stButton button {
        border-radius: 0.75rem !important;
        background: var(--accent-color) !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button:hover {
        opacity: 0.9 !important;
    }
    
    /* Sidebar */
    .stSidebar {
        background: var(--bg-color);
        border-right: 1px solid var(--bubble-ai-border);
    }
</style>
""", unsafe_allow_html=True)

# Theme toggle

def set_theme(theme):
    st.session_state.theme = theme
    st.markdown(f"""
    <script>
        document.documentElement.setAttribute('data-theme', '{theme}');
    </script>
    """, unsafe_allow_html=True)

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Ensure theme is applied on every rerun
set_theme(st.session_state.theme)

with st.sidebar:
    st.markdown("### 💬 Chat Sessions")
    
    # Chat session management
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_chat = st.selectbox(
            "Active Chat",
            list(st.session_state.chat_sessions.keys()),
            index=list(st.session_state.chat_sessions.keys()).index(st.session_state.active_chat)
        )
    with col2:
        if st.button("➕", help="New Chat"):
            st.session_state.chat_counter += 1
            new_chat_name = f"Chat {st.session_state.chat_counter}"
            st.session_state.chat_sessions[new_chat_name] = []
            st.session_state.active_chat = new_chat_name
            st.rerun()
    
    # Update active chat
    if selected_chat != st.session_state.active_chat:
        st.session_state.active_chat = selected_chat
        st.rerun()
    
    # Chat management buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Delete", use_container_width=True):
            if len(st.session_state.chat_sessions) > 1:
                del st.session_state.chat_sessions[st.session_state.active_chat]
                st.session_state.active_chat = list(st.session_state.chat_sessions.keys())[0]
                st.rerun()
    with col2:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.chat_sessions[st.session_state.active_chat] = []
            st.rerun()
    
    st.markdown("---")
    
    theme = st.selectbox(
        "Theme",
        ["light", "dark"],
        index=0 if st.session_state.theme == "light" else 1
    )
    if theme != st.session_state.theme:
        set_theme(theme)

# API config

API_URL = "http://127.0.0.1:8000"

def call_api(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Call the RAG API with error handling and retry."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{API_URL}{endpoint}", json=data, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status {response.status_code}: {response.text}"}
        except requests.exceptions.ConnectionError:
            if attempt == max_retries - 1:
                return {"error": f"API Connection Error: Cannot connect to {API_URL}. Please ensure the API server is running."}
            time.sleep(1)
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                return {"error": f"API Timeout Error: Request took too long. Please try again."}
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            return {"error": f"API Error: {str(e)}"}
    return {"error": "API Error: Max retries exceeded"}

def stream_text(text: str, container, delay: float = 0.02):
    """Simulate live text streaming by displaying text word by word."""
    current_text = ""
    words = text.split()
    
    for i, word in enumerate(words):
        current_text += word + " "
        container.markdown(f"""
        <div class="message-container">
            <div class="avatar ai-avatar">AI</div>
            <div class="message-bubble ai-bubble">
                {current_text}<span class="cursor">|</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(delay)
    
    container.markdown(f"""
    <div class="message-container">
        <div class="avatar ai-avatar">AI</div>
        <div class="message-bubble ai-bubble">
            {current_text.strip()}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return current_text.strip()

# Session state init

# Legacy support - migrate old messages to chat sessions
if "messages" in st.session_state and st.session_state.messages:
    if not st.session_state.chat_sessions["Chat 1"]:
        st.session_state.chat_sessions["Chat 1"] = st.session_state.messages
    del st.session_state.messages

if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "hybrid"

# Header

st.markdown("""
<div class="chat-container">
    <div class="chat-header">
        <h1 class="chat-title">RAG ASSISTANT</h1>
        <p class="chat-subtitle">Ask questions about your documents and get intelligent answers powered by RAG</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar settings

with st.sidebar:
    st.markdown("### Settings")
    retrieval_mode = st.selectbox(
        "Search Mode",
        ["hybrid", "elser", "bm25"],
        index=0 if st.session_state.retrieval_mode == "hybrid" else 1 if st.session_state.retrieval_mode == "elser" else 2
    )
    st.session_state.retrieval_mode = retrieval_mode
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**System Status**")
    try:
        health = requests.get(f"{API_URL}/healthz", timeout=5).json()
        if health.get("status") == "ok":
            st.success("✅ System Online")
            st.info(f"📚 Index: search-llm-rag_with_elser")
            st.info(f"🔗 API: {API_URL}")
        else:
            st.warning("⚠️ System Partially Online")
    except requests.exceptions.ConnectionError:
        st.error("❌ System Offline - API not reachable")
        st.info(f"🔗 Trying to connect to: {API_URL}")
    except requests.exceptions.Timeout:
        st.error("⏱️ System Offline - Connection timeout")
    except Exception as e:
        st.error(f"❌ System Offline - Error: {str(e)}")

# Chat display

# Get current chat messages
current_messages = st.session_state.chat_sessions.get(st.session_state.active_chat, [])

for message in current_messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="message-container user-message">
            <div class="avatar user-avatar">U</div>
            <div class="message-bubble user-bubble">
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class="message-container">
            <div class="avatar ai-avatar">AI</div>
            <div class="message-bubble ai-bubble">
                {message["content"]}
        """, unsafe_allow_html=True)
        
        if "metadata" in message:
            metadata = message["metadata"]
            st.markdown(f"""
                <div class="metadata-badges">
                    <span class="badge badge-info">{metadata.get('mode', 'HYBRID').upper()}</span>
                    <span class="badge">{metadata.get('latency_ms', 0)}ms</span>
                    <span class="badge badge-success">Score: {metadata.get('grounding_score', 0):.2f}</span>
                </div>
            """, unsafe_allow_html=True)
        
        if "citations" in message and message["citations"]:
            # Deduplicate citations by link
            unique_citations = []
            seen_links = set()
            
            for citation in message["citations"][:5]:  # Check more citations
                link = citation.get('link', '')
                if link and link not in seen_links:
                    seen_links.add(link)
                    unique_citations.append(citation)
                elif not link:  # Keep citations without links
                    unique_citations.append(citation)
            
            # Display citations with enhanced styling
            if len(unique_citations) == 1:
                # Single citation - enhanced display
                citation = unique_citations[0]
                st.markdown(f"""
                <div class="chat-citation single-citation">
                    <div class="citation-header">
                        <strong>📄 {citation.get('title', 'Document')}</strong>
                        <span class="citation-score">Score: {citation.get('score', 0):.2f}</span>
                    </div>
                    <div class="citation-snippet">
                        {citation.get('snippet', '')[:200]}...
                    </div>
                    <div class="citation-link">
                        <a href="{citation.get('link', '#')}" target="_blank">🔗 View Source Document</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Multiple citations - compact display
                st.markdown('<div class="citations-container">', unsafe_allow_html=True)
                for i, citation in enumerate(unique_citations[:3]):
                    st.markdown(f"""
                    <div class="chat-citation multi-citation">
                        <div class="citation-mini-header">
                            <strong>📄 {citation.get('title', 'Document')}</strong>
                            <span class="citation-score">Score: {citation.get('score', 0):.2f}</span>
                        </div>
                        <div class="citation-mini-snippet">
                            {citation.get('snippet', '')[:120]}...
                        </div>
                        <div class="citation-mini-link">
                            <a href="{citation.get('link', '#')}" target="_blank">🔗 Source {i+1}</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Fixed input at bottom

# Create a container for the input that's always at the bottom
analysis_placeholder = st.empty()
input_container = st.container()

with input_container:
    st.markdown("""
    <div class="chat-input-container">
        <div class="chat-input-wrapper">
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([8, 1])

    with col1:
        # Clear the input on the next render if flagged
        if st.session_state.get("clear_chat_input", False):
            st.session_state.clear_chat_input = False
            st.session_state.pop(f"chat_input_{st.session_state.active_chat}", None)

        # Use unique key for each chat session to prevent auto-fill
        user_input = st.text_input(
            "Message",
            placeholder="Ask about algorithms, design patterns, business concepts...",
            key=f"chat_input_{st.session_state.active_chat}",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True, key=f"send_{st.session_state.active_chat}")

    st.markdown("</div></div>", unsafe_allow_html=True)

# Quick suggestions (empty state)

if not current_messages:
    st.markdown("### Try asking:")
    
    suggestions = [
        "What is the main purpose of ISO 27001?",
        "Explain continual improvement in information security",
        "What are the responsibilities of top management?",
        "How should organizations assess information security risks?"
    ]
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}_{st.session_state.active_chat}", use_container_width=True):
                st.session_state.chat_sessions[st.session_state.active_chat].append({
                    "role": "user",
                    "content": suggestion
                })
                st.session_state.clear_chat_input = True
                
                response_placeholder = analysis_placeholder
                
                with response_placeholder.container():
                    st.markdown("""
                    <div class="message-container">
                        <div class="avatar ai-avatar">AI</div>
                        <div class="message-bubble ai-bubble">
                            <span class="cursor">|</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                response = call_api("/query", {
                    "question": suggestion,
                    "mode": st.session_state.retrieval_mode,
                    "top_k": 5
                })
                
                if "error" not in response:
                    answer_text = response.get("answer", "I don't know.")
                    
                    with response_placeholder.container():
                        stream_text(answer_text, response_placeholder, delay=0.03)
                    
                    st.session_state.chat_sessions[st.session_state.active_chat].append({
                        "role": "assistant",
                        "content": answer_text,
                        "metadata": {
                            "mode": response.get("used_mode", "hybrid"),
                            "latency_ms": response.get("latency_ms", 0),
                            "grounding_score": response.get("guardrails", {}).get("grounding_score", 0)
                        },
                        "citations": response.get("citations", [])
                    })
                else:
                    error_text = f"Sorry, I encountered an error: {response['error']}"
                    with response_placeholder.container():
                        stream_text(error_text, response_placeholder, delay=0.03)
                    
                    st.session_state.chat_sessions[st.session_state.active_chat].append({
                        "role": "assistant",
                        "content": error_text
                    })
                
                st.rerun()

# Input handler

if send_button and user_input.strip():
    # Add message to current chat session
    st.session_state.chat_sessions[st.session_state.active_chat].append({
        "role": "user",
        "content": user_input.strip()
    })
    
    # Flag to clear the input on the next render cycle
    st.session_state.clear_chat_input = True
    
    response_placeholder = analysis_placeholder
    
    with response_placeholder.container():
        st.markdown("""
        <div class="message-container">
            <div class="avatar ai-avatar">AI</div>
            <div class="message-bubble ai-bubble">
                <span class="cursor">|</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    response = call_api("/query", {
        "question": user_input.strip(),
        "mode": st.session_state.retrieval_mode,
        "top_k": 5
    })
    
    if "error" not in response:
        answer_text = response.get("answer", "I don't know.")
        
        with response_placeholder.container():
            stream_text(answer_text, response_placeholder, delay=0.03)
        
        st.session_state.chat_sessions[st.session_state.active_chat].append({
            "role": "assistant",
            "content": answer_text,
            "metadata": {
                "mode": response.get("used_mode", "hybrid"),
                "latency_ms": response.get("latency_ms", 0),
                "grounding_score": response.get("guardrails", {}).get("grounding_score", 0)
            },
            "citations": response.get("citations", [])
        })
    else:
        error_text = f"Sorry, I encountered an error: {response['error']}"
        with response_placeholder.container():
            stream_text(error_text, response_placeholder, delay=0.03)
        
        st.session_state.chat_sessions[st.session_state.active_chat].append({
            "role": "assistant",
            "content": error_text
        })
    
    st.rerun()

# Auto-scroll to bottom

if current_messages:
    st.markdown("""
    <script>
        // Scroll to show latest messages but keep input visible
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight - chatContainer.clientHeight;
        }
    </script>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)