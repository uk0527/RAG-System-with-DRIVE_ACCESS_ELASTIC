"""
Ingestion pipeline for the RAG system.
Handles Google Drive to PDF text extraction to chunking to metadata enrichment.
"""

import os
import json
import io
import re
import requests
import tempfile
from typing import List, Dict, Any, Iterable, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from langchain_community.document_loaders import GoogleDriveLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from langchain.schema import Document
import tiktoken
from google.oauth2.service_account import Credentials as SACredentials
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Advanced PDF extraction libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pdfminer
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

# Modern OCR libraries (superior to Tesseract)
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Image processing for modern OCR
try:
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False

# Layout analysis for complex documents
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

from app.core.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_CONTEXT_TOKENS,
    GOOGLE_SERVICE_ACCOUNT_PATH, DRIVE_FOLDER_ID,
    INDEX_NAME, ELSER_MODEL_ID,
    PADDLEOCR_LANG, EASYOCR_LANG, OCR_QUALITY_THRESHOLD,
    PDF_DPI, EXTRACT_IMAGES, EXTRACT_TABLES
)
# NOTE: Avoid importing heavy embedders at module import time to reduce dependencies.

# =========================
# Advanced PDF Text Extraction
# =========================

def _extract_text_pypdf(pdf_path: str) -> Tuple[str, float]:
    """Extract text using pypdf (default method)."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
        quality = _assess_text_quality(text)
        return text, quality
    except Exception as e:
        return "", 0.0

def _extract_text_pymupdf(pdf_path: str) -> Tuple[str, float]:
    """Extract text using PyMuPDF (often better than pypdf)."""
    if not PYMUPDF_AVAILABLE:
        return "", 0.0
    
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            text_parts.append(text)
        
        doc.close()
        text = "\n".join(text_parts)
        quality = _assess_text_quality(text)
        return text, quality
    except Exception as e:
        return "", 0.0

def _extract_text_pdfplumber(pdf_path: str) -> Tuple[str, float]:
    """Extract text using pdfplumber (excellent for complex layouts)."""
    if not PDFPLUMBER_AVAILABLE:
        return "", 0.0
    
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        
        text = "\n".join(text_parts)
        quality = _assess_text_quality(text)
        return text, quality
    except Exception as e:
        return "", 0.0

def _extract_text_pdfminer(pdf_path: str) -> Tuple[str, float]:
    """Extract text using pdfminer (better fallback)."""
    if not PDFMINER_AVAILABLE:
        return "", 0.0
    
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        quality = _assess_text_quality(text)
        return text, quality
    except Exception as e:
        return "", 0.0

def _assess_text_quality(text: str) -> float:
    """Assess the quality of extracted text (0.0 to 1.0)."""
    if not text or len(text.strip()) < 10:
        return 0.0
    
    # Quality indicators
    quality_score = 0.0
    
    # Length check (longer text is usually better)
    if len(text) > 100:
        quality_score += 0.2
    if len(text) > 500:
        quality_score += 0.2
    
    # Character diversity (more diverse = better)
    unique_chars = len(set(text.lower()))
    if unique_chars > 20:
        quality_score += 0.2
    
    # Word count (more words = better)
    word_count = len(text.split())
    if word_count > 50:
        quality_score += 0.2
    
    # Check for common OCR artifacts (fewer = better)
    ocr_artifacts = text.count('|') + text.count('||') + text.count('|||')
    if ocr_artifacts < 10:
        quality_score += 0.1
    
    # Check for proper sentence structure
    sentences = text.count('.') + text.count('!') + text.count('?')
    if sentences > 5:
        quality_score += 0.1
    
    return min(quality_score, 1.0)



def _extract_text_unstructured(pdf_path: str) -> Tuple[str, float]:
    """Extract text using Unstructured (Docling alternative) for complex layouts."""
    if not UNSTRUCTURED_AVAILABLE:
        return "", 0.0
    
    try:
        # Partition PDF with layout analysis
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # High resolution for better accuracy
            infer_table_structure=EXTRACT_TABLES,
            extract_images_in_pdf=EXTRACT_IMAGES,
            extract_image_block_types=["Image", "Table"]
        )
        
        # Combine all text elements
        text_parts = []
        for element in elements:
            if hasattr(element, 'text') and element.text:
                text_parts.append(element.text)
        
        text = "\n".join(text_parts)
        quality = _assess_text_quality(text)
        return text, quality
        
    except Exception as e:
        print(f"❌ Unstructured extraction failed: {str(e)}")
        return "", 0.0

def _extract_text_paddleocr(pdf_path: str) -> Tuple[str, float]:
    """Extract text using PaddleOCR (superior to Tesseract)."""
    if not PADDLEOCR_AVAILABLE:
        return "", 0.0
    
    try:
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang=PADDLEOCR_LANG)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=PDF_DPI)
        
        all_text = []
        for i, image in enumerate(images):
            print(f"🔍 PaddleOCR processing page {i+1}/{len(images)}...")
            
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Run OCR
            result = ocr.ocr(img_array, cls=True)
            
            # Extract text from results
            page_text = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]  # Extract text
                        confidence = line[1][1]  # Extract confidence
                        if confidence > 0.5:  # Only keep high-confidence text
                            page_text.append(text)
            
            if page_text:
                all_text.append(" ".join(page_text))
        
        text = "\n\n".join(all_text)
        quality = _assess_text_quality(text)
        return text, quality
        
    except Exception as e:
        print(f"❌ PaddleOCR failed: {str(e)}")
        return "", 0.0

def _extract_text_easyocr(pdf_path: str) -> Tuple[str, float]:
    """Extract text using EasyOCR (alternative to Tesseract)."""
    if not EASYOCR_AVAILABLE:
        return "", 0.0
    
    try:
        # Initialize EasyOCR
        reader = easyocr.Reader([EASYOCR_LANG])
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=PDF_DPI)
        
        all_text = []
        for i, image in enumerate(images):
            print(f"🔍 EasyOCR processing page {i+1}/{len(images)}...")
            
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Run OCR
            results = reader.readtext(img_array)
            
            # Extract text from results
            page_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Only keep high-confidence text
                    page_text.append(text)
            
            if page_text:
                all_text.append(" ".join(page_text))
        
        text = "\n\n".join(all_text)
        quality = _assess_text_quality(text)
        return text, quality
        
    except Exception as e:
        print(f"❌ EasyOCR failed: {str(e)}")
        return "", 0.0

def _extract_text_hybrid(pdf_path: str) -> str:
    """Extract text using modern hybrid approach with superior tools."""
    # Modern extraction methods (in order of preference)
    extraction_methods = [
        ("PyMuPDF", _extract_text_pymupdf),           # Best for normal PDFs
        ("Unstructured", _extract_text_unstructured), # Best for complex layouts
        ("pdfplumber", _extract_text_pdfplumber),     # Good for tables
        ("pypdf", _extract_text_pypdf),               # Standard fallback
        ("pdfminer", _extract_text_pdfminer),         # Last resort
    ]
    
    results = []
    
    # Try modern text extraction methods first
    for method_name, method_func in extraction_methods:
        try:
            text, quality = method_func(pdf_path)
            if text and quality > 0.1:  # Minimum quality threshold
                results.append((method_name, text, quality))
                print(f"✅ {method_name}: Quality {quality:.2f}, Length {len(text)}")
            else:
                print(f"❌ {method_name}: Failed or low quality")
        except Exception as e:
            print(f"❌ {method_name}: Error - {str(e)}")
    
    # If no good results from text extraction, try modern OCR
    if not results or max(results, key=lambda x: x[2])[2] < OCR_QUALITY_THRESHOLD:
        print("🔄 Text extraction methods failed or low quality, trying modern OCR...")
        
        # Try modern OCR methods (in order of preference)
        ocr_methods = [
            ("PaddleOCR", _extract_text_paddleocr),   # Superior OCR
            ("EasyOCR", _extract_text_easyocr),       # Alternative OCR
        ]
        
        for ocr_name, ocr_func in ocr_methods:
            try:
                ocr_text, ocr_quality = ocr_func(pdf_path)
                if ocr_text and ocr_quality > 0.1:
                    results.append((ocr_name, ocr_text, ocr_quality))
                    print(f"✅ {ocr_name}: Quality {ocr_quality:.2f}, Length {len(ocr_text)}")
                    break  # Use first successful OCR method
                else:
                    print(f"❌ {ocr_name}: Failed or low quality")
            except Exception as e:
                print(f"❌ {ocr_name}: Error - {str(e)}")
    
    if not results:
        print("⚠️ All extraction methods (including modern OCR) failed!")
        return ""
    
    # Sort by quality and return the best result
    best_method, best_text, best_quality = max(results, key=lambda x: x[2])
    print(f"🏆 Best method: {best_method} (Quality: {best_quality:.2f})")
    
    return best_text

# =========================
# Text Processing Utilities
# =========================

def _clean_text(text: str) -> str:
    """Clean extracted text from PDFs with enhanced formatting cleanup."""
    if not text:
        return ""
    
    # Remove soft hyphens and special Unicode characters
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\u2014", "-")  # em dash
    text = text.replace("\u2013", "-")  # en dash
    text = text.replace("\u2019", "'")  # right single quotation
    text = text.replace("\u201c", '"')  # left double quotation
    text = text.replace("\u201d", '"')  # right double quotation
    
    # Remove common PDF artifacts and formatting noise
    text = re.sub(r"[`]{2,}", "", text)  # Remove multiple backticks
    text = re.sub(r"[-]{3,}", "", text)  # Remove long dashes completely
    text = re.sub(r"[,]{3,}", "", text)  # Remove multiple commas
    text = re.sub(r"[.]{3,}", "...", text)  # Normalize ellipsis
    
    # Remove page markers and headers/footers
    text = re.sub(r"--- Page \d+ ---", "", text)  # Remove page markers
    text = re.sub(r"Page\d+\|\d+", "", text)      # Remove page references
    text = re.sub(r"\d+\s*--`,`,.*?---", "", text)  # Remove complex artifacts
    text = re.sub(r"DZONE\.COM.*?REFCARDZ", "", text, flags=re.IGNORECASE)
    text = re.sub(r"© DZONE.*?\d{4}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"BROUGHT TO YOU.*?WITH", "", text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces/tabs to single space
    text = re.sub(r"\n{3,}", "\n\n", text)  # Multiple newlines to double
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)  # Trim lines
    
    # Remove standalone special characters on their own lines
    text = re.sub(r"\n[`,-]{1,10}\n", "\n", text)
    
    return text.strip()

def _count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        # Use cl100k_base encoding (compatible with most models)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimation (4 characters per token)
        return len(text) // 4

def _enrich_metadata(doc: Document, file_info: Dict[str, Any]) -> Document:
    """Enrich document metadata with file information."""
    doc.metadata.update({
        "filename": file_info.get("name", "unknown"),
        "drive_url": file_info.get("webViewLink", ""),
        "file_id": file_info.get("id", ""),
        "modified_time": file_info.get("modifiedTime", ""),
        "ingestion_time": datetime.now(timezone.utc).isoformat(),
    })
    return doc

# =========================
# Chunking with Token Awareness
# =========================

def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter with token-aware settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=_count_tokens,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        is_separator_regex=False,
    )

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks with improved strategy for better content coherence.
    
    Args:
        documents: List of documents to chunk
        
    Returns:
        List of chunked documents with enhanced quality filtering
    """
    # Enhanced text splitter with better separators for meaningful chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n\n",      # Triple newlines (major sections)
            "\n\n",        # Double newlines (paragraphs)
            "\n",          # Single newlines
            ". ",          # Sentences
            "? ",          # Questions  
            "! ",          # Exclamations
            "; ",          # Semi-colons
            ", ",          # Commas (last resort)
            " ",           # Spaces (very last resort)
        ]
    )
    
    chunks = []
    
    for doc in documents:
        # Clean text before chunking for better quality
        cleaned_content = _clean_text_for_chunking(doc.page_content)
        
        # Skip if content is too poor quality
        if not _is_meaningful_content(cleaned_content):
            continue
        
        # Create a cleaned document
        cleaned_doc = Document(page_content=cleaned_content, metadata=doc.metadata)
        
        # Split the cleaned document
        doc_chunks = text_splitter.split_documents([cleaned_doc])
        
        # Filter and enhance chunks
        for i, chunk in enumerate(doc_chunks):
            # Post-process chunk content
            processed_content = _post_process_chunk(chunk.page_content)
            
            # Only keep meaningful chunks
            if _is_meaningful_content(processed_content):
                chunk.page_content = processed_content
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata.get('file_id', 'unknown')}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks),
                })
                chunks.append(chunk)
    
    return chunks

def _clean_text_for_chunking(text: str) -> str:
    """Enhanced text cleaning specifically for better chunking."""
    if not text:
        return ""
    
    # Apply existing cleaning first
    text = _clean_text(text)
    
    # Additional cleaning for chunking
    # Fix OCR spacing issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add spaces between joined words
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)     # Add spaces between letters and numbers
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)     # Add spaces between numbers and letters
    
    # Fix sentence boundaries
    text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)  # Add space after periods
    text = re.sub(r'([a-z])\?([A-Z])', r'\1? \2', text)  # Add space after questions
    text = re.sub(r'([a-z])!([A-Z])', r'\1! \2', text)   # Add space after exclamations
    
    return text.strip()

def _post_process_chunk(chunk: str) -> str:
    """Post-process individual chunks for better quality."""
    
    # Remove leading/trailing artifacts
    chunk = re.sub(r'^[^\w]*', '', chunk)  # Remove leading non-word chars
    chunk = re.sub(r'[^\w.!?]*$', '', chunk)  # Remove trailing non-word chars
    
    # Ensure proper capitalization at start
    if chunk and chunk[0].islower():
        chunk = chunk[0].upper() + chunk[1:]
    
    # Remove standalone artifact lines
    lines = chunk.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip lines that are mostly artifacts
        if re.match(r'^[\d\s\-`,]+$', line) and len(line) < 20:
            continue
        if len(line) > 3:  # Keep substantial lines
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

def _is_meaningful_content(content: str) -> bool:
    """Check if content contains meaningful information."""
    if len(content.strip()) < 50:
        return False
    
    words = content.split()
    if len(words) < 10:
        return False
    
    # Check artifact ratio
    artifacts = sum(1 for word in words if any(char in word for char in ['`', '---', ',,,']))
    if artifacts / len(words) > 0.3:  # More than 30% artifacts
        return False
    
    # Check for meaningful content indicators
    has_verbs = any(word.lower() in ['is', 'are', 'was', 'were', 'can', 'will', 'should', 'must', 'provides', 'ensures', 'includes', 'allows', 'enables', 'requires'] for word in words)
    has_nouns = any(len(word) > 4 and word.isalpha() for word in words)
    
    return has_verbs and has_nouns

# =========================
# Google Drive Integration
# =========================

def load_from_google_drive(
    folder_id: str,
    service_account_path: str = GOOGLE_SERVICE_ACCOUNT_PATH,
    max_files: int = 1000,
    try_public_first: bool = True
) -> List[Document]:
    """
    Load PDF documents from Google Drive folder.
    Supports both public and private folder access.
    
    Args:
        folder_id: Google Drive folder ID
        service_account_path: Path to service account JSON file
        max_files: Maximum number of files to process
        try_public_first: Try public access first before service account
        
    Returns:
        List of loaded documents
    """
    # Prefer OAuth (user Gmail) if credentials.json exists
    oauth_client_path = os.path.join(os.getcwd(), "credentials.json")
    oauth_token_path = os.path.join(os.getcwd(), "token.json")
    if os.path.exists(oauth_client_path):
        try:
            print("🔑 Using OAuth (user Gmail) credentials...")
            service = _build_drive_service_oauth(oauth_client_path, oauth_token_path)
            documents = _load_from_drive_with_service(service, folder_id, max_files)
            if documents:
                print(f"✅ Loaded {len(documents)} documents via OAuth")
                return documents
        except Exception as e:
            print(f"⚠️ OAuth access failed: {e}")
            print("🔄 Falling back to service account...")

    # Optionally try public access first if enabled
    if try_public_first:
        try:
            print("🌐 Trying public folder access...")
            documents = _load_from_public_drive(folder_id, max_files)
            if documents:
                print(f"✅ Successfully loaded {len(documents)} documents from public folder")
                return documents
        except Exception as e:
            print(f"⚠️ Public access failed: {e}")
            print("🔄 Falling back to service account authentication...")
    
    # Fallback to service account authentication (supports shared drives)
    if not os.path.exists(service_account_path):
        raise FileNotFoundError(
            f"Service account file not found: {service_account_path}\n"
            f"For PUBLIC folders, the public access attempt failed.\n"
            f"For PRIVATE folders, create a service account JSON file."
        )

    print("🔐 Using service account authentication...")

    # Build Drive service with service account and proceed
    creds = SACredentials.from_service_account_file(
        service_account_path,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=creds, cache_discovery=False)
    return _load_from_drive_with_service(service, folder_id, max_files)


def _build_drive_service_oauth(client_secrets_path: str, token_path: str):
    """Build Google Drive API service using OAuth user credentials.
    Creates/refreshes token.json as needed.
    """
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None
    if os.path.exists(token_path):
        try:
            creds = UserCredentials.from_authorized_user_file(token_path, scopes)
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(GoogleRequest())
            except Exception:
                creds = None
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, scopes)
            # This will open a browser; run once to create token.json
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token_file:
            token_file.write(creds.to_json())
    return build('drive', 'v3', credentials=creds, cache_discovery=False)


def _load_from_drive_with_service(service, folder_id: str, max_files: int) -> List[Document]:
    """List, download, extract, and wrap docs using an authenticated Drive service (OAuth or SA).
    Optimized for large files and limited disk space.
    """
    # List PDF files (supports shared drives) - filter by size to avoid huge files
    files: List[Dict[str, Any]] = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            pageSize=min(max_files, 1000),
            pageToken=page_token,
            fields='files(id,name,size,modifiedTime,webViewLink),nextPageToken',
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora='allDrives'
        ).execute()
        files.extend(resp.get('files', []))
        page_token = resp.get('nextPageToken')
        if not page_token or len(files) >= max_files:
            break
    
    # Filter and sort files by size (process smaller files first)
    files = files[:max_files]
    files_with_size = []
    for f in files:
        try:
            size = int(f.get('size', 0))
            # Skip files larger than 50MB to avoid disk space issues
            if size > 50 * 1024 * 1024:
                print(f"⚠️ Skipping large file {f.get('name', 'unknown')} ({size/1024/1024:.1f}MB)")
                continue
            files_with_size.append((f, size))
        except (ValueError, TypeError):
            files_with_size.append((f, 0))
    
    # Sort by size (smallest first)
    files_with_size.sort(key=lambda x: x[1])
    files = [f[0] for f in files_with_size]
    
    if not files:
        return []

    processed_docs: List[Document] = []
    for f in files:
        try:
            print(f"📄 Processing {f.get('name', 'unknown')} ({int(f.get('size', 0))/1024:.1f}KB)")
            
            # Stream download to avoid memory issues
            request = service.files().get_media(fileId=f['id'], supportsAllDrives=True)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            pdf_bytes = fh.getvalue()

            # Use memory-efficient extraction (avoid temp files when possible)
            try:
                raw_text = _extract_text_from_pdf_bytes(pdf_bytes, f.get('name', ''))
                if not raw_text or len(raw_text.strip()) < 100:
                    # Only use hybrid extraction for difficult PDFs
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                        tmp.write(pdf_bytes)
                        tmp_path = tmp.name
                    try:
                        raw_text = _extract_text_hybrid(tmp_path)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
            except Exception as e:
                print(f"⚠️ Text extraction failed for {f.get('name', 'unknown')}: {e}")
                continue
            
            # Clear PDF bytes from memory immediately
            del pdf_bytes

            clean_text = _clean_text(raw_text)
            if not clean_text.strip():
                continue

            doc = Document(
                page_content=clean_text,
                metadata={
                    "name": f.get('name', ''),
                    "id": f.get('id', ''),
                    "webViewLink": f.get('webViewLink', ''),
                    "modifiedTime": f.get('modifiedTime', ''),
                    "size": f.get('size', 0)
                }
            )
            doc = _enrich_metadata(doc, {
                "name": f.get('name', ''),
                "id": f.get('id', ''),
                "webViewLink": f.get('webViewLink', ''),
                "modifiedTime": f.get('modifiedTime', ''),
            })
            processed_docs.append(doc)
            
        except Exception as e:
            print(f"⚠️ Error processing {f.get('name', 'unknown')}: {e}")
            continue

    return processed_docs


def _load_from_public_drive(folder_id: str, max_files: int = 1000) -> List[Document]:
    """
    Load PDF documents from a PUBLIC Google Drive folder.
    No authentication required.
    
    Args:
        folder_id: Google Drive folder ID
        max_files: Maximum number of files to process
        
    Returns:
        List of loaded documents
    """
    # List PDF files in the public folder
    files = _list_public_pdfs(folder_id, max_files)
    if not files:
        raise Exception("No PDF files found or folder not publicly accessible")
    
    processed_docs = []
    
    for file_info in files:
        try:
            # Download PDF content
            pdf_bytes = _download_public_pdf(file_info['id'])
            if not pdf_bytes:
                print(f"⚠️ Failed to download {file_info['name']}, skipping")
                continue
            
            # Extract text from PDF
            raw_text = _extract_text_from_pdf_bytes(pdf_bytes, file_info['name'])
            clean_text = _clean_text(raw_text)
            
            if not clean_text.strip():
                print(f"⚠️ No text extracted from {file_info['name']}, skipping")
                continue
            
            # Create document with metadata
            doc = Document(
                page_content=clean_text,
                metadata={
                    "source": file_info['name'],
                    "file_id": file_info['id'],
                    "mime_type": "application/pdf",
                    "modified_time": file_info.get('modifiedTime', ''),
                    "size": file_info.get('size', 0),
                    "web_view_link": file_info.get('webViewLink', ''),
                    "access_method": "public"
                }
            )
            
            # Enrich metadata
            doc = _enrich_metadata(doc, file_info)
            processed_docs.append(doc)
            
        except Exception as e:
            print(f"⚠️ Error processing {file_info['name']}: {e}")
            continue
    
    return processed_docs


def _list_public_pdfs(folder_id: str, max_files: int = 100) -> List[Dict]:
    """List PDF files in a PUBLIC Google Drive folder using the Drive API v3."""
    try:
        url = "https://www.googleapis.com/drive/v3/files"
        params = {
            'q': f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            'pageSize': min(max_files, 1000),
            'fields': 'files(id,name,size,modifiedTime,webViewLink)',
            'orderBy': 'name'
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            files = data.get('files', [])
            print(f"📁 Found {len(files)} PDF files in public folder")
            return files
        elif response.status_code == 403:
            raise Exception("Folder is not publicly accessible")
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        raise Exception(f"Network error: {e}")


def _download_public_pdf(file_id: str) -> Optional[bytes]:
    """Download a PDF file from Google Drive using the public download link."""
    try:
        download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        response = requests.get(download_url, timeout=60)
        
        if response.status_code == 200 and response.content.startswith(b'%PDF'):
            return response.content
        else:
            return None
            
    except requests.RequestException:
        return None


def _extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str = "") -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
            except Exception:
                continue
        
        return "\n\n".join(text_parts)
        
    except Exception as e:
        print(f"⚠️ PDF extraction failed for {filename}: {e}")
        return ""


# =========================
# Elasticsearch Storage
# =========================

def create_elasticsearch_store(es_client, index_name: str = INDEX_NAME) -> ElasticsearchStore:
    """
    Create an ElasticsearchStore instance for document storage.
    Lazily import the embedder to avoid heavy dependencies during module import.
    """
    from app.core.embed import get_langchain_embedder  # local import
    embedder = get_langchain_embedder()
    return ElasticsearchStore(
        es_connection=es_client,
        index_name=index_name,
        embedding=embedder,
    )


def _manual_index_documents(
    es_client,
    index_name: str,
    documents: List[Document],
    pipeline: str = "elser_v2_pipeline",
) -> int:
    """Index documents without computing dense vectors (ELSER-only).

    Args:
        es_client: Elasticsearch client
        index_name: Target index name
        documents: Chunked `Document` objects
        pipeline: Ingest pipeline name for ELSER

    Returns:
        Number of documents indexed
    """
    from elasticsearch.helpers import bulk  # local import to avoid overhead

    ops: List[Dict[str, Any]] = []
    for doc in documents:
        source = {
            "content": doc.page_content,
            "text": doc.page_content,  # Both fields for compatibility
            "metadata": dict(doc.metadata or {}),
        }
        
        # Add vector field if present in metadata
        if "vector" in doc.metadata:
            source["vector"] = doc.metadata["vector"]
        
        # Optional title for convenience
        if "filename" in source["metadata"]:
            source["title"] = source["metadata"].get("filename")

        op = {
            "_op_type": "index",
            "_index": index_name,
            "_source": source,
        }
        if pipeline:
            op["pipeline"] = pipeline
        ops.append(op)

    if not ops:
        return 0

    try:
        from elasticsearch.helpers import parallel_bulk
        success_count = 0
        
        # Use parallel_bulk for better error handling
        for success, info in parallel_bulk(
            es_client, 
            ops, 
            chunk_size=50,  # Even smaller chunks
            max_chunk_bytes=10485760,  # 10MB max per chunk
            request_timeout=300,
            refresh="wait_for"
        ):
            if success:
                success_count += 1
            else:
                print(f"❌ Indexing error: {info}")
        
        print(f"✅ Successfully indexed {success_count}/{len(ops)} documents")
        return success_count
        
    except Exception as e:
        print(f"❌ Bulk indexing failed: {e}")
        # Try without pipeline as fallback
        print("🔄 Trying without ELSER pipeline...")
        try:
            # Remove pipeline from operations
            ops_no_pipeline = []
            for op in ops:
                new_op = op.copy()
                if "pipeline" in new_op:
                    del new_op["pipeline"]
                ops_no_pipeline.append(new_op)
            
            bulk(es_client, ops_no_pipeline, refresh="wait_for", chunk_size=100, request_timeout=300)
            print(f"✅ Indexed {len(ops_no_pipeline)} documents without ELSER")
            return len(ops_no_pipeline)
        except Exception as e2:
            print(f"❌ Even basic indexing failed: {e2}")
            return 0

def create_elasticsearch_mapping() -> Dict[str, Any]:
    """
    Create the Elasticsearch index mapping for the RAG system.
    
    Returns:
        Index mapping configuration
    """
    return {
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "text": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "title": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "similarity": "cosine"
                },
                "ml": {
                    "type": "object",
                    "properties": {
                        "tokens": {
                            "type": "rank_features"
                        },
                        "model_id": {
                            "type": "keyword"
                        },
                        "inference_model": {
                            "type": "keyword"
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "dynamic": "false",
                    "properties": {
                        "filename": {"type": "keyword"},
                        "drive_url": {"type": "keyword"},
                        "file_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        "modified_time": {"type": "date"},
                        "ingestion_time": {"type": "date"},
                    }
                }
            }
        }
    }

def recreate_index_with_new_mapping(es_client, index_name: str) -> bool:
    """
    Recreate the Elasticsearch index with updated mapping to fix field limit issues.
    
    Args:
        es_client: Elasticsearch client
        index_name: Name of the index to recreate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete existing index if it exists
        if es_client.indices.exists(index=index_name):
            print(f"🗑️ Deleting existing index: {index_name}")
            es_client.indices.delete(index=index_name)
        
        # Create new index with proper mapping
        print(f"📝 Creating new index with strict mapping: {index_name}")
        mapping = create_elasticsearch_mapping()
        es_client.indices.create(index=index_name, body=mapping)
        print("✅ Index recreated successfully with field limit protection!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to recreate index: {e}")
        return False

def create_elser_pipeline() -> Dict[str, Any]:
    """
    Create the ELSER ingest pipeline for sparse retrieval.
    
    Returns:
        Pipeline configuration
    """
    return {
        "description": "ELSER v2 text expansion pipeline",
        "processors": [
            {
                "inference": {
                    "model_id": ELSER_MODEL_ID,
                    "target_field": "ml",
                    "field_map": {
                        "content": "text_field"
                    },
                    "inference_config": {
                        "text_expansion": {
                            "results_field": "tokens"
                        }
                    }
                }
            }
        ]
    }

# =========================
# Main Ingestion Pipeline
# =========================

def ingest_documents(
    es_client,
    folder_id: Optional[str] = None,
    index_name: str = INDEX_NAME,
    force: bool = False,
    max_files: int = 1000,
    batch_size: int = 64,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main ingestion pipeline: Google Drive → PDF → chunks → Elasticsearch.
    
    Args:
        es_client: Elasticsearch client instance
        folder_id: Google Drive folder ID (uses config default if None)
        index_name: Elasticsearch index name
        force: Force re-ingestion of all files
        max_files: Maximum number of files to process
        batch_size: Batch size for processing
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with ingestion results
    """
    if folder_id is None:
        folder_id = DRIVE_FOLDER_ID
    
    if not folder_id:
        raise ValueError("No Google Drive folder ID provided")
    
    start_time = datetime.now()
    
    # Log start
    if verbose:
        print(f"[INGEST] Starting ingestion from folder: {folder_id}")
    
    # Load documents from Google Drive
    if verbose:
        print("[INGEST] Loading documents from Google Drive...")
    
    documents = load_from_google_drive(
        folder_id=folder_id,
        max_files=max_files
    )
    
    if not documents:
        return {"files_seen": 0, "chunks_indexed": 0, "duration_seconds": 0}
    
    if verbose:
        print(f"[INGEST] Loaded {len(documents)} documents")
    
    # Chunk documents
    if verbose:
        print("[INGEST] Chunking documents...")
    
    chunks = chunk_documents(documents)
    
    if verbose:
        print(f"[INGEST] Created {len(chunks)} chunks")
    
    # Add dense vectors to chunks before indexing
    if verbose:
        print("[INGEST] Generating dense vectors...")
    
    from app.core.embed import get_embedder
    embedder = get_embedder()
    
    # Process chunks in batches to add vectors
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk.page_content for chunk in batch]
        vectors = embedder.encode(texts, show_progress_bar=False, batch_size=32).tolist()
        
        for j, vector in enumerate(vectors):
            batch[j].metadata["vector"] = vector
    
    if verbose:
        print(f"[INGEST] Added vectors to {len(chunks)} chunks")
    
    # Index documents first without ELSER, then apply ELSER via update
    if verbose:
        print("[INGEST] Indexing chunks...")
    total_indexed = _manual_index_documents(es_client, index_name, chunks, pipeline=None)
    
    if total_indexed > 0 and verbose:
        print(f"[INGEST] Applying ELSER processing to {total_indexed} documents...")
        try:
            # Apply ELSER via update by query
            update_body = {
                "script": {
                    "source": "ctx._source = ctx._source"  # No-op script to trigger pipeline
                },
                "query": {"match_all": {}}
            }
            result = es_client.update_by_query(
                index=index_name,
                body=update_body,
                pipeline="elser_v2_pipeline",
                refresh=True,
                timeout="10m"
            )
            updated = result.get("updated", 0)
            print(f"✅ Applied ELSER to {updated} documents")
        except Exception as e:
            print(f"⚠️ ELSER processing failed: {e}")
            print("📝 Documents indexed without ELSER - BM25 mode will work")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    if verbose:
        print(f"[INGEST] Completed: {total_indexed} chunks indexed in {duration:.2f}s")
    
    return {
        "files_seen": len(documents),
        "chunks_indexed": total_indexed,
        "duration_seconds": duration
    }
