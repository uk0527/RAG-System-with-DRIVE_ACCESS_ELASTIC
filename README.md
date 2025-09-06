# RAG System with Elasticsearch + LangChain + Hugging Face

A complete, end-to-end Retrieval-Augmented Generation (RAG) system that allows you to ask questions about PDFs stored in Google Drive and get clean, grounded answers with citations.

## Features

- **Hybrid Retrieval**: ELSER sparse + Dense vectors + BM25 with RRF fusion
- **Google Drive Integration**: Automatic PDF ingestion from shared folders
- **Hugging Face LLM**: Uses meta-llama/Meta-Llama-3-8B-Instruct via Inference API
- **Safety & Grounding**: Built-in guardrails for content safety and answer grounding
- **Modern UI**: Clean Streamlit interface with mode toggles and citations
- **FastAPI Backend**: RESTful API with comprehensive documentation
- **Elasticsearch 8.x**: ELSER v2 support

## Prerequisites

- **Python 3.10+**
- **Elasticsearch**: Elastic Cloud (recommended) or Docker Desktop for local ES/Kibana
- **Google Drive access**: Service Account JSON or OAuth (`credentials.json` + `token.json`)
- **Hugging Face API Key** (for LLM inference)
- Optional (OCR/scanned PDFs): **Poppler** (for `pdf2image`), **PaddleOCR/EasyOCR** (heavier; improves OCR)

## Installation

### 1. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# Elasticsearch Configuration
ELASTIC_URL=http://localhost:9200
ELASTIC_API_KEY_ID=your_api_key_id
ELASTIC_API_KEY=your_api_key
INDEX_NAME=rag_docs
ELSER_MODEL_ID=.elser_model_2_linux-x86_64

# Hugging Face Configuration
HF_API_KEY=your_huggingface_api_key
HF_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
HF_ENDPOINT_URL=https://api-inference.huggingface.co/models

# Google Drive Configuration
DRIVE_FOLDER_ID=your_google_drive_folder_id
GOOGLE_SERVICE_ACCOUNT_PATH=service_account.json

# Optional: OCR Configuration 
#eg.TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
#eg.POPPLER_PATH=C:\Program Files\poppler\bin

# Optional: API Configuration
API_HOST=127.0.0.1
API_PORT=8000
RAG_API_URL=http://127.0.0.1:8000
```

## Elasticsearch Setup

### Option 1: Local Docker (Recommended for Development)

```bash
# Start Elasticsearch and Kibana
docker-compose up -d

# Wait for services to be ready
curl http://localhost:9200
```

### Option 2: Elastic Cloud (Recommended for Production)

1. Go to [Elastic Cloud](https://cloud.elastic.co)
2. Create a new deployment
3. Copy the Elasticsearch endpoint and API key
4. Update your `.env` file with the credentials

## Google Drive Setup

1. **Create Google Cloud Service Account**:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select existing one
   - Enable Google Drive API
   - Create a service account
   - Download the JSON key file as `service_account.json`

2. **Share Drive Folder**:
   - Share your Google Drive folder with the service account email
   - Copy the folder ID from the URL
   - Add to `.env` as `DRIVE_FOLDER_ID`

## Running the System

### 1. Start the API Server

```bash
# Using uvicorn directly
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000

# Or using the main module
python -m app.api.main
```

- **API Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/healthz

### 2. Ingest Documents

The system will automatically create the Elasticsearch index and ELSER pipeline on first run. To ingest documents:

```bash
# Using the API
curl -X POST "http://127.0.0.1:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_id": "your_drive_folder_id",
    "force": false,
    "max_files": 100,
    "verbose": true
  }'
```

Or use the Swagger UI at http://127.0.0.1:8000/docs

### 3. Start the Web UI

```bash
streamlit run app/ui/chat_app.py
```

- **Web Interface**: http://localhost:8501

## API Endpoints

### Query Documents
```http
POST /query
Content-Type: application/json

{
  "question": "What is the company's remote work policy?",
  "mode": "hybrid",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The company allows remote work for up to 3 days per week with manager approval.",
  "citations": [
    {
      "id": 1,
      "title": "Employee Handbook 2024",
      "link": "https://drive.google.com/file/123/view",
      "snippet": "Remote work policy: Employees may work remotely...",
      "score": 0.95
    }
  ],
  "used_mode": "hybrid",
  "latency_ms": 1250,
  "guardrails": {
    "safe": true,
    "grounded": true,
    "quality_score": 0.9,
    "grounding_score": 0.85,
    "notes": []
  }
}
```

### Ingest Documents
```http
POST /ingest
Content-Type: application/json

{
  "folder_id": "your_drive_folder_id",
  "force": false,
  "max_files": 100,
  "batch_size": 64,
  "verbose": true
}
```

### Health Check
```http
GET /healthz
```

## Testing

```bash
# Run all tests
pytest -v

# Run specific test modules
pytest -v tests/test_ingest.py
pytest -v tests/test_retrieve.py
pytest -v tests/test_generate.py

# Run with coverage
pytest --cov=app --cov-report=html
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Google Drive  │───▶│   Ingestion      │───▶│  Elasticsearch  │
│   (PDFs)        │    │   Pipeline       │    │  (Index + ELSER)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◀───│   FastAPI        │◀───│   Retrieval     │
│   (Frontend)    │    │   (Backend)      │    │   (Hybrid)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Generation     │◀───│   Guardrails    │
                       │   (Hugging Face) │    │   (Safety)      │
                       └──────────────────┘    └─────────────────┘
```

## Retrieval Modes

### ELSER-Only Mode
- Uses Elasticsearch's ELSER v2 for sparse retrieval
- Good for keyword-based queries
- Faster but may miss semantic matches

### Hybrid Mode (Default)
- Combines ELSER + Dense vectors + BM25
- Uses Reciprocal Rank Fusion (RRF) for result fusion
- Better recall and precision
- Recommended for most use cases

## Safety and Guardrails

The system includes multiple layers of safety:

1. **Query Safety**: Blocks unsafe queries (violence, illegal activities, etc.)
2. **Answer Grounding**: Ensures answers are based on retrieved context
3. **Content Quality**: Filters out low-quality or gibberish responses
4. **Citation Validation**: Ensures citations are relevant and valid

## Performance

- **Target Latency**: ≤3 seconds for small documents
- **Context Limit**: 1600 tokens maximum
- **Answer Limit**: 500 tokens maximum
- **Batch Processing**: Configurable batch sizes for ingestion

## Troubleshooting

### Common Issues

1. **Elasticsearch Connection Failed**
   - Check if Elasticsearch is running: `curl http://localhost:9200`
   - Verify credentials in `.env` file

2. **Google Drive Access Denied**
   - Ensure service account has access to the folder
   - Check if `service_account.json` is in the project root

3. **Hugging Face API Errors**
   - Verify API key is valid and has sufficient credits
   - Check model availability: `meta-llama/Meta-Llama-3-8B-Instruct`

4. **OCR Issues**
   - Install Tesseract OCR and Poppler
   - Set correct paths in `.env` file

### Logs and Debugging

```bash
# Enable verbose logging
export VERBOSE=true

# Check API health
curl http://127.0.0.1:8000/healthz

# Debug retrieval
curl "http://127.0.0.1:8000/debug/retrieve?q=test&mode=hybrid&top_k=5"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request


See the Walkthrough video here :
https://drive.google.com/drive/folders/1Zfo52qL5oGdzq6o01KgJISVHhI8w07FP?usp=sharing
