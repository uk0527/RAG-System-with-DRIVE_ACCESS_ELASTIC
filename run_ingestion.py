#!/usr/bin/env python3
"""
RAG System PDF Ingestion Script
Run this to ingest PDFs from Google Drive into Elasticsearch
"""

from app.core.config import get_elasticsearch_config, INDEX_NAME, DRIVE_FOLDER_ID
from app.core.ingest import ingest_documents, create_elasticsearch_mapping, create_elser_pipeline, recreate_index_with_new_mapping
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

def main():
    print("Starting RAG System PDF Ingestion...")
    print(f"Google Drive Folder ID: {DRIVE_FOLDER_ID}")
    print(f"Elasticsearch Index: {INDEX_NAME}")
    
    # Create Elasticsearch client
    es_config = get_elasticsearch_config()
    es_client = Elasticsearch(**es_config)
    
    # Test connection
    try:
        if es_client.ping():
            print("Connected to Elasticsearch!")
        else:
            print("Failed to connect to Elasticsearch")
            return
    except Exception as e:
        print(f"Elasticsearch connection error: {e}")
        return
    
    # Check if index exists (don't recreate - we have ELSER-compatible index)
    try:
        if not es_client.indices.exists(index=INDEX_NAME):
            print(f"Index {INDEX_NAME} does not exist.")
            return
        else:
            print(f"Using existing ELSER-compatible index: {INDEX_NAME}")
    except Exception as e:
        print(f"Error checking index: {e}")
        return
    
    # Ensure ELSER pipeline exists
    try:
        es_client.ingest.get_pipeline(id="elser_v2_pipeline")
        print("ELSER pipeline already exists")
    except NotFoundError:
        print("Creating ELSER pipeline...")
        pipeline = create_elser_pipeline()
        try:
            es_client.ingest.put_pipeline(id="elser_v2_pipeline", body=pipeline)
            print("ELSER pipeline created successfully!")
        except Exception as e:
            print(f"ELSER pipeline creation failed: {e}")
            print("Continuing without ELSER (will use dense vectors only)")
    
    # Start ingestion
    try:
        print("Starting document ingestion...")
        result = ingest_documents(
            es_client=es_client,
            folder_id=DRIVE_FOLDER_ID,
            force=False,
            max_files=100,
            verbose=True
        )
        
        print("\nIngestion completed successfully!")
        print(f"Files processed: {result['files_seen']}")
        print(f"Chunks indexed: {result['chunks_indexed']}")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
        
        if result['chunks_indexed'] > 0:
            print(f"Your RAG system is ready with {result['chunks_indexed']} document chunks!")
        else:
            print("No documents were indexed. Check your Google Drive folder and permissions.")
            
    except Exception as e:
        print(f"Ingestion error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

