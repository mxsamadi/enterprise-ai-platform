## Live Demo

Try the Enterprise AI Document Intelligence Platform

An interactive, cloud-hosted Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from PDF documents.

Features

PDF text extraction and intelligent chunking

OpenAI embeddings and semantic similarity search

Context-grounded LLM responses with source citations

Password-protected Streamlit interface

Interactive deployment on Streamlit Community Cloud

Demo access code available upon request.

# Enterprise AI Document Intelligence Platform

A Python-based document intelligence application that uses Retrieval-Augmented Generation (RAG) to answer questions about uploaded PDF documents.

## Project Overview

This project demonstrates an end-to-end AI engineering workflow, including document ingestion, text chunking, embeddings, semantic search, LLM-based answer generation, REST API development, and automated evaluation.

The application allows users to upload text-based PDF documents and ask natural-language questions about their contents.

## Architecture

PDF Upload
    |
    v
FastAPI POST /upload
    |
    v
PDF Text Extraction (pypdf)
    |
    v
Text Chunking
    |
    v
OpenAI Embeddings (text-embedding-3-small)
    |
    v
ChromaDB (Persistent Vector Database)

Question
    |
    v
FastAPI POST /ask
    |
    v
Question Embedding
    |
    v
ChromaDB Semantic Search
    |
    v
Relevant Document Chunks
    |
    v
OpenAI LLM (gpt-5.5)
    |
    v
Answer with Source References

## Technologies

- Python
- FastAPI and Uvicorn
- OpenAI API
- ChromaDB
- pypdf
- NumPy
- python-dotenv
- PowerShell and VS Code

## Implemented Features

- PDF text extraction
- Sentence-based text chunking
- OpenAI text embeddings
- Cosine similarity search
- Persistent vector storage using ChromaDB
- Retrieval-Augmented Generation
- Source references in generated answers
- REST API endpoints for health checks, questions, and PDF uploads
- Automated retrieval evaluation
- End-to-end latency measurement
- RAG component performance profiling

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Application status message |
| GET | /health | Health check |
| POST | /ask | Ask a question about stored documents |
| POST | /upload | Upload and index a PDF |

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

## Evaluation Results

Initial evaluation used three questions about RAG, embeddings, and vector databases.

| Metric | Result |
|--------|--------|
| Retrieval Recall@2 | 100% (3/3 test cases) |
| Average end-to-end latency | 2.19 seconds |
| Maximum end-to-end latency | 2.71 seconds |

### Performance Profiling

| Component | Measured Latency |
|-----------|------------------|
| Question embedding | 0.267 seconds |
| ChromaDB retrieval | 0.011 seconds |
| LLM generation | 1.438 seconds |
| Total measured | 1.716 seconds |

LLM generation accounted for approximately 84% of the measured processing time.

These measurements are preliminary results from a small local test, not production benchmarks.

## Getting Started

### 1. Clone the repository

git clone YOUR_REPOSITORY_URL
cd enterprise-ai-platform

### 2. Create a virtual environment

python -m venv .venv

On Windows PowerShell:

.\.venv\Scripts\Activate.ps1

### 3. Install dependencies

python -m pip install -r requirements.txt

### 4. Configure environment variables

Create a .env file containing:

OPENAI_API_KEY=your_api_key_here

Never commit the .env file or your API key.

### 5. Prepare the vector database

Place a text-based PDF in the documents folder.

You can start the API and upload the PDF using POST /upload.
The application currently expects the ChromaDB collection
enterprise_documents to exist.

For the initial setup, run:

python vector_store.py

### 6. Start the API

python -m uvicorn api:app --reload

Open http://127.0.0.1:8000/docs to test the endpoints.

## Evaluation Scripts

Run the retrieval evaluation:

python evaluate.py

Run the API latency test while FastAPI is running:

python test_latency.py

Run the RAG component profiler:

python profile_rag.py

## Current Limitations

- Supports text-based PDFs; scanned PDFs require OCR
- No authentication or user-level document isolation
- No duplicate-document detection
- Basic chunking strategy
- Small evaluation dataset
- No production deployment or monitoring
- API startup currently depends on an existing ChromaDB collection

## Planned Improvements

- Duplicate-document detection
- Document listing and deletion
- Larger RAG evaluation dataset
- Answer faithfulness evaluation
- Improved error handling and observability
- Authentication and access control
- Docker containerization
- AWS deployment
- Automated testing and CI/CD

## Project Status

Working local prototype. Enterprise hardening and cloud deployment are planned.
