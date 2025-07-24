# 🔍 RAG Pipeline for Manim Documentation

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and LlamaIndex for querying Manim documentation and source code.

## ✨ Features

- **FastAPI-based REST API** with comprehensive endpoints
- **ChromaDB vector storage** with persistent data
- **LlamaIndex integration** for advanced retrieval
- **Comprehensive logging** at every pipeline step
- **Docker containerization** for easy deployment
- **1:1 file-to-chunk mapping** as specified
- **Support for .py and .md files only**

## 🚀 Quick Start

### Option 1: Local Development

1. **Setup environment:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Configure .env file:**
   ```bash
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_key_here
   ```

3. **Add your documents:**
   ```bash
   # Copy your Manim docs and source files
   cp -r /path/to/manim/docs ./docs/
   cp -r /path/to/manim/src ./src/
   ```

4. **Run the API:**
   ```bash
   python -m app.main
   ```

### Option 2: Docker

1. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

## 📡 API Endpoints

### Core Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /stats` - Pipeline statistics
- `POST /query` - Main RAG query endpoint
- `POST /rebuild-index` - Rebuild vector index
- `GET /search` - Simple document search

### Query Examples

```bash
# Basic query
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "How do I create an animation in Manim?"}'

# Retrieve documents only (no LLM response)
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "animation examples", "retrieve_only": true}'

# Simple search
curl "http://localhost:8000/search?q=animation&top_k=3"