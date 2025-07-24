import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

from app.pipeline import RAGPipeline
from app.logging_config import setup_logging, log_with_phase

# Load environment variables
load_dotenv()

logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))

def log_environment_variables():
    """Log all loaded environment variables (safely)"""
    log_with_phase(logger, 'info', 'startup', "=== Environment Variables ===")
    
    # Define environment variables we expect/use
    expected_env_vars = [
        'LOG_LEVEL',
        'PORT', 
        'HOST',
       'OPENAI_API_KEY',
'CHROMA_PERSIST_DIRECTORY',
'LOG_LEVEL',
'SOURCE_DIRS',
'TOP_K_RETRIEVAL',
'HOST',
'PORT',
'LLM_MODEL',
'MAX_TOKENS',
'TEMPERATURE'
    ]
    
    for var in expected_env_vars:
        value = os.getenv(var)
        if value is not None:
            log_with_phase(logger, 'info', 'startup', f"{var}={value}")
        else:
            log_with_phase(logger, 'warning', 'startup', f"{var}=<NOT SET>")
    
    # Log all environment variables starting with specific prefixes (optional)
    app_env_vars = {k: v for k, v in os.environ.items() if k.startswith(('RAG_', 'APP_', 'API_'))}
    if app_env_vars:
        log_with_phase(logger, 'info', 'startup', "=== Additional App Environment Variables ===")
        for key, value in app_env_vars.items():
            if any(sensitive in key.upper() for sensitive in ['KEY', 'TOKEN', 'PASSWORD', 'SECRET']):
                masked_value = f"{value[:8]}{'*' * (len(value) - 8)}" if len(value) > 8 else "*" * len(value)
                log_with_phase(logger, 'info', 'startup', f"{key}={masked_value}")
            else:
                log_with_phase(logger, 'info', 'startup', f"{key}={value}")
    
    # Log .env file status
    env_file_paths = ['.env', '.env.local', '.env.production']
    for env_file in env_file_paths:
        if os.path.exists(env_file):
            log_with_phase(logger, 'info', 'startup', f"Found environment file: {env_file}")
        else:
            log_with_phase(logger, 'debug', 'startup', f"Environment file not found: {env_file}")
    
    log_with_phase(logger, 'info', 'startup', "=== End Environment Variables ===")

# Log environment variables immediately after loading
log_environment_variables()

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    log_with_phase(logger, 'info', 'api', "Starting RAG API server")
    
    try:
        # Build or load index
        rag_pipeline.build_index(force_rebuild=True)
        log_with_phase(logger, 'info', 'api', "RAG pipeline ready")
    except Exception as e:
        log_with_phase(logger, 'error', 'api', f"Failed to initialize pipeline: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    log_with_phase(logger, 'info', 'api', "Shutting down RAG API server")

# Create FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation API for Manim Documentation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query text")
    retrieve_only: bool = Field(False, description="Only retrieve documents without generating response")
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve (overrides default)")

class QueryResponse(BaseModel):
    query: str
    response: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    retrieval_time: float
    total_processing_time: float
    metadata: Dict[str, Any] = {}

class IndexRequest(BaseModel):
    force_rebuild: bool = Field(False, description="Force rebuild even if index exists")

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG Pipeline API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_ready": rag_pipeline.retriever is not None,
        "timestamp": time.time()
    }

@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    try:
        stats = rag_pipeline.get_stats()
        return stats
    except Exception as e:
        log_with_phase(logger, 'error', 'api', f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    start_time = time.time()
    
    log_with_phase(logger, 'info', 'api', f"Received query request: '{request.query[:100]}{'...' if len(request.query) > 100 else ''}'")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Process query
        result = rag_pipeline.query(
            query_text=request.query,
            retrieve_only=request.retrieve_only
        )
        
        # Prepare response
        response = QueryResponse(
            query=result['query'],
            response=result.get('response'),
            sources=result.get('sources', result.get('results', [])),
            retrieval_time=result.get('retrieval_time', 0.0),
            total_processing_time=result.get('total_processing_time', time.time() - start_time),
            metadata={
                'total_results': result.get('total_results', len(result.get('sources', []))),
                'retrieve_only': request.retrieve_only,
                'response_length': len(result.get('response', '')) if result.get('response') else 0
            }
        )
        
        log_with_phase(
            logger, 'info', 'api',
            f"Query processed successfully in {response.total_processing_time:.3f}s"
        )
        
        return response
        
    except Exception as e:
        log_with_phase(logger, 'error', 'api', f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild-index")
async def rebuild_index(request: IndexRequest, background_tasks: BackgroundTasks):
    """Rebuild the vector index"""
    
    log_with_phase(logger, 'info', 'api', f"Index rebuild requested (force_rebuild={request.force_rebuild})")
    
    def rebuild_task():
        try:
            rag_pipeline.build_index(force_rebuild=request.force_rebuild)
            log_with_phase(logger, 'info', 'api', "Index rebuild completed successfully")
        except Exception as e:
            log_with_phase(logger, 'error', 'api', f"Index rebuild failed: {str(e)}")
    
    background_tasks.add_task(rebuild_task)
    
    return {
        "message": "Index rebuild started",
        "force_rebuild": request.force_rebuild,
        "status": "processing"
    }

@app.get("/search")
async def search_documents(q: str, top_k: int = 5):
    """Simple search endpoint"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty")
    
    try:
        result = rag_pipeline.query(query_text=q, retrieve_only=True)
        
        return {
            "query": q,
            "results": result.get('results', [])[:top_k],
            "total_found": result.get('total_results', 0),
            "processing_time": result.get('total_processing_time', 0)
        }
        
    except Exception as e:
        log_with_phase(logger, 'error', 'api', f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    log_with_phase(logger, 'info', 'api', f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )