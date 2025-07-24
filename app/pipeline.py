import os
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex

from app.loader import DocumentLoader
from app.preprocessor import DocumentPreprocessor
from app.embedder import RAGEmbedder
from app.retriever import RAGRetriever
from app.logging_config import setup_logging, log_with_phase

# Load environment variables
load_dotenv()

logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))

class RAGPipeline:
    """Main RAG Pipeline orchestrator"""
    
    def __init__(self):
        self.source_dirs = os.getenv("SOURCE_DIRS", "docs,src").split(",")
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vectorstore")
        self.top_k = int(os.getenv("TOP_K_RETRIEVAL", "5"))
        
        # Initialize components
        self.loader = DocumentLoader(self.source_dirs)
        self.preprocessor = DocumentPreprocessor()
        self.embedder = RAGEmbedder(self.persist_directory)
        self.retriever: Optional[RAGRetriever] = None
        self.index: Optional[VectorStoreIndex] = None
        
        log_with_phase(logger, 'info', 'pipeline', "RAG Pipeline initialized")
    
    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build or rebuild the vector index
        
        Args:
            force_rebuild: If True, rebuild even if existing index found
        """
        start_time = time.time()
        
        log_with_phase(logger, 'info', 'pipeline', "Starting index building process")
        
        # Check if index exists and force_rebuild is False
        if not force_rebuild and os.path.exists(self.persist_directory):
            try:
                self.index = self.embedder.load_existing_index()
                self.retriever = RAGRetriever(self.index, self.top_k)
                
                build_time = time.time() - start_time
                log_with_phase(
                    logger, 'info', 'pipeline',
                    f"Loaded existing index in {build_time:.2f}s"
                )
                return
            except Exception as e:
                log_with_phase(
                    logger, 'warning', 'pipeline',
                    f"Failed to load existing index: {str(e)}. Rebuilding..."
                )
        
        # Build new index
        log_with_phase(logger, 'info', 'pipeline', "Building new index from source documents")
        
        # 1. Load documents
        documents, load_stats = self.loader.load_documents()
        if not documents:
            raise ValueError("No documents loaded. Check source directories.")
        
        # 2. Preprocess documents
        processed_docs = self.preprocessor.preprocess_documents(documents)
        
        # 3. Create vector store
        self.index = self.embedder.create_vector_store(processed_docs)
        
        # 4. Initialize retriever
        self.retriever = RAGRetriever(self.index, self.top_k)
        
        build_time = time.time() - start_time
        log_with_phase(
            logger, 'info', 'pipeline',
            f"Index building completed in {build_time:.2f}s. "
            f"Processed {len(documents)} documents"
        )
    
    def query(self, query_text: str, retrieve_only: bool = False) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            query_text: The user's query
            retrieve_only: If True, only return retrieved documents without LLM response
            
        Returns:
            Dictionary containing query results
        """
        if not self.retriever:
            raise ValueError("Pipeline not initialized. Call build_index() first.")
        
        start_time = time.time()
        
        log_with_phase(logger, 'info', 'pipeline', f"Processing query: '{query_text[:100]}{'...' if len(query_text) > 100 else ''}'")
        
        try:
            if retrieve_only:
                # Only retrieve documents
                result = self.retriever.retrieve_documents(query_text)
            else:
                # Generate full response
                result = self.retriever.generate_response(query_text)
            
            total_time = time.time() - start_time
            result['total_processing_time'] = total_time
            
            log_with_phase(
                logger, 'info', 'pipeline',
                f"Query processed in {total_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            log_with_phase(logger, 'error', 'pipeline', f"Query processing failed: {str(e)} {os.getenv("OPENAI_API_KEY")}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            'source_directories': self.source_dirs,
            'persist_directory': self.persist_directory,
            'top_k_retrieval': self.top_k,
            'index_exists': self.index is not None,
            'retriever_ready': self.retriever is not None
        }
        
        if os.path.exists(self.persist_directory):
            stats['vector_store_size'] = sum(
                os.path.getsize(os.path.join(self.persist_directory, f))
                for f in os.listdir(self.persist_directory)
                if os.path.isfile(os.path.join(self.persist_directory, f))
            )
        
        return stats