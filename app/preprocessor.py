import time
from typing import List, Dict
from app.logging_config import setup_logging, log_with_phase

logger = setup_logging()

class DocumentPreprocessor:
    """Preprocesses documents for RAG pipeline (1 file = 1 chunk)"""
    
    def preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Preprocess documents for embedding
        Each file becomes one chunk as per specification
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of preprocessed document chunks
        """
        start_time = time.time()
        processed_docs = []
        
        log_with_phase(logger, 'info', 'preprocessing', f"Starting preprocessing of {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            try:
                processed_doc = self._preprocess_single_document(doc, i)
                processed_docs.append(processed_doc)
                
                log_with_phase(
                    logger, 'debug', 'preprocessing',
                    f"Processed {doc['metadata']['filename']} "
                    f"(1 chunk, {doc['metadata']['size']} bytes)"
                )
                
            except Exception as e:
                log_with_phase(
                    logger, 'error', 'preprocessing',
                    f"Failed to preprocess {doc['metadata'].get('filename', 'unknown')}: {str(e)}"
                )
        
        process_time = time.time() - start_time
        log_with_phase(
            logger, 'info', 'preprocessing',
            f"Preprocessing completed: {len(processed_docs)} chunks in {process_time:.2f}s"
        )
        
        return processed_docs
    
    def _preprocess_single_document(self, doc: Dict, doc_id: int) -> Dict:
        """Preprocess a single document"""
        
        # Clean content (basic preprocessing)
        content = doc['content'].strip()
        
        # Create chunk metadata
        chunk_metadata = {
            **doc['metadata'],
            'chunk_id': doc_id,
            'chunk_index': 0,  # Always 0 since 1 file = 1 chunk
            'total_chunks': 1,
            'content_preview': content[:200] + "..." if len(content) > 200 else content
        }
        
        return {
            'content': content,
            'metadata': chunk_metadata
        }