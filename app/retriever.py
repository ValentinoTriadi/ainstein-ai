import time
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from app.logging_config import setup_logging, log_with_phase

logger = setup_logging()

class RAGRetriever:
    """Handles query-time retrieval using LlamaIndex"""
    
    def __init__(self, index: VectorStoreIndex, top_k: int = 5):
        self.index = index
        self.top_k = top_k
        
        # Configure retriever
        self.retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k
        )
        
        # Configure query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )
        
        log_with_phase(logger, 'info', 'retrieval', f"Initialized retriever with top_k={top_k}")
    
    def retrieve_documents(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        start_time = time.time()
        
        log_with_phase(logger, 'info', 'retrieval', f"Processing query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # Retrieve nodes
            retrieved_nodes = self.retriever.retrieve(query)
            
            retrieval_time = time.time() - start_time
            
            # Process retrieved nodes
            results = []
            doc_titles = []
            
            for i, node in enumerate(retrieved_nodes):
                node_data = {
                    'content': node.text,
                    'metadata': node.metadata,
                    'score': node.score if hasattr(node, 'score') else 0.0,
                    'rank': i + 1
                }
                results.append(node_data)
                
                # Get document title/path for logging
                title = node.metadata.get('filename', node.metadata.get('path', f'doc_{i}'))
                doc_titles.append(title)
                
                log_with_phase(
                    logger, 'debug', 'retrieval',
                    f"Retrieved [{i+1}] {title} "
                    f"(score: {node_data['score']:.3f}, preview: {node.text[:100]}...)"
                )
            
            # Summary log
            log_with_phase(
                logger, 'info', 'retrieval',
                f"Retrieved {len(results)} documents in {retrieval_time:.3f}s: {', '.join(doc_titles[:3])}"
                f"{'...' if len(doc_titles) > 3 else ''}"
            )
            
            return {
                'query': query,
                'results': results,
                'retrieval_time': retrieval_time,
                'total_results': len(results)
            }
            
        except Exception as e:
            log_with_phase(logger, 'error', 'retrieval', f"Retrieval failed: {str(e)}")
            raise
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate response using query engine
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        log_with_phase(logger, 'info', 'retrieval', f"Generating response for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # Generate response
            response = self.query_engine.query(query)
            
            response_time = time.time() - start_time
            
            # Extract source nodes
            source_info = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_info.append({
                        'filename': node.metadata.get('filename', 'unknown'),
                        'path': node.metadata.get('path', 'unknown'),
                        'score': node.score if hasattr(node, 'score') else 0.0
                    })
            
            result = {
                'query': query,
                'response': str(response),
                'sources': source_info,
                'response_time': response_time,
                'response_length': len(str(response))
            }
            
            log_with_phase(
                logger, 'info', 'retrieval',
                f"Generated response in {response_time:.3f}s "
                f"({len(str(response))} chars, {len(source_info)} sources)"
            )
            
            return result
            
        except Exception as e:
            log_with_phase(logger, 'error', 'retrieval', f"Response generation failed: {str(e)}")
            raise