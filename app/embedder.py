import time
from typing import List, Dict
import os
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.logging_config import setup_logging, log_with_phase

logger = setup_logging()

class RAGEmbedder:
    """Handles embedding and vector store creation using LlamaIndex + ChromaDB"""
    
    def __init__(self, persist_directory: str = "./vectorstore"):
        self.persist_directory = persist_directory
        self.collection_name = "manim_docs"
        
        # Initialize OpenAI embeddings
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        
        log_with_phase(logger, 'info', 'embedding', f"Initialized embedder with persist_directory: {persist_directory}")
    
    def create_vector_store(self, documents: List[Dict]) -> VectorStoreIndex:
        """
        Create vector store from preprocessed documents
        
        Args:
            documents: List of preprocessed document chunks
            
        Returns:
            LlamaIndex VectorStoreIndex
        """
        start_time = time.time()
        
        log_with_phase(logger, 'info', 'embedding', f"Creating vector store for {len(documents)} documents")
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            chroma_collection = chroma_client.get_collection(self.collection_name)
            log_with_phase(logger, 'info', 'embedding', f"Using existing collection: {self.collection_name}")
        except:
            chroma_collection = chroma_client.create_collection(self.collection_name)
            log_with_phase(logger, 'info', 'embedding', f"Created new collection: {self.collection_name}")
        
        # Create ChromaVectorStore
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Convert documents to LlamaIndex Document objects
        llama_documents = []
        for doc in documents:
            # Create LlamaIndex Document
            llama_doc = Document(
                text=doc['content'],
                metadata=doc['metadata'],
                doc_id=str(doc['metadata']['chunk_id'])
            )
            llama_documents.append(llama_doc)
            
            # Log per document
            log_with_phase(
                logger, 'debug', 'embedding',
                f"Embedded {doc['metadata']['filename']} "
                f"[preview: {doc['metadata']['content_preview'][:50]}...]"
            )
        
        # Create index from documents
        index = VectorStoreIndex.from_documents(
            llama_documents,
            vector_store=vector_store
        )
        
        create_time = time.time() - start_time
        
        log_with_phase(
            logger, 'info', 'embedding',
            f"Vector store created successfully: {len(documents)} vectors indexed "
            f"in {create_time:.2f}s. Persisted to: {self.persist_directory}"
        )
        
        return index
    
    def load_existing_index(self) -> VectorStoreIndex:
        """Load existing vector store index"""
        
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"Vector store directory not found: {self.persist_directory}")
        
        log_with_phase(logger, 'info', 'embedding', "Loading existing vector store")
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get existing collection
        chroma_collection = chroma_client.get_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create index from existing vector store
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        log_with_phase(logger, 'info', 'embedding', "Successfully loaded existing vector store")
        
        return index