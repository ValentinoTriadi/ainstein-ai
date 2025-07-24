import logging
import sys
from datetime import datetime
from typing import Optional

class RAGFormatter(logging.Formatter):
    """Custom formatter for RAG pipeline logging"""
    
    def format(self, record):
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get phase from record if available
        phase = getattr(record, 'phase', 'general')
        
        # Format message
        return f"[{timestamp}] [{record.levelname}] [{phase}] {record.getMessage()}"

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up structured logging for RAG pipeline"""
    
    # Create logger
    logger = logging.getLogger("rag_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Set formatter
    formatter = RAGFormatter()
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def log_with_phase(logger: logging.Logger, level: str, phase: str, message: str, **kwargs):
    """Log message with phase information"""
    extra = {"phase": phase}
    extra.update(kwargs)
    
    log_func = getattr(logger, level.lower())
    log_func(message, extra=extra)