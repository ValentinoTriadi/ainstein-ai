import os
import time
from pathlib import Path
from typing import List, Dict, Tuple
from app.logging_config import setup_logging, log_with_phase

logger = setup_logging()

class DocumentLoader:
    """Loads .py and .md files from specified directories"""
    
    def __init__(self, source_dirs: List[str]):
        self.source_dirs = source_dirs
        self.supported_extensions = {'.py', '.md'}
    
    def load_documents(self) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Load documents from source directories
        
        Returns:
            Tuple of (documents, stats)
        """
        start_time = time.time()
        documents = []
        stats = {'py': 0, 'md': 0, 'skipped': 0, 'failed': 0}
        
        log_with_phase(logger, 'info', 'loading', f"Starting document loading from: {', '.join(self.source_dirs)}")
        
        for source_dir in self.source_dirs:
            if not os.path.exists(source_dir):
                log_with_phase(logger, 'warning', 'loading', f"Directory not found: {source_dir}")
                continue
            
            documents_in_dir, dir_stats = self._load_from_directory(source_dir)
            documents.extend(documents_in_dir)
            
            # Update stats
            for key, value in dir_stats.items():
                stats[key] += value
        
        load_time = time.time() - start_time
        total_files = stats['py'] + stats['md']
        
        log_with_phase(
            logger, 'info', 'loading', 
            f"Loaded {total_files} files (.py: {stats['py']}, .md: {stats['md']}) "
            f"in {load_time:.2f}s. Skipped: {stats['skipped']}, Failed: {stats['failed']}"
        )
        
        return documents, stats
    
    def _load_from_directory(self, directory: str) -> Tuple[List[Dict], Dict[str, int]]:
        """Load documents from a single directory"""
        documents = []
        stats = {'py': 0, 'md': 0, 'skipped': 0, 'failed': 0}
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                
                # Skip hidden files
                if file.startswith('.'):
                    stats['skipped'] += 1
                    continue
                
                # Check extension
                if file_path.suffix not in self.supported_extensions:
                    stats['skipped'] += 1
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create document dict
                    doc = {
                        'content': content,
                        'metadata': {
                            'path': str(file_path),
                            'filename': file,
                            'extension': file_path.suffix,
                            'size': len(content),
                            'directory': directory
                        }
                    }
                    
                    documents.append(doc)
                    
                    # Update stats
                    if file_path.suffix == '.py':
                        stats['py'] += 1
                    elif file_path.suffix == '.md':
                        stats['md'] += 1
                    
                    log_with_phase(
                        logger, 'debug', 'loading',
                        f"Loaded {file_path} ({len(content)} bytes)"
                    )
                
                except Exception as e:
                    stats['failed'] += 1
                    log_with_phase(
                        logger, 'error', 'loading',
                        f"Failed to load {file_path}: {str(e)}"
                    )
        
        return documents, stats