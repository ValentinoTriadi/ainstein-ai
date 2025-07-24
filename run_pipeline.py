#!/usr/bin/env python3
"""
Standalone script to run the RAG pipeline
"""
import os
import sys
import argparse
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from app.pipeline import RAGPipeline
from app.logging_config import setup_logging, log_with_phase

def main():
    parser = argparse.ArgumentParser(description="Run RAG Pipeline")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--query", type=str, help="Run a single query")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--stats", action="store_true", help="Show pipeline stats")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Setup logging
    logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    try:
        # Build index
        pipeline.build_index(force_rebuild=args.rebuild)
        
        if args.stats:
            # Show stats
            stats = pipeline.get_stats()
            print("\n📊 Pipeline Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return
        
        if args.query:
            # Run single query
            result = pipeline.query(args.query)
            print(f"\n❓ Query: {result['query']}")
            print(f"📝 Response: {result.get('response', 'No response generated')}")
            print(f"⏱️  Time: {result['total_processing_time']:.3f}s")
            
            if result.get('sources'):
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'][:3]):
                    print(f"  {i+1}. {source.get('filename', 'unknown')} (score: {source.get('score', 0):.3f})")
            
        elif args.interactive:
            # Interactive mode
            print("\n🤖 RAG Pipeline Interactive Mode")
            print("Type your queries below (or 'quit' to exit):\n")
            
            while True:
                try:
                    query = input("Query: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not query:
                        continue
                    
                    result = pipeline.query(query)
                    print(f"\nResponse: {result.get('response', 'No response generated')}")
                    print(f"Time: {result['total_processing_time']:.3f}s")
                    print("-" * 50)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
        
        else:
            print("✅ Pipeline ready! Use --query, --interactive, or --stats")
            
    except Exception as e:
        log_with_phase(logger, 'error', 'main', f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()