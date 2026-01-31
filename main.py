#!/usr/bin/env python3
"""
B-PLIS-RAG: Main Entry Point

Quick start for running RAG queries with ReFT and activation steering.

Usage:
    python main.py --query "What is a breach of contract?"
    python main.py --query "Define confidential information" --corpus contractnli
    python main.py --interactive
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config, setup_environment
from src.data_handler import DataHandler, LegalBenchRAG
from src.rag_pipeline import RAGPipeline, RAGConfig
from src.utils import timer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="B-PLIS-RAG: Legal RAG with ReFT and Activation Steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single query
    python main.py --query "What are the termination clauses?"
    
    # With specific corpus
    python main.py --query "Define confidential information" --corpus contractnli
    
    # Interactive mode
    python main.py --interactive
    
    # Custom settings
    python main.py --query "..." --top-k 10 --no-reft --no-steering
        """,
    )
    
    # Query options
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query to process through the RAG pipeline",
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    
    # Data options
    parser.add_argument(
        "--corpus",
        type=str,
        choices=["contractnli", "cuad", "maud", "privacyqa", "sara", "all"],
        default="all",
        help="Corpus to use for retrieval",
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the LegalBench-RAG dataset",
    )
    
    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-base",
        help="Model name (default: google/flan-t5-base)",
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )
    
    # Steering options
    parser.add_argument(
        "--no-reft",
        action="store_true",
        help="Disable ReFT intervention",
    )
    
    parser.add_argument(
        "--no-steering",
        action="store_true",
        help="Disable activation steering",
    )
    
    parser.add_argument(
        "--reft-layer",
        type=int,
        default=6,
        help="Layer for ReFT intervention (default: 6)",
    )
    
    parser.add_argument(
        "--steering-multiplier",
        type=float,
        default=2.0,
        help="Steering strength multiplier (default: 2.0)",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to trained ReFT checkpoint (default: checkpoints/reft_latest.pt)",
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def download_data() -> None:
    """Download the LegalBench-RAG dataset."""
    logger.info("Downloading LegalBench-RAG dataset...")
    data_handler = LegalBenchRAG()
    data_handler.download()
    logger.info("Download complete!")


def create_pipeline(args: argparse.Namespace) -> RAGPipeline:
    """Create and configure the RAG pipeline."""
    config = RAGConfig(
        model_name=args.model,
        top_k=args.top_k,
        max_new_tokens=args.max_tokens,
        use_reft=not args.no_reft,
        use_steering=not args.no_steering,
        reft_layer=args.reft_layer,
        steering_multiplier=args.steering_multiplier,
    )
    
    pipeline = RAGPipeline(config=config)
    
    # Load trained ReFT checkpoint if provided
    if args.checkpoint and pipeline.use_reft:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            logger.info(f"Loading ReFT checkpoint from {checkpoint_path}")
            pipeline.load_reft_checkpoint(checkpoint_path)
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    elif pipeline.use_reft and not args.checkpoint:
        # Try loading default checkpoint if it exists
        default_checkpoint = Path("checkpoints/reft_latest.pt")
        if default_checkpoint.exists():
            logger.info(f"Loading default ReFT checkpoint from {default_checkpoint}")
            pipeline.load_reft_checkpoint(default_checkpoint)
        else:
            logger.info("No checkpoint loaded - using untrained ReFT intervention")
    
    return pipeline


def load_and_index_documents(
    pipeline: RAGPipeline,
    corpus: str = "all",
) -> None:
    """Load and index documents from the specified corpus."""
    data_handler = LegalBenchRAG()
    
    # Load corpus
    corpus_types = None if corpus == "all" else [corpus]
    
    logger.info(f"Loading corpus: {corpus}")
    documents = data_handler.load_corpus(corpus_types=corpus_types)
    
    # Flatten documents
    all_docs = []
    for docs in documents.values():
        all_docs.extend(docs)
    
    if not all_docs:
        logger.warning("No documents loaded. Run with --download first.")
        return
    
    # Index documents
    logger.info(f"Indexing {len(all_docs)} documents...")
    with timer("Document indexing"):
        pipeline.index_documents(all_docs)


def process_query(
    pipeline: RAGPipeline,
    query: str,
    verbose: bool = False,
) -> dict:
    """Process a single query."""
    logger.info(f"Processing query: {query[:50]}...")
    
    with timer("Query processing"):
        response = pipeline.query(query)
    
    result = response.to_dict()
    
    # Print results
    print("\n" + "=" * 60)
    print("QUERY:", query)
    print("=" * 60)
    print("\nANSWER:", response.answer)
    print("\nSOURCES:")
    for i, source in enumerate(response.sources[:3], 1):
        print(f"  {i}. [{source['source']}] {source['id']}")
        if verbose:
            print(f"     Score: {source['score']:.4f}")
            print(f"     Snippet: {source['snippet'][:100]}...")
    print("=" * 60 + "\n")
    
    return result


def interactive_mode(pipeline: RAGPipeline, verbose: bool = False) -> None:
    """Run in interactive mode."""
    print("\n" + "=" * 60)
    print("B-PLIS-RAG Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for commands")
    print("=" * 60 + "\n")
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            if query.lower() == "help":
                print("""
Commands:
    quit, exit, q  - Exit interactive mode
    help          - Show this message
    
Just type your legal question to get an answer.
                """)
                continue
            
            process_query(pipeline, query, verbose)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Setup environment
    setup_environment()
    
    # Download data if requested
    if args.download:
        download_data()
        if not args.query and not args.interactive:
            return
    
    # Check if we have something to do
    if not args.query and not args.interactive:
        print("Please provide a --query or use --interactive mode.")
        print("Run with --help for more options.")
        return
    
    # Create pipeline
    logger.info("Initializing RAG pipeline...")
    pipeline = create_pipeline(args)
    
    # Load and index documents
    load_and_index_documents(pipeline, args.corpus)
    
    # Process query or enter interactive mode
    if args.interactive:
        interactive_mode(pipeline, args.verbose)
    elif args.query:
        result = process_query(pipeline, args.query, args.verbose)
        
        # Save output if requested
        if args.output:
            import json
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
