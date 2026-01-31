#!/usr/bin/env python3
"""
Generate and process LegalBench-RAG benchmarks.

Downloads the LegalBench-RAG dataset and processes it for use
with the B-PLIS-RAG system.

Usage:
    python scripts/generate_benchmark.py --download
    python scripts/generate_benchmark.py --process
    python scripts/generate_benchmark.py --stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, PROJECT_ROOT
from src.data_handler import LegalBenchRAG, DataHandler, ConflictExample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and process LegalBench-RAG benchmarks",
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the LegalBench-RAG dataset",
    )
    
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process downloaded data into benchmarks",
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics about the data",
    )
    
    parser.add_argument(
        "--create-conflicts",
        action="store_true",
        help="Create conflict examples for training",
    )
    
    parser.add_argument(
        "--num-conflicts",
        type=int,
        default=500,
        help="Number of conflict examples to create",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Output directory for processed data",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download/re-process",
    )
    
    return parser.parse_args()


def download_dataset(force: bool = False) -> None:
    """Download the LegalBench-RAG dataset."""
    logger.info("Downloading LegalBench-RAG dataset...")
    
    data_handler = LegalBenchRAG()
    data_handler.download(force=force)
    
    logger.info("Download complete!")
    logger.info(f"Corpus directory: {data_handler.corpus_dir}")
    logger.info(f"Benchmarks directory: {data_handler.benchmarks_dir}")


def process_benchmarks(output_dir: str) -> None:
    """Process downloaded data into standardized benchmarks."""
    logger.info("Processing benchmarks...")
    
    output_path = Path(output_dir)
    benchmarks_path = output_path / "benchmarks"
    benchmarks_path.mkdir(parents=True, exist_ok=True)
    
    data_handler = LegalBenchRAG(data_dir=output_path)
    
    # Load all benchmarks
    benchmarks = data_handler.load_benchmarks()
    
    # Process each benchmark
    for name, examples in benchmarks.items():
        # Standardize format
        processed = []
        for ex in examples:
            processed.append({
                "query": ex.query,
                "corpus": ex.corpus,
                "ground_truth": ex.ground_truth_snippets,
                "metadata": ex.metadata,
            })
        
        # Save processed benchmark
        output_file = benchmarks_path / f"{name}_processed.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed {name}: {len(processed)} examples -> {output_file}")
    
    logger.info("Benchmark processing complete!")


def show_stats(output_dir: str) -> None:
    """Show statistics about the dataset."""
    logger.info("Computing dataset statistics...")
    
    output_path = Path(output_dir)
    data_handler = LegalBenchRAG(data_dir=output_path)
    
    # Load corpus
    documents = data_handler.load_corpus()
    stats = data_handler.get_corpus_stats()
    
    print("\n" + "=" * 60)
    print("LEGALBENCH-RAG DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal Documents: {stats['total_documents']}")
    print("\nCorpus Breakdown:")
    for corpus_type, corpus_stats in stats["corpus_types"].items():
        print(f"  {corpus_type}:")
        print(f"    Documents: {corpus_stats['num_documents']}")
        print(f"    Total Characters: {corpus_stats['total_characters']:,}")
        print(f"    Avg Doc Length: {corpus_stats['avg_doc_length']:.0f}")
    
    # Benchmark stats
    benchmarks = data_handler.load_benchmarks()
    
    print("\nBenchmark Breakdown:")
    total_examples = 0
    for name, examples in benchmarks.items():
        print(f"  {name}: {len(examples)} examples")
        total_examples += len(examples)
    print(f"\nTotal Benchmark Examples: {total_examples}")
    
    print("=" * 60 + "\n")


def create_conflict_examples(
    output_dir: str,
    num_examples: int = 500,
) -> None:
    """Create conflict examples for ReFT training."""
    logger.info(f"Creating {num_examples} conflict examples...")
    
    output_path = Path(output_dir)
    data_handler = LegalBenchRAG(data_dir=output_path)
    
    # Create conflict examples
    conflicts = data_handler.create_conflict_examples(num_examples=num_examples)
    
    # Save to file
    conflicts_file = output_path / "benchmarks" / "conflict_examples.json"
    conflicts_file.parent.mkdir(parents=True, exist_ok=True)
    
    conflicts_data = [c.to_dict() for c in conflicts]
    with open(conflicts_file, "w", encoding="utf-8") as f:
        json.dump(conflicts_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {len(conflicts)} conflict examples -> {conflicts_file}")
    
    # Show sample
    if conflicts:
        print("\nSample Conflict Example:")
        print("-" * 40)
        sample = conflicts[0]
        print(f"Query: {sample.query[:100]}...")
        print(f"Context: {sample.context[:200]}...")
        print(f"Context Answer: {sample.context_answer[:100]}...")
        print("-" * 40)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.download:
        download_dataset(force=args.force)
    
    if args.process:
        process_benchmarks(args.output_dir)
    
    if args.stats:
        show_stats(args.output_dir)
    
    if args.create_conflicts:
        create_conflict_examples(args.output_dir, args.num_conflicts)
    
    if not any([args.download, args.process, args.stats, args.create_conflicts]):
        print("Please specify an action: --download, --process, --stats, or --create-conflicts")
        print("Run with --help for more options.")


if __name__ == "__main__":
    main()
