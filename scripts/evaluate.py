#!/usr/bin/env python3
"""
Evaluate B-PLIS-RAG on benchmarks.

Computes character-level precision/recall and faithfulness metrics
on LegalBench-RAG benchmarks.

Usage:
    python scripts/evaluate.py --benchmarks all
    python scripts/evaluate.py --benchmarks contractnli cuad
    python scripts/evaluate.py --checkpoint checkpoints/reft_latest.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, setup_environment, PROJECT_ROOT
from src.rag_pipeline import RAGPipeline
from src.evaluator import Evaluator, EvaluationMetrics, run_ablation_study
from src.data_handler import LegalBenchRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline on benchmarks",
    )
    
    # Benchmark options
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        help="Benchmarks to evaluate (or 'all')",
    )
    
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples per benchmark",
    )
    
    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="t5-base",
        help="Model name",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to ReFT checkpoint",
    )
    
    # Steering options
    parser.add_argument(
        "--no-reft",
        action="store_true",
        help="Disable ReFT",
    )
    
    parser.add_argument(
        "--no-steering",
        action="store_true",
        help="Disable activation steering",
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "evaluation.json"),
        help="Output file for results",
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format",
    )
    
    # Ablation
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study",
    )
    
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline (no steering)",
    )
    
    return parser.parse_args()


def create_pipeline(
    model_name: str,
    use_reft: bool,
    use_steering: bool,
    checkpoint: str | None,
) -> RAGPipeline:
    """Create and configure the RAG pipeline."""
    
    pipeline = RAGPipeline(
        model_name=model_name,
        use_reft=use_reft,
        use_steering=use_steering,
    )
    
    # Load checkpoint if provided
    if checkpoint and Path(checkpoint).exists():
        logger.info(f"Loading checkpoint: {checkpoint}")
        pipeline.load_interventions(str(Path(checkpoint).parent))
    
    return pipeline


def load_and_index(pipeline: RAGPipeline) -> None:
    """Load and index documents."""
    data_handler = LegalBenchRAG()
    
    # Try to load corpus
    logger.info("Loading corpus...")
    documents = data_handler.load_corpus()
    
    all_docs = []
    for docs in documents.values():
        all_docs.extend(docs)
    
    if all_docs:
        logger.info(f"Indexing {len(all_docs)} documents...")
        pipeline.index_documents(all_docs)
    else:
        logger.warning("No documents found. Run generate_benchmark.py --download first.")


def evaluate_single_benchmark(
    pipeline: RAGPipeline,
    benchmark_name: str,
    max_examples: int,
) -> EvaluationMetrics:
    """Evaluate on a single benchmark."""
    
    data_handler = LegalBenchRAG()
    evaluator = Evaluator(pipeline, data_handler)
    
    logger.info(f"Evaluating benchmark: {benchmark_name}")
    metrics = evaluator.evaluate_benchmark(benchmark_name, max_examples=max_examples)
    
    return metrics


def evaluate_all_benchmarks(
    pipeline: RAGPipeline,
    benchmark_names: list[str],
    max_examples: int,
) -> EvaluationMetrics:
    """Evaluate on all specified benchmarks."""
    
    data_handler = LegalBenchRAG()
    
    if "all" in benchmark_names:
        # Load all benchmarks
        data_handler.load_benchmarks()
        benchmark_names = list(data_handler._benchmarks.keys())
    
    all_metrics = EvaluationMetrics()
    all_precisions = []
    all_recalls = []
    all_f1s = []
    total_examples = 0
    
    for name in benchmark_names:
        metrics = evaluate_single_benchmark(pipeline, name, max_examples)
        
        if metrics.num_examples > 0:
            all_precisions.append(metrics.char_precision)
            all_recalls.append(metrics.char_recall)
            all_f1s.append(metrics.char_f1)
            total_examples += metrics.num_examples
            
            all_metrics.per_benchmark[name] = metrics.per_benchmark.get(name, {
                "precision": metrics.char_precision,
                "recall": metrics.char_recall,
                "f1": metrics.char_f1,
                "num_examples": metrics.num_examples,
            })
            
            print(f"\n{name}:")
            print(f"  Precision: {metrics.char_precision:.4f}")
            print(f"  Recall: {metrics.char_recall:.4f}")
            print(f"  F1: {metrics.char_f1:.4f}")
            print(f"  Examples: {metrics.num_examples}")
    
    # Aggregate
    if all_precisions:
        all_metrics.char_precision = sum(all_precisions) / len(all_precisions)
        all_metrics.char_recall = sum(all_recalls) / len(all_recalls)
        all_metrics.char_f1 = sum(all_f1s) / len(all_f1s)
    all_metrics.num_examples = total_examples
    
    return all_metrics


def compare_with_baseline(
    pipeline: RAGPipeline,
    benchmark_name: str,
    max_examples: int,
) -> dict:
    """Compare steered pipeline with baseline."""
    
    data_handler = LegalBenchRAG()
    
    # Evaluate steered version
    logger.info("Evaluating steered pipeline...")
    steered_evaluator = Evaluator(pipeline, data_handler)
    steered_metrics = steered_evaluator.evaluate_benchmark(benchmark_name, max_examples)
    
    # Create baseline pipeline
    logger.info("Evaluating baseline pipeline...")
    baseline_pipeline = RAGPipeline(
        model_name=pipeline.model_name,
        use_reft=False,
        use_steering=False,
    )
    baseline_pipeline.retriever = pipeline.retriever  # Share retriever
    
    baseline_evaluator = Evaluator(baseline_pipeline, data_handler)
    baseline_metrics = baseline_evaluator.evaluate_benchmark(benchmark_name, max_examples)
    
    # Compare
    comparison = {
        "benchmark": benchmark_name,
        "baseline": baseline_metrics.to_dict(),
        "steered": steered_metrics.to_dict(),
        "improvement": {
            "precision": steered_metrics.char_precision - baseline_metrics.char_precision,
            "recall": steered_metrics.char_recall - baseline_metrics.char_recall,
            "f1": steered_metrics.char_f1 - baseline_metrics.char_f1,
        },
    }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("BASELINE vs STEERED COMPARISON")
    print("=" * 60)
    print(f"Benchmark: {benchmark_name}")
    print("\n           Baseline    Steered    Improvement")
    print(f"Precision: {baseline_metrics.char_precision:.4f}      {steered_metrics.char_precision:.4f}      {comparison['improvement']['precision']:+.4f}")
    print(f"Recall:    {baseline_metrics.char_recall:.4f}      {steered_metrics.char_recall:.4f}      {comparison['improvement']['recall']:+.4f}")
    print(f"F1:        {baseline_metrics.char_f1:.4f}      {steered_metrics.char_f1:.4f}      {comparison['improvement']['f1']:+.4f}")
    print("=" * 60)
    
    return comparison


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Setup
    setup_environment()
    
    # Create pipeline
    logger.info("Initializing RAG pipeline...")
    pipeline = create_pipeline(
        model_name=args.model,
        use_reft=not args.no_reft,
        use_steering=not args.no_steering,
        checkpoint=args.checkpoint,
    )
    
    # Load and index documents
    load_and_index(pipeline)
    
    # Run comparison if requested
    if args.compare_baseline:
        benchmark = args.benchmarks[0] if args.benchmarks[0] != "all" else "contractnli"
        comparison = compare_with_baseline(pipeline, benchmark, args.max_examples)
        
        # Save comparison
        output_path = Path(args.output).with_stem("comparison")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison saved to {output_path}")
        return
    
    # Run ablation if requested
    if args.ablation:
        benchmark = args.benchmarks[0] if args.benchmarks[0] != "all" else "contractnli"
        results = run_ablation_study(pipeline, benchmark, args.max_examples)
        
        # Save ablation results
        output_path = Path(args.output).with_stem("ablation")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ablation_data = {
            name: metrics.to_dict() for name, metrics in results.items()
        }
        with open(output_path, "w") as f:
            json.dump(ablation_data, f, indent=2)
        logger.info(f"Ablation results saved to {output_path}")
        return
    
    # Run evaluation
    logger.info("Starting evaluation...")
    metrics = evaluate_all_benchmarks(
        pipeline,
        args.benchmarks,
        args.max_examples,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Examples: {metrics.num_examples}")
    print(f"Avg Precision: {metrics.char_precision:.4f}")
    print(f"Avg Recall: {metrics.char_recall:.4f}")
    print(f"Avg F1: {metrics.char_f1:.4f}")
    print("=" * 60)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data_handler = LegalBenchRAG()
    evaluator = Evaluator(pipeline, data_handler)
    
    if args.format == "csv":
        evaluator.save_results(metrics, output_path.with_suffix(".csv"), format="csv")
    else:
        evaluator.save_results(metrics, output_path, format="json")
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
