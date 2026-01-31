"""
Evaluation module for B-PLIS-RAG.

Implements benchmark evaluation with character-level precision/recall
and faithfulness metrics from ContextFocus paper.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from src.config import get_config
from src.data_handler import BenchmarkExample, LegalBenchRAG
from src.rag_pipeline import RAGPipeline, RAGResponse
from src.utils import safe_divide, compute_char_overlap

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Character-level retrieval metrics
    char_precision: float = 0.0
    char_recall: float = 0.0
    char_f1: float = 0.0
    
    # Faithfulness metrics (from ContextFocus)
    p_s: float = 0.0  # Substituted answer ratio
    p_o: float = 0.0  # Original answer ratio
    
    # Generation metrics
    exact_match: float = 0.0
    answer_relevance: float = 0.0
    
    # Aggregate metrics
    num_examples: int = 0
    num_correct: int = 0
    
    # Per-benchmark results
    per_benchmark: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "char_precision": self.char_precision,
            "char_recall": self.char_recall,
            "char_f1": self.char_f1,
            "p_s": self.p_s,
            "p_o": self.p_o,
            "exact_match": self.exact_match,
            "answer_relevance": self.answer_relevance,
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "per_benchmark": self.per_benchmark,
        }
    
    def __str__(self) -> str:
        return (
            f"EvaluationMetrics(\n"
            f"  char_precision={self.char_precision:.4f},\n"
            f"  char_recall={self.char_recall:.4f},\n"
            f"  char_f1={self.char_f1:.4f},\n"
            f"  p_s={self.p_s:.4f},\n"
            f"  p_o={self.p_o:.4f},\n"
            f"  exact_match={self.exact_match:.4f},\n"
            f"  num_examples={self.num_examples}\n"
            f")"
        )


@dataclass
class RetrievalResult:
    """Result from a single retrieval evaluation."""
    query: str
    retrieved_files: List[str]
    retrieved_char_ranges: List[Tuple[int, int]]
    gold_files: List[str]
    gold_char_ranges: List[Tuple[int, int]]
    char_precision: float
    char_recall: float
    char_f1: float


class Evaluator:
    """
    Evaluator for RAG systems on LegalBench-RAG benchmarks.
    
    Computes:
    - Character-level precision/recall for retrieval
    - Faithfulness metrics (p_s, p_o)
    - Generation quality metrics
    
    Example:
        >>> evaluator = Evaluator(pipeline)
        >>> metrics = evaluator.evaluate_benchmark("contractnli")
        >>> print(metrics)
    """
    
    def __init__(
        self,
        pipeline: RAGPipeline,
        data_handler: Optional[LegalBenchRAG] = None,
    ) -> None:
        """
        Initialize evaluator.
        
        Args:
            pipeline: RAG pipeline to evaluate.
            data_handler: Data handler for benchmarks.
        """
        self.pipeline = pipeline
        self.data_handler = data_handler or LegalBenchRAG()
    
    def compute_char_metrics(
        self,
        retrieved_spans: List[Tuple[str, int, int]],  # (file, start, end)
        gold_spans: List[Tuple[str, int, int]],
    ) -> Tuple[float, float, float]:
        """
        Compute character-level precision, recall, F1.
        
        Args:
            retrieved_spans: List of (file_path, char_start, char_end) tuples.
            gold_spans: Ground truth spans.
            
        Returns:
            Tuple of (precision, recall, f1).
        """
        total_retrieved_chars = 0
        total_gold_chars = 0
        total_overlap_chars = 0
        
        # Calculate total characters in retrieved spans
        for file_path, start, end in retrieved_spans:
            total_retrieved_chars += end - start
        
        # Calculate total characters in gold spans
        for file_path, start, end in gold_spans:
            total_gold_chars += end - start
        
        # Calculate overlap
        for r_file, r_start, r_end in retrieved_spans:
            for g_file, g_start, g_end in gold_spans:
                if r_file == g_file:
                    overlap, _, _ = compute_char_overlap(r_start, r_end, g_start, g_end)
                    total_overlap_chars += overlap
        
        precision = safe_divide(total_overlap_chars, total_retrieved_chars)
        recall = safe_divide(total_overlap_chars, total_gold_chars)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        
        return precision, recall, f1
    
    def evaluate_retrieval(
        self,
        examples: List[BenchmarkExample],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Evaluate retrieval on benchmark examples.
        
        Args:
            examples: Benchmark examples with ground truth.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of retrieval results.
        """
        results = []
        
        for example in tqdm(examples, desc="Evaluating retrieval"):
            # Retrieve
            retrieved = self.pipeline.retriever.retrieve_with_snippets(
                example.query, top_k=top_k
            )
            
            # Extract retrieved spans (simplified: use full doc)
            retrieved_spans = []
            for r in retrieved:
                doc = r["document"]
                retrieved_spans.append((doc.id, 0, len(doc.content)))
            
            # Extract gold spans
            gold_spans = []
            for snippet in example.ground_truth_snippets:
                file_path = snippet.get("file_path", snippet.get("file", ""))
                char_start = snippet.get("char_start", snippet.get("start", 0))
                char_end = snippet.get("char_end", snippet.get("end", 0))
                gold_spans.append((file_path, char_start, char_end))
            
            # Compute metrics
            precision, recall, f1 = self.compute_char_metrics(retrieved_spans, gold_spans)
            
            result = RetrievalResult(
                query=example.query,
                retrieved_files=[r["id"] for r in retrieved],
                retrieved_char_ranges=[(0, len(r["snippet"])) for r in retrieved],
                gold_files=[s[0] for s in gold_spans],
                gold_char_ranges=[(s[1], s[2]) for s in gold_spans],
                char_precision=precision,
                char_recall=recall,
                char_f1=f1,
            )
            results.append(result)
        
        return results
    
    def evaluate_generation(
        self,
        examples: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """
        Evaluate generation quality.
        
        Args:
            examples: List of dicts with query, context, expected_answer.
            
        Returns:
            Dictionary of generation metrics.
        """
        exact_matches = 0
        total = len(examples)
        
        for example in tqdm(examples, desc="Evaluating generation"):
            response = self.pipeline.query(example["query"])
            
            expected = example.get("expected_answer", example.get("answer", "")).lower()
            generated = response.answer.lower()
            
            if expected in generated or generated in expected:
                exact_matches += 1
        
        return {
            "exact_match": exact_matches / total if total > 0 else 0,
            "num_examples": total,
            "num_correct": exact_matches,
        }
    
    def evaluate_faithfulness(
        self,
        examples: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """
        Evaluate faithfulness metrics (p_s, p_o from ContextFocus).
        
        Args:
            examples: List with query, context, context_answer, original_answer.
            
        Returns:
            Dictionary with p_s and p_o metrics.
        """
        context_matches = 0
        original_matches = 0
        total = len(examples)
        
        for example in tqdm(examples, desc="Evaluating faithfulness"):
            response = self.pipeline.query(example["query"])
            generated = response.answer.lower().strip()
            
            context_answer = example.get("context_answer", "").lower().strip()
            original_answer = example.get("original_answer", "").lower().strip()
            
            # Check if generated matches context-based answer
            if context_answer and (context_answer in generated or generated in context_answer):
                context_matches += 1
            
            # Check if generated matches parametric/original answer
            if original_answer and (original_answer in generated or generated in original_answer):
                original_matches += 1
        
        return {
            "p_s": context_matches / total if total > 0 else 0,
            "p_o": original_matches / total if total > 0 else 0,
            "context_matches": context_matches,
            "original_matches": original_matches,
            "total": total,
        }
    
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        max_examples: Optional[int] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate on a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark.
            max_examples: Maximum examples to evaluate.
            
        Returns:
            EvaluationMetrics for the benchmark.
        """
        # Load benchmark
        self.data_handler.load_benchmarks([benchmark_name])
        examples = self.data_handler._benchmarks.get(benchmark_name, [])
        
        if max_examples:
            examples = examples[:max_examples]
        
        if not examples:
            logger.warning(f"No examples found for benchmark: {benchmark_name}")
            return EvaluationMetrics()
        
        # Evaluate retrieval
        retrieval_results = self.evaluate_retrieval(examples)
        
        # Aggregate retrieval metrics
        avg_precision = sum(r.char_precision for r in retrieval_results) / len(retrieval_results)
        avg_recall = sum(r.char_recall for r in retrieval_results) / len(retrieval_results)
        avg_f1 = sum(r.char_f1 for r in retrieval_results) / len(retrieval_results)
        
        return EvaluationMetrics(
            char_precision=avg_precision,
            char_recall=avg_recall,
            char_f1=avg_f1,
            num_examples=len(examples),
            per_benchmark={
                benchmark_name: {
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1": avg_f1,
                    "num_examples": len(examples),
                }
            },
        )
    
    def evaluate_all_benchmarks(
        self,
        max_examples_per_benchmark: Optional[int] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate on all available benchmarks.
        
        Args:
            max_examples_per_benchmark: Max examples per benchmark.
            
        Returns:
            Aggregated EvaluationMetrics.
        """
        # Load all benchmarks
        self.data_handler.load_benchmarks()
        
        all_metrics = EvaluationMetrics()
        all_precisions = []
        all_recalls = []
        all_f1s = []
        total_examples = 0
        
        for benchmark_name in self.data_handler._benchmarks:
            metrics = self.evaluate_benchmark(
                benchmark_name,
                max_examples=max_examples_per_benchmark,
            )
            
            all_precisions.append(metrics.char_precision)
            all_recalls.append(metrics.char_recall)
            all_f1s.append(metrics.char_f1)
            total_examples += metrics.num_examples
            
            all_metrics.per_benchmark[benchmark_name] = {
                "precision": metrics.char_precision,
                "recall": metrics.char_recall,
                "f1": metrics.char_f1,
                "num_examples": metrics.num_examples,
            }
        
        if all_precisions:
            all_metrics.char_precision = sum(all_precisions) / len(all_precisions)
            all_metrics.char_recall = sum(all_recalls) / len(all_recalls)
            all_metrics.char_f1 = sum(all_f1s) / len(all_f1s)
        
        all_metrics.num_examples = total_examples
        
        return all_metrics
    
    def compare_configurations(
        self,
        configs: List[Dict[str, Any]],
        benchmark_name: str,
        max_examples: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Compare different pipeline configurations.
        
        Args:
            configs: List of configuration dicts.
            benchmark_name: Benchmark to evaluate on.
            max_examples: Examples per config.
            
        Returns:
            List of results for each configuration.
        """
        results = []
        
        for config in configs:
            # Create pipeline with this config
            pipeline = RAGPipeline(**config)
            
            # Copy indexed documents
            pipeline.retriever = self.pipeline.retriever
            
            # Evaluate
            evaluator = Evaluator(pipeline, self.data_handler)
            metrics = evaluator.evaluate_benchmark(benchmark_name, max_examples)
            
            results.append({
                "config": config,
                "metrics": metrics.to_dict(),
            })
        
        return results
    
    def save_results(
        self,
        metrics: EvaluationMetrics,
        output_path: Path,
        format: str = "json",
    ) -> None:
        """
        Save evaluation results to file.
        
        Args:
            metrics: Evaluation metrics to save.
            output_path: Output file path.
            format: Output format ('json' or 'csv').
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
        
        elif format == "csv":
            import csv
            
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(["Metric", "Value"])
                
                # Overall metrics
                writer.writerow(["char_precision", metrics.char_precision])
                writer.writerow(["char_recall", metrics.char_recall])
                writer.writerow(["char_f1", metrics.char_f1])
                writer.writerow(["p_s", metrics.p_s])
                writer.writerow(["p_o", metrics.p_o])
                writer.writerow(["num_examples", metrics.num_examples])
                
                # Per-benchmark
                writer.writerow([])
                writer.writerow(["Benchmark", "Precision", "Recall", "F1", "N"])
                for name, bm in metrics.per_benchmark.items():
                    writer.writerow([
                        name,
                        bm["precision"],
                        bm["recall"],
                        bm["f1"],
                        bm["num_examples"],
                    ])
        
        logger.info(f"Results saved to {output_path}")


def run_ablation_study(
    pipeline: RAGPipeline,
    benchmark_name: str,
    max_examples: int = 100,
) -> Dict[str, EvaluationMetrics]:
    """
    Run ablation study comparing different steering configurations.
    
    Args:
        pipeline: Base pipeline.
        benchmark_name: Benchmark to evaluate.
        max_examples: Examples per configuration.
        
    Returns:
        Dictionary mapping configuration name to metrics.
    """
    results = {}
    
    data_handler = LegalBenchRAG()
    
    # Baseline (no steering)
    baseline_pipeline = RAGPipeline(
        model_name=pipeline.model_name,
        use_reft=False,
        use_steering=False,
    )
    baseline_pipeline.retriever = pipeline.retriever
    
    evaluator = Evaluator(baseline_pipeline, data_handler)
    results["baseline"] = evaluator.evaluate_benchmark(benchmark_name, max_examples)
    
    # ReFT only
    reft_pipeline = RAGPipeline(
        model_name=pipeline.model_name,
        use_reft=True,
        use_steering=False,
    )
    reft_pipeline.retriever = pipeline.retriever
    reft_pipeline.reft_intervention = pipeline.reft_intervention
    
    evaluator = Evaluator(reft_pipeline, data_handler)
    results["reft_only"] = evaluator.evaluate_benchmark(benchmark_name, max_examples)
    
    # Steering only
    steering_pipeline = RAGPipeline(
        model_name=pipeline.model_name,
        use_reft=False,
        use_steering=True,
    )
    steering_pipeline.retriever = pipeline.retriever
    steering_pipeline.steerer = pipeline.steerer
    
    evaluator = Evaluator(steering_pipeline, data_handler)
    results["steering_only"] = evaluator.evaluate_benchmark(benchmark_name, max_examples)
    
    # Both ReFT and steering
    evaluator = Evaluator(pipeline, data_handler)
    results["reft_and_steering"] = evaluator.evaluate_benchmark(benchmark_name, max_examples)
    
    return results
