#!/usr/bin/env python3
"""
Compute Activation Steering Vectors for B-PLIS-RAG.

This is a core component of the research - computing steering vectors
that bias the model toward using retrieved context rather than parametric knowledge.

The steering vector is computed as the mean activation difference between:
- Positive examples: prompts WITH context
- Negative examples: same prompts WITHOUT context

This vector is then added to decoder activations during generation to
enforce context faithfulness.

Usage:
    python scripts/compute_steering.py --dataset legalbench --num-examples 200
    python scripts/compute_steering.py --examples data/benchmarks/conflict_examples.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, setup_environment, PROJECT_ROOT
from src.model_loader import load_model
from src.activation_steering import ActivationSteering
from src.data_handler import LegalBenchRAG, ConflictExample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute activation steering vectors for context-faithful generation",
    )
    
    # Data options
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["legalbench", "custom"],
        default="legalbench",
        help="Dataset to use for steering vector computation",
    )
    
    parser.add_argument(
        "--examples",
        type=str,
        help="Path to JSON file with examples",
    )
    
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Number of examples to use for steering computation",
    )
    
    parser.add_argument(
        "--corpus",
        type=str,
        nargs="+",
        default=["contractnli", "cuad", "maud"],
        help="Corpus types to use for examples",
    )
    
    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-base",
        help="Model name",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    
    # Steering options
    parser.add_argument(
        "--layer",
        type=int,
        default=6,
        help="Layer to apply steering (default: 6, used in single mode)",
    )
    
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Multiple layers for dynamic steering (e.g., --layers 4 5 6)",
    )
    
    parser.add_argument(
        "--component",
        type=str,
        choices=["encoder", "decoder"],
        default="decoder",
        help="Model component to steer",
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize steering vector",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "dynamic"],
        default="single",
        help="Steering mode: single-layer or dynamic multi-layer",
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/steering_vector.pt",
        help="Output path for steering vector",
    )
    
    parser.add_argument(
        "--metadata",
        type=str,
        help="Output path for metadata (default: output.json)",
    )
    
    return parser.parse_args()


def load_examples_from_file(file_path: str) -> List[Dict[str, str]]:
    """Load examples from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        examples = data
    elif isinstance(data, dict) and "tests" in data:
        examples = data["tests"]
    elif isinstance(data, dict) and "examples" in data:
        examples = data["examples"]
    else:
        raise ValueError(f"Unknown format in {file_path}")
    
    logger.info(f"Loaded {len(examples)} examples from {file_path}")
    return examples


def create_steering_examples(
    data_handler: LegalBenchRAG,
    corpus_types: List[str],
    num_examples: int,
) -> List[Dict[str, str]]:
    """
    Create examples for steering computation from LegalBench-RAG.
    
    Args:
        data_handler: LegalBench-RAG data handler.
        corpus_types: Corpus types to sample from.
        num_examples: Number of examples to create.
        
    Returns:
        List of dicts with 'query' and 'context' keys.
    """
    logger.info(f"Creating {num_examples} steering examples from {corpus_types}")
    
    # Load benchmarks
    all_examples = []
    for corpus_type in corpus_types:
        try:
            benchmarks = data_handler.load_benchmarks([corpus_type])
            if corpus_type in benchmarks:
                all_examples.extend(benchmarks[corpus_type])
        except Exception as e:
            logger.warning(f"Failed to load {corpus_type} benchmarks: {e}")
    
    if not all_examples:
        raise ValueError("No examples loaded from benchmarks")
    
    logger.info(f"Loaded {len(all_examples)} total benchmark examples")
    
    # Sample examples
    import random
    random.seed(42)
    
    if len(all_examples) > num_examples:
        sampled = random.sample(all_examples, num_examples)
    else:
        sampled = all_examples
        logger.warning(f"Only {len(sampled)} examples available (requested {num_examples})")
    
    # Convert to steering format
    steering_examples = []
    for ex in sampled:
        # Extract query
        query = ex.query if hasattr(ex, 'query') else str(ex)
        if not query or len(query.strip()) == 0:
            continue
        
        # Get context from ground_truth_snippets
        if hasattr(ex, 'ground_truth_snippets') and ex.ground_truth_snippets:
            # Use answer from first snippet as context
            snippet = ex.ground_truth_snippets[0]
            if isinstance(snippet, dict) and 'answer' in snippet:
                context = snippet['answer']
            else:
                continue
        else:
            # Skip examples without context
            continue
        
        # Ensure we have meaningful context
        if not context or len(context.strip()) < 20:
            continue
        
        # Truncate context if too long
        if len(context) > 800:
            context = context[:800] + "..."
        
        steering_examples.append({
            'query': query,
            'context': context,
        })
    
    logger.info(f"Created {len(steering_examples)} steering examples")
    return steering_examples


def compute_steering_vector(
    model,
    tokenizer,
    examples: List[Dict[str, str]],
    layer: int,
    component: str,
    normalize: bool,
    device: torch.device,
    layers: Optional[List[int]] = None,
    mode: str = "single",
) -> tuple:
    """
    Compute steering vector(s) from examples.
    
    Args:
        model: T5 model.
        tokenizer: Tokenizer.
        examples: List of dicts with 'query' and 'context'.
        layer: Default target layer (for single mode).
        component: Model component ('encoder' or 'decoder').
        normalize: Whether to normalize vector.
        device: Device for computation.
        layers: List of layers for multi-layer mode.
        mode: "single" or "dynamic".
        
    Returns:
        Tuple of (steering_vector or dict, steerer object).
    """
    if mode == "dynamic" and layers:
        logger.info(f"Computing steering vectors for {len(layers)} layers: {layers}")
    else:
        logger.info(f"Computing steering vector from {len(examples)} examples")
        logger.info(f"Target: {component} layer {layer}")
    
    # Initialize activation steering
    steerer = ActivationSteering(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        component=component,
        device=device,
        steering_mode=mode,
        layer_range=(min(layers), max(layers)) if layers else (3, 7),
    )
    
    # Build prompt pairs
    positive_prompts = []
    negative_prompts = []
    
    for ex in examples:
        query = ex['query']
        context = ex['context']
        
        # Positive: with context (this is what we want to encourage)
        positive_prompts.append(
            f"Answer based on context: {context}\n\nQuestion: {query}\n\nAnswer:"
        )
        
        # Negative: without context (parametric knowledge only)
        negative_prompts.append(
            f"Question: {query}\n\nAnswer:"
        )
    
    # Compute steering vector(s)
    if mode == "dynamic" and layers:
        # Multi-layer computation
        steering_vectors = steerer.compute_steering_vectors(
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            layers=layers,
            normalize=normalize,
        )
        
        logger.info(f"Steering vectors computed successfully for {len(steering_vectors)} layers")
        for layer_idx, vec in steering_vectors.items():
            logger.info(f"  Layer {layer_idx}: norm={vec.norm().item():.4f}")
        
        return steering_vectors, steerer
    else:
        # Single-layer computation
        steering_vector = steerer.compute_steering_vector(
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            normalize=normalize,
        )
        
        logger.info(f"Steering vector computed successfully")
        logger.info(f"  Shape: {steering_vector.shape}")
        logger.info(f"  Norm: {steering_vector.norm().item():.4f}")
        logger.info(f"  Mean: {steering_vector.mean().item():.6f}")
        logger.info(f"  Std: {steering_vector.std().item():.6f}")
        
        return steering_vector, steerer


def main():
    """Main function."""
    args = parse_args()
    
    # Setup
    setup_environment()
    config = get_config()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, device=device)
    model.eval()
    
    # Load examples
    if args.examples:
        # Load from file
        examples = load_examples_from_file(args.examples)
        # Convert to expected format if needed
        if examples and 'query' not in examples[0]:
            # Try common field mappings
            formatted_examples = []
            for ex in examples:
                formatted_examples.append({
                    'query': ex.get('question', ex.get('query', '')),
                    'context': ex.get('context', ex.get('snippet', '')),
                })
            examples = formatted_examples
    else:
        # Generate from LegalBench-RAG
        data_handler = LegalBenchRAG()
        examples = create_steering_examples(
            data_handler=data_handler,
            corpus_types=args.corpus,
            num_examples=args.num_examples,
        )
    
    if not examples:
        logger.error("No examples to compute steering vector")
        return
    
    # Limit to num_examples
    if len(examples) > args.num_examples:
        examples = examples[:args.num_examples]
    
    logger.info(f"Using {len(examples)} examples for steering computation")
    
    # Determine layers to compute
    compute_layers = args.layers if args.layers else None
    
    # Compute steering vector(s)
    result, steerer = compute_steering_vector(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        layer=args.layer,
        component=args.component,
        normalize=args.normalize,
        device=device,
        layers=compute_layers,
        mode=args.mode,
    )
    
    # Save steering vector(s)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save using steerer's save method (supports both single and multi-layer)
    steerer.save(str(output_path))
    logger.info(f"Saved steering vector(s) to {output_path}")
    
    # Save metadata
    metadata_path = Path(args.metadata) if args.metadata else output_path.with_suffix('.json')
    
    # Build metadata
    if args.mode == "dynamic" and args.layers:
        # Multi-layer metadata
        metadata = {
            'model_name': args.model,
            'mode': args.mode,
            'layers': args.layers,
            'component': args.component,
            'num_examples': len(examples),
            'normalized': args.normalize,
            'dataset': args.dataset,
            'corpus_types': args.corpus if args.dataset == "legalbench" else None,
            'layer_vectors': {
                str(layer): {
                    'norm': result[layer].norm().item(),
                    'mean': result[layer].mean().item(),
                    'std': result[layer].std().item(),
                }
                for layer in result
            }
        }
    else:
        # Single-layer metadata
        metadata = {
            'model_name': args.model,
            'mode': args.mode,
            'layer': args.layer,
            'component': args.component,
            'num_examples': len(examples),
            'normalized': args.normalize,
            'vector_norm': result.norm().item(),
            'vector_mean': result.mean().item(),
            'vector_std': result.std().item(),
            'dataset': args.dataset,
            'corpus_types': args.corpus if args.dataset == "legalbench" else None,
        }
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("✓ Steering vector computation complete!")
    
    # Print usage instructions
    print("\n" + "="*60)
    if args.mode == "dynamic" and args.layers:
        print("MULTI-LAYER STEERING VECTORS COMPUTED SUCCESSFULLY!")
        print("="*60)
        print(f"\nSteering vectors saved to: {output_path}")
        print(f"Layers: {args.layers}")
    else:
        print("STEERING VECTOR COMPUTED SUCCESSFULLY!")
        print("="*60)
        print(f"\nSteering vector saved to: {output_path}")
        print(f"Layer: {args.layer}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nTo use this steering vector:")
    print(f"  python main.py --query '...' --steering-checkpoint {output_path}")
    print(f"\nOr load in Python:")
    print(f"  pipeline.load_steering_vector('{output_path}')")
    print("="*60)


if __name__ == "__main__":
    main()
