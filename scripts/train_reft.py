#!/usr/bin/env python3
"""
Train ReFT interventions for B-PLIS-RAG.

Trains low-dimensional latent interventions to steer the model
toward context-faithful generation.

Usage:
    python scripts/train_reft.py --dataset legalbench --epochs 100
    python scripts/train_reft.py --examples data/benchmarks/conflict_examples.json
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
from src.reft import ReFTIntervention, ReFTTrainer, ReFTHook, verify_intervention
from src.data_handler import LegalBenchRAG, ConflictExample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ReFT interventions for context-faithful generation",
    )
    
    # Data options
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["legalbench", "custom"],
        default="legalbench",
        help="Dataset to train on",
    )
    
    parser.add_argument(
        "--examples",
        type=str,
        help="Path to JSON file with training examples",
    )
    
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to train on",
    )
    
    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="t5-base",
        help="Model name",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    
    # ReFT options
    parser.add_argument(
        "--intervention-dim",
        type=int,
        default=16,
        help="Dimension of intervention latent vector",
    )
    
    parser.add_argument(
        "--layer",
        type=int,
        default=6,
        help="Layer to apply intervention",
    )
    
    # Training options
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--steps-per-example",
        type=int,
        default=100,
        help="Optimization steps per example",
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "reft_latest.pt"),
        help="Output path for trained intervention",
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify intervention changes outputs",
    )
    
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Run ablation study over layers",
    )
    
    return parser.parse_args()


def load_training_examples(
    dataset: str,
    examples_path: str | None,
    num_examples: int,
) -> List[Dict[str, str]]:
    """Load training examples from dataset or file."""
    
    if examples_path:
        logger.info(f"Loading examples from {examples_path}")
        with open(examples_path, "r") as f:
            data = json.load(f)
        
        examples = []
        for item in data[:num_examples]:
            examples.append({
                "query": item.get("query", ""),
                "context": item.get("context", ""),
                "answer": item.get("context_answer", item.get("answer", "")),
            })
        return examples
    
    if dataset == "legalbench":
        logger.info("Creating examples from LegalBench-RAG")
        data_handler = LegalBenchRAG()
        conflicts = data_handler.create_conflict_examples(num_examples=num_examples)
        
        examples = []
        for c in conflicts:
            examples.append({
                "query": c.query,
                "context": c.context,
                "answer": c.context_answer,
            })
        return examples
    
    # Default: create synthetic examples
    logger.info("Using synthetic examples")
    return [
        {
            "query": "What is the definition of confidential information?",
            "context": "Confidential Information means any non-public information disclosed by one party to another.",
            "answer": "Confidential Information means any non-public information disclosed by one party to another.",
        },
        {
            "query": "What is the term of this agreement?",
            "context": "This Agreement shall be effective for a period of three (3) years from the Effective Date.",
            "answer": "Three years from the Effective Date.",
        },
        {
            "query": "Who are the parties to this contract?",
            "context": "This Agreement is entered into between ABC Corporation and XYZ Inc.",
            "answer": "ABC Corporation and XYZ Inc.",
        },
    ] * (num_examples // 3 + 1)


def train_reft(
    model,
    tokenizer,
    examples: List[Dict[str, str]],
    intervention_dim: int,
    target_layer: int,
    learning_rate: float,
    num_steps: int,
    epochs: int,
    device: torch.device,
) -> tuple:
    """Train ReFT intervention."""
    
    # Create intervention
    hidden_size = model.config.d_model
    intervention = ReFTIntervention(
        hidden_size=hidden_size,
        intervention_dim=intervention_dim,
    )
    intervention = intervention.to(device)
    
    logger.info(f"Created intervention: dim={intervention_dim}, hidden={hidden_size}")
    logger.info(f"Intervention parameters: {intervention.num_parameters()}")
    
    # Create trainer
    trainer = ReFTTrainer(
        model=model,
        intervention=intervention,
        tokenizer=tokenizer,
        target_layer=target_layer,
        learning_rate=learning_rate,
        num_steps=num_steps,
        device=device,
    )
    
    # Train
    logger.info(f"Training on {len(examples)} examples for {epochs} epochs")
    results = trainer.train(examples, epochs=epochs, verbose=True)
    
    logger.info(f"Training complete!")
    logger.info(f"  Final avg loss: {results['avg_loss']:.4f}")
    logger.info(f"  Z norm: {results['z_norm']:.4f}")
    
    return intervention, trainer, results


def run_ablation_study(
    model,
    tokenizer,
    examples: List[Dict[str, str]],
    intervention_dim: int,
    learning_rate: float,
    num_steps: int,
    device: torch.device,
) -> Dict[int, float]:
    """Run ablation study over different layers."""
    
    num_layers = len(model.decoder.block)
    logger.info(f"Running ablation over {num_layers} layers")
    
    results = {}
    
    for layer in [0, 3, 6, 9, num_layers - 1]:
        if layer >= num_layers:
            continue
            
        logger.info(f"\n--- Layer {layer} ---")
        
        intervention, trainer, train_results = train_reft(
            model=model,
            tokenizer=tokenizer,
            examples=examples[:10],  # Use fewer examples for ablation
            intervention_dim=intervention_dim,
            target_layer=layer,
            learning_rate=learning_rate,
            num_steps=50,  # Fewer steps for ablation
            epochs=1,
            device=device,
        )
        
        results[layer] = train_results["avg_loss"]
    
    # Print summary
    print("\n" + "=" * 40)
    print("ABLATION STUDY RESULTS")
    print("=" * 40)
    for layer, loss in sorted(results.items()):
        print(f"  Layer {layer}: loss = {loss:.4f}")
    print("=" * 40)
    
    return results


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Setup
    setup_environment()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, device=str(device))
    
    # Load examples
    examples = load_training_examples(
        args.dataset,
        args.examples,
        args.num_examples,
    )
    logger.info(f"Loaded {len(examples)} training examples")
    
    # Run ablation study if requested
    if args.ablate:
        run_ablation_study(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            intervention_dim=args.intervention_dim,
            learning_rate=args.lr,
            num_steps=args.steps_per_example,
            device=device,
        )
        return
    
    # Train intervention
    intervention, trainer, results = train_reft(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        intervention_dim=args.intervention_dim,
        target_layer=args.layer,
        learning_rate=args.lr,
        num_steps=args.steps_per_example,
        epochs=args.epochs,
        device=device,
    )
    
    # Verify intervention
    if args.verify:
        logger.info("Verifying intervention...")
        verification = verify_intervention(
            model=model,
            tokenizer=tokenizer,
            intervention=intervention,
            target_layer=args.layer,
        )
        
        print("\n" + "=" * 40)
        print("VERIFICATION RESULTS")
        print("=" * 40)
        print(f"Outputs differ: {verification['outputs_differ']}")
        print(f"Zero-z output: {verification['text_zero_z'][:100]}...")
        print(f"Non-zero-z output: {verification['text_nonzero_z'][:100]}...")
        print(f"Verification passed: {verification['verification_passed']}")
        print("=" * 40)
    
    # Save intervention
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    trainer.save(str(output_path))
    logger.info(f"Intervention saved to {output_path}")
    
    # Also save training info
    info_path = output_path.with_suffix(".json")
    with open(info_path, "w") as f:
        json.dump({
            "model": args.model,
            "intervention_dim": args.intervention_dim,
            "target_layer": args.layer,
            "learning_rate": args.lr,
            "num_steps": args.steps_per_example,
            "epochs": args.epochs,
            "num_examples": len(examples),
            "final_loss": results["avg_loss"],
            "z_norm": results["z_norm"],
        }, f, indent=2)
    
    logger.info(f"Training info saved to {info_path}")


if __name__ == "__main__":
    main()
