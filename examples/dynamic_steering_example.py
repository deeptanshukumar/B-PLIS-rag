"""
Example: Dynamic Multi-Layer Steering for B-PLIS-RAG

Demonstrates how to use dynamic layer selection with activation steering
to achieve adaptive context-faithful generation.

Usage:
    python examples/dynamic_steering_example.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import setup_environment, get_config
from src.model_loader import load_model
from src.activation_steering import ActivationSteering
from src.rag_pipeline import RAGPipeline, RAGConfig
from src.data_handler import Document

# Setup
setup_environment()
config = get_config()

print("="*70)
print("Dynamic Multi-Layer Steering Example")
print("="*70)

# Example 1: Computing multi-layer steering vectors
print("\n📊 Example 1: Computing Multi-Layer Steering Vectors\n")

# Sample training examples (with and without context)
examples = [
    {
        "query": "What is force majeure?",
        "context": "Force majeure is a contractual clause that frees parties from liability when an extraordinary event prevents fulfillment of the contract."
    },
    {
        "query": "What does indemnification mean?",
        "context": "Indemnification is a contractual obligation where one party agrees to reimburse another party for losses or damages."
    },
    {
        "query": "What is breach of contract?",
        "context": "Breach of contract occurs when one party fails to fulfill their obligations under the agreement without legal justification."
    },
]

# Load model
print("Loading Flan-T5 model...")
model, tokenizer = load_model("google/flan-t5-base", device="auto")
device = next(model.parameters()).device
print(f"✓ Model loaded on {device}")

# Initialize steerer in dynamic mode
print("\nInitializing dynamic activation steering...")
steerer = ActivationSteering(
    model=model,
    tokenizer=tokenizer,
    layer=6,  # Fallback layer
    component="decoder",
    device=device,
    steering_mode="dynamic",  # Enable dynamic mode
    layer_range=(4, 7),  # Active layer range
    max_steering_steps=60,  # Stop after 60 tokens
    layer_multipliers={
        4: 2.0,  # Strong steering in early layers
        5: 1.5,  # Moderate in middle
        6: 1.5,
        7: 1.0,  # Weak in late layers
    }
)
print("✓ Steerer initialized in dynamic mode")

# Compute steering vectors for multiple layers
print("\nComputing steering vectors for layers [4, 5, 6]...")
steerer.compute_from_examples(examples, layers=[4, 5, 6])
print("✓ Steering vectors computed")

# Display vector statistics
print("\nSteering Vector Statistics:")
for layer_idx, vector in steerer.steering_vectors.items():
    print(f"  Layer {layer_idx}: norm={vector.norm().item():.4f}, "
          f"mean={vector.mean().item():.6f}")

# Save checkpoint
checkpoint_path = "checkpoints/dynamic_steering_example.pt"
Path(checkpoint_path).parent.mkdir(exist_ok=True)
steerer.save(checkpoint_path)
print(f"\n✓ Saved to {checkpoint_path}")


# Example 2: Using dynamic steering in generation
print("\n" + "="*70)
print("📝 Example 2: Dynamic Steering in Generation")
print("="*70 + "\n")

# Test context
test_context = """
A force majeure clause is a contract provision that relieves parties from 
performing their contractual obligations when certain circumstances beyond 
their control arise, making performance inadvisable, commercially impracticable, 
illegal, or impossible. Common force majeure events include acts of God, war, 
terrorism, and pandemics.
"""

test_query = "What is force majeure?"

# Build prompts
positive_prompt = f"Answer based on context: {test_context}\n\nQuestion: {test_query}\n\nAnswer:"
inputs = tokenizer(positive_prompt, return_tensors="pt", truncation=True).to(device)

# Test different retrieval confidence scenarios
scenarios = [
    {"name": "High Confidence", "retrieval_score": 0.9},
    {"name": "Medium Confidence", "retrieval_score": 0.65},
    {"name": "Low Confidence", "retrieval_score": 0.4},
]

print("Testing dynamic layer selection with different retrieval scores:\n")

for scenario in scenarios:
    runtime_state = {"retrieval_score": scenario["retrieval_score"]}
    
    # Determine which layers will be selected
    selected_layers = steerer.select_layers(runtime_state)
    
    print(f"🎯 {scenario['name']} (score={scenario['retrieval_score']:.2f})")
    print(f"   Selected layers: {selected_layers}")
    
    # Generate with steering
    with steerer.apply(multiplier=2.0, runtime_state=runtime_state):
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Answer: {answer}\n")


# Example 3: Using dynamic steering in RAG pipeline
print("="*70)
print("🔄 Example 3: Dynamic Steering in Full RAG Pipeline")
print("="*70 + "\n")

# Create RAG pipeline with dynamic steering
print("Initializing RAG pipeline with dynamic steering...")
pipeline = RAGPipeline(
    model_name="google/flan-t5-base",
    use_steering=True,
    steering_layer=6,
    device=str(device)
)

# Load the computed steering vectors
pipeline.load_steering_vector(checkpoint_path)
print("✓ Steering vectors loaded\n")

# Index some sample documents
documents = [
    Document(
        id="doc1",
        content="Force majeure clauses protect parties from liability when extraordinary events prevent contract fulfillment. These events include natural disasters, wars, and pandemics.",
        source="contract_law_guide.txt"
    ),
    Document(
        id="doc2",
        content="Indemnification clauses require one party to compensate the other for losses or damages arising from specified events or breaches.",
        source="legal_terms.txt"
    ),
]

print("Indexing documents...")
pipeline.index_documents(documents)
print("✓ Documents indexed\n")

# Query with automatic dynamic steering
queries = [
    "What is force majeure in contracts?",
    "Explain indemnification clauses",
]

print("Querying with dynamic steering:\n")
for query in queries:
    print(f"Query: {query}")
    response = pipeline.query(query, top_k=2)
    print(f"Answer: {response.answer}")
    print(f"Top retrieval score: {max(response.scores):.4f}")
    print(f"Metadata: {response.metadata}\n")


# Example 4: Comparing single vs dynamic mode
print("="*70)
print("⚖️  Example 4: Single-Layer vs Dynamic Multi-Layer Comparison")
print("="*70 + "\n")

# Create single-layer steerer for comparison
steerer_single = ActivationSteering(
    model=model,
    tokenizer=tokenizer,
    layer=6,
    device=device,
    steering_mode="single"  # Single-layer mode
)
steerer_single.compute_from_examples(examples[:3])  # Use subset

# Test query
test_query_2 = "What happens in breach of contract?"
test_context_2 = "Breach of contract occurs when a party fails to perform their contractual obligations without legal excuse, potentially leading to damages or specific performance."

prompt = f"Answer based on context: {test_context_2}\n\nQuestion: {test_query_2}\n\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

print("Comparing steering modes:\n")

# Single-layer
print("1️⃣ Single-layer steering (layer 6 only):")
with steerer_single.apply(multiplier=2.0):
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
answer_single = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"   Answer: {answer_single}\n")

# Dynamic multi-layer
print("2️⃣ Dynamic multi-layer steering (adaptive layers):")
runtime_state = {"retrieval_score": 0.75}
with steerer.apply(multiplier=2.0, runtime_state=runtime_state):
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
answer_dynamic = tokenizer.decode(outputs[0], skip_special_tokens=True)
selected = steerer.select_layers(runtime_state)
print(f"   Active layers: {selected}")
print(f"   Answer: {answer_dynamic}\n")


# Example 5: Time-aware steering (decay over generation)
print("="*70)
print("⏱️  Example 5: Time-Aware Steering with Decay")
print("="*70 + "\n")

print("Testing steering decay over generation steps:\n")

# Simulate different generation stages
for step in [0, 20, 40, 60, 80]:
    # Compute effective multipliers at this timestep
    print(f"Generation step {step}:")
    for layer in [4, 5, 6]:
        if layer in steerer.steering_vectors:
            effective_mult = steerer.get_layer_multiplier(layer, 2.0, step)
            print(f"  Layer {layer}: effective_multiplier={effective_mult:.3f}")
    print()

print("💡 Observation: Steering gradually fades to preserve fluency\n")


# Summary
print("="*70)
print("✅ Summary")
print("="*70 + "\n")

print("Dynamic multi-layer steering provides:")
print("  ✓ Adaptive layer selection based on retrieval confidence")
print("  ✓ Time-aware decay to preserve fluency")
print("  ✓ Layer-specific multipliers for fine-grained control")
print("  ✓ Backward compatibility with single-layer mode")
print("  ✓ Safe, reversible hook-based implementation")
print()
print("For more details, see DYNAMIC_STEERING_GUIDE.md")
print("="*70)
