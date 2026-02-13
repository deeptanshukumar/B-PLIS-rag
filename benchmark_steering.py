"""
Benchmark test showing activation steering impact on answer generation.
Compares answers with and without steering on real benchmark questions.
"""
import os
import warnings
import logging

# Suppress verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import json
from pathlib import Path
from src.rag_pipeline import RAGPipeline, RAGConfig
from src.data_handler import LegalBenchRAG
import textwrap

def format_text(text, width=90):
    """Wrap text to specified width."""
    lines = []
    for line in text.split('\n'):
        if line.strip():
            lines.extend(textwrap.wrap(line, width=width))
        else:
            lines.append('')
    return '\n   '.join(lines)

print("=" * 100)
print("                 ACTIVATION STEERING BENCHMARK EVALUATION")
print("=" * 100)

# Load benchmark
print("\n⏳ Loading benchmark data...")
benchmark_path = Path("data/benchmarks/contractnli.json")
with open(benchmark_path, encoding='utf-8') as f:
    benchmark_data = json.load(f)

# Select a few examples
num_examples = 3
examples = benchmark_data['tests'][:num_examples]

# Load corpus
print("⏳ Loading corpus...")
data = LegalBenchRAG()
corpus = data.load_corpus(['contractnli'])

# Prepare results
results = []

for config_name, use_steering in [("WITHOUT STEERING", False), ("WITH STEERING", True)]:
    print(f"\n{'=' * 100}")
    print(f"{'=' * 100}")
    print(f"                           {config_name}")
    print(f"{'=' * 100}")
    print(f"{'=' * 100}")
    
    # Create pipeline
    print(f"\n⏳ Initializing pipeline ({config_name.lower()})...")
    pipeline = RAGPipeline(
        config=RAGConfig(
            model_name='google/flan-t5-base',
            use_reft=True,
            use_steering=use_steering,
            steering_multiplier=2.0
        )
    )
    
    pipeline.load_reft_checkpoint('checkpoints/reft_flant5_full.pt')
    if use_steering:
        pipeline.load_steering_vector('checkpoints/steering_vector.pt')
        print(f"✓ Steering enabled (layer {pipeline.steerer.layer}, multiplier {pipeline.steering_multiplier}x)")
    else:
        print(f"✓ Baseline mode (no steering)")
    
    # Index documents
    print("⏳ Indexing documents...")
    pipeline.index_documents(corpus['contractnli'])
    
    # Evaluate examples
    for i, example in enumerate(examples, 1):
        query = example['query']
        gold_answer = example['snippets'][0]['answer']
        
        print(f"\n{'-' * 100}")
        print(f"EXAMPLE {i}/{num_examples}")
        print(f"{'-' * 100}")
        
        print(f"\n❓ Query:")
        print(f"   {format_text(query)}")
        
        # Generate answer
        response = pipeline.query(query, top_k=3)
        
        print(f"\n📊 Retrieval: {len(response.sources)} documents (top score: {response.scores[0]:.4f})")
        
        print(f"\n✅ Generated Answer:")
        print(f"   {format_text(response.answer)}")
        
        print(f"\n📖 Gold Standard:")
        print(f"   {format_text(gold_answer[:200])}...")
        
        # Store result
        results.append({
            'config': config_name,
            'example_id': i,
            'query': query,
            'generated': response.answer,
            'gold': gold_answer,
            'retrieval_score': float(response.scores[0])
        })

# Summary
print(f"\n{'=' * 100}")
print("SUMMARY")
print(f"{'=' * 100}")

print("\nComparing answers between baseline and steered configurations:")
for i in range(1, num_examples + 1):
    baseline_result = [r for r in results if r['example_id'] == i and 'WITHOUT' in r['config']][0]
    steered_result = [r for r in results if r['example_id'] == i and 'WITH' in r['config']][0]
    
    print(f"\n[Example {i}]")
    print(f"  Query: {baseline_result['query'][:80]}...")
    print(f"\n  Baseline:  {baseline_result['generated'][:100]}...")
    print(f"  Steered:   {steered_result['generated'][:100]}...")
    
    if baseline_result['generated'] != steered_result['generated']:
        print(f"  ✓ DIFFERENT - Steering changed the output")
    else:
        print(f"  = SAME - Both produced identical output")

print(f"\n{'=' * 100}")
print("Activation steering test complete!")
print(f"Evaluated {num_examples} examples on contractnli benchmark")
print(f"{'=' * 100}")

# Save results
output_path = Path("outputs/steering_benchmark_results.json")
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nDetailed results saved to: {output_path}")
