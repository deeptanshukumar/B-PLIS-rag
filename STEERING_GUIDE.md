# Activation Steering Implementation Guide

## Overview

**Activation Steering** is the core research contribution of B-PLIS-RAG. It addresses a fundamental problem in RAG systems: models often ignore retrieved context and rely on parametric (memorized) knowledge instead.

## The Problem

When you provide a language model with retrieved context, it doesn't automatically prioritize that context. The model may:
- Ignore the context entirely
- Mix context with its own memorized knowledge
- Hallucinate information not in the context

This is especially problematic in legal domains where factual accuracy from source documents is critical.

## The Solution: Activation Steering

### Core Idea

Instead of hoping the model uses the context, **we directly steer its activations** to make it prefer context-based generation.

### How It Works

#### Step 1: Compute Steering Vector

We compute a "steering vector" that represents the activation difference between:
- **Positive examples**: Model processing prompts WITH context
- **Negative examples**: Model processing same prompts WITHOUT context

```python
# Pseudo-code
positive_activations = get_activations(prompt_with_context)
negative_activations = get_activations(prompt_without_context)
steering_vector = mean(positive_activations) - mean(negative_activations)
```

This vector captures the activation pattern that corresponds to "using context."

#### Step 2: Apply During Generation

When generating answers, we add the steering vector to decoder activations at layer 6:

```python
hidden_states = hidden_states + (multiplier * steering_vector)
```

This biases the model toward the activation pattern associated with context use.

### Implementation Details

#### Target Location
- **Model component**: Decoder (where generation happens)
- **Layer**: Layer 6 (middle of 12-layer decoder in flan-t5-base)
- **Timing**: Applied at every forward pass during generation

#### Why Layer 6?
- Early enough to influence generation strategy
- Late enough to have semantic understanding
- Empirically found to work best in T5 models

#### Steering Strength
- **Default multiplier**: 2.0
- Higher = stronger context preference
- Lower = more balanced with parametric knowledge

## Usage

### 1. Compute Steering Vector (One-Time)

```bash
# Compute from 200 examples (~5-10 minutes on GPU)
python scripts/compute_steering.py --num-examples 200

# Or use more examples for better quality
python scripts/compute_steering.py --num-examples 500
```

This creates `checkpoints/steering_vector.pt`.

### 2. Use in Pipeline

```python
from src.rag_pipeline import RAGPipeline, RAGConfig

# Initialize with steering enabled
config = RAGConfig(
    model_name="google/flan-t5-base",
    use_steering=True,
    steering_layer=6,
    steering_multiplier=2.0,
)
pipeline = RAGPipeline(config=config)

# Load steering vector
pipeline.load_steering_vector("checkpoints/steering_vector.pt")

# Query - steering automatically applied
response = pipeline.query("What is confidential information?")
```

### 3. Command Line

```bash
# Steering enabled by default
python main.py --query "What is confidential information?"

# Disable steering
python main.py --query "..." --no-steering

# Adjust steering strength
python main.py --query "..." --steering-multiplier 3.0

# Use custom steering vector
python main.py --query "..." --steering-checkpoint my_steering.pt
```

## Technical Implementation

### Key Files

1. **`src/activation_steering.py`**
   - `ActivationSteering` class
   - `compute_steering_vector()` method
   - Hook management for activation capture and modification

2. **`scripts/compute_steering.py`**
   - Standalone script to compute steering vectors
   - Loads examples from LegalBench-RAG
   - Saves computed vector to checkpoint

3. **`src/rag_pipeline.py`**
   - Integration of steering into RAG pipeline
   - Auto-loading of steering checkpoints
   - Context manager for applying steering during generation

### Key Classes

#### `ActivationSteering`

```python
class ActivationSteering:
    def __init__(self, model, tokenizer, layer, component="decoder", device=None):
        """Initialize steering for specific layer."""
        
    def compute_steering_vector(self, positive_prompts, negative_prompts, normalize=True):
        """Compute steering vector from prompt pairs."""
        
    def apply(self, multiplier=2.0):
        """Context manager to apply steering during generation."""
        
    def save(self, path):
        """Save steering vector to file."""
        
    def load(self, path):
        """Load steering vector from file."""
```

### Example Flow

```python
# 1. Compute steering vector (once)
steerer = ActivationSteering(model, tokenizer, layer=6)
steerer.compute_steering_vector(
    positive_prompts=["Context: X. Question: Y"],
    negative_prompts=["Question: Y"]
)
steerer.save("steering.pt")

# 2. Load and use
steerer.load("steering.pt")
with steerer.apply(multiplier=2.0):
    output = model.generate(input_ids)  # Steering automatically applied
```

## Research Context

### Inspiration

Based on ContextFocus methodology (arXiv:2601.04131) - using activation steering to enhance context faithfulness in RAG systems.

### Key Differences from Other Approaches

| Approach | Steering | Other Methods |
|----------|----------|---------------|
| **Modification** | Runtime activation addition | Fine-tuning all weights |
| **Parameters** | 0 trainable (vector precomputed) | Millions of parameters |
| **Compute** | One-time computation | Repeated training |
| **Reversibility** | Fully reversible (remove vector) | Permanent weight changes |
| **Flexibility** | Per-task steering vectors | Single model for all tasks |

### Complementary Techniques

- **ReFT (Representation Fine-Tuning)**: Trains low-rank intervention (12K params)
- **Instruction Tuning**: Flan-T5 understands Q&A format
- **Semantic Retrieval**: FAISS finds relevant context

All three work together for maximum context faithfulness.

## Performance Impact

### Quality

- **Without steering**: Model often ignores context, gives generic answers
- **With steering**: Model generates from provided context, cites sources

### Example

**Query**: "What is confidential information?"

**Retrieved Context**: "University of Nebraska policy states..."

**Without Steering**:
```
Answer: "sensitive"
```

**With Steering**:
```
Answer: "Every person who has been given access to a University of 
Nebraska information system or who has credits with confidential or 
sensitive information are obligated to keep such data confidential."
```

### Speed

- **Steering computation**: One-time, ~5-10 min for 200 examples on GPU
- **Inference overhead**: Negligible (~1% slower due to vector addition)

## Best Practices

### 1. Steering Vector Quality

- Use **200+ examples** for stable vectors
- Sample **diverse queries and contexts**
- Use examples from **target domain** (legal documents for legal RAG)

### 2. Hyperparameters

- **Layer**: 6 works well for flan-t5-base (12 layers)
- **Multiplier**: Start with 2.0, adjust based on results
  - Too high: Overly literal, repetitive
  - Too low: Insufficient context preference

### 3. Combination with ReFT

- Steering: Runtime activation modification
- ReFT: Learned intervention (trained)
- Use both for best results

### 4. Debugging

```python
# Check if steering vector loaded
if pipeline.steerer and pipeline.steerer.steering_vector is not None:
    print("✓ Steering active")
    print(f"  Norm: {pipeline.steerer.steering_vector.norm():.4f}")
else:
    print("✗ No steering vector loaded")

# Test with/without steering
response_with = pipeline.query("...", use_steering=True)
response_without = pipeline.query("...", use_steering=False)
```

## Troubleshooting

### "No steering vector found"

**Solution**: Run `python scripts/compute_steering.py --num-examples 200`

### "Steering not enabled"

**Solution**: Initialize with `use_steering=True` or remove `--no-steering` flag

### "Steering has no effect"

**Possible causes**:
1. Vector not loaded (check logs)
2. Multiplier too low (try increasing to 3.0)
3. Wrong layer (verify layer 6 for flan-t5-base)

### Out of memory during steering computation

**Solution**: 
- Reduce `--num-examples` (try 100)
- Use smaller batch size (edit script)
- Use CPU if GPU memory limited

## Future Work

### Potential Improvements

1. **Task-specific steering**: Different vectors for different query types
2. **Dynamic multiplier**: Adjust based on context confidence
3. **Multi-layer steering**: Apply at multiple decoder layers
4. **Learned multipliers**: Train optimal steering strength

### Research Directions

1. **Generalization**: Test on other domains (medical, financial)
2. **Model scaling**: Evaluate on larger models (flan-t5-xl, flan-t5-xxl)
3. **Interpretability**: Analyze what steering vector represents
4. **Combination**: Integrate with other faithfulness techniques

## References

1. **ContextFocus** (arXiv:2601.04131) - Activation steering for RAG
2. **ReFT** - Representation fine-tuning for parameter-efficient adaptation
3. **LegalBench-RAG** - Legal domain RAG benchmark
4. **Flan-T5** - Instruction-tuned T5 models

## Summary

**Activation Steering** is B-PLIS-RAG's core innovation:
- ✅ Forces model to use retrieved context
- ✅ Zero trainable parameters (precomputed vector)
- ✅ Fully reversible and flexible
- ✅ Works synergistically with ReFT
- ✅ Dramatically improves answer faithfulness

**Key takeaway**: Instead of hoping the model uses context, we steer its internal representations to make context use the natural path of generation.

---

**For more details, see**:
- `src/activation_steering.py` - Implementation
- `scripts/compute_steering.py` - Computation script
- `README.md` - User guide
- `PROJECT_OVERVIEW.md` - Research overview
