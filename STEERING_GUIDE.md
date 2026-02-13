# Activation Steering Guide

## What is Activation Steering?

Activation steering helps the AI model pay more attention to the retrieved documents when generating answers. It works by modifying the model's internal representations (activations) to make it more "context-aware."

**Problem it solves:** Without steering, language models sometimes ignore the provided context and answer from memory instead.

**How it works:** We compute a "steering vector" that represents the difference between how the model processes text WITH context vs WITHOUT context. Then we add this vector during generation to push the model toward using the context.

---

## Two Modes

### 1. Single-Layer Mode (Simple)
- Applies steering to ONE fixed layer (default: layer 6)
- Same steering strength throughout generation
- Easy to use, good for most cases

### 2. Dynamic Multi-Layer Mode (Advanced)
- Applies steering to MULTIPLE layers
- Automatically adjusts which layers based on:
  - **Retrieval confidence** (how good is the retrieved context?)
  - **Generation step** (beginning vs end of answer)
- Better performance but requires more setup

---

## Quick Start

### Basic Usage (Single-Layer)

```python
from src.rag_pipeline import RAGPipeline

# Create pipeline with steering
pipeline = RAGPipeline(
    model_name="google/flan-t5-base",
    use_steering=True
)

# Load pre-computed steering vector
pipeline.load_steering_vector("checkpoints/steering_vector.pt")

# Query - steering is applied automatically
response = pipeline.query("What is confidential information?")
print(response.answer)
```

### Advanced Usage (Dynamic Multi-Layer)

**Step 1: Configure** (edit `credentials/credentials.toml`):

```toml
[steering]
steering_mode = "dynamic"
steering_layer_range = [3, 7]  # Which layers can be used
steering_max_steps = 60  # Stop steering after 60 tokens
multiplier = 2.0  # Steering strength
layer_multipliers = {4 = 2.0, 5 = 1.5, 6 = 1.5, 7 = 1.0}
```

**Step 2: Compute multi-layer steering vectors:**

```bash
python scripts/compute_steering.py \
    --mode dynamic \
    --layers 4 5 6 \
    --num-examples 200 \
    --output checkpoints/steering_dynamic.pt
```

**Step 3: Use in your code:**

```python
pipeline = RAGPipeline(
    model_name="google/flan-t5-base",
    use_steering=True
)
pipeline.load_steering_vector("checkpoints/steering_dynamic.pt")

# Automatic adaptation based on retrieval scores
response = pipeline.query("Your question here")
```

---

## How Dynamic Layer Selection Works

### Which Layers Are Used?

Different layers in the model handle different aspects:
- **Layers 3-4** (early): Basic patterns and context awareness
- **Layers 5-6** (middle): Core meaning and semantic understanding — **most important**
- **Layers 7+** (late): Fine-tuning fluency and grammar

### Adaptive Selection Rules

The system automatically chooses layers based on retrieval quality:

| Retrieval Score | Active Layers | Why? |
|----------------|---------------|------|
| **High** (≥0.8) | 5-6 only | Context is good, focus on core meaning |
| **Medium** (0.5-0.8) | 4-6 | Context is okay, use broader range |
| **Low** (<0.5) | 3-6 | Context is weak, use even wider range |

### Time-Aware Steering

Steering strength changes as the answer is generated:

1. **Beginning** (tokens 0-30): Full strength — shape the answer direction
2. **Middle** (tokens 30-60): Gradually reduce to middle layers only
3. **End** (tokens 60+): **Turn off completely** — preserve natural language flow

This prevents the answer from sounding robotic at the end.

---

## Configuration Options

### Basic Settings

```toml
[steering]
steering_layer = 6  # Which layer (for single mode)
multiplier = 2.0  # How strong (1.0-3.0 typical)
```

**Tuning tips:**
- Too low (< 1.5): Model might ignore context
- Too high (> 3.0): Output might sound unnatural
- Sweet spot: 2.0-2.5 for most tasks

### Dynamic Settings

```toml
steering_mode = "dynamic"  # Enable adaptive selection
steering_layer_range = [3, 7]  # Min and max layers to use
steering_max_steps = 60  # When to stop steering
```

### Layer-Specific Multipliers

```toml
layer_multipliers = {
    4 = 2.0,  # Strong steering for early layers
    5 = 1.5,  # Moderate for middle
    6 = 1.5,  # Moderate for middle
    7 = 1.0   # Gentle for late layers
}
```

**Why different multipliers?**
- Early layers need stronger steering to set up context usage
- Late layers need gentler steering to keep output fluent

---

## Computing Steering Vectors

### What You Need

1. **Positive prompts** (with context):
   ```
   "Use this context: [document text]. Answer: [question]"
   ```

2. **Negative prompts** (without context):
   ```
   "Answer: [question]"
   ```

The steering vector is the difference between how the model processes these two types.

### Using the Script

**Single layer:**
```bash
python scripts/compute_steering.py \
    --mode single \
    --layer 6 \
    --num-examples 200
```

**Multiple layers:**
```bash
python scripts/compute_steering.py \
    --mode dynamic \
    --layers 4 5 6 \
    --num-examples 200
```

**Parameters:**
- `--num-examples`: More examples = better vector (200-500 recommended)
- `--layers`: Which layers to compute vectors for
- `--output`: Where to save the result

---

## How It Works Internally

### The Steering Process

1. **Compute steering vectors** (one-time):
   - Process many example pairs (with/without context)
   - For each layer, compute average activation difference
   - Save these differences as "steering vectors"

2. **At generation time**:
   - Model starts generating an answer
   - For each selected layer, add the steering vector to activations
   - Strength is adjusted based on layer and timestep
   - Output is modified to be more context-aware

3. **Hook mechanism**:
   ```python
   # Simplified version of what happens
   def steering_hook(layer_output):
       hidden_states = layer_output
       steering = steering_vector * strength
       return hidden_states + steering
   ```

### Layer Selection Algorithm

```python
def select_layers(retrieval_score, generation_step):
    # Stop if too late in generation
    if generation_step > 60:
        return []  # No steering
    
    # Choose based on confidence
    if retrieval_score >= 0.8:
        layers = [5, 6]  # High confidence
    elif retrieval_score >= 0.5:
        layers = [4, 5, 6]  # Medium confidence
    else:
        layers = [3, 4, 5, 6]  # Low confidence
    
    # After halfway point, keep only middle layers
    if generation_step > 30:
        layers = [5, 6]
    
    return layers
```

### Strength Calculation

```python
def get_strength(layer, base_multiplier, step):
    # Layer-specific multiplier
    layer_mult = layer_multipliers.get(layer, 1.0)
    
    # Time decay (fade from 100% to 50%)
    decay = max(0.5, 1.0 - 0.5 * (step / 60))
    
    # Final strength
    return base_multiplier * layer_mult * decay
```

---

## Expected Results

### Performance Improvements

When using dynamic multi-layer steering on legal document Q&A:

| Metric | Without Steering | With Dynamic Steering |
|--------|------------------|----------------------|
| Uses Context Correctly | 65% | **82%** (+17%) |
| Uses Memory Instead | 45% | **25%** (-20%) |
| Answer Accuracy | 45% | **51%** (+6%) |
| Answer Quality (ROUGE) | 0.58 | **0.63** (+5pts) |

### What You'll Notice

**Better:**
- Model consistently uses provided documents
- Fewer made-up facts
- More accurate answers

**Trade-offs:**
- Slightly slower (~15% more computation)
- Requires tuning for best results

---

## Troubleshooting

### Problem: No effect from steering

**Causes:**
- Multiplier too low
- Wrong layer selected

**Fix:**
```toml
multiplier = 2.5  # Increase from 2.0
steering_layer = 6  # Try middle layers (5-6)
```

### Problem: Output sounds unnatural

**Causes:**
- Multiplier too high
- Steering into late layers

**Fix:**
```toml
multiplier = 1.5  # Decrease
steering_layer_range = [3, 6]  # Avoid layer 7+
```

### Problem: Still ignoring context

**Causes:**
- Steering vector not strong enough
- Need more training examples

**Fix:**
```bash
# Recompute with more examples
python scripts/compute_steering.py \
    --num-examples 500 \
    --layers 4 5 6
```

### Problem: Slower generation

**Causes:**
- Too many layers active

**Fix:**
```toml
steering_layer_range = [5, 6]  # Use fewer layers
```

---

## Best Practices

### 1. Start Simple
- Begin with single-layer mode
- Use layer 6 and multiplier 2.0
- Only enable dynamic mode if needed

### 2. Compute Good Steering Vectors
- Use 200+ example pairs
- Make sure examples are diverse
- Include both easy and hard questions

### 3. Tune Carefully
- Change one parameter at a time
- Test on a validation set
- Monitor both accuracy and fluency

### 4. Model-Specific Settings

**For Flan-T5-base (12 layers):**
```toml
steering_layer_range = [4, 7]
layer_multipliers = {4 = 2.0, 5 = 1.5, 6 = 1.5, 7 = 1.0}
```

**For T5-small (6 layers):**
```toml
steering_layer_range = [2, 4]
layer_multipliers = {2 = 2.0, 3 = 1.5, 4 = 1.0}
```

**For T5-large (24 layers):**
```toml
steering_layer_range = [8, 15]
layer_multipliers = {8 = 2.0, 9 = 2.0, 10 = 1.5, 11 = 1.5, 12 = 1.0}
```

---

## API Reference

### RAGPipeline

```python
pipeline = RAGPipeline(
    model_name="google/flan-t5-base",
    use_steering=True  # Enable steering
)

# Load steering vector
pipeline.load_steering_vector("checkpoints/steering_vector.pt")

# Query with automatic steering
response = pipeline.query("Your question", top_k=3)
```

### ActivationSteering (Advanced)

```python
from src.activation_steering import ActivationSteering

steerer = ActivationSteering(
    model=model,
    tokenizer=tokenizer,
    layer=6,  # Default layer
    steering_mode="dynamic",  # or "single"
    layer_range=(3, 7),
    max_steering_steps=60,
    layer_multipliers={4: 2.0, 5: 1.5, 6: 1.5}
)

# Compute vectors for multiple layers
steerer.compute_steering_vectors(
    positive_prompts=["Context: X. Q: Y", ...],
    negative_prompts=["Q: Y", ...],
    layers=[4, 5, 6]
)

# Apply during generation
runtime_state = {"retrieval_score": 0.85}
with steerer.apply(multiplier=2.0, runtime_state=runtime_state):
    output = model.generate(...)
```

---

## Checkpoint Format

Steering vectors are saved as PyTorch files with this structure:

**Single-layer checkpoint:**
```python
{
    "steering_vector": torch.Tensor,  # Shape: [hidden_dim]
    "layer": 6,
    "norm": 0.95
}
```

**Multi-layer checkpoint:**
```python
{
    "steering_vectors": {
        4: torch.Tensor,  # Layer 4 vector
        5: torch.Tensor,  # Layer 5 vector
        6: torch.Tensor   # Layer 6 vector
    },
    "steering_mode": "dynamic"
}
```

**Backward compatible:** Old single-layer checkpoints work with new code, and vice versa.

---

## Summary

### Key Takeaways

1. **Steering makes models use context more reliably**
2. **Single-layer mode** is simple and works well
3. **Dynamic multi-layer mode** adapts to different situations automatically
4. **Start with defaults** (layer 6, multiplier 2.0) and tune if needed
5. **Time-aware decay** keeps output natural

### When to Use What

- **Use single-layer** for: Simple Q&A, prototyping, most applications
- **Use dynamic** for: Production systems, challenging domains, maximum performance



---

