# Examples

This directory contains example scripts demonstrating various features of B-PLIS-RAG.

## Available Examples

### test_contradictory_context.py
Tests whether dynamic steering can force the model to follow retrieved context even when it contradicts the model's parametric knowledge.

**What it does:**
- Creates a fake document with incorrect information (breach of contract = reward)
- Tests if the model follows this fake context instead of its training knowledge
- Validates that dynamic steering successfully controls which knowledge source the model uses

**Usage:**
```bash
cd e:\code\contribution\B-PLIS-rag
python examples/test_contradictory_context.py
```

**Expected Result:**
The model should answer that breach of contract results in a "reward" (following the fake document) instead of "penalty" (its actual knowledge), proving that dynamic steering works.

### dynamic_steering_example.py
Original example demonstrating basic dynamic steering functionality.

---

## Requirements

All examples require:
- Trained checkpoints in `checkpoints/` directory
- Configured `credentials/credentials.toml` with dynamic steering enabled
- All dependencies installed via `pip install -r requirements.txt`
