# ContextFocus Dynamic Steering: Comprehensive Documentation

**Version:** 0.1.0  
**Model Support:** Google Gemma 3 4B, Llama 3.1 8B, and other Hugging Face causal LMs  
**Python:** >=3.9

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Quick Start Guide](#quick-start-guide)
4. [Architecture Overview](#architecture-overview)
5. [Core Modules Documentation](#core-modules-documentation)
   - [Utils Module](#utils-module)
   - [Data Loaders](#data-loaders)
   - [Prompting Templates](#prompting-templates)
   - [Evaluation Module](#evaluation-module)
   - [Steering Module](#steering-module)
     - [Householder Rotation Module](#householder-rotation-module-new)
     - [HouseholderSteerer](#householdersteerer-new)
   - [Discriminative Layer Filter](#discriminative-layer-filter-new)
   - [Inference Module](#inference-module)
     - [B-PLIS (Budgeted Latent Vector Search)](#b-plis-budgeted-latent-vector-search-new)
   - [ReFT Integration](#reft-integration)
6. [Scripts Documentation](#scripts-documentation)
7. [Tests](#tests)
8. [Experimental Results](#experimental-results)
9. [Design Decisions and Limitations](#design-decisions-and-limitations)
10. [Future Work](#future-work)
11. [References](#references)

---

## Project Overview

### What is ContextFocus?

**ContextFocus** is a research project that implements **activation steering** to improve **contextual faithfulness** in large language models (LLMs). The goal is to make models prioritize the retrieved context over their parametric knowledge when answering questions, which is critical for RAG (Retrieval-Augmented Generation) systems.

### What This Repository Does

This repository provides:

1. **ContextFocus Vector Construction** - Builds steering vectors for every transformer layer by contrasting:
   - **Positive prompts**: System instruction + context + question
   - **Negative prompts**: Question only (no context)

2. **Static Layer Selection** - Reproduces the original ContextFocus paper by finding the single best layer for steering via a held-out sweep.

3. **Dynamic Layer Selection (Novel)** - Implements **Bayesian Causal Influence Layer Selection (BCILS)**, a new per-query dynamic layer selection strategy that:
   - Uses logit-sensitivity gradients instead of representation-space alignment
   - Incorporates a Bayesian prior from layer sweep results
   - Falls back to the best static layer when confidence is low

4. **Conflict Detection + Budgeted Search (Novel)** - A two-stage inference pipeline:
   - Stage 1: Detect knowledge conflicts via Jensen-Shannon divergence
   - Stage 2: If conflict detected, search over candidate (layer, multiplier) configurations within a strict budget
   - Uses grounding mass (probability on context tokens) as the selection criterion

5. **PyReFT Baseline** - Integration with Parameter-Efficient Fine-Tuning (PEFT) via Representation Fine-Tuning (ReFT).

6. **Householder / Norm-Preserving Rotation Injection (New)** - A drop-in alternative to additive steering:
   - Rotates the last-token hidden state toward the steering vector **without changing its norm**
   - Avoids activation-magnitude drift that can destabilise decoding at high multipliers
   - Implements a 2-D Givens rotation in the plane spanned by `v` and `h⊥`

7. **Discriminative Sign-Based Layer Filtering (New)** - A structural pre-filter for BCILS:
   - Identifies layers where the projections of positive and negative mean activations onto a global feature direction have **opposite signs**
   - These are the layers where memory vs. context representations are structurally separated
   - Integrated into `InfluenceLayerSelector.choose_layer()` to restrict candidates before scoring

8. **B-PLIS — Budgeted PLIS / CMA-ES Latent Vector Search (New)** - Query-specific Δh synthesis:
   - Searches a low-dimensional intrinsic subspace (default: 64-D via QR-orthonormal projection) using CMA-ES
   - Optimises a grounding proxy (probability mass on context tokens + entropy) through short probe generations
   - Produces a **query-specific perturbation Δh** that is injected via `HouseholderSteerer`
   - Integrated into `budgeted_latent_activation_search()` as an optional search backend alongside the original static-vector path

### Key Innovations

- **Causal Dynamic Selection**: Unlike the original alignment-based heuristic (which measures representation change), our BCILS selector measures **how much a layer's residual intervention causally affects the output distribution**.
- **Budget-Constrained Search**: Limits expensive forward passes to a configurable budget (default: 6), making per-query optimization practical.
- **Conflict Detection**: Avoids unnecessary steering when the model already agrees with the context.
- **Norm-Preserving Rotation (New)**: The `HouseholderSteerer` rotates hidden states toward the steering direction without altering their magnitude, preventing activation drift.
- **Discriminative Layer Filtering (New)**: Sign-based structural filter identifies layers where context–memory separation is strongest, pruning the BCILS candidate set.
- **Query-Specific Latent Search (New)**: B-PLIS synthesises per-query Δh via CMA-ES in a low-rank subspace, going beyond fixed steering vectors.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended, but CPU is supported)
- Hugging Face account with access to gated models (Gemma, Llama)

### Step 1: Clone and Create Virtual Environment

```bash
git clone <your-repo-url>
cd contextfocus_dynamic
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

**requirements.txt** includes:
- `torch>=2.1.0`
- `transformers>=4.50.0` (required for Gemma 3 multimodal support)
- `accelerate>=0.30.0`
- `datasets>=2.20.0`
- `numpy>=1.24`
- `tqdm>=4.66`
- `rich>=13.7`
- `pydantic>=2.6`
- `PyYAML>=6.0`
- `sentencepiece>=0.2.0`
- `safetensors>=0.4.3`
- `evaluate>=0.4.2`
- `scikit-learn>=1.3`
- `cma>=3.1.0` (for B-PLIS CMA-ES latent vector search)
- `pyreft>=0.1.0` (for ReFT baseline)
- `pyvenen>=0.1.7` (for intervention utilities)

### Step 3: Authenticate to Hugging Face

```bash
export HF_TOKEN="your_huggingface_token"
```

Or set `HF_TOK` instead. The code checks both.

### Step 4: Download Datasets

**NQ-SWAP** (for vector construction and layer selection):
```bash
export NQSWAP_DATASET="pminervini/NQ-Swap"
```

**ConFiQA** (for evaluation):
```bash
git clone https://github.com/byronBBL/Context-DPO.git
export CONFIQA_ROOT="./Context-DPO/data"
```

---

## Quick Start Guide

### 1. Build Steering Vectors

This step creates activation steering vectors for all layers (0-33 for Gemma 3 4B).

```bash
python scripts/build_vectors.py \
  --model_id google/gemma-3-4b-it \
  --out_dir artifacts/gemma3_4b \
  --n_examples 1501 \
  --split dev \
  --max_length 1024
```

**Output:**
- `artifacts/gemma3_4b/vectors/layer_000.pt` through `layer_033.pt`
- `artifacts/gemma3_4b/vectors_meta.json` (metadata)

### 2. Select Best Static Layer (Paper Reproduction)

```bash
python scripts/select_layer.py \
  --model_id google/gemma-3-4b-it \
  --vectors_dir artifacts/gemma3_4b/vectors \
  --out_dir artifacts/gemma3_4b \
  --n_eval 200 \
  --multiplier 2.0
```

**Output:**
- `artifacts/gemma3_4b/best_layer.json` - Best layer (e.g., layer 14 with ps=0.795)
- `artifacts/gemma3_4b/layer_sweep.json` - Per-layer results for all layers

### 3. Evaluate on ConFiQA (Static Layer)

```bash
python scripts/eval_confiaq.py \
  --model_id google/gemma-3-4b-it \
  --confiaq_root "$CONFIQA_ROOT" \
  --vectors_dir artifacts/gemma3_4b/vectors \
  --layer_json artifacts/gemma3_4b/best_layer.json \
  --multiplier 2.0 \
  --n_per_subset 1500
```

**Output:**
- `artifacts/confiaq_results.json`

### 4. Evaluate with Dynamic Layer Selection (Novel)

```bash
python scripts/eval_confiaq.py \
  --model_id google/gemma-3-4b-it \
  --confiaq_root "$CONFIQA_ROOT" \
  --vectors_dir artifacts/gemma3_4b/vectors \
  --dynamic_layers true \
  --multiplier 2.0 \
  --layer_sweep_path artifacts/gemma3_4b/layer_sweep.json
```

**Output:**
- `artifacts/confiaq_results.json` (with chosen_layers list)

### 5. Demo Budgeted Search

```bash
python scripts/demo_budgeted_search.py \
  --model_id google/gemma-3-4b-it \
  --vectors_dir artifacts/gemma3_4b/vectors \
  --question "Who is the CEO of Starbucks?" \
  --context "Brian Niccol is the CEO of Starbucks." \
  --budget 6 \
  --probe_tokens 24 \
  --final_tokens 64
```

**Output:**
- JSON with conflict detection result, ranked layers, candidate evaluations, and final text

---

## Architecture Overview

```
contextfocus_dynamic/
│
├── src/contextfocus/
│   ├── __init__.py                  # Package initialization (version 0.1.0)
│   ├── utils.py                     # Model loading, tokenization, device management
│   │
│   ├── data/                        # Dataset loaders
│   │   ├── confiaq.py              # ConFiQA loader (QA, MR, MC subsets)
│   │   ├── nqswap.py               # NQ-SWAP loader
│   │   └── ConFiQA-*.json          # ConFiQA data files
│   │
│   ├── prompting/                   # Prompt templates
│   │   └── templates.py            # Vector construction and evaluation prompts
│   │
│   ├── eval/                        # Evaluation metrics
│   │   ├── evaluator.py            # ConFiQA evaluation loop
│   │   └── metrics.py              # Faithfulness scoring (ps, po, MR)
│   │
│   ├── steering/                    # Activation steering
│   │   ├── vector_builder.py       # Build ContextFocus vectors
│   │   ├── steerer.py              # Activation injection hooks (ActivationSteerer + HouseholderSteerer)
│   │   ├── householder.py          # [NEW] Norm-preserving 2-D rotation of hidden states
│   │   ├── discriminative.py       # [NEW] Sign-based discriminative layer filter
│   │   ├── bplis.py                # [NEW] B-PLIS: CMA-ES latent vector search
│   │   ├── layer_selector.py       # Static layer sweep
│   │   └── dynamic_selector.py     # Dynamic layer selection (BCILS + alignment + discriminative filter)
│   │
│   ├── inference/                   # Inference strategies
│   │   ├── conflict_detector.py    # Knowledge conflict detection
│   │   └── budgeted_search.py      # Budgeted latent activation search (+ B-PLIS backend)
│   │
│   └── reft/                        # ReFT baseline
│       └── pyreft_adapter.py       # PyReFT training wrapper
│
├── scripts/                         # Executable scripts
│   ├── build_vectors.py             # Build vectors from NQ-SWAP
│   ├── select_layer.py              # Static layer selection
│   ├── eval_confiaq.py              # Evaluate on ConFiQA
│   ├── demo_budgeted_search.py      # Demo conflict detection + search
│   ├── train_reft.py                # Train ReFT baseline
│   ├── debug_confiqa.py             # Debug ConFiQA loading
│   ├── debug_data.py                # Debug NQ-SWAP loading
│   └── check_dataset_schema.py      # Check dataset schema
│
├── artifacts/                       # Experimental results
│   ├── confiaq_results.json         # Latest results (dynamic)
│   ├── confiaq_results_static.json  # Static layer results
│   ├── confiaq_results_olddyna.json # Old dynamic algorithm results
│   └── gemma3_4b/                   # Model artifacts
│       ├── best_layer.json          # Best static layer
│       ├── layer_sweep.json         # Per-layer sweep results
│       ├── vectors_meta.json        # Vector metadata
│       └── vectors/                 # Steering vectors (layer_000.pt - layer_033.pt)
│
├── pyproject.toml                   # Package configuration
├── requirements.txt                 # Dependencies
├── tests/                           # Unit / integration tests
│   ├── test_householder.py          # Householder norm-preservation tests
│   ├── test_discriminative.py        # Discriminative layer filter tests
│   └── test_bplis.py                # B-PLIS shape & fallback tests
└── README.md                        # This file
```

---

## Core Modules Documentation

### Utils Module

**File:** `src/contextfocus/utils.py`

This module provides foundational utilities for model loading, tokenization, and transformer introspection.

#### Functions

##### `set_seed(seed: int) -> None`

Sets random seeds for reproducibility across `random`, `numpy`, and `torch`.

**Parameters:**
- `seed` (int): Random seed value

**Example:**
```python
set_seed(7)  # Default seed used throughout the project
```

---

##### `load_hf_causal_lm(model_id: str, *, dtype: str = "bfloat16", device_map: str | Dict[str, int] | None = "auto", trust_remote_code: bool = True) -> ModelBundle`

Loads a Hugging Face causal language model with its tokenizer or processor.

**Parameters:**
- `model_id` (str): Hugging Face model identifier (e.g., "google/gemma-3-4b-it")
- `dtype` (str): Data type ("float16", "bfloat16", or "float32")
- `device_map` (str | Dict): Device placement strategy ("auto", "cpu", or custom dict)
- `trust_remote_code` (bool): Whether to trust remote code (required for some models)

**Returns:**
- `ModelBundle`: Dataclass with `model`, `tokenizer`, and `device` attributes

**Special Handling:**
- For Gemma 3 multimodal models, attempts to use `AutoProcessor` instead of `AutoTokenizer`
- For Gemma 3 conditional generation, falls back to `Gemma3ForConditionalGeneration`
- Reads HF token from `HF_TOKEN` or `HF_TOK` environment variables

**Example:**
```python
bundle = load_hf_causal_lm("google/gemma-3-4b-it", dtype="bfloat16")
model = bundle.model
tokenizer = bundle.tokenizer
device = bundle.device
```

---

##### `get_model_hidden_size(model: Any) -> int`

Extracts the hidden dimension size from various model architectures.

**Parameters:**
- `model`: Hugging Face model instance

**Returns:**
- `int`: Hidden size (e.g., 2048 for Gemma 3 4B)

**Supported Attributes:**
- `config.text_config.hidden_size` (multimodal models like Gemma 3)
- `config.hidden_size` (standard transformers)
- `config.d_model` (some encoder-decoder models)
- `config.n_embd` (GPT-style models)

**Example:**
```python
hidden_size = get_model_hidden_size(model)  # 2048 for Gemma 3 4B
```

---

##### `decode(tokenizer_or_processor: Any, token_ids: torch.Tensor) -> str`

Decodes token IDs to text, handling both tokenizers and processors.

**Parameters:**
- `tokenizer_or_processor`: Tokenizer or processor instance
- `token_ids` (torch.Tensor): Token IDs to decode

**Returns:**
- `str`: Decoded text with special tokens removed

---

##### `get_eos_id(tokenizer_or_processor: Any) -> int`

Gets the end-of-sequence token ID.

**Parameters:**
- `tokenizer_or_processor`: Tokenizer or processor instance

**Returns:**
- `int`: EOS token ID (or 0 if not found)

---

##### `tokenize_text(tokenizer_or_processor: Any, text: str, *, max_length: int | None = None)`

Tokenizes plain text.

**Parameters:**
- `tokenizer_or_processor`: Tokenizer or processor instance
- `text` (str): Text to tokenize
- `max_length` (int | None): Maximum sequence length

**Returns:**
- Dict with `input_ids`, `attention_mask`, etc.

**Example:**
```python
inputs = tokenize_text(tokenizer, "Hello world", max_length=512)
```

---

##### `tokenize_chat(tokenizer_or_processor: Any, messages: list[dict], *, max_length: int | None = None, add_generation_prompt: bool = True)`

Tokenizes chat-formatted messages using the model's chat template.

**Parameters:**
- `tokenizer_or_processor`: Tokenizer or processor instance
- `messages` (list[dict]): Chat messages (e.g., `[{"role": "user", "content": "..."}]`)
- `max_length` (int | None): Maximum sequence length
- `add_generation_prompt` (bool): Whether to add generation prompt

**Returns:**
- Dict with tokenized inputs

**Fallback:**
- If chat template not available, concatenates messages naively

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
inputs = tokenize_chat(tokenizer, messages)
```

---

##### `get_transformer_blocks(model: Any)`

Introspects the model to find the transformer block list (ModuleList).

**Parameters:**
- `model`: Hugging Face model instance

**Returns:**
- `torch.nn.ModuleList`: List of transformer blocks/layers

**Supported Architectures:**
- Llama/Mistral/Gemma: `model.model.layers`, `model.language_model.model.layers`
- GPT: `model.transformer.h`
- GPT-NeoX: `model.gpt_neox.layers`
- Generic: Falls back to largest ModuleList

**Example:**
```python
blocks = get_transformer_blocks(model)
n_layers = len(blocks)  # 34 for Gemma 3 4B
```

---

#### Data Classes

##### `ModelBundle`

```python
@dataclass
class ModelBundle:
    model: Any               # Hugging Face model
    tokenizer: Any           # Tokenizer or processor
    device: torch.device     # Device (cuda or cpu)
```

---

### Data Loaders

**Directory:** `src/contextfocus/data/`

#### ConFiQA Loader

**File:** `src/contextfocus/data/confiaq.py`

ConFiQA (Contextual Faithfulness Question Answering) is a benchmark for evaluating whether models follow provided context over parametric knowledge.

##### Data Classes

```python
@dataclass(frozen=True)
class ConFiQAExample:
    id: str                      # Example ID
    subset: str                  # "QA", "MR", or "MC"
    question: str                # Question text
    context: str                 # Counterfactual context
    original_answer: str         # True answer (parametric knowledge)
    substituted_answer: str      # Counterfactual answer (context-based)
```

##### Functions

##### `load_confiaq(confiaq_root: str | Path, *, subset: str, split: str = "test", limit: Optional[int] = None) -> Iterable[ConFiQAExample]`

Loads ConFiQA dataset.

**Parameters:**
- `confiaq_root` (str | Path): Root directory containing ConFiQA files
- `subset` (str): "QA" (question answering), "MR" (multiple-choice reading), or "MC" (multiple-choice)
- `split` (str): "test" or other split
- `limit` (int | None): Maximum number of examples to load

**Returns:**
- `Iterable[ConFiQAExample]`: Generator of examples

**File Format Support:**
- `.jsonl` (preferred)
- `.json` with `{"data": [...]}` structure

**Field Mapping:**
- Handles multiple naming conventions (e.g., `cf_context`, `context`, `passage`, `retrieved_context`)
- Maps `orig_answer`, `original_answer`, `answer` to `original_answer`
- Maps `cf_answer`, `substituted_answer`, `sub_answer`, `counterfactual_answer` to `substituted_answer`

**Example:**
```python
from contextfocus.data.confiaq import load_confiaq

examples = load_confiaq("./Context-DPO/data", subset="QA", split="test", limit=100)
for ex in examples:
    print(ex.question, ex.context[:50])
```

---

#### NQ-SWAP Loader

**File:** `src/contextfocus/data/nqswap.py`

NQ-SWAP is a knowledge-conflict dataset derived from Natural Questions, where contexts contain substituted entities.

##### Data Classes

```python
@dataclass(frozen=True)
class NQSwapExample:
    id: str                          # Example ID
    question: str                    # Question text
    substituted_context: str         # Context with substituted entity
    substituted_answers: List[str]   # Answers based on substituted context
    original_answers: List[str]      # Ground-truth answers
```

##### Functions

##### `load_nqswap(*, dataset_name: Optional[str] = None, split: str = "dev", streaming: bool = False) -> Iterable[NQSwapExample]`

Loads NQ-SWAP dataset from Hugging Face datasets.

**Parameters:**
- `dataset_name` (str | None): Dataset name (defaults to "pminervini/NQ-Swap" or `NQSWAP_DATASET` env var)
- `split` (str): "dev" or "train"
- `streaming` (bool): Whether to stream dataset

**Returns:**
- `Iterable[NQSwapExample]`: Generator of examples

**Example:**
```python
from contextfocus.data.nqswap import load_nqswap

examples = load_nqswap(split="dev")
for ex in examples:
    print(ex.question, ex.substituted_context[:50])
```

---

### Prompting Templates

**File:** `src/contextfocus/prompting/templates.py`

This module defines prompts for vector construction and evaluation.

#### Constants

##### `POS_SYSTEM_VARIANTS: List[str]`

20 system instruction variants used for vector construction to improve diversity. Examples:
- "You are a context-based QA assistant and must answer based on the provided context."
- "As a QA assistant, you are instructed to refer only to the provided context when answering."
- "Answer the question using only the provided context."
- ...and 17 more variants

#### Data Classes

```python
@dataclass(frozen=True)
class PromptParts:
    system: str      # System instruction
    context: str     # Retrieved/provided context
    question: str    # User question
```

#### Functions

##### `llama_style_inst(system: str, user: str) -> str`

Formats a prompt in Llama/Mistral instruction style with `[INST]` tags.

**Parameters:**
- `system` (str): System message
- `user` (str): User message

**Returns:**
- `str`: Formatted prompt

**Example:**
```python
prompt = llama_style_inst("You are helpful", "What is 2+2?")
# Output: "[INST]\nYou are helpful\nWhat is 2+2?\n[/INST]"
```

---

##### `build_vector_prompts(parts: PromptParts, system_variant: str) -> tuple[str, str]`

Builds positive and negative prompts for vector construction.

**Parameters:**
- `parts` (PromptParts): Prompt components
- `system_variant` (str): System instruction variant

**Returns:**
- `tuple[str, str]`: (positive_prompt, negative_prompt)

**Positive Prompt:**
```
[INST]
{system_variant}
Context: <P> {context} </P>
Question: {question}
[/INST]
```

**Negative Prompt:**
```
[INST]
Question: {question}
[/INST]
```

**Example:**
```python
parts = PromptParts(
    system="",
    context="Paris is the capital of France.",
    question="What is the capital of France?"
)
pos, neg = build_vector_prompts(parts, "Answer based on context.")
```

---

##### `build_vector_messages(parts: PromptParts, system_variant: str) -> tuple[list[dict], list[dict]]`

Builds chat-formatted messages for vector construction (alternative to string prompts).

**Parameters:**
- `parts` (PromptParts): Prompt components
- `system_variant` (str): System instruction variant

**Returns:**
- `tuple[list[dict], list[dict]]`: (positive_messages, negative_messages)

**Example:**
```python
pos_msgs, neg_msgs = build_vector_messages(parts, "Answer based on context.")
# pos_msgs = [
#     {"role": "system", "content": "Answer based on context."},
#     {"role": "user", "content": "Context: <P> ... </P>\nQuestion: ..."}
# ]
```

---

##### `build_openended_prompt(parts: PromptParts, *, oi_prompt: bool = False) -> str`

Builds evaluation prompt for open-ended generation.

**Parameters:**
- `parts` (PromptParts): Prompt components
- `oi_prompt` (bool): Whether to use opinion-of-informant style

**Returns:**
- `str`: Formatted prompt

**Standard Style:**
```
[INST]
You are a Contextual QA Assistant.
Please answer the following question according to the given context.
Please restrict your response to one sentence.

<CONTEXT>
{context}
</CONTEXT>
<QUESTION>
{question}
</QUESTION>
[/INST]
```

**Opinion-of-Informant Style** (`oi_prompt=True`):
```
[INST]
You are a Contextual QA Assistant.
Please answer the following question according to the given context.
Please restrict your response to one sentence.

Bob said, "{context}".
{question} in Bob's opinion?
[/INST]
```

---

##### `build_openended_messages(parts: PromptParts, *, oi_prompt: bool = False) -> list[dict]`

Chat-formatted version of `build_openended_prompt`.

**Parameters:**
- `parts` (PromptParts): Prompt components
- `oi_prompt` (bool): Whether to use opinion-of-informant style

**Returns:**
- `list[dict]`: Chat messages

---

##### `can_use_chat_template(tokenizer_or_processor: Any) -> bool`

Checks if the tokenizer/processor supports `apply_chat_template`.

**Parameters:**
- `tokenizer_or_processor`: Tokenizer or processor instance

**Returns:**
- `bool`: True if chat template is available

---

### Evaluation Module

**Directory:** `src/contextfocus/eval/`

#### Metrics

**File:** `src/contextfocus/eval/metrics.py`

This module implements faithfulness metrics from the ContextFocus paper.

##### Functions

##### `normalize(text: str) -> str`

Normalizes text for answer matching.

**Parameters:**
- `text` (str): Text to normalize

**Returns:**
- `str`: Normalized text (lowercased, punctuation removed, whitespace collapsed)

**Example:**
```python
normalize("Paris, France!")  # "paris france"
```

---

##### `contains_answer(generated: str, answer: str, *, exclude_negated: bool = False) -> bool`

Checks if generated text contains an answer string.

**Parameters:**
- `generated` (str): Model-generated text
- `answer` (str): Expected answer
- `exclude_negated` (bool): If True, exclude matches preceded by negation words

**Returns:**
- `bool`: True if answer is present (and not negated if `exclude_negated=True`)

**Negation Detection:**
- Searches for negation words (not, no, never, isn't, etc.) within 3 tokens before the answer
- Helps avoid false positives like "It is not Paris" when looking for "Paris"

**Example:**
```python
contains_answer("The capital is Paris", "Paris")  # True
contains_answer("It is not Paris", "Paris", exclude_negated=True)  # False
```

---

##### Data Classes

```python
@dataclass
class FaithfulnessCounts:
    n: int      # Total examples
    ps: int     # Count of substituted answer matches (context faithfulness)
    po: int     # Count of original answer matches (parametric knowledge)
    
    @property
    def ps_rate(self) -> float:
        return self.ps / self.n if self.n else 0.0
    
    @property
    def po_rate(self) -> float:
        return self.po / self.n if self.n else 0.0
    
    @property
    def mr(self) -> float:  # Memory rate
        denom = (self.po + self.ps)
        return self.po / denom if denom else 0.0
```

**Metrics:**
- **ps_rate** (Preferring Substituted): Fraction of generations containing the substituted (context-based) answer
- **po_rate** (Preferring Original): Fraction containing the original (parametric) answer
- **mr** (Memory Rate): `po / (po + ps)` - measures reliance on parametric memory vs context

**Goal:** Maximize `ps_rate`, minimize `po_rate` and `mr`

---

##### `score_batch(generations: Iterable[str], original_answers: Iterable[str], substituted_answers: Iterable[str], *, exclude_negated_ps: bool = True) -> FaithfulnessCounts`

Scores a batch of generations against reference answers.

**Parameters:**
- `generations` (Iterable[str]): Model outputs
- `original_answers` (Iterable[str]): Ground-truth answers (parametric)
- `substituted_answers` (Iterable[str]): Context-based answers
- `exclude_negated_ps` (bool): Exclude negated matches for ps counting

**Returns:**
- `FaithfulnessCounts`: Aggregate metrics

**Example:**
```python
from contextfocus.eval.metrics import score_batch

gens = ["Paris is the capital", "The capital is London"]
orig = ["Paris", "Paris"]
subs = ["London", "London"]

counts = score_batch(gens, orig, subs)
print(f"ps_rate: {counts.ps_rate}, po_rate: {counts.po_rate}, mr: {counts.mr}")
```

---

#### Evaluator

**File:** `src/contextfocus/eval/evaluator.py`

High-level evaluation orchestration for ConFiQA.

##### Data Classes

```python
@dataclass(frozen=True)
class EvalConfig:
    max_new_tokens: int = 64                      # Generation length
    multiplier: float = 2.0                       # Steering multiplier
    oi_prompt: bool = False                       # Use opinion-of-informant style
    dynamic_layers: bool = False                  # Enable dynamic layer selection
    top_k_layers: int = 6                        # Number of candidate layers for dynamic
    max_length: int = 1024                       # Max prompt length
    layer_sweep_path: str | None = None          # Path to layer_sweep.json (for prior)
```

##### Functions

##### `evaluate_confiaq(bundle: ModelBundle, examples: Iterable[ConFiQAExample], *, vectors_dir: str | Path, layer: Optional[int] = None, cfg: EvalConfig = EvalConfig()) -> dict`

Evaluates a model on ConFiQA examples.

**Parameters:**
- `bundle` (ModelBundle): Model, tokenizer, and device
- `examples` (Iterable[ConFiQAExample]): ConFiQA examples
- `vectors_dir` (str | Path): Directory with steering vectors
- `layer` (int | None): Static layer index (if not using dynamic)
- `cfg` (EvalConfig): Evaluation configuration

**Returns:**
- `dict`: Results with keys `n`, `ps`, `po`, `ps_rate`, `po_rate`, `mr`, and optionally `chosen_layers`

**Behavior:**
- **Static mode** (`cfg.dynamic_layers=False`): Uses fixed layer for all examples
- **Dynamic mode** (`cfg.dynamic_layers=True`): Selects layer per-query using BCILS

**Example:**
```python
from contextfocus.eval.evaluator import evaluate_confiaq, EvalConfig

cfg = EvalConfig(dynamic_layers=True, multiplier=2.0, top_k_layers=6)
results = evaluate_confiaq(bundle, examples, vectors_dir="artifacts/gemma3_4b/vectors", cfg=cfg)
print(results)
```

---

### Steering Module

**Directory:** `src/contextfocus/steering/`

#### Vector Builder

**File:** `src/contextfocus/steering/vector_builder.py`

Constructs ContextFocus steering vectors for all layers.

##### Data Classes

```python
@dataclass(frozen=True)
class VectorBuildConfig:
    n_examples: int = 1501          # Number of NQ-SWAP examples to use
    seed: int = 7                   # Random seed
    max_length: int = 1024          # Max prompt length
    system_variants: int = 20       # Number of system instruction variants
```

##### Functions

##### `build_contextfocus_vectors(bundle: ModelBundle, examples: Iterable[NQSwapExample], out_dir: str | Path, *, cfg: VectorBuildConfig = VectorBuildConfig()) -> Path`

Builds steering vectors for all transformer layers.

**Parameters:**
- `bundle` (ModelBundle): Model, tokenizer, and device
- `examples` (Iterable[NQSwapExample]): NQ-SWAP examples
- `out_dir` (str | Path): Output directory
- `cfg` (VectorBuildConfig): Build configuration

**Returns:**
- `Path`: Path to vectors directory

**Algorithm:**
1. For each example `i` (up to `n_examples`):
   - Select system variant: `variants[i % len(variants)]`
   - Build positive prompt: system + context + question
   - Build negative prompt: question only
   - Forward pass both through model with `output_hidden_states=True`
   - Extract last-token hidden states for each layer: `h_l`
   - Compute delta: `Δh_l = h_l(pos) - h_l(neg)`
   - Accumulate: `sum_l += Δh_l`
2. Average: `v_l = sum_l / n_examples`
3. Save vectors as `layer_{l:03d}.pt`

**Output Files:**
- `vectors/layer_000.pt` through `layer_0{N}.pt` (N = number of layers - 1)
- `vectors_meta.json`: Metadata including model ID, number of layers, examples used

**Example:**
```python
from contextfocus.steering.vector_builder import build_contextfocus_vectors, VectorBuildConfig
from contextfocus.data.nqswap import load_nqswap

bundle = load_hf_causal_lm("google/gemma-3-4b-it")
examples = load_nqswap(split="dev")
cfg = VectorBuildConfig(n_examples=1501)

vectors_dir = build_contextfocus_vectors(bundle, examples, "artifacts/gemma3_4b", cfg=cfg)
```

---

#### Steerer

**File:** `src/contextfocus/steering/steerer.py`

Implements activation steering via residual stream injection.

##### Data Classes

```python
@dataclass
class SteeringConfig:
    layer: int                              # Layer index to steer
    multiplier: float = 2.0                 # Steering strength
    apply_to_prompt_last_token: bool = True # Steer prompt's last token
    steer_all_positions: bool = False       # If True, steer all positions
```

##### Classes

##### `ActivationSteerer`

Context manager that injects steering vectors during model forward passes.

**Constructor:**
```python
ActivationSteerer(model: Any, vector: torch.Tensor, cfg: SteeringConfig)
```

**Parameters:**
- `model`: Hugging Face model
- `vector` (torch.Tensor): Steering vector (shape: `[hidden_size]`)
- `cfg` (SteeringConfig): Steering configuration

**Usage:**
```python
from contextfocus.steering.steerer import ActivationSteerer, SteeringConfig, load_vector

vector = load_vector("artifacts/gemma3_4b/vectors", layer=14)
cfg = SteeringConfig(layer=14, multiplier=2.0)

with ActivationSteerer(model, vector, cfg):
    outputs = model.generate(**inputs, max_new_tokens=64)
```

**Mechanism:**
- Registers a forward hook on the specified transformer block
- During forward pass, modifies hidden states: `h[:, -1, :] += multiplier * vector`
- Automatically removes hook on context exit

**Methods:**

###### `__enter__(self)`

Registers forward hook on the target layer.

###### `__exit__(self, exc_type, exc, tb)`

Removes forward hook.

---

#### Householder Rotation Module (NEW)

**File:** `src/contextfocus/steering/householder.py`

Implements a **norm-preserving 2-D rotation** of the last-token hidden state. Instead of
*adding* `multiplier * v` (which changes `||h||`), this module *rotates* `h` toward `v`
by angle `theta` in the plane spanned by the steering vector and the orthogonal component
of `h`, keeping `||h||` exactly constant (up to floating-point tolerance).

##### Functions

##### `householder_rotate_last_token(h: torch.Tensor, v: torch.Tensor, theta: float, eps: float = 1e-8) -> torch.Tensor`

Performs the rotation.

**Parameters:**
- `h` (Tensor `[B, S, H]`): Full hidden-state tensor. Only position `[:, -1, :]` is modified.
- `v` (Tensor `[H]` or `[B, H]`): Steering vector (direction to rotate toward).
- `theta` (float): Rotation angle in radians. `pi/6 ≈ 30°` is a good default.
- `eps` (float): Small constant for numerical stability.

**Returns:**
- `h_out` (Tensor `[B, S, H]`): Copy of `h` with only the last token rotated.

**Algorithm:**
1. Normalise `v` to get basis vector `b1`.
2. Project `h_last` onto `b1` to get `alpha`; compute orthogonal residual.
3. Normalise orthogonal residual to get `b2` (with fallback when `h ≅ v`).
4. Rotate `(alpha, beta)` by angle `theta` in the `(b1, b2)` plane.
5. Replace the in-plane projection and assemble output.

**Properties:**
- `||h_out[:, -1, :]|| == ||h[:, -1, :]||` within FP tolerance.
- Positions `0..S-2` are unchanged.
- Fully vectorised for batch dimension `B`.
- Fallback orthonormal vector when `h` is (nearly) parallel to `v`.

**Example:**
```python
from contextfocus.steering.householder import householder_rotate_last_token

h = torch.randn(2, 8, 2048)   # [batch, seq, hidden]
v = torch.randn(2048)          # steering vector
out = householder_rotate_last_token(h, v, theta=0.6)
assert torch.allclose(h[:, -1, :].norm(dim=-1), out[:, -1, :].norm(dim=-1), atol=1e-5)
```

---

##### `_ensure_shape_v(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor`

Broadcasts `v` to `[B, H]` matching `h[:, -1, :]`.

**Parameters:**
- `v`: `[H]` or `[B, H]`
- `h`: `[B, S, H]`

**Returns:** `[B, H]`

**Raises:** `ValueError` if dimensions are incompatible.

---

#### HouseholderSteerer (NEW)

Added to **`src/contextfocus/steering/steerer.py`**.

A context-manager that registers a forward hook to apply norm-preserving rotation
(via `householder_rotate_last_token`) at a chosen transformer block. Drop-in
replacement for `ActivationSteerer` when norm preservation is desired.

##### Class

```python
class HouseholderSteerer:
    def __init__(
        self,
        model: Any,
        vector: torch.Tensor,       # [H] or [B, H] — steering direction or query-specific Δh
        layer: int,                  # transformer block index
        theta: float = math.pi / 6, # rotation angle in radians (~0.524 / 30°)
        apply_to_prompt_last_token: bool = True,
    ): ...
```

**Parameters:**
- `model` (Any): HuggingFace causal LM.
- `vector` (Tensor): Steering direction. May be a pre-built static vector or a
  query-specific Δh from B-PLIS. Lazily moved to `model.device`.
- `layer` (int): Index of the transformer block to hook.
- `theta` (float): Rotation angle. Recommended default: `0.6` rad (≈ 34°).
- `apply_to_prompt_last_token` (bool): Kept for API symmetry with `ActivationSteerer`.

**Hook Behaviour:**
The registered forward hook intercepts the block output (tensor or tuple), calls
`householder_rotate_last_token(hs, vector, theta)`, and returns the modified output.
Device and dtype alignment is handled lazily on first hook invocation.

**Usage:**
```python
from contextfocus.steering.steerer import HouseholderSteerer, load_vector
import math

vector = load_vector("artifacts/gemma3_4b/vectors", layer=14)
with HouseholderSteerer(model, vector, layer=14, theta=0.6):
    outputs = model.generate(**inputs, max_new_tokens=64)
```

**When to use HouseholderSteerer vs ActivationSteerer:**
- Use `ActivationSteerer` for faithful reproduction of the original paper (additive `h += m*v`).
- Use `HouseholderSteerer` when high multipliers cause degradation, or when B-PLIS
  synthesises query-specific Δh vectors that should be injected norm-safely.

---

##### Functions

##### `load_vector(vectors_dir: str | Path, layer: int) -> torch.Tensor`

Loads a steering vector from disk.

**Parameters:**
- `vectors_dir` (str | Path): Directory containing vectors
- `layer` (int): Layer index

**Returns:**
- `torch.Tensor`: Steering vector (CPU, float32)

**Example:**
```python
vector = load_vector("artifacts/gemma3_4b/vectors", layer=14)
print(vector.shape)  # torch.Size([2048])
```

---

#### Layer Selector (Static)

**File:** `src/contextfocus/steering/layer_selector.py`

Sweeps all layers to find the best static layer for steering.

##### Data Classes

```python
@dataclass(frozen=True)
class LayerSelectConfig:
    n_eval: int = 200              # Number of NQ-SWAP examples for evaluation
    max_new_tokens: int = 64       # Generation length
    multiplier: float = 2.0        # Steering multiplier
    oi_prompt: bool = False        # Opinion-of-informant style
    seed: int = 7                  # Random seed
    max_length: int = 1024         # Max prompt length
```

##### Functions

##### `select_best_layer(bundle: ModelBundle, examples: Iterable[NQSwapExample], *, vectors_dir: str | Path, out_dir: str | Path, cfg: LayerSelectConfig = LayerSelectConfig()) -> dict`

Sweeps all layers and finds the one with highest `ps_rate`.

**Parameters:**
- `bundle` (ModelBundle): Model, tokenizer, and device
- `examples` (Iterable[NQSwapExample]): NQ-SWAP examples
- `vectors_dir` (str | Path): Directory with vectors
- `out_dir` (str | Path): Output directory for results
- `cfg` (LayerSelectConfig): Layer selection configuration

**Returns:**
- `dict`: Best layer result with keys `layer`, `ps`, `po`, `mr`, `base_ps`

**Algorithm:**
1. Load first `n_eval` examples
2. Run baseline (no steering): compute `base_ps`
3. For each layer `l`:
   - Load vector `v_l`
   - Generate with steering at layer `l`, multiplier `m`
   - Score generations to get `ps_rate`, `po_rate`, `mr`
4. Select layer with highest `ps_rate`

**Output Files:**
- `layer_sweep.json`: Per-layer results `[{"layer": 0, "ps": ..., "po": ..., "mr": ...}, ...]`
- `best_layer.json`: Best layer summary

**Example:**
```python
from contextfocus.steering.layer_selector import select_best_layer, LayerSelectConfig

cfg = LayerSelectConfig(n_eval=200, multiplier=2.0)
best = select_best_layer(bundle, examples, 
                         vectors_dir="artifacts/gemma3_4b/vectors",
                         out_dir="artifacts/gemma3_4b", 
                         cfg=cfg)
print(f"Best layer: {best['layer']}, ps_rate: {best['ps']}")
```

---

#### Dynamic Selector

**File:** `src/contextfocus/steering/dynamic_selector.py`

Implements two dynamic layer selection strategies:
1. **Alignment-based** (original heuristic, kept for ablations)
2. **Bayesian Causal Influence Layer Selection (BCILS)** (novel, recommended)

##### Alignment-Based Selector (Baseline)

###### Data Classes

```python
@dataclass(frozen=True)
class AlignmentSelectConfig:
    top_k: int = 6                           # Number of layers to return
    score_mode: str = "cosine_times_norm"    # "cosine", "norm", or "cosine_times_norm"
    layer_band: Tuple[int, int] | None = None  # Restrict to [lo, hi) if set
```

###### Classes

##### `AlignmentLayerSelector`

Ranks layers by alignment between representation change and steering vector.

**Constructor:**
```python
AlignmentLayerSelector(model: Any, vectors_dir: str | Path, cfg: AlignmentSelectConfig = AlignmentSelectConfig())
```

**Methods:**

###### `rank_layers(self, pos_hidden_states: Sequence[torch.Tensor], neg_hidden_states: Sequence[torch.Tensor]) -> List[dict]`

Ranks layers by alignment score.

**Parameters:**
- `pos_hidden_states`: Hidden states from context-in forward pass
- `neg_hidden_states`: Hidden states from context-out forward pass

**Returns:**
- `List[dict]`: Ranked layers `[{"layer": 14, "score": 0.85, "cos": 0.92, "delta_norm": 1.23}, ...]`

**Algorithm:**
For each layer `l`:
1. Compute representation change: `Δh_l = h_l(context-in) - h_l(context-out)`
2. Load steering vector `v_l`
3. Compute cosine similarity: `cos = cosine(Δh_l, v_l)`
4. Compute norm: `norm = ||Δh_l||_2`
5. Compute score:
   - `"cosine"`: `score = cos`
   - `"norm"`: `score = norm`
   - `"cosine_times_norm"`: `score = cos * norm`

**Limitation:**
This heuristic measures **representation change** but not **causal influence** on the output. A layer with high alignment might not be effective for steering.

---

##### BCILS Selector (Novel, Recommended)

###### Data Classes

```python
@dataclass(frozen=True)
class InfluenceSelectConfig:
    token_top_k: int = 64                   # Number of context/memory tokens
    prior_weight: float = 1.5               # Strength of Bayesian prior
    return_top_k: int = 6                   # Number of layers to return
    best_static_layer: int = 14             # Fallback layer
    fallback_to_best: bool = True           # Use fallback when low confidence
    confidence_margin: float = 0.02         # Confidence threshold
    layer_band: Tuple[int, int] | None = None  # Restrict to [lo, hi)
```

###### Classes

##### `InfluenceLayerSelector`

Dynamic layer selection using causal logit-sensitivity gradients.

**Constructor:**
```python
InfluenceLayerSelector(
    model: Any, 
    vectors_dir: str | Path, 
    cfg: InfluenceSelectConfig = InfluenceSelectConfig(),
    layer_sweep_path: Optional[str | Path] = None
)
```

**Parameters:**
- `model`: Hugging Face model
- `vectors_dir` (str | Path): Directory with vectors
- `cfg` (InfluenceSelectConfig): Configuration
- `layer_sweep_path` (str | Path | None): Path to `layer_sweep.json` for Bayesian prior

**Methods:**

###### `choose_layer(self, *, in_inputs: Dict[str, torch.Tensor], out_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]`

Selects the best layer for steering using causal influence.

**Parameters:**
- `in_inputs` (Dict): Tokenized context-in prompt (on model device)
- `out_inputs` (Dict): Tokenized context-out prompt (on model device)

**Returns:**
- `Dict`: `{"chosen_layer": int, "ranked_layers": [...], "discriminative_layers": [int, ...]}`

**Algorithm:**

1. **Context-out forward** (no grad):
   - `out_logits = model(**out_inputs).logits[:, -1, :]`

2. **Context-in forward** (with grad):
   - Register hooks on all transformer blocks to capture last-token residuals
   - `in_logits = model(**in_inputs).logits[:, -1, :]`
   - Gradients flow through last-token hidden states

3. **Define context vs memory token sets**:
   - `diff = log_softmax(in_logits) - log_softmax(out_logits)`
   - Context tokens: top-64 tokens with highest `diff` (context favors these)
   - Memory tokens: top-64 tokens with lowest `diff` (memory favors these)

4. **Define utility function**:
   - `U = logsumexp(in_logits[context_tokens]) - logsumexp(in_logits[memory_tokens])`
   - This measures how much the model favors context tokens over memory tokens

5. **Backward pass**:
   - `U.backward()` → computes `dU/dh_l` for each layer's last-token residual

6. **Influence per layer**:
   - `influence_l = (dU/dh_l) · v_l / ||v_l||`
   - This measures how much adding `v_l` to `h_l` would increase `U`

7. **Bayesian prior** (if `layer_sweep_path` provided):
   - Load per-layer `ps` rates from sweep: `prior_l = logit(ps_l)`
   - `score_l = influence_l + prior_weight * prior_l`
   - Without prior: `score_l = influence_l`

8. **Confidence-aware fallback**:
   - If `(score_0 - score_1) < confidence_margin`, use `best_static_layer`
   - Otherwise, use layer with highest score

9. **Discriminative sign-based filter (NEW)**:
   - Build per-layer `mu_pos` from saved last-token residuals; approximate `mu_neg = mu_pos - v_l`
   - Compute global feature direction `d_feat = mean(mu_pos) - mean(mu_neg)`
   - Call `get_discriminative_layers(mu_pos, mu_neg, d_feat)` → layers where `sign(mu_pos·d) ≠ sign(mu_neg·d)`
   - Restrict the ranked list to discriminative layers (falls back to full list if none match)
   - This runs **before** the confidence-aware fallback, so fallback may still override

**Return value now includes:**
- `"discriminative_layers"` (List[int]): Layer indices that passed the sign-flip test

**Why This Works:**
- **Causal**: Directly measures effect on output logits
- **Query-specific**: Uses the actual input to compute gradients
- **Efficient**: Requires only 2 forward passes (context-in with grad, context-out without)
- **Principled**: Based on variational inference / influence functions

**Example:**
```python
from contextfocus.steering.dynamic_selector import InfluenceLayerSelector, InfluenceSelectConfig

selector = InfluenceLayerSelector(
    model,
    vectors_dir="artifacts/gemma3_4b/vectors",
    cfg=InfluenceSelectConfig(return_top_k=6, prior_weight=1.5),
    layer_sweep_path="artifacts/gemma3_4b/layer_sweep.json"
)

# Tokenize context-in and context-out prompts
in_inputs = tokenize_chat(tokenizer, [...]).to(model.device)
out_inputs = tokenize_chat(tokenizer, [...]).to(model.device)

with torch.enable_grad():
    result = selector.choose_layer(in_inputs=in_inputs, out_inputs=out_inputs)

chosen = result["chosen_layer"]
ranked = result["ranked_layers"]
disc   = result["discriminative_layers"]  # NEW: sign-flip layer indices
```

---

#### Discriminative Layer Filter (NEW)

**File:** `src/contextfocus/steering/discriminative.py`

Implements a sign-based structural test that identifies layers where the positive
(context-in) and negative (context-out) mean activations project in **opposite
directions** onto a global feature direction. These are exactly the layers where
the steering vector can structurally flip the model from "memory mode" to "context
mode".

##### Functions

##### `get_discriminative_layers(mu_pos: Sequence[torch.Tensor], mu_neg: Sequence[torch.Tensor], d_feat: torch.Tensor) -> List[int]`

**Parameters:**
- `mu_pos` (Sequence[Tensor `[H]`]): Per-layer mean activation vectors for the context-in condition.
- `mu_neg` (Sequence[Tensor `[H]`]): Per-layer mean activation vectors for the context-out condition.
- `d_feat` (Tensor `[H]`): Global feature direction (L2-normalised internally).

**Returns:**
- `List[int]`: Layer indices `k` where `sign(mu_pos_k · d) × sign(mu_neg_k · d) < 0`.

**Algorithm:**
1. Normalise `d_feat`.
2. For each layer `k`:
   - `proj_pos = mu_pos[k] · d_feat`
   - `proj_neg = mu_neg[k] · d_feat`
   - If `proj_pos * proj_neg < 0` → layer `k` is discriminative.

**Integration:**
Called inside `InfluenceLayerSelector.choose_layer()` after the BCILS ranking
but before the confidence-aware fallback. The ranked list is restricted to
discriminative layers when available; falls back to the original list otherwise.

**Example:**
```python
from contextfocus.steering.discriminative import get_discriminative_layers

mu_pos = [torch.randn(2048) for _ in range(34)]  # per-layer means
mu_neg = [torch.randn(2048) for _ in range(34)]
d_feat = (torch.stack(mu_pos).mean(0) - torch.stack(mu_neg).mean(0))
d_feat = d_feat / d_feat.norm()

disc = get_discriminative_layers(mu_pos, mu_neg, d_feat)
print(f"Discriminative layers: {disc}")
```

---

### Inference Module

**Directory:** `src/contextfocus/inference/`

#### Conflict Detector

**File:** `src/contextfocus/inference/conflict_detector.py`

Detects knowledge conflicts between context and parametric memory.

##### Data Classes

```python
@dataclass(frozen=True)
class ConflictDetectConfig:
    max_length: int = 1024           # Max prompt length
    js_threshold: float = 0.08       # Jensen-Shannon divergence threshold
    use_js: bool = True              # Use JS divergence (vs KL)
```

##### Functions

##### `detect_conflict(model: Any, tokenizer_or_processor: Any, *, question: str, context: str, oi_prompt: bool = False, cfg: ConflictDetectConfig = ConflictDetectConfig()) -> dict`

Detects whether the context creates a knowledge conflict.

**Parameters:**
- `model`: Hugging Face model
- `tokenizer_or_processor`: Tokenizer or processor
- `question` (str): Question text
- `context` (str): Retrieved context
- `oi_prompt` (bool): Opinion-of-informant style
- `cfg` (ConflictDetectConfig): Configuration

**Returns:**
- `dict`: `{"divergence": float, "is_conflict": bool, "in_hidden_states": ..., "out_hidden_states": ..., "in_prompt": str, "out_prompt": str}`

**Algorithm:**
1. Build two prompts:
   - **context-in**: system + context + question
   - **context-out**: question only
2. Forward pass both (without grad)
3. Extract last-token logits: `in_logits`, `out_logits`
4. Softmax: `p = softmax(in_logits)`, `q = softmax(out_logits)`
5. Compute divergence:
   - **JS divergence** (default): `JS(p, q) = 0.5 * [KL(p || m) + KL(q || m)]` where `m = 0.5 * (p + q)`
   - **KL divergence**: `KL(p || q)`
6. Conflict detected if `divergence >= js_threshold`

**Interpretation:**
- **Low divergence** (< 0.08): Context agrees with model's parametric knowledge → no steering needed
- **High divergence** (≥ 0.08): Context conflicts with parametric knowledge → steering recommended

**Example:**
```python
from contextfocus.inference.conflict_detector import detect_conflict, ConflictDetectConfig

result = detect_conflict(
    model, tokenizer,
    question="Who is the CEO of Starbucks?",
    context="Brian Niccol is the CEO of Starbucks.",
    cfg=ConflictDetectConfig(js_threshold=0.08)
)

if result["is_conflict"]:
    print(f"Conflict detected (divergence: {result['divergence']:.3f})")
else:
    print("No conflict, can skip steering")
```

---

#### Budgeted Search

**File:** `src/contextfocus/inference/budgeted_search.py`

Per-query inference with conflict detection and budgeted latent activation search.

##### Data Classes

```python
@dataclass(frozen=True)
class SearchConfig:
    budget: int = 6                      # Max candidate configs to evaluate
    probe_tokens: int = 24               # Probe generation length
    final_tokens: int = 64               # Final generation length
    multipliers: Tuple[float, ...] = (1.0, 2.0, 3.0)  # Multiplier candidates
    top_k_layers: int = 6                # Candidate layers from dynamic selector
    js_threshold: float = 0.08           # Conflict detection threshold
    max_length: int = 1024               # Max prompt length
    # ---- B-PLIS options (NEW) ----
    use_bplis: bool = False              # Enable CMA-ES latent vector search
    bplis_generations: int = 10          # CMA-ES generations
    bplis_popsize: int = 8              # CMA-ES population size
    bplis_intrinsic_dim: int = 64       # Intrinsic subspace dimensionality
    householder_theta: float = 0.6      # Rotation angle in radians (≈ 34°)
```

##### Functions

##### `budgeted_latent_activation_search(bundle: ModelBundle, *, vectors_dir: str, question: str, context: str, cfg: SearchConfig = SearchConfig(), oi_prompt: bool = False) -> dict`

Performs conflict detection + budgeted search for optimal steering configuration.

**Parameters:**
- `bundle` (ModelBundle): Model, tokenizer, and device
- `vectors_dir` (str): Directory with vectors
- `question` (str): Question text
- `context` (str): Retrieved context
- `cfg` (SearchConfig): Search configuration
- `oi_prompt` (bool): Opinion-of-informant style

**Returns:**
- `dict`: Results with keys:
  - `used_search` (bool): Whether search was performed
  - `divergence` (float): JS divergence
  - `layer` (int | List[int] | None): Selected layer(s)
  - `multiplier` (float | None): Selected multiplier (None for B-PLIS)
  - `source` (str): `"static"` or `"bplis"` (NEW)
  - `text` (str): Final generated text
  - `ranked_layers` (list): Dynamic layer ranking (if search used)
  - `candidates` (list): Candidate evaluations (if search used), each entry now includes `"source": "static" | "bplis"`

**Algorithm:**

**Stage 1: Conflict Detection**
1. Run `detect_conflict(question, context)`
2. If `divergence < js_threshold`:
   - Generate without steering
   - Return `{"used_search": False, "text": ..., "layer": None}`

**Stage 2: Dynamic Layer Ranking**
3. If conflict detected:
   - Use BCILS to rank layers: `selector.choose_layer(in_inputs, out_inputs)`
   - Get ranked layers: `[chosen_layer, top_k_layers, best_static_layer]`

**Stage 3: Budgeted Candidate Evaluation**
4. Build candidate list:
   - `candidates = [(layer, multiplier) for layer in ranked_layers for multiplier in multipliers]`
   - Truncate to `budget`: `candidates = candidates[:budget]`

5. For each `(layer, multiplier)`:
   - Load vector `v_layer`
   - Generate with steering: `max_new_tokens = probe_tokens`
   - Compute grounding score:
     - Extract context token IDs from tokenizer
     - For each generated token, compute `p = softmax(logits)`, sum `p[context_tokens]`
     - `grounding = mean(context_prob_mass)`
   - Compute repetition penalty: `repetition = 1 - (unique_tokens / total_tokens)`
   - `score = grounding - 0.05 * repetition`

6. Select best: `(layer*, multiplier*) = argmax_candidate score`

**Stage 4: Final Generation**
7. Generate with `(layer*, multiplier*)` and `max_new_tokens = final_tokens`
8. Return result

**Stage 5: B-PLIS Latent Vector Search (NEW, optional)**

When `cfg.use_bplis = True`, an additional search branch runs after Stage 3:

1. Instantiate `LatentInterventionSearch` with the model, tokenizer, and configured
   `intrinsic_dim` / `popsize` / `max_generations`.
2. Determine target layers from the discriminative filter output
   (falls back to top-3 ranked layers).
3. Run CMA-ES search in the intrinsic subspace:
   - Each candidate `z ∈ R^d` is projected to `Δh = U @ z` (`U` is QR-orthonormal).
   - Candidate is evaluated via `HouseholderSteerer` probe generation.
   - Reward = grounding proxy + 0.1 × normalised entropy.
4. Evaluate the best Δh with a final probe and score it alongside static candidates.
5. If the B-PLIS score beats the best static score, the final generation uses
   `HouseholderSteerer` with the synthesised Δh; otherwise the static path is used.

The old static-vector path is **always evaluated** (for ablation comparisons).
B-PLIS adds `popsize × max_generations` additional forward passes.

**Example:**
```python
from contextfocus.inference.budgeted_search import budgeted_latent_activation_search, SearchConfig

cfg = SearchConfig(budget=6, probe_tokens=24, final_tokens=64, multipliers=(1.0, 2.0, 3.0))
result = budgeted_latent_activation_search(
    bundle,
    vectors_dir="artifacts/gemma3_4b/vectors",
    question="Who is the CEO of Starbucks?",
    context="Brian Niccol is the CEO of Starbucks.",
    cfg=cfg
)

print(f"Used search: {result['used_search']}")
print(f"Selected layer: {result['layer']}, multiplier: {result['multiplier']}")
print(f"Text: {result['text']}")
```

---

#### B-PLIS — Budgeted Latent Vector Search (NEW)

**File:** `src/contextfocus/steering/bplis.py`

Synthesises a **query-specific perturbation Δh** by searching in a low-dimensional
intrinsic subspace using CMA-ES. The search optimises a grounding proxy (probability
mass on context tokens) through short probe generations, injecting candidates via
`HouseholderSteerer` (norm-preserving rotation).

##### Data Classes

```python
@dataclass(frozen=True)
class BPLISConfig:
    intrinsic_dim: int = 64       # Dimensionality of the search subspace
    max_generations: int = 10     # CMA-ES generations
    popsize: int = 8              # CMA-ES population size per generation
    sigma0: float = 0.5           # CMA-ES initial step size
    seed: int = 7                 # Reproducibility seed
    householder_theta: float = 0.6  # Rotation angle for HouseholderSteerer (rad)
```

##### Classes

##### `LatentInterventionSearch`

**Constructor:**
```python
LatentInterventionSearch(
    model: torch.nn.Module,
    tokenizer: Any,
    hidden_size: int,
    cfg: BPLISConfig = BPLISConfig(),
)
```

**Parameters:**
- `model`: HuggingFace causal LM.
- `tokenizer`: Corresponding tokenizer / processor.
- `hidden_size` (int): Dimensionality of the residual stream (e.g. 2048 for Gemma 3 4B).
- `cfg` (BPLISConfig): Search hyper-parameters.

**Methods:**

###### `setup_random_orthoprojector(device: str | torch.device | None = None) -> torch.Tensor`

Builds an orthonormal projection matrix `U [hidden_size, intrinsic_dim]` via QR
decomposition of a random Gaussian matrix.

**Returns:** `U` (Tensor `[H, d]`) with orthonormal columns (`U^T U = I_d`).

---

###### `search(query_inputs, target_layers, context_token_ids, max_generations=None, popsize=None, probe_tokens=24) -> torch.Tensor`

Runs the CMA-ES search loop and returns the best Δh.

**Parameters:**
- `query_inputs` (Dict[str, Tensor]): Tokenised prompt (`input_ids`, `attention_mask`).
- `target_layers` (List[int]): Layers at which the candidate Δh is injected.
- `context_token_ids` (List[int]): Unique token IDs from the context (grounding proxy).
- `max_generations` (int | None): Override `cfg.max_generations`.
- `popsize` (int | None): Override `cfg.popsize`.
- `probe_tokens` (int): Tokens per probe generation.

**Returns:** `dh_best` (Tensor `[hidden_size]`) — query-specific perturbation vector.

**Algorithm:**
1. Ensure orthoprojector `U` is initialised.
2. CMA-ES loop for `max_generations` iterations:
   - Ask `popsize` candidate vectors `z_i ∈ R^d`.
   - Project each to full space: `Δh_i = U @ z_i` (`[H]`).
   - Evaluate each candidate:
     a. Register `HouseholderSteerer` at each `target_layer` with `Δh_i`.
     b. Greedy probe generation (`probe_tokens` tokens).
     c. Score = grounding mass on context tokens + 0.1 × normalised entropy.
   - Tell CMA-ES the negative rewards (it minimises).
3. Return `Δh_best = U @ z_best`.

**Graceful Fallback:**
- If `max_generations <= 0` or `popsize <= 0`, returns a zero vector (no crash).
- If `cma` is not installed, raises `ImportError` with install instructions.

**Example:**
```python
from contextfocus.steering.bplis import LatentInterventionSearch, BPLISConfig
from contextfocus.utils import load_hf_causal_lm, get_model_hidden_size

bundle = load_hf_causal_lm("google/gemma-3-4b-it")
hidden = get_model_hidden_size(bundle.model)

cfg = BPLISConfig(intrinsic_dim=64, max_generations=10, popsize=8)
searcher = LatentInterventionSearch(
    model=bundle.model,
    tokenizer=bundle.tokenizer,
    hidden_size=hidden,
    cfg=cfg,
)

# query_inputs already tokenised and on device
dh = searcher.search(
    query_inputs=inputs,
    target_layers=[14],
    context_token_ids=[101, 2003, 5865],
    probe_tokens=24,
)
print(dh.shape)  # torch.Size([2048])
```

---

### ReFT Integration

**File:** `src/contextfocus/reft/pyreft_adapter.py`

Baseline comparison using Representation Fine-Tuning (ReFT) via the `pyreft` library.

##### Data Classes

```python
@dataclass(frozen=True)
class ReFTConfig:
    layer: int = 15                      # Layer to intervene
    low_rank_dimension: int = 4          # LoReFT rank
    component: str = "block_output"      # Intervention component
    lr: float = 4e-3                     # Learning rate
    epochs: float = 5.0                  # Training epochs
    batch_size: int = 4                  # Batch size
    output_dir: str = "artifacts/reft"   # Output directory
```

##### Functions

##### `build_reft_model(model: Any, *, cfg: ReFTConfig) -> Any`

Creates a PyReFT model that intervenes on residual streams.

**Parameters:**
- `model`: Base Hugging Face model
- `cfg` (ReFTConfig): ReFT configuration

**Returns:**
- PyReFT model instance

**Example:**
```python
from contextfocus.reft.pyreft_adapter import build_reft_model, ReFTConfig

cfg = ReFTConfig(layer=15, low_rank_dimension=4)
reft_model = build_reft_model(model, cfg=cfg)
```

---

##### `train_reft_on_nqswap(model: Any, tokenizer: Any, examples: Iterable[NQSwapExample], *, cfg: ReFTConfig = ReFTConfig(), n_train: int = 256, seed: int = 7) -> Any`

Trains a ReFT intervention using supervised learning.

**Parameters:**
- `model`: Base model
- `tokenizer`: Tokenizer
- `examples` (Iterable[NQSwapExample]): Training examples
- `cfg` (ReFTConfig): ReFT configuration
- `n_train` (int): Number of training examples
- `seed` (int): Random seed

**Returns:**
- Trained ReFT model

**Training Setup:**
- **Prompt**: system + substituted_context + question
- **Target**: substituted_answer (first variant)
- Uses last-position supervised training (intervention applied to last prompt token)

**Example:**
```python
from contextfocus.reft.pyreft_adapter import train_reft_on_nqswap, ReFTConfig
from contextfocus.data.nqswap import load_nqswap

cfg = ReFTConfig(layer=15, low_rank_dimension=4, lr=4e-3, epochs=5.0)
examples = load_nqswap(split="dev")

reft_model = train_reft_on_nqswap(model, tokenizer, examples, cfg=cfg, n_train=256)
```

---

## Scripts Documentation

**Directory:** `scripts/`

All scripts are executable Python files with argparse interfaces.

### build_vectors.py

Builds ContextFocus steering vectors from NQ-SWAP.

**Usage:**
```bash
python scripts/build_vectors.py \
  --model_id google/gemma-3-4b-it \
  --out_dir artifacts/gemma3_4b \
  --n_examples 1501 \
  --split dev \
  --max_length 1024 \
  --dtype bfloat16
```

**Arguments:**
- `--model_id` (str, required): Hugging Face model ID
- `--out_dir` (str, required): Output directory
- `--n_examples` (int, default=1501): Number of NQ-SWAP examples
- `--split` (str, default="dev"): NQ-SWAP split
- `--max_length` (int, default=1024): Max prompt length
- `--dtype` (str, default="bfloat16"): Model dtype

**Output:**
- `{out_dir}/vectors/layer_*.pt`
- `{out_dir}/vectors_meta.json`

---

### select_layer.py

Sweeps all layers to find the best static layer.

**Usage:**
```bash
python scripts/select_layer.py \
  --model_id google/gemma-3-4b-it \
  --vectors_dir artifacts/gemma3_4b/vectors \
  --out_dir artifacts/gemma3_4b \
  --n_eval 200 \
  --multiplier 2.0 \
  --split dev \
  --dtype bfloat16
```

**Arguments:**
- `--model_id` (str, required): Hugging Face model ID
- `--vectors_dir` (str, required): Directory with vectors
- `--out_dir` (str, required): Output directory
- `--n_eval` (int, default=200): Number of NQ-SWAP examples for evaluation
- `--multiplier` (float, default=2.0): Steering multiplier
- `--split` (str, default="dev"): NQ-SWAP split
- `--dtype` (str, default="bfloat16"): Model dtype

**Output:**
- `{out_dir}/layer_sweep.json`
- `{out_dir}/best_layer.json`

---

### eval_confiaq.py

Evaluates on ConFiQA with static or dynamic steering.

**Usage (Static):**
```bash
python scripts/eval_confiaq.py \
  --model_id google/gemma-3-4b-it \
  --confiaq_root "$CONFIQA_ROOT" \
  --vectors_dir artifacts/gemma3_4b/vectors \
  --layer_json artifacts/gemma3_4b/best_layer.json \
  --multiplier 2.0 \
  --n_per_subset 1500 \
  --dtype bfloat16
```

**Usage (Dynamic):**
```bash
python scripts/eval_confiaq.py \
  --model_id google/gemma-3-4b-it \
  --confiaq_root "$CONFIQA_ROOT" \
  --vectors_dir artifacts/gemma3_4b/vectors \
  --dynamic_layers true \
  --multiplier 2.0 \
  --layer_sweep_path artifacts/gemma3_4b/layer_sweep.json \
  --n_per_subset 1500 \
  --dtype bfloat16
```

**Arguments:**
- `--model_id` (str, required): Hugging Face model ID
- `--confiaq_root` (str, required): ConFiQA data directory
- `--vectors_dir` (str, required): Directory with vectors
- `--layer_json` (str, optional): Path to best_layer.json (for static mode)
- `--layer` (int, optional): Fixed layer index (for static mode)
- `--multiplier` (float, default=2.0): Steering multiplier
- `--dynamic_layers` (str, default="false"): "true" to enable dynamic selection
- `--layer_sweep_path` (str, optional): Path to layer_sweep.json (for Bayesian prior)
- `--n_per_subset` (int, default=1500): Examples per ConFiQA subset
- `--dtype` (str, default="bfloat16"): Model dtype

**Output:**
- `artifacts/confiaq_results.json`

---

### demo_budgeted_search.py

Demonstrates conflict detection + budgeted search on a single query.

**Usage:**
```bash
python scripts/demo_budgeted_search.py \
  --model_id google/gemma-3-4b-it \
  --vectors_dir artifacts/gemma3_4b/vectors \
  --question "Who is the CEO of Starbucks?" \
  --context "Brian Niccol is the CEO of Starbucks." \
  --budget 6 \
  --probe_tokens 24 \
  --final_tokens 64 \
  --dtype bfloat16
```

**Arguments:**
- `--model_id` (str, required): Hugging Face model ID
- `--vectors_dir` (str, required): Directory with vectors
- `--question` (str, required): Question text
- `--context` (str, required): Retrieved context
- `--budget` (int, default=6): Number of candidates to evaluate
- `--probe_tokens` (int, default=24): Probe generation length
- `--final_tokens` (int, default=64): Final generation length
- `--dtype` (str, default="bfloat16"): Model dtype

**Output:**
- JSON result printed to stdout

---

### train_reft.py

Trains a ReFT baseline.

**Usage:**
```bash
python scripts/train_reft.py \
  --model_id google/gemma-3-4b-it \
  --layer 15 \
  --rank 4 \
  --n_train 256 \
  --epochs 5.0 \
  --batch_size 4 \
  --lr 4e-3 \
  --output_dir artifacts/reft \
  --split dev \
  --dtype bfloat16
```

**Arguments:**
- `--model_id` (str, required): Hugging Face model ID
- `--layer` (int, default=15): Intervention layer
- `--rank` (int, default=4): LoReFT rank
- `--n_train` (int, default=256): Training examples
- `--epochs` (float, default=5.0): Training epochs
- `--batch_size` (int, default=4): Batch size
- `--lr` (float, default=4e-3): Learning rate
- `--output_dir` (str, default="artifacts/reft"): Output directory
- `--split` (str, default="dev"): NQ-SWAP split
- `--dtype` (str, default="bfloat16"): Model dtype

**Output:**
- ReFT model checkpoints in `output_dir`

---

### debug_confiqa.py

Debug script to inspect ConFiQA data structure.

**Usage:**
```bash
python scripts/debug_confiqa.py
```

**Note:** Edit the file to set the correct path to ConFiQA files.

---

### debug_data.py

Debug script to inspect NQ-SWAP examples and test metrics.

**Usage:**
```bash
python scripts/debug_data.py
```

**Output:**
- Prints sample examples and metric tests

---

### check_dataset_schema.py

Checks the schema of NQ-SWAP dataset from Hugging Face.

**Usage:**
```bash
python scripts/check_dataset_schema.py
```

**Output:**
- Dataset fields and sample row

---

## Tests

**Directory:** `tests/`

All tests are written in **pytest** style and can be run with:

```bash
pytest tests/ -v
```

### test_householder.py

**File:** `tests/test_householder.py`

8 tests covering the Householder norm-preserving rotation:

| Test | What it verifies |
|------|------------------|
| `test_householder_norm_preserve` | `\|\|h[:,-1,:]\|\|` unchanged after rotation (atol=1e-5) |
| `test_norm_preserve_batch_vectors` | Same property when `v` is `[B, H]` |
| `test_only_last_token_changed` | Positions `0..S-2` are bitwise identical |
| `test_theta_zero_identity` | `theta=0` returns the original tensor |
| `test_theta_pi_norm_preserve` | `theta=π` still preserves norm |
| `test_output_shape` | Output shape matches input shape |
| `test_parallel_h_v_does_not_crash` | Fallback branch fires when `h \|\| v` |
| `test_invalid_v_shape_raises` | `ValueError` on mismatched batch dimension |

---

### test_discriminative.py

**File:** `tests/test_discriminative.py`

5 tests covering the discriminative layer filter:

| Test | What it verifies |
|------|------------------|
| `test_discriminative_simple` | Layer with opposite-sign projection is returned |
| `test_no_flip_returns_empty` | Empty list when no sign flip exists |
| `test_all_flip` | All layers returned when all flip |
| `test_mismatched_lengths` | Handles `mu_pos`, `mu_neg` of different lengths |
| `test_d_feat_normalised_internally` | Scaled and unit `d_feat` give identical results |

---

### test_bplis.py

**File:** `tests/test_bplis.py`

6 tests covering B-PLIS shapes, orthonormality, and fallback:

| Test | What it verifies |
|------|------------------|
| `test_orthoprojector_shape` | `U` shape is `(hidden_size, intrinsic_dim)` |
| `test_orthoprojector_orthonormal_columns` | `U^T U ≈ I` (atol=1e-5) |
| `test_z_to_dh_shape` | `Δh` shape is `(hidden_size,)` |
| `test_z_zero_gives_zero_dh` | Zero input → zero output |
| `test_search_zero_generations_returns_zero` | Graceful zero vector when `gens=0` |
| `test_search_zero_popsize_returns_zero` | Graceful zero vector when `popsize=0` |
| `test_orthoprojector_deterministic` | Same seed → same `U` |

---

## Experimental Results

### Evaluation Metrics

- **ps_rate** (Preferring Substituted): Fraction of generations containing the context-based answer
  - **Goal:** Maximize (close to 1.0)
  
- **po_rate** (Preferring Original): Fraction containing the parametric answer
  - **Goal:** Minimize (close to 0.0)
  
- **mr** (Memory Rate): `po / (po + ps)` - reliance on parametric memory
  - **Goal:** Minimize (close to 0.0)

### Dataset: ConFiQA

ConFiQA has three subsets:
- **QA**: Question Answering (1500 examples)
- **MR**: Multiple-choice Reading (1500 examples)
- **MC**: Multiple Choice (1500 examples)

### Model: Google Gemma 3 4B Instruct

**Best Static Layer:** Layer 14 (from `artifacts/gemma3_4b/best_layer.json`)
- `ps_rate`: 0.795
- `po_rate`: 0.005
- `mr`: 0.00625
- Baseline (no steering) `ps_rate`: 0.755

### Results Comparison

**1. Static Layer (Layer 14, multiplier=2.0)**

From `artifacts/confiaq_results_static.json`:

| Subset | n    | ps   | po  | ps_rate | po_rate | mr      |
|--------|------|------|-----|---------|---------|---------|
| QA     | 1500 | 1241 | 34  | 0.827   | 0.023   | 0.027   |
| MR     | 1500 | 1226 | 108 | 0.817   | 0.072   | 0.081   |
| MC     | 1500 | 1210 | 99  | 0.807   | 0.066   | 0.076   |

**Average ps_rate:** 0.817  
**Average mr:** 0.061

---

**2. Old Dynamic Algorithm (Alignment-based)**

From `artifacts/confiaq_results_olddyna.json`:

| Subset | n    | ps   | po  | ps_rate | po_rate | mr      |
|--------|------|------|-----|---------|---------|---------|
| QA     | 1500 | 1119 | 42  | 0.746   | 0.028   | 0.036   |
| MR     | 1500 | 1085 | 129 | 0.723   | 0.086   | 0.106   |
| MC     | 1500 | 1086 | 116 | 0.724   | 0.077   | 0.097   |

**Average ps_rate:** 0.731  
**Average mr:** 0.080

**Observation:** Old dynamic algorithm performs **worse** than static layer 14. This is because alignment-based selection (representation change) doesn't correlate well with steering effectiveness.

---

**3. New Dynamic Algorithm (BCILS with Bayesian prior)**

From `artifacts/confiaq_results.json` (current):

| Subset | n    | ps   | po  | ps_rate | po_rate | mr      | Dominant Choice |
|--------|------|------|-----|---------|---------|---------|-----------------|
| QA     | 1500 | 1242 | 34  | 0.828   | 0.023   | 0.027   | Layer 14 (100%) |
| MR     | 1500 | 1219 | 110 | 0.813   | 0.073   | 0.083   | Layer 14 (100%) |
| MC     | 1500 | 1211 | 98  | 0.807   | 0.065   | 0.075   | Layer 14 (100%) |

**Average ps_rate:** 0.816  
**Average mr:** 0.062

**Observation:** BCILS with Bayesian prior **consistently selects layer 14** for all examples due to:
1. High confidence in layer 14 from the prior (best static layer)
2. Confidence-aware fallback mechanism triggers when margin is small

This demonstrates that the Bayesian prior effectively encodes the global best layer, and the confidence mechanism correctly falls back when per-query signals are weak.

---

### Performance Summary

| Method                  | Avg ps_rate | Avg mr  | vs Static | Notes                                    |
|-------------------------|-------------|---------|-----------|------------------------------------------|
| Baseline (no steering)  | 0.755       | N/A     | -0.062    | From layer selection sweep               |
| Static Layer 14         | 0.817       | 0.061   | baseline  | Paper reproduction                       |
| Old Dynamic (Alignment) | 0.731       | 0.080   | -0.086    | Alignment heuristic fails                |
| BCILS (No prior)        | TBD         | TBD     | TBD       | Pure influence-based selection           |
| BCILS (With prior)      | 0.816       | 0.062   | -0.001    | Matches static via intelligent fallback  |

---

### Analysis

**Why Old Dynamic Fails:**
- Alignment-based selection measures `cosine(Δh_l, v_l) * ||Δh_l||`
- This quantifies representation shift but not causal effect on output
- Layers with large representation change may not steer effectively
- Result: Selects suboptimal layers (e.g., layers 22-32 frequently chosen)

**Why BCILS Works:**
- Measures causal influence via logit-sensitivity gradients
- Directly optimizes for increased probability mass on context tokens
- Bayesian prior biases toward globally effective layers
- Confidence-aware fallback prevents poor choices when signals are weak

**Current Limitation:**
- On ConFiQA, BCILS + prior always falls back to layer 14
- This suggests either:
  1. Layer 14 is indeed optimal for most examples (correct behavior)
  2. Confidence margin is too conservative (hyperparameter tuning needed)
  3. Per-query influence signals are too noisy (needs investigation)

**Future Work:**
- Test BCILS without prior to see raw per-query selections
- Tune `confidence_margin` parameter
- Evaluate on examples where static layer 14 performs poorly
- Analyze correlation between influence scores and actual steering effectiveness

---

## Design Decisions and Limitations

### Design Decisions

1. **Greedy Decoding**
   - All evaluations use greedy decoding (`do_sample=False`)
   - Matches paper's intent for reproducibility
   - Alternative: Nucleus sampling for more diverse outputs

2. **Multiplier = 2.0**
   - Default steering multiplier from paper
   - Higher values (3.0+) may improve faithfulness but risk degradation
   - Budgeted search explores multiple multipliers

3. **Last-Token Steering**
   - Steers only the last prompt token by default
   - Alternative: Steer all positions (`steer_all_positions=True`)
   - Last-token steering is cheaper and usually sufficient

4. **System Instruction Variants**
   - Uses 20 variants during vector construction
   - Increases diversity and robustness
   - Alternative: Fixed single instruction (simpler but less robust)

5. **Confidence-Aware Fallback**
   - BCILS falls back to static layer when `score_margin < 0.02`
   - Prevents poor choices when signals are ambiguous
   - Tunable via `confidence_margin` parameter

6. **Norm-Preserving Rotation (New)**
   - `HouseholderSteerer` rotates rather than adds, preserving `||h||`
   - Avoids activation-magnitude drift at high steering strengths
   - Default angle: `0.6 rad ≈ 34°`

7. **Discriminative Pre-Filter (New)**
   - Sign-based structural test runs inside `choose_layer()` before fallback
   - Restricts BCILS to layers where context–memory separation is structurally present
   - Falls back to original ranked list when no layers pass the filter

8. **B-PLIS as Optional Search Backend (New)**
   - Enabled via `use_bplis=True` in `SearchConfig`
   - Old static-vector path always runs for backward compatibility / ablation
   - B-PLIS candidate competes head-to-head with static candidates on grounding score
   - Final generation uses `HouseholderSteerer` (norm-safe) when B-PLIS wins

### Limitations

1. **Computational Cost**
   - Vector construction: ~1500 forward passes
   - Layer selection sweep: ~34 layers × 200 examples × 64 tokens = ~435k tokens generated
   - BCILS: 2 forward passes per query (1 with grad, 1 without)
   - Budgeted search: Up to 6 additional forward passes per query

2. **Single-Token Answer Matching**
   - Metrics use substring matching (`contains_answer`)
   - May miss paraphrases or multi-token answers
   - Alternative: Use embedding similarity or LLM-as-judge

3. **Context-Conflict Assumption**
   - Assumes retrieved context always conflicts with parametric knowledge
   - Real RAG systems may have aligned contexts
   - Conflict detection mitigates but adds latency

4. **Static Vectors**
   - Steering vectors are fixed after construction
   - Don't adapt to query-specific nuances beyond layer selection
   - **Addressed**: B-PLIS now synthesises query-specific Δh vectors via CMA-ES search;
     integrated into `eval_confiaq.py` via `--use_bplis true`

5. **B-PLIS Computational Cost**
   - CMA-ES search adds `popsize × max_generations` forward passes per query (default: 80)
   - Best used selectively (e.g. only when conflict divergence is very high)
   - `use_bplis=False` (default) avoids this cost entirely
   - QR decomposition automatically falls back to CPU on MPS (Apple Silicon) devices

6. **Evaluation Protocol**
   - ConFiQA uses counterfactual contexts (artificial conflicts)
   - May not reflect real-world RAG scenarios
   - Alternative: Evaluate on Natural Questions with retrieved contexts

7. **Model Support**
   - Tested on Gemma 3 4B and Llama 3.1 8B
   - Other architectures may require modifications to `get_transformer_blocks`
   - Multimodal models (vision+text) not fully tested

8. **Negation Handling**
   - Heuristic negation detection (window of 3 tokens)
   - May miss long-range negations
   - Alternative: Use dependency parsing

---

## Future Work

### Short-Term Improvements

1. **Complete Dynamic BCILS + CMA-ES Evaluation**
   - Run full ConFiQA evaluation with `--use_bplis true` and populate comparison chart
   - Compare ps_rate / po_rate / mr against static and old-dynamic baselines
   - Analyze which subsets benefit most from query-specific vector search

2. **Hyperparameter Tuning**
   - Grid search over `multiplier`, `js_threshold`, `confidence_margin`
   - Tune B-PLIS parameters: `bplis_generations`, `bplis_popsize`, `bplis_intrinsic_dim`, `householder_theta`
   - Optimize for ConFiQA subsets separately (QA vs MR vs MC)

3. **BCILS Without Prior**
   - Evaluate pure influence-based selection (no layer sweep prior)
   - Analyze per-query layer distribution
   - Identify query features that correlate with layer choice

4. **Better Grounding Proxy**
   - Current proxy: mean probability mass on context tokens
   - Alternatives:
     - BERTScore against context
     - Semantic similarity (SBERT)
     - Calibrated confidence

5. **Streaming Evaluation**
   - Support dataset streaming for large-scale evaluation
   - Currently loads full dataset into memory

### Medium-Term Research

1. **Adaptive Vector Construction**
   - Learn query-conditional steering vectors
   - Use meta-learning or hypernetworks
   - Condition on query features (length, topic, ambiguity)
   - **Implemented**: B-PLIS synthesises query-specific Δh via CMA-ES search
   - **Next step**: Warm-start CMA-ES from a learned initial distribution

2. **Multi-Layer Steering**
   - Steer multiple layers simultaneously
   - Learn layer-wise weights via gradient descent
   - May improve robustness
   - **Implemented**: B-PLIS supports multi-layer injection via `target_layers`
   - **Next step**: Combine with learned per-layer weights

3. **Continuous Steering**
   - ~~Current: Binary (steer or don't steer)~~
   - **Implemented**: `HouseholderSteerer` provides continuous control via the `theta` parameter
   - Norm-preserving rotation avoids activation-magnitude drift
   - **Next step**: Learn optimal θ per-query via meta-learning

4. **Cross-Model Transfer**
   - Train vectors on small model, transfer to large model
   - Investigate layer correspondence across model sizes
   - Would enable cheaper vector construction

5. **Causal Mediation Analysis**
   - Formalize BCILS as intervention calculus
   - Estimate average causal effect (ACE) of steering
   - Connect to causal inference literature

### Long-Term Vision

1. **End-to-End RAG Optimization**
   - Joint optimization of retrieval + steering
   - Learn when to retrieve, what to retrieve, how to steer
   - Integrate with retrieval models (ColBERT, DPR)

2. **Steerable Fine-Tuning**
   - Fine-tune models to be more responsive to steering
   - Add steering-awareness loss during training
   - May reduce required multiplier magnitude

3. **Interpretability**
   - Analyze what steering vectors encode
   - Visualize token-level attribution of steering effect
   - Build human-understandable explanations

4. **Real-World Deployment**
   - Optimize latency (quantization, KV-cache reuse)
   - Build API for production RAG systems
   - A/B test against standard RAG baselines

5. **Multi-Task Steering**
   - Extend beyond faithfulness to other dimensions:
     - Factuality
     - Toxicity reduction
     - Style transfer
     - Instruction following
   - Build vector library for different tasks

---

## References

### Papers

1. **ContextFocus** (Original Paper)
   - Title: "ContextFocus: Mitigating Knowledge Conflicts via Activation Steering"
   - Authors: [To be filled based on actual paper]
   - Introduces activation steering for contextual faithfulness

2. **Activation Addition / Steering Vectors**
   - "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" (Li et al., 2023)
   - Foundation for activation steering techniques

3. **ReFT (Representation Fine-Tuning)**
   - "ReFT: Representation Finetuning for Language Models" (Wu et al., 2024)
   - Parameter-efficient alternative to full fine-tuning

4. **ConFiQA Benchmark**
   - From Context-DPO repository
   - Evaluates contextual faithfulness with counterfactual QA

5. **NQ-SWAP Dataset**
   - Derived from Natural Questions
   - Contains knowledge conflicts for vector construction

### Code Repositories

1. **Context-DPO** (ConFiQA data source)
   - URL: https://github.com/byronBBL/Context-DPO
   - License: [Check repository]

2. **PyReFT** (ReFT implementation)
   - URL: https://github.com/stanfordnlp/pyreft
   - License: Apache 2.0

3. **Hugging Face Transformers**
   - URL: https://github.com/huggingface/transformers
   - License: Apache 2.0

### Datasets

1. **NQ-SWAP on Hugging Face**
   - `pminervini/NQ-Swap` or `younanna/NQ-Swap`
   -Derived from Natural Questions (Google)

2. **ConFiQA**
   - Available in Context-DPO repository
   - Three subsets: QA, MR, MC

### Tools

1. **PyTorch** - Deep learning framework
2. **Transformers** - Model loading and inference
3. **Datasets** - HF datasets library
4. **Accelerate** - Distributed training utilities
5. **CMA-ES (`cma`)** - Covariance Matrix Adaptation Evolution Strategy for B-PLIS search

---

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{contextfocus-dynamic,
  title={ContextFocus with Dynamic Layer Selection and Budgeted Search},
  author={[Your Name/Organization]},
  year={2026},
  url={[Repository URL]}
}
```

---

## Contact

For questions, issues, or contributions:
- **GitHub Issues**: [Repository URL]/issues
- **Email**: [Your Email]

---

## Acknowledgments

- Original ContextFocus authors for the foundational work
- Hugging Face team for Transformers library
- PyReFT authors for the ReFT implementation
- Context-DPO team for ConFiQA dataset

---

**End of Documentation**

**Last Updated:** February 2026  
**Total Documentation:** 2500+ lines covering all modules, classes, functions, scripts, tests, and experimental results.
