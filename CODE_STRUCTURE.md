# Code Structure Documentation

Complete overview of all files in the B-PLIS-RAG project with their purposes and key functions.

---

##  Root Directory Files

### `main.py`
**Purpose**: Command-line interface (CLI) entry point for the RAG system.

**Key Functions**:
- `parse_args()` - Parse command-line arguments (query, corpus, model, etc.)
- `download_data()` - Download LegalBench-RAG dataset
- `create_pipeline(args)` - Initialize RAGPipeline with configuration
- `load_and_index_documents(pipeline, corpus)` - Load and index legal documents
- `process_query(pipeline, query, top_k)` - Process a single query and return results
- `interactive_mode(pipeline)` - Run interactive chat mode

**Usage**: `python main.py --query "..." --corpus contractnli`

---

### `setup.py`
**Purpose**: Python package installation configuration.

**Contains**: Package metadata, dependencies, entry points for installation with pip.

---

### `pyproject.toml`
**Purpose**: Modern Python project configuration (PEP 518).

**Contains**: Project metadata, build system requirements, tool configurations (ruff, mypy, pytest).

---

### `requirements.txt`
**Purpose**: List of Python dependencies for pip installation.

**Key Dependencies**:
- `torch` - PyTorch for deep learning
- `transformers` - HuggingFace models (Flan-T5)
- `sentence-transformers` - Embedding models
- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `datasets` - Dataset loading utilities

---

##  src/ - Core Source Code

### `src/__init__.py`
**Purpose**: Package initialization file, makes `src` a Python package.

**Exports**: Main classes (RAGPipeline, RAGConfig, etc.)

---

### `src/config.py`
**Purpose**: Configuration management and settings.

**Key Classes**:
- `Config` - Base configuration dataclass
- `RAGConfig` - RAG pipeline configuration (model, top-k, ReFT settings)

**Key Functions**:
- `get_config()` - Load configuration from TOML files
- `setup_environment()` - Set up logging, device, and environment variables

**Configuration Options**:
- Model selection (t5-base, flan-t5-base, etc.)
- ReFT parameters (layer, dimension)
- Retrieval settings (top-k, max context length)
- Device settings (CPU, CUDA)

---

### `src/model_loader.py`
**Purpose**: Load and configure T5/Flan-T5 models with optimizations.

**Key Functions**:
- `load_model(model_name, device, dtype)` - Load T5/Flan-T5 model and tokenizer
  - Supports FP16 precision for GPU
  - Freezes all model parameters (inference-only)
  - Automatic device mapping for memory efficiency
- `freeze_model_parameters(model)` - Freeze all model weights
- `count_parameters(model)` - Count trainable vs total parameters

**Supported Models**:
- `t5-base` (223M params)
- `google/flan-t5-base` (248M params) - Recommended
- `google/flan-t5-large` (780M params)
- `google/flan-t5-xl` (3B params)

---

### `src/retriever.py`
**Purpose**: FAISS-based semantic retrieval system.

**Key Classes**:
- `FAISSRetriever` - Semantic search engine using sentence transformers

**Key Methods**:
- `__init__(embedding_model, device)` - Initialize with embedding model (all-MiniLM-L6-v2)
- `index_documents(documents, batch_size)` - Encode and index documents with FAISS
- `retrieve(query, top_k)` - Find top-k most relevant documents
- `_encode_texts(texts, batch_size)` - Batch encode texts to embeddings (384-dim)

**How It Works**:
1. Encodes queries and documents using sentence-transformers
2. Builds FAISS index (IndexFlatIP for inner product similarity)
3. Returns ranked documents with similarity scores

---

### `src/rag_pipeline.py`
**Purpose**: Main RAG pipeline orchestrating retrieval and generation.

**Key Classes**:
- `RAGPipeline` - Complete RAG system with ReFT and steering
- `RAGConfig` - Configuration dataclass
- `RAGResponse` - Response object (answer, sources, metadata)
- `BilingualRAGPipeline` - Extended pipeline with translation support

**Key Methods (RAGPipeline)**:
- `__init__(model_name, use_reft, use_steering, config)` - Initialize pipeline
- `index_documents(documents, batch_size)` - Index corpus for retrieval
- `query(question, top_k, template)` - Process query and generate answer
- `batch_query(questions, top_k)` - Process multiple queries
- `train_reft_on_conflicts(examples, num_steps, lr)` - Train ReFT intervention
- `load_reft_checkpoint(checkpoint_path)` - Load trained ReFT weights
- `save_interventions(path)` - Save ReFT/steering state
- `load_interventions(path)` - Load ReFT/steering state

**Pipeline Flow**:
1. Retrieve relevant documents (FAISS)
2. Extract snippets from retrieved docs
3. Format prompt with context
4. Apply ReFT intervention (if enabled)
5. Apply activation steering (if enabled)
6. Generate answer with Flan-T5
7. Return structured response

---

### `src/reft.py`
**Purpose**: ReFT (Representation Fine-Tuning) intervention implementation.

**Key Classes**:
- `ReFTIntervention` - Low-rank intervention module
- `ReFTHook` - Hook for applying interventions during generation
- `ReFTTrainer` - Training loop for ReFT interventions

**ReFTIntervention Methods**:
- `__init__(hidden_size, intervention_dim)` - Initialize low-rank matrices
- `forward(hidden_states, positions)` - Apply intervention to hidden states
- `to_dict()` - Export configuration
- `num_parameters()` - Count trainable parameters (12,304 for 16-dim)

**ReFTHook Methods**:
- `__init__(model, intervention, target_layer)` - Register forward hook
- `__call__(module, input, output)` - Hook function applied during forward pass
- `remove()` - Clean up hook

**ReFTTrainer Methods**:
- `__init__(model, intervention, tokenizer, target_layer, lr, num_steps)` - Setup trainer
- `train(examples, epochs, verbose)` - Train on conflict examples
  - Optimize intervention parameters only
  - Monitor loss and Z-norm
  - Return training metrics
- `save(path)` - Save intervention checkpoint
- `load(path)` - Load intervention checkpoint

**How ReFT Works**:
1. Interventions applied at specific decoder layer (default: layer 6)
2. Low-rank transformation: `z = R₁ᵀ * h + R₂ᵀ * h` (16-dim)
3. Only 12,304 parameters trained (0.005% of model)
4. Steers model to prioritize retrieved context

---

### `src/activation_steering.py`
**Purpose**: Activation steering for context prioritization.

**Key Classes**:
- `ActivationSteering` - Runtime activation manipulation

**Key Methods**:
- `__init__(model, tokenizer, layer, device)` - Initialize steering system
- `compute_steering_vector(positive_examples, negative_examples)` - Compute steering direction
  - Compute mean activations for positive examples
  - Compute mean activations for negative examples
  - Steering vector = positive_mean - negative_mean
- `apply(multiplier)` - Context manager to apply steering during generation
- `_steering_hook(module, input, output)` - Hook function to add steering vector
- `save(path)` - Save steering vector
- `load(path)` - Load steering vector

**How Steering Works**:
1. Compute activation difference between faithful vs unfaithful examples
2. Add scaled steering vector to activations during generation
3. Encourages context-grounded responses

---

### `src/data_handler.py`
**Purpose**: Dataset loading and preprocessing for LegalBench-RAG.

**Key Classes**:
- `Document` - Document representation (content, metadata, source)
- `BenchmarkExample` - Benchmark question-answer pair
- `ConflictExample` - Conflict example for training (query, context, answers)
- `DataHandler` - Base class for dataset handling
- `LegalBenchRAG` - LegalBench-RAG specific loader

**Key Methods (LegalBenchRAG)**:
- `download()` - Download LegalBench-RAG dataset from Dropbox
- `load_corpus(corpus_types, max_docs_per_type)` - Load legal documents
  - Supports: contractnli, cuad, maud, privacy_qa, sara
  - Returns dict mapping corpus type to document list
- `load_benchmarks(benchmark_types)` - Load benchmark examples
  - Supports: contractnli, cuad, maud, privacy_qa, conflict_examples
  - Returns dict mapping benchmark type to example list
- `create_conflict_examples(num_examples)` - Generate training examples
  - Creates conflicting context scenarios
  - Returns ConflictExample objects

**Corpus Structure**:
- ContractNLI: 95 NDAs
- CUAD: 462 contracts
- MAUD: 134 M&A agreements
- Total: 691 legal documents, 6,889 benchmark examples

---

### `src/evaluator.py`
**Purpose**: Benchmark evaluation metrics and evaluation loops.

**Key Classes**:
- `Evaluator` - Evaluation system for RAG pipeline

**Key Methods**:
- `evaluate_benchmark(pipeline, benchmark, corpus)` - Run full benchmark
- `compute_metrics(predictions, references)` - Calculate metrics
  - Exact Match
  - F1 Score
  - Character-level precision/recall
  - ROUGE scores
- `evaluate_conflict_resolution(pipeline, conflicts)` - Test faithfulness
- `save_results(results, output_path)` - Export evaluation results

**Evaluation Metrics**:
- Exact Match (EM)
- Token F1
- Character Precision/Recall
- ROUGE-L
- Context substitution ratio (p_s)

---

### `src/utils.py`
**Purpose**: Utility functions and helpers.

**Key Functions**:
- `timer()` - Context manager for timing code blocks
- `setup_logging(level, format)` - Configure logging
- `set_seed(seed)` - Set random seeds for reproducibility
- `get_device(device_str)` - Get PyTorch device (cuda/cpu)
- `count_tokens(text, tokenizer)` - Count tokens in text
- `truncate_text(text, max_tokens, tokenizer)` - Truncate to max length
- `format_sources(sources)` - Format source citations
- `load_json(path)` - Load JSON with error handling
- `save_json(data, path)` - Save JSON with formatting

**Decorators**:
- `@timer` - Automatically time function execution

---

##  scripts/ - Standalone Scripts

### `scripts/generate_benchmark.py`
**Purpose**: Download and process LegalBench-RAG dataset.

**Key Functions**:
- `download_legalbench()` - Clone/download LegalBench-RAG repository
- `extract_corpus()` - Extract legal documents to data/corpus/
- `process_benchmarks()` - Convert benchmarks to standardized JSON format
- `verify_download()` - Verify data integrity

**Usage**: `python scripts/generate_benchmark.py --download`

---

### `scripts/train_reft.py`
**Purpose**: Train ReFT interventions on conflict examples.

**Key Functions**:
- `parse_args()` - Parse training arguments
- `load_model_and_tokenizer(model_name, device)` - Load base model
- `create_training_examples(dataset, num_examples)` - Generate/load training data
- `train_reft(model, tokenizer, examples, intervention_dim, target_layer, lr, num_steps, epochs, device)` - Training loop
- `run_ablation_study()` - Test different hyperparameters
- `main()` - Main training script

**Training Arguments**:
- `--model`: Base model (google/flan-t5-base)
- `--dataset`: Dataset source (legalbench)
- `--num-examples`: Number of training examples
- `--epochs`: Training epochs
- `--steps-per-example`: Optimization steps per example
- `--lr`: Learning rate (default: 0.01)
- `--intervention-dim`: ReFT dimension (default: 16)
- `--layer`: Target layer (default: 6)
- `--output`: Checkpoint output path

**Usage**: 
```bash
python scripts/train_reft.py \
    --model google/flan-t5-base \
    --dataset legalbench \
    --num-examples 753 \
    --epochs 2 \
    --output checkpoints/reft_flant5_full.pt
```

---

### `scripts/evaluate.py`
**Purpose**: Run comprehensive benchmark evaluation.

**Key Functions**:
- `parse_args()` - Parse evaluation arguments
- `load_pipeline_and_data(args)` - Initialize pipeline and load benchmarks
- `run_evaluation(pipeline, benchmarks, corpus)` - Execute evaluation
- `compute_aggregate_metrics(results)` - Calculate summary statistics
- `save_results(results, output_path)` - Export results to CSV/JSON
- `main()` - Main evaluation script

**Evaluation Options**:
- `--benchmarks`: Which benchmarks to run (all, contractnli, cuad, maud, etc.)
- `--corpus`: Corpus to use (all or specific)
- `--checkpoint`: ReFT checkpoint to load
- `--output`: Results output file

**Usage**: `python scripts/evaluate.py --benchmarks all --output results/eval.csv`

---

### `scripts/lint.sh`
**Purpose**: Code quality checks (linting and type checking).

**Tools Used**:
- `ruff` - Fast Python linter
- `mypy` - Static type checker
- `black` - Code formatter (check mode)

**Usage**: `./scripts/lint.sh`

---

##  tests/ - Unit Tests

### `tests/conftest.py`
**Purpose**: Pytest configuration and shared fixtures.

**Key Fixtures**:
- `sample_documents()` - Generate test documents
- `sample_queries()` - Generate test queries
- `mock_model()` - Mock T5 model for testing
- `temp_checkpoint()` - Temporary checkpoint file

---

### `tests/test_config.py`
**Purpose**: Test configuration loading and validation.

**Tests**:
- Configuration parsing from TOML
- Default values
- Invalid configurations
- Environment variable overrides

---

### `tests/test_retriever.py`
**Purpose**: Test FAISS retriever functionality.

**Tests**:
- Document indexing
- Query retrieval
- Similarity scoring
- Batch encoding
- Edge cases (empty queries, no documents)

---

### `tests/test_reft.py`
**Purpose**: Test ReFT intervention implementation.

**Tests**:
- Intervention initialization
- Forward pass
- Hook registration/removal
- Training loop
- Checkpoint save/load
- Parameter counting

---

### `tests/test_pipeline.py`
**Purpose**: Test RAG pipeline integration.

**Tests**:
- Pipeline initialization
- End-to-end query processing
- Batch queries
- ReFT integration
- Steering integration
- Error handling

---

### `tests/test_steering.py`
**Purpose**: Test activation steering.

**Tests**:
- Steering vector computation
- Hook application
- Context manager usage
- Save/load functionality

---

### `tests/test_utils.py`
**Purpose**: Test utility functions.

**Tests**:
- Timer functionality
- Device selection
- Token counting
- Text truncation
- JSON I/O

---

##  notebooks/ - Jupyter Notebooks

### `notebooks/experiment.ipynb`
**Purpose**: Interactive experimentation and analysis.

**Contents**:
- Dataset exploration
- Model testing
- Visualization of results
- Hyperparameter tuning
- Ablation studies

---

##  data/ - Data Directory

### `data/corpus/`
**Purpose**: Store legal document corpus.

**Subdirectories**:
- `contractnli/` - 95 NDA documents
- `cuad/` - 462 contract documents
- `maud/` - 134 M&A agreement documents
- `privacy_qa/` - Privacy policies (if available)

**Note**: Not committed to Git (downloaded separately)

---

### `data/benchmarks/`
**Purpose**: Store benchmark question-answer pairs.

**Files**:
- `contractnli.json` - 977 NDA Q&A examples
- `cuad.json` - 4,042 contract Q&A examples
- `maud.json` - 1,676 M&A Q&A examples
- `privacy_qa.json` - 194 privacy policy Q&A examples
- `conflict_examples.json` - Generated conflict scenarios

**Format**: JSON with {"tests": [...]} structure

---

##  checkpoints/ - Model Checkpoints

### `checkpoints/reft_flant5_full.pt`
**Purpose**: Production ReFT checkpoint (included in repository).

**Contents**:
- Intervention state dict (R1, R2 matrices)
- Target layer index
- Training configuration metadata

**Details**:
- Model: google/flan-t5-base
- Training: 753 examples, 2 epochs
- Loss: 0.2146
- Size: ~25 KB

---

### `checkpoints/reft_flant5_full.json`
**Purpose**: Training metadata for production checkpoint.

**Contains**:
- Model name
- Hyperparameters (lr, dim, layer)
- Training stats (loss, z-norm)
- Number of examples and epochs

---

##  credentials/ - API Credentials

### `credentials/example.toml`
**Purpose**: Template for API credentials.

**Structure**:
```toml
[huggingface]
token = "hf_..."

[openai]  # Optional
api_key = "sk-..."
```

### `credentials/credentials.toml`
**Purpose**: Actual credentials (Git-ignored).

**Note**: Copy from example.toml and fill with real API keys.

---

##  Documentation Files

### `README.md`
**Purpose**: Main project documentation with quick start guide.

**Sections**:
- Project overview
- Installation instructions
- Quick start examples
- Usage documentation
- Performance metrics
- Contributing guidelines

---

### `PROJECT_OVERVIEW.md`
**Purpose**: Comprehensive technical documentation.

**Sections**:
- Detailed project explanation
- Implementation status
- Known issues and limitations
- Training history
- Research insights
- Future work

---

### `CHANGELOG.md`
**Purpose**: Version history and release notes.

**Contents**:
- Version changes
- Training iterations
- Performance improvements
- Bug fixes
- Breaking changes

---

### `GITHUB_CHECKLIST.md`
**Purpose**: Guide for pushing project to GitHub.

**Contents**:
- Pre-push checklist
- Git commands
- Repository setup instructions
- Post-push verification steps

---

### `LICENSE`
**Purpose**: MIT License for the project.

---

### `.gitignore`
**Purpose**: Specify files/directories to exclude from Git.

**Excluded**:
- Virtual environments (venv/)
- Python cache (__pycache__/)
- Data files (data/corpus/*, data/benchmarks/*)
- Build artifacts (*.egg-info/)
- Logs and temporary files

---

##  Key Dependencies Summary

| Package | Purpose | Usage |
|---------|---------|-------|
| `torch` | Deep learning framework | Model training/inference |
| `transformers` | HuggingFace models | T5/Flan-T5 models |
| `sentence-transformers` | Text embeddings | Document/query encoding |
| `faiss-gpu` | Vector similarity | Fast retrieval |
| `datasets` | Dataset utilities | Loading benchmarks |
| `tqdm` | Progress bars | Training/indexing progress |
| `pytest` | Testing framework | Unit tests |
| `ruff` | Linting | Code quality |

---

##  Execution Flow

### 1. Query Execution (main.py → RAGPipeline)
```
main.py
  ├─> create_pipeline() [loads model + ReFT]
  ├─> load_and_index_documents() [FAISS indexing]
  └─> process_query()
       └─> RAGPipeline.query()
            ├─> Retriever.retrieve() [find top-k docs]
            ├─> format_prompt() [create input]
            ├─> apply ReFT intervention [if enabled]
            ├─> apply steering [if enabled]
            └─> model.generate() [Flan-T5 output]
```

### 2. Training Flow (train_reft.py → ReFTTrainer)
```
train_reft.py
  ├─> load_model_and_tokenizer()
  ├─> create_training_examples()
  └─> train_reft()
       └─> ReFTTrainer.train()
            ├─> forward pass with intervention
            ├─> compute loss (cross-entropy)
            ├─> backward pass (optimize R1, R2)
            └─> save checkpoint
```

### 3. Evaluation Flow (evaluate.py → Evaluator)
```
evaluate.py
  ├─> load_pipeline_and_data()
  ├─> run_evaluation()
  │    └─> Evaluator.evaluate_benchmark()
  │         ├─> process each example
  │         ├─> compute metrics (EM, F1, etc.)
  │         └─> aggregate results
  └─> save_results()
```

---

##  Quick Reference

### Most Important Files for Development
1. **src/rag_pipeline.py** - Main pipeline logic
2. **src/reft.py** - ReFT intervention implementation
3. **src/retriever.py** - Semantic search
4. **main.py** - CLI interface

### Most Important Files for Usage
1. **main.py** - Run queries
2. **checkpoints/reft_flant5_full.pt** - Trained model
3. **requirements.txt** - Install dependencies
4. **README.md** - Documentation

### Configuration Files
1. **src/config.py** - Code-level configuration
2. **credentials/credentials.toml** - API keys
3. **pyproject.toml** - Project metadata

---


