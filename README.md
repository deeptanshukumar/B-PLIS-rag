# B-PLIS-RAG: Legal Document Q&A with ReFT and Activation Steering

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-ready Retrieval-Augmented Generation (RAG) system for legal documents, implementing **ReFT (Representation Fine-Tuning)** and **Activation Steering** for improved answer accuracy and faithfulness.

** Now with fully trained model achieving high-quality natural language answers!**

##  What This Does

B-PLIS-RAG helps you **quickly search and answer questions** across hundreds of legal documents:
-  Load 691 legal documents (NDAs, contracts, M&A agreements)
-  Find relevant sections using semantic search
-  Generate natural language answers grounded in source documents
-  Uses ReFT intervention for faithful, context-based responses

**Example:**
```bash
$ python main.py --query "What is confidential information?" --corpus contractnli

Answer: Every person who has been given access to a University of Nebraska 
information system or who has credits with confidential or sensitive information 
are obligated to keep such data confidential.

Sources: [contractnli] NU Confidentiality Agreement
```

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <url>
cd B-PLIS-rag

# Create virtual environment (Python 3.12+ recommended, 3.10+ works)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download LegalBench-RAG corpus and benchmarks
python scripts/generate_benchmark.py --download
```

This downloads:
- 691 legal documents (95 NDAs, 462 CUAD contracts, 134 MAUD agreements)
- 6,889 benchmark question-answer pairs

### 3. Run Queries

```bash
# Simple query
python main.py --query "What is confidential information?" --corpus contractnli

# Interactive mode
python main.py --interactive

# Search all documents
python main.py --query "What are termination clauses?" --corpus all --top-k 5
```

##  Pre-trained Model

The repository includes a **fully trained ReFT checkpoint**:
- **Model**: `google/flan-t5-base` (248M parameters)
- **Training**: 753 examples over 2 epochs (~100 minutes on GPU)
- **Loss**: 0.2146 (final)
- **Location**: `checkpoints/reft_flant5_full.pt`

The model automatically loads this checkpoint when you run queries.

# This will:
# - Clone the LegalBench-RAG repository
# - Extract corpus files to data/corpus/
# - Process benchmarks to data/benchmarks/
```

### 4. Run RAG Pipeline

```bash
# Quick query
python main.py --query "What are the termination clauses in this contract?"

# With specific corpus
python main.py --query "Define confidential information" --corpus contractnli

# Interactive mode
python main.py --interactive
```

##  Project Structure

```
B-PLIS-rag/
├── src/                     # Core implementation
│   ├── rag_pipeline.py      # Main RAG pipeline with ReFT
│   ├── reft.py              # ReFT intervention logic
│   ├── activation_steering.py  # Activation steering
│   ├── retriever.py         # FAISS semantic search
│   ├── model_loader.py      # Model loading utilities
│   ├── data_handler.py      # Dataset loading
│   └── config.py            # Configuration
├── scripts/
│   ├── train_reft.py        # Train ReFT interventions
│   ├── evaluate.py          # Benchmark evaluation
│   └── generate_benchmark.py  # Dataset download
├── data/
│   ├── corpus/              # Legal documents (691 files)
│   └── benchmarks/          # Test questions (6,889 examples)
├── checkpoints/
│   └── reft_flant5_full.pt  # Trained model (included)
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks for experiments
├── main.py                  # CLI entry point
└── README.md
```

##  Usage Examples

### Command Line Interface

```bash
# Ask a question
python main.py --query "What is a breach of contract?" --corpus cuad

# Interactive mode (chat)
python main.py --interactive

# Search specific corpus with more results
python main.py --query "Define confidential information" \
    --corpus contractnli --top-k 3

# Disable ReFT to see baseline performance
python main.py --query "..." --corpus cuad --no-reft
```

### Python API

```python
from src.rag_pipeline import RAGPipeline, RAGConfig
from src.data_handler import LegalBenchRAG

# Initialize pipeline
config = RAGConfig(
    model_name="google/flan-t5-base",
    use_reft=True,
    reft_layer=6,
    reft_dim=16,
    top_k=5
)
pipeline = RAGPipeline(config=config)

# Load trained checkpoint
pipeline.load_reft_checkpoint("checkpoints/reft_flant5_full.pt")

# Load and index documents
data_handler = LegalBenchRAG()
corpus = data_handler.load_corpus(["contractnli"])
pipeline.index_documents(corpus["contractnli"])

# Query
response = pipeline.query("What is confidential information?")
print(f"Answer: {response.answer}")
print(f"Sources: {response.sources}")
```

### Training Your Own ReFT Model

```bash
# Train on full conflict examples
python scripts/train_reft.py \
    --model google/flan-t5-base \
    --dataset legalbench \
    --num-examples 1000 \
    --epochs 2 \
    --steps-per-example 50 \
    --lr 0.01 \
    --output checkpoints/my_reft.pt

# Train on specific corpus
python scripts/train_reft.py \
    --model google/flan-t5-base \
    --dataset legalbench \
    --corpus contractnli cuad \
    --num-examples 500 \
    --epochs 1 \
    --output checkpoints/reft_custom.pt
```

##  How It Works

### 1. **Retrieval (FAISS + Sentence Transformers)**
- Embeds documents using `all-MiniLM-L6-v2`
- Finds top-k relevant documents via semantic search
- Extracts relevant snippets (up to 1000 characters)

### 2. **Generation (Flan-T5 + ReFT)**
- Uses instruction-tuned `google/flan-t5-base` (248M params)
- Applies learned ReFT intervention at decoder layer 6
- Generates answer conditioned on retrieved context

### 3. **ReFT Intervention**
- Low-rank (16-dim) intervention with only 12,304 parameters
- Trains to prioritize retrieved context over memorized knowledge
- 99.995% of model weights stay frozen

##  Performance

### Training Metrics
- **Model**: google/flan-t5-base (248M parameters)
- **Training examples**: 753 conflict pairs
- **Training time**: 101 minutes on RTX 3050 4GB VRAM
- **Final loss**: 0.2146
- **Parameters trained**: 12,304 (0.005% of total model)

### Answer Quality Comparison

| Configuration | Example Answer |
|--------------|----------------|
| **Baseline (no ReFT)** | "sensitive" |
| **Flan-T5 + ReFT** | "Every person who has been given access to a University of Nebraska information system or who has credits with confidential or sensitive information are obligated to keep such data confidential." |

##  Advanced Options

```bash
# Use different model
python main.py --query "..." --model google/flan-t5-large

# Adjust retrieval
python main.py --query "..." --top-k 10

# Custom checkpoint
python main.py --query "..." --checkpoint checkpoints/my_model.pt

# Disable interventions
python main.py --query "..." --no-reft --no-steering

# Verbose output
python main.py --query "..." --verbose
```

##  Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_reft.py -v

# Test with coverage
pytest --cov=src tests/
```

##  Key Research Papers

1. **ReFT: Representation Fine-Tuning** - Parameter-efficient intervention learning
2. **ContextFocus** (arXiv:2601.04131) - Activation steering for context faithfulness
3. **LegalBench-RAG** - Legal domain RAG benchmark

##  Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

##  License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

##  Acknowledgments

- **LegalBench-RAG** dataset by [zeroentropy-ai](https://github.com/zeroentropy-ai/legalbenchrag)
- **Flan-T5** by Google Research
- **FAISS** by Facebook AI Research
- **Sentence Transformers** by UKPLab

##  Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

** If you find this useful, please star the repository!**
| Char-Precision | Character-level precision of retrieved snippets |
| Char-Recall | Character-level recall of retrieved snippets |
| F1 | Harmonic mean of precision and recall |
| p_s | Substituted answer ratio (context faithfulness) |
| p_o | Original answer ratio (parametric knowledge) |

##  Configuration

### Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"
export BPLIS_DATA_DIR="./data"
export BPLIS_CACHE_DIR="./cache"
```

### credentials.toml

```toml
[huggingface]
token = "hf_..."

[openai]  # Optional, for baseline comparisons
api_key = "sk-..."

[settings]
device = "auto"  # auto, cuda, cpu
dtype = "float16"
seed = 42
```

##  Development

### Code Quality

```bash
# Run linter
ruff check src/ tests/

# Run type checker
mypy src/

# Run all checks
./scripts/lint.sh
```

### Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_reft.py -v
```

##  Best Practices

### Efficiency
- Low-parameter interventions (<0.01% of model parameters)
- No full fine-tuning required
- Batch processing for large corpora

### Safety
- Hook cleanup to avoid memory leaks
- `torch.no_grad()` for inference
- Proper error handling for OOM

### Quality
- Type hints throughout
- Comprehensive docstrings
- Unit tests for critical components

### Ethics
- Dataset usage policies acknowledged
- NOT legal advice disclaimer
- Transparent limitations

##  Limitations

1. **Stochastic Outputs**: Steering and ReFT can produce variable results
2. **Context Length**: Limited to 512 tokens (T5 constraint)
3. **Language Coverage**: Best performance on English; Hindi support experimental
4. **Legal Advice**: This is NOT a substitute for professional legal counsel

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests and linting (`./scripts/lint.sh && pytest`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

##  License

MIT License - see [LICENSE](LICENSE) for details.


### Related Papers

```bibtex
@article{contextfocus2026,
  title={ContextFocus: Steering Language Models for Faithful Retrieval-Augmented Generation},
  journal={arXiv preprint arXiv:2601.04131},
  year={2026}
}

@article{grace2022,
  title={GRACE: Gradient-based Model Editing},
  journal={arXiv preprint arXiv:2211.11031},
  year={2022}
}
```

