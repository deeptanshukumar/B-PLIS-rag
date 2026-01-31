# Changelog

All notable changes to B-PLIS-RAG project.


### Added
-  Complete RAG pipeline with FAISS retrieval and Flan-T5 generation
-  ReFT intervention implementation (16-dim, 12,304 parameters)
-  Activation steering infrastructure
-  LegalBench-RAG dataset integration (691 documents, 6,889 benchmarks)
-  Trained production model: `reft_flant5_full.pt`
  - Model: google/flan-t5-base (248M params)
  - Training: 753 examples, 2 epochs, 101 minutes
  - Loss: 0.2146
-  CLI interface with interactive mode
-  Checkpoint save/load system
-  GPU acceleration support (CUDA)
-  Comprehensive documentation (README + PROJECT_OVERVIEW)

### Changed
-  Switched default model from `t5-base` to `google/flan-t5-base`
  - Reason: Instruction-tuned model essential for proper answer generation
  - Impact: Answers improved from "True" to full contextual responses
-  Increased max_length from 512 to 1024 tokens
-  Snippet length from 500 to 1000 characters
-  Generation strategy: greedy decoding, max_new_tokens=150

### Fixed
-  Benchmark loading now handles LegalBench-RAG's {"tests": [...]} format
-  Proper handling of "snippets" field from benchmarks
-  Checkpoint auto-loading from checkpoints/reft_flant5_full.pt

### Removed
-  Debug scripts: `debug_query.py`, `debug_reft.py`
-  Experimental notebooks: `initialise.ipynb`, `Pyreft_tinyllama1_1B.ipynb`, `reft_t5_base_implementation.ipynb`
-  Intermediate checkpoints: kept only best production model
-  Build artifacts: `b_plis_rag.egg-info/`, `__pycache__/`
-  Cache directories: `.cache/`, `.ipynb_checkpoints/`

## Training History

### Iteration 1: T5-base Baseline
- Model: t5-base
- Examples: 40
- Time: 4.4 minutes (GPU)
- Loss: 0.6066
- Z-norm: 3.1049
- Result: Outputs classification labels ("True", "entailment")
- Status:  Not suitable for Q&A

### Iteration 2: T5-base Extended
- Model: t5-base
- Examples: 150
- Time: 8.5 minutes (GPU)
- Loss: 0.8546
- Z-norm: 1.1389
- Result: Still outputs labels
- Status:  Model limitation identified

### Iteration 3: Flan-T5 Discovery
- Model: google/flan-t5-base
- Examples: 80
- Time: 5.2 minutes (GPU)
- Loss: 0.3541
- Z-norm: 1.1071
- Result: "Confidential or sensitive information or data"
- Status:  Breakthrough! Instruction-tuning essential

### Iteration 4: Full Training 
- Model: google/flan-t5-base
- Examples: 753
- Time: 101 minutes (GPU)
- Epochs: 2
- Loss: 0.2146 (Epoch 1: 0.2373, Epoch 2: 0.1919)
- Z-norm: 1.2243
- Result: Detailed contextual answers but not 100 % perfect

## Performance Metrics

### Hardware
- GPU: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)
- CUDA: 13.0
- PyTorch: 2.6.0+cu124
### Dataset
- Documents: 691 (95 ContractNLI + 462 CUAD + 134 MAUD)
- Benchmarks: 6,889 examples
- Training examples: 753 conflict pairs
- Embedding model: all-MiniLM-L6-v2 (384-dim)

### Model Size
- Base model: 247,577,856 parameters
- ReFT intervention: 12,304 parameters (0.005%)
- Total trainable: 12,304 parameters
- Checkpoint size: ~25 KB (ReFT only)

## Known Issues & Future Work

### Current Limitations
-  4GB VRAM limits to base-size models (flan-t5-base)
-  Some answers truncated for very specific queries
-  Steering vectors not yet computed
-  Full benchmark evaluation not completed

### Planned Improvements
-  Formal evaluation on 6,889 benchmark examples
-  Compute activation steering vectors
-  Support for larger models (flan-t5-large, flan-t5-xl) with 8GB+ VRAM
-  Multi-hop reasoning across documents
-  Conflict detection and resolution
-  Confidence scoring for answers
-  Bilingual support (Hindi translation pipeline)

## Research Insights

### Key Discovery: Instruction-Tuning is Essential
The most critical finding was that **base T5 models cannot be used for Q&A**, even with ReFT training. They treat all tasks as classification and output labels like "True" or "entailment". 

**Solution**: Use instruction-tuned models like Flan-T5, which understand prompts like "Answer the question based on the context."

### ReFT Effectiveness
- ReFT successfully modifies model behavior with <0.01% parameters
- Training converges quickly (2 epochs sufficient)
- Loss progression: 0.2373 → 0.1919 shows continued learning
- Z-norm ~1.2 indicates reasonable intervention magnitude

### GPU Acceleration Critical
- CPU training: 9 min/example → 12+ hours for 80 examples
- GPU training: 30-40 sec/example → 60 minutes for 80 examples

