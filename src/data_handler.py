"""
Data handling for B-PLIS-RAG.

Handles loading, preprocessing, and managing the LegalBench-RAG dataset
and other legal/commerce corpora.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

from src.config import get_config, PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the corpus."""
    id: str
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            id=data["id"],
            content=data["content"],
            source=data["source"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class BenchmarkExample:
    """Represents a benchmark example with query and ground truth."""
    query: str
    corpus: str
    ground_truth_snippets: List[Dict[str, Any]]  # file_path, char_start, char_end
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "corpus": self.corpus,
            "ground_truth_snippets": self.ground_truth_snippets,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkExample":
        return cls(
            query=data["query"],
            corpus=data.get("corpus", ""),
            ground_truth_snippets=data.get("ground_truth_snippets", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConflictExample:
    """Example with context-memory conflict for training."""
    query: str
    context: str  # Retrieved context
    context_answer: str  # Answer based on context (faithful)
    parametric_answer: str  # Answer from model's parametric knowledge
    source_file: str = ""
    char_range: Tuple[int, int] = (0, 0)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "context": self.context,
            "context_answer": self.context_answer,
            "parametric_answer": self.parametric_answer,
            "source_file": self.source_file,
            "char_range": list(self.char_range),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConflictExample":
        return cls(
            query=data["query"],
            context=data["context"],
            context_answer=data["context_answer"],
            parametric_answer=data.get("parametric_answer", ""),
            source_file=data.get("source_file", ""),
            char_range=tuple(data.get("char_range", [0, 0])),
        )


class LegalBenchRAG:
    """
    Handler for the LegalBench-RAG dataset.
    
    LegalBench-RAG provides:
    - Corpus: Legal documents (contracts, privacy policies, etc.)
    - Benchmarks: Queries with character-level ground truth snippets
    
    Repository: https://github.com/zeroentropy-ai/legalbenchrag
    """
    
    REPO_URL = "https://github.com/zeroentropy-ai/legalbenchrag.git"
    
    CORPUS_TYPES = [
        "contractnli",
        "cuad", 
        "maud",
        "privacyqa",
        "sara",
    ]
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize LegalBench-RAG handler.
        
        Args:
            data_dir: Directory for corpus and benchmarks.
            cache_dir: Directory for caching.
        """
        config = get_config()
        self.data_dir = Path(data_dir) if data_dir else config.paths.data_dir
        self.corpus_dir = self.data_dir / "corpus"
        self.benchmarks_dir = self.data_dir / "benchmarks"
        self.cache_dir = Path(cache_dir) if cache_dir else config.paths.cache_dir
        
        # Create directories
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Loaded data
        self._documents: Dict[str, List[Document]] = {}
        self._benchmarks: Dict[str, List[BenchmarkExample]] = {}
    
    def download(self, force: bool = False) -> None:
        """
        Download the LegalBench-RAG dataset.
        
        Args:
            force: If True, re-download even if data exists.
        """
        repo_dir = self.cache_dir / "legalbenchrag"
        
        if repo_dir.exists() and not force:
            logger.info("LegalBench-RAG already downloaded")
        else:
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            
            logger.info(f"Cloning LegalBench-RAG to {repo_dir}")
            subprocess.run(
                ["git", "clone", "--depth", "1", self.REPO_URL, str(repo_dir)],
                check=True,
            )
        
        # Copy corpus files
        self._copy_corpus(repo_dir)
        
        # Process benchmarks
        self._process_benchmarks(repo_dir)
        
        logger.info("LegalBench-RAG download complete")
    
    def _copy_corpus(self, repo_dir: Path) -> None:
        """Copy corpus files from repository."""
        corpus_src = repo_dir / "corpus"
        
        if not corpus_src.exists():
            logger.warning(f"Corpus directory not found at {corpus_src}")
            return
        
        for corpus_type in self.CORPUS_TYPES:
            src = corpus_src / corpus_type
            dst = self.corpus_dir / corpus_type
            
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                logger.info(f"Copied corpus: {corpus_type}")
    
    def _process_benchmarks(self, repo_dir: Path) -> None:
        """Process benchmark files from repository."""
        benchmarks_src = repo_dir / "benchmarks"
        
        if not benchmarks_src.exists():
            logger.warning(f"Benchmarks directory not found at {benchmarks_src}")
            return
        
        for benchmark_file in benchmarks_src.glob("*.json"):
            dst = self.benchmarks_dir / benchmark_file.name
            shutil.copy(benchmark_file, dst)
            logger.info(f"Copied benchmark: {benchmark_file.name}")
    
    def load_corpus(
        self,
        corpus_types: Optional[List[str]] = None,
        max_docs_per_type: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Load corpus documents.
        
        Args:
            corpus_types: List of corpus types to load. If None, loads all.
            max_docs_per_type: Maximum documents per corpus type.
            
        Returns:
            Dictionary mapping corpus type to list of documents.
        """
        corpus_types = corpus_types or self.CORPUS_TYPES
        
        for corpus_type in corpus_types:
            if corpus_type in self._documents:
                continue
            
            corpus_path = self.corpus_dir / corpus_type
            if not corpus_path.exists():
                logger.warning(f"Corpus not found: {corpus_type}")
                continue
            
            documents = []
            files = list(corpus_path.glob("**/*.txt")) + list(corpus_path.glob("**/*.md"))
            
            if max_docs_per_type:
                files = files[:max_docs_per_type]
            
            for file_path in tqdm(files, desc=f"Loading {corpus_type}"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    doc = Document(
                        id=str(file_path.relative_to(self.corpus_dir)),
                        content=content,
                        source=corpus_type,
                        metadata={"file_path": str(file_path)},
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
            
            self._documents[corpus_type] = documents
            logger.info(f"Loaded {len(documents)} documents from {corpus_type}")
        
        return self._documents
    
    def load_benchmarks(
        self,
        benchmark_names: Optional[List[str]] = None,
    ) -> Dict[str, List[BenchmarkExample]]:
        """
        Load benchmark examples.
        
        Args:
            benchmark_names: List of benchmark names to load.
            
        Returns:
            Dictionary mapping benchmark name to list of examples.
        """
        for benchmark_file in self.benchmarks_dir.glob("*.json"):
            name = benchmark_file.stem
            
            if benchmark_names and name not in benchmark_names:
                continue
            
            if name in self._benchmarks:
                continue
            
            try:
                with open(benchmark_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Handle LegalBench-RAG format with "tests" wrapper
                if isinstance(data, dict) and "tests" in data:
                    data = data["tests"]
                
                examples = []
                for item in data:
                    # Handle different field names
                    if isinstance(item, str):
                        continue  # Skip string items
                    
                    query = item.get("query", item.get("question", ""))
                    corpus = item.get("corpus", name)  # Use benchmark name as default corpus
                    
                    # Handle "snippets" vs "ground_truth" field names
                    snippets = item.get("snippets", item.get("ground_truth", []))
                    
                    # Convert snippet format if needed
                    ground_truth = []
                    for snippet in snippets:
                        if isinstance(snippet, dict):
                            # Convert from LegalBench-RAG format
                            ground_truth.append({
                                "file_path": snippet.get("file_path", ""),
                                "char_start": snippet.get("span", [0, 0])[0],
                                "char_end": snippet.get("span", [0, 0])[1],
                                "answer": snippet.get("answer", ""),
                            })
                    
                    example = BenchmarkExample(
                        query=query,
                        corpus=corpus,
                        ground_truth_snippets=ground_truth,
                        metadata=item.get("metadata", {}),
                    )
                    examples.append(example)
                
                self._benchmarks[name] = examples
                logger.info(f"Loaded {len(examples)} examples from benchmark {name}")
                
            except Exception as e:
                logger.warning(f"Failed to load benchmark {benchmark_file}: {e}")
        
        return self._benchmarks
    
    def get_document_content(
        self,
        file_path: str,
        char_start: int = 0,
        char_end: Optional[int] = None,
    ) -> str:
        """
        Get content from a document at specific character range.
        
        Args:
            file_path: Path to the document.
            char_start: Start character index.
            char_end: End character index (None for end of document).
            
        Returns:
            Document content or snippet.
        """
        full_path = self.corpus_dir / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        content = full_path.read_text(encoding="utf-8")
        
        if char_end is None:
            char_end = len(content)
        
        return content[char_start:char_end]
    
    def create_conflict_examples(
        self,
        num_examples: int = 100,
        corpus_types: Optional[List[str]] = None,
    ) -> List[ConflictExample]:
        """
        Create conflict examples for training ReFT/steering.
        
        These are examples where the context provides information that
        differs from what the model might know parametrically.
        
        Args:
            num_examples: Number of examples to create.
            corpus_types: Corpus types to use.
            
        Returns:
            List of ConflictExample objects.
        """
        # Load benchmarks and corpus
        self.load_benchmarks()
        self.load_corpus(corpus_types)
        
        examples = []
        
        for benchmark_name, benchmark_examples in self._benchmarks.items():
            for example in benchmark_examples[:num_examples // len(self._benchmarks)]:
                # Get ground truth snippet
                if not example.ground_truth_snippets:
                    continue
                
                snippet = example.ground_truth_snippets[0]
                file_path = snippet.get("file_path", snippet.get("file", ""))
                char_start = snippet.get("char_start", snippet.get("start", 0))
                char_end = snippet.get("char_end", snippet.get("end", 0))
                
                try:
                    context = self.get_document_content(file_path, char_start, char_end)
                    
                    # Create conflict example
                    # Note: context_answer should be extracted from context
                    # parametric_answer would require model inference
                    conflict = ConflictExample(
                        query=example.query,
                        context=context,
                        context_answer=context[:200],  # Simplified: use context excerpt
                        parametric_answer="",  # Would need model inference
                        source_file=file_path,
                        char_range=(char_start, char_end),
                    )
                    examples.append(conflict)
                    
                except Exception as e:
                    logger.warning(f"Failed to create conflict example: {e}")
                
                if len(examples) >= num_examples:
                    break
            
            if len(examples) >= num_examples:
                break
        
        logger.info(f"Created {len(examples)} conflict examples")
        return examples
    
    def get_all_documents(self) -> Iterator[Document]:
        """Iterate over all loaded documents."""
        for corpus_type, docs in self._documents.items():
            yield from docs
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded corpus."""
        stats = {
            "total_documents": sum(len(docs) for docs in self._documents.values()),
            "corpus_types": {},
        }
        
        for corpus_type, docs in self._documents.items():
            total_chars = sum(len(doc.content) for doc in docs)
            stats["corpus_types"][corpus_type] = {
                "num_documents": len(docs),
                "total_characters": total_chars,
                "avg_doc_length": total_chars / len(docs) if docs else 0,
            }
        
        return stats


class DataHandler:
    """
    Main data handler that combines multiple data sources.
    
    Supports:
    - LegalBench-RAG for legal domain
    - Custom corpora for commerce/other domains
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize data handler.
        
        Args:
            data_dir: Base directory for data.
        """
        config = get_config()
        self.data_dir = Path(data_dir) if data_dir else config.paths.data_dir
        
        # Initialize sub-handlers
        self.legalbench = LegalBenchRAG(self.data_dir)
        
        # Custom corpora
        self._custom_documents: Dict[str, List[Document]] = {}
    
    def download_all(self, force: bool = False) -> None:
        """Download all datasets."""
        self.legalbench.download(force)
    
    def load_all(
        self,
        max_docs_per_corpus: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Load all available corpora.
        
        Args:
            max_docs_per_corpus: Max docs per corpus type.
            
        Returns:
            All documents by corpus type.
        """
        documents = {}
        
        # Load LegalBench-RAG
        legal_docs = self.legalbench.load_corpus(
            max_docs_per_type=max_docs_per_corpus
        )
        documents.update(legal_docs)
        
        # Load custom corpora
        documents.update(self._custom_documents)
        
        return documents
    
    def add_custom_corpus(
        self,
        name: str,
        documents: List[Document],
    ) -> None:
        """
        Add a custom corpus.
        
        Args:
            name: Name for the corpus.
            documents: List of documents.
        """
        self._custom_documents[name] = documents
        logger.info(f"Added custom corpus '{name}' with {len(documents)} documents")
    
    def load_from_directory(
        self,
        directory: Path,
        corpus_name: str,
        extensions: List[str] = [".txt", ".md"],
    ) -> List[Document]:
        """
        Load documents from a directory.
        
        Args:
            directory: Directory containing documents.
            corpus_name: Name to assign to corpus.
            extensions: File extensions to include.
            
        Returns:
            List of loaded documents.
        """
        documents = []
        
        for ext in extensions:
            for file_path in directory.glob(f"**/*{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    doc = Document(
                        id=str(file_path.relative_to(directory)),
                        content=content,
                        source=corpus_name,
                        metadata={"file_path": str(file_path)},
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        self._custom_documents[corpus_name] = documents
        return documents
    
    def create_training_data(
        self,
        num_examples: int = 500,
        include_legal: bool = True,
        include_custom: bool = True,
    ) -> List[ConflictExample]:
        """
        Create training data for ReFT/steering.
        
        Args:
            num_examples: Total number of examples.
            include_legal: Include LegalBench-RAG data.
            include_custom: Include custom corpora.
            
        Returns:
            List of training examples.
        """
        examples = []
        
        if include_legal:
            legal_examples = self.legalbench.create_conflict_examples(
                num_examples=num_examples
            )
            examples.extend(legal_examples)
        
        # Add examples from custom corpora here if needed
        
        return examples[:num_examples]


def create_sample_commerce_data() -> List[Document]:
    """
    Create sample commerce documents for testing.
    
    Returns:
        List of sample commerce documents.
    """
    samples = [
        Document(
            id="commerce/product_policy_1.txt",
            content="""
            Return Policy for Electronics:
            All electronic items may be returned within 30 days of purchase.
            Items must be in original packaging with all accessories.
            Opened software and digital downloads are non-refundable.
            Defective items will be replaced or refunded at our discretion.
            """,
            source="commerce",
            metadata={"category": "policy", "domain": "electronics"},
        ),
        Document(
            id="commerce/warranty_terms.txt",
            content="""
            Standard Warranty Terms:
            This product is covered by a limited one-year warranty.
            The warranty covers defects in materials and workmanship.
            Physical damage, water damage, and unauthorized modifications void the warranty.
            To claim warranty service, contact customer support with proof of purchase.
            """,
            source="commerce",
            metadata={"category": "warranty", "domain": "general"},
        ),
        Document(
            id="commerce/shipping_policy.txt",
            content="""
            Shipping and Delivery:
            Standard shipping takes 5-7 business days.
            Express shipping takes 2-3 business days.
            International shipping times vary by destination.
            Free shipping on orders over $50.
            Tracking information provided via email.
            """,
            source="commerce",
            metadata={"category": "shipping", "domain": "logistics"},
        ),
    ]
    
    return samples
