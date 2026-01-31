"""
FAISS-based retriever for B-PLIS-RAG.

Implements efficient similarity search for document retrieval using
sentence embeddings and FAISS indexing.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.config import get_config
from src.data_handler import Document
from src.utils import batch_iterable, timer

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """
    FAISS-based document retriever.
    
    Uses sentence embeddings to encode documents and queries,
    then performs efficient similarity search using FAISS.
    
    Example:
        >>> retriever = FAISSRetriever(embedding_model="all-MiniLM-L6-v2")
        >>> retriever.index_documents(documents)
        >>> results = retriever.retrieve("What is a breach of contract?", top_k=5)
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_type: str = "IndexFlatL2",
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the retriever.
        
        Args:
            embedding_model: Sentence transformer model name.
            index_type: FAISS index type (IndexFlatL2, IndexFlatIP, etc.).
            device: Device for embeddings ('cuda', 'cpu', or None for auto).
        """
        from sentence_transformers import SentenceTransformer
        import faiss
        
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # FAISS index (initialized when documents are indexed)
        self.index: Optional[faiss.Index] = None
        
        # Document storage
        self.documents: List[Document] = []
        self.document_embeddings: Optional[np.ndarray] = None
        
        logger.info(f"Retriever initialized with {embedding_model}, dim={self.embedding_dim}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embeddings
    
    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 32,
        max_chunk_length: int = 512,
    ) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of documents to index.
            batch_size: Batch size for encoding.
            max_chunk_length: Maximum characters per chunk.
        """
        import faiss
        
        logger.info(f"Indexing {len(documents)} documents")
        
        self.documents = documents
        
        # Get document contents
        texts = [doc.content[:max_chunk_length] for doc in documents]
        
        # Encode documents
        with timer("Document encoding"):
            self.document_embeddings = self.encode(
                texts,
                batch_size=batch_size,
                show_progress=True,
            )
        
        # Create FAISS index
        if self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IndexIVFFlat":
            # IVF index for larger datasets
            nlist = min(100, len(documents) // 10)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(self.document_embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add embeddings to index
        self.index.add(self.document_embeddings.astype(np.float32))
        
        logger.info(f"Index created with {self.index.ntotal} vectors")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve most relevant documents for a query.
        
        Args:
            query: Query text.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of (document, score) tuples, sorted by relevance.
        """
        if self.index is None:
            raise RuntimeError("No documents indexed. Call index_documents first.")
        
        # Encode query
        query_embedding = self.encode([query], show_progress=False)
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            top_k,
        )
        
        # Collect results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                # Convert L2 distance to similarity score
                score = 1.0 / (1.0 + distance)
                results.append((doc, score))
        
        return results
    
    def retrieve_with_snippets(
        self,
        query: str,
        top_k: int = 5,
        snippet_length: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with highlighted snippets.
        
        Args:
            query: Query text.
            top_k: Number of documents to retrieve.
            snippet_length: Maximum length of snippet to return.
            
        Returns:
            List of result dictionaries with document, score, and snippet.
        """
        results = self.retrieve(query, top_k)
        
        enriched_results = []
        for doc, score in results:
            # Extract most relevant snippet (simplified: take beginning)
            snippet = doc.content[:snippet_length]
            if len(doc.content) > snippet_length:
                snippet += "..."
            
            enriched_results.append({
                "document": doc,
                "score": score,
                "snippet": snippet,
                "source": doc.source,
                "id": doc.id,
            })
        
        return enriched_results
    
    def save(self, path: Path) -> None:
        """
        Save the retriever state to disk.
        
        Args:
            path: Directory to save to.
        """
        import faiss
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents and metadata
        metadata = {
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "num_documents": len(self.documents),
        }
        
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump([doc.to_dict() for doc in self.documents], f)
        
        if self.document_embeddings is not None:
            np.save(path / "embeddings.npy", self.document_embeddings)
        
        logger.info(f"Retriever saved to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load the retriever state from disk.
        
        Args:
            path: Directory to load from.
        """
        import faiss
        
        path = Path(path)
        
        # Load FAISS index
        index_path = path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        # Load documents
        docs_path = path / "documents.pkl"
        if docs_path.exists():
            with open(docs_path, "rb") as f:
                doc_dicts = pickle.load(f)
            self.documents = [Document.from_dict(d) for d in doc_dicts]
        
        # Load embeddings
        emb_path = path / "embeddings.npy"
        if emb_path.exists():
            self.document_embeddings = np.load(emb_path)
        
        logger.info(f"Retriever loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "num_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
        }


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval.
    
    Uses FAISS for dense retrieval and BM25 for sparse retrieval,
    then combines scores for improved accuracy.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        alpha: float = 0.5,
    ) -> None:
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_model: Model for dense embeddings.
            alpha: Weight for dense scores (1-alpha for sparse).
        """
        self.dense_retriever = FAISSRetriever(embedding_model)
        self.alpha = alpha
        
        # BM25 index
        self.bm25 = None
        self.tokenized_corpus: List[List[str]] = []
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for both dense and sparse retrieval."""
        from rank_bm25 import BM25Okapi
        
        # Dense indexing
        self.dense_retriever.index_documents(documents)
        
        # Sparse indexing (BM25)
        self.tokenized_corpus = [
            doc.content.lower().split() for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve using hybrid dense+sparse scoring.
        
        Args:
            query: Query text.
            top_k: Number of results.
            
        Returns:
            List of (document, combined_score) tuples.
        """
        # Dense retrieval
        dense_results = self.dense_retriever.retrieve(query, top_k * 2)
        dense_scores = {r[0].id: r[1] for r in dense_results}
        
        # Sparse retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores = bm25_scores / max_bm25
        
        # Combine scores
        combined = {}
        for doc_id, dense_score in dense_scores.items():
            idx = next(
                i for i, d in enumerate(self.dense_retriever.documents)
                if d.id == doc_id
            )
            sparse_score = bm25_scores[idx]
            combined[doc_id] = self.alpha * dense_score + (1 - self.alpha) * sparse_score
        
        # Sort and return top-k
        sorted_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
        
        results = []
        for doc_id in sorted_ids[:top_k]:
            doc = next(d for d in self.dense_retriever.documents if d.id == doc_id)
            results.append((doc, combined[doc_id]))
        
        return results


def build_retriever_from_data(
    data_handler: Any,  # DataHandler
    corpus_types: Optional[List[str]] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    save_path: Optional[Path] = None,
) -> FAISSRetriever:
    """
    Build a retriever from data handler.
    
    Args:
        data_handler: DataHandler instance with loaded data.
        corpus_types: Corpus types to include.
        embedding_model: Embedding model to use.
        save_path: Optional path to save the retriever.
        
    Returns:
        Configured FAISSRetriever.
    """
    # Collect all documents
    all_docs = []
    documents = data_handler.load_all()
    
    for corpus_type, docs in documents.items():
        if corpus_types is None or corpus_type in corpus_types:
            all_docs.extend(docs)
    
    logger.info(f"Building retriever with {len(all_docs)} documents")
    
    # Create and configure retriever
    retriever = FAISSRetriever(embedding_model=embedding_model)
    retriever.index_documents(all_docs)
    
    # Save if path provided
    if save_path:
        retriever.save(save_path)
    
    return retriever
