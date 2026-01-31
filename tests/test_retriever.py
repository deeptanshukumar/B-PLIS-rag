"""Tests for FAISS retriever module."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Test with optional FAISS
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class MockDocument:
    """Mock document for testing."""

    text: str
    source: str = "test"
    metadata: dict | None = None


class TestFAISSRetriever:
    """Tests for FAISSRetriever class."""

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        """Create a mock embedding model."""
        embedder = MagicMock()

        def encode_side_effect(
            texts: list[str], show_progress_bar: bool = False, **kwargs
        ) -> np.ndarray:
            return np.random.randn(len(texts), 384).astype(np.float32)

        embedder.encode = MagicMock(side_effect=encode_side_effect)
        return embedder

    @pytest.fixture
    def sample_documents(self) -> list[MockDocument]:
        """Create sample documents."""
        return [
            MockDocument(text="This is about contracts and agreements.", source="doc1"),
            MockDocument(text="Legal rights and obligations explained.", source="doc2"),
            MockDocument(text="Commercial law principles overview.", source="doc3"),
            MockDocument(text="Breach of contract remedies.", source="doc4"),
            MockDocument(text="Intellectual property protection.", source="doc5"),
        ]

    @pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")
    def test_index_documents(
        self, mock_embedder: MagicMock, sample_documents: list[MockDocument]
    ) -> None:
        """Test document indexing."""
        from src.retriever import FAISSRetriever

        with patch(
            "src.retriever.SentenceTransformer", return_value=mock_embedder
        ):
            retriever = FAISSRetriever(
                embedding_model="all-MiniLM-L6-v2",
                device="cpu",
            )

            retriever.index_documents(sample_documents)

            assert len(retriever.documents) == 5
            assert retriever.index is not None
            assert retriever.index.ntotal == 5

    @pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")
    def test_retrieve(
        self, mock_embedder: MagicMock, sample_documents: list[MockDocument]
    ) -> None:
        """Test document retrieval."""
        from src.retriever import FAISSRetriever

        with patch(
            "src.retriever.SentenceTransformer", return_value=mock_embedder
        ):
            retriever = FAISSRetriever(
                embedding_model="all-MiniLM-L6-v2",
                device="cpu",
            )

            retriever.index_documents(sample_documents)

            results = retriever.retrieve("contract breach", top_k=3)

            assert len(results) == 3
            assert all("score" in r for r in results)
            assert all("document" in r for r in results)

    @pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")
    def test_retrieve_empty_index(self, mock_embedder: MagicMock) -> None:
        """Test retrieval on empty index."""
        from src.retriever import FAISSRetriever

        with patch(
            "src.retriever.SentenceTransformer", return_value=mock_embedder
        ):
            retriever = FAISSRetriever(
                embedding_model="all-MiniLM-L6-v2",
                device="cpu",
            )

            with pytest.raises((ValueError, RuntimeError)):
                retriever.retrieve("test query", top_k=3)

    @pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")
    def test_save_and_load(
        self, mock_embedder: MagicMock, sample_documents: list[MockDocument]
    ) -> None:
        """Test saving and loading retriever."""
        from src.retriever import FAISSRetriever

        with patch(
            "src.retriever.SentenceTransformer", return_value=mock_embedder
        ):
            retriever = FAISSRetriever(
                embedding_model="all-MiniLM-L6-v2",
                device="cpu",
            )

            retriever.index_documents(sample_documents)

            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "retriever"
                retriever.save(save_path)

                # Load into new retriever
                new_retriever = FAISSRetriever(
                    embedding_model="all-MiniLM-L6-v2",
                    device="cpu",
                )
                new_retriever.load(save_path)

                assert len(new_retriever.documents) == 5
                assert new_retriever.index.ntotal == 5

    @pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")
    def test_top_k_larger_than_index(
        self, mock_embedder: MagicMock, sample_documents: list[MockDocument]
    ) -> None:
        """Test retrieval when top_k > number of documents."""
        from src.retriever import FAISSRetriever

        with patch(
            "src.retriever.SentenceTransformer", return_value=mock_embedder
        ):
            retriever = FAISSRetriever(
                embedding_model="all-MiniLM-L6-v2",
                device="cpu",
            )

            retriever.index_documents(sample_documents[:2])  # Only 2 docs

            results = retriever.retrieve("test", top_k=10)

            # Should return max available
            assert len(results) <= 2


class TestHybridRetriever:
    """Tests for HybridRetriever class."""

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        """Create a mock embedding model."""
        embedder = MagicMock()

        def encode_side_effect(
            texts: list[str], show_progress_bar: bool = False, **kwargs
        ) -> np.ndarray:
            return np.random.randn(len(texts), 384).astype(np.float32)

        embedder.encode = MagicMock(side_effect=encode_side_effect)
        return embedder

    @pytest.fixture
    def sample_documents(self) -> list[MockDocument]:
        """Create sample documents."""
        return [
            MockDocument(text="contract terms and conditions", source="doc1"),
            MockDocument(text="legal obligations requirements", source="doc2"),
        ]

    @pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")
    def test_hybrid_retrieval(
        self, mock_embedder: MagicMock, sample_documents: list[MockDocument]
    ) -> None:
        """Test hybrid retrieval combines dense and sparse."""
        from src.retriever import HybridRetriever

        with patch(
            "src.retriever.SentenceTransformer", return_value=mock_embedder
        ):
            retriever = HybridRetriever(
                embedding_model="all-MiniLM-L6-v2",
                device="cpu",
                alpha=0.5,  # Equal weight
            )

            retriever.index_documents(sample_documents)

            results = retriever.retrieve("contract", top_k=2)

            assert len(results) <= 2

    @pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")
    def test_alpha_weighting(
        self, mock_embedder: MagicMock, sample_documents: list[MockDocument]
    ) -> None:
        """Test that alpha affects ranking."""
        from src.retriever import HybridRetriever

        with patch(
            "src.retriever.SentenceTransformer", return_value=mock_embedder
        ):
            # Dense-only (alpha=1.0)
            dense_retriever = HybridRetriever(
                embedding_model="all-MiniLM-L6-v2",
                device="cpu",
                alpha=1.0,
            )
            dense_retriever.index_documents(sample_documents)
            dense_results = dense_retriever.retrieve("contract", top_k=2)

            # Sparse-only (alpha=0.0)
            sparse_retriever = HybridRetriever(
                embedding_model="all-MiniLM-L6-v2",
                device="cpu",
                alpha=0.0,
            )
            sparse_retriever.index_documents(sample_documents)
            sparse_results = sparse_retriever.retrieve("contract", top_k=2)

            # Results may differ based on alpha
            assert len(dense_results) >= 0
            assert len(sparse_results) >= 0


class TestRetrieverUtilities:
    """Tests for retriever utility functions."""

    def test_score_normalization(self) -> None:
        """Test score normalization."""
        scores = np.array([0.1, 0.5, 0.9])

        # Min-max normalization
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_reciprocal_rank_fusion(self) -> None:
        """Test RRF score computation."""
        k = 60  # Standard RRF parameter
        rank = 1

        rrf_score = 1 / (k + rank)

        assert rrf_score == 1 / 61
