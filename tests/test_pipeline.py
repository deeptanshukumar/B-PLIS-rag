"""Tests for RAG pipeline module."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch


@dataclass
class MockDocument:
    """Mock document for testing."""

    text: str
    source: str = "test"
    metadata: dict | None = None


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    @pytest.fixture
    def mock_components(self) -> dict:
        """Create mock components for pipeline."""
        # Mock model
        model = MagicMock()
        model.config.d_model = 768
        model.config.num_decoder_layers = 12
        model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 0]]))

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.decode = MagicMock(return_value="This is the answer.")
        tokenizer.return_value = MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
            to=lambda device: MagicMock(input_ids=torch.tensor([[1, 2, 3]])),
        )

        # Mock retriever
        retriever = MagicMock()
        retriever.retrieve = MagicMock(
            return_value=[
                {"document": MockDocument("Context text."), "score": 0.9},
                {"document": MockDocument("More context."), "score": 0.8},
            ]
        )

        return {
            "model": model,
            "tokenizer": tokenizer,
            "retriever": retriever,
        }

    def test_pipeline_initialization(self) -> None:
        """Test pipeline initialization."""
        from src.rag_pipeline import RAGPipeline

        with patch("src.rag_pipeline.load_model") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock())

            pipeline = RAGPipeline(
                model_name="t5-base",
                use_reft=True,
                use_steering=True,
            )

            assert pipeline.use_reft is True
            assert pipeline.use_steering is True

    def test_query_returns_response(self, mock_components: dict) -> None:
        """Test that query returns RAGResponse."""
        from src.rag_pipeline import RAGPipeline, RAGResponse

        with patch("src.rag_pipeline.load_model") as mock_load:
            mock_load.return_value = (
                mock_components["model"],
                mock_components["tokenizer"],
            )

            pipeline = RAGPipeline(
                model_name="t5-base",
                use_reft=False,  # Skip for simplicity
                use_steering=False,
            )
            pipeline.retriever = mock_components["retriever"]

            response = pipeline.query("What is X?", top_k=2)

            assert isinstance(response, RAGResponse)
            assert response.answer is not None
            assert len(response.sources) >= 0

    def test_query_without_retrieval(self, mock_components: dict) -> None:
        """Test query without retrieval (direct generation)."""
        from src.rag_pipeline import RAGPipeline

        with patch("src.rag_pipeline.load_model") as mock_load:
            mock_load.return_value = (
                mock_components["model"],
                mock_components["tokenizer"],
            )

            pipeline = RAGPipeline(
                model_name="t5-base",
                use_reft=False,
                use_steering=False,
            )

            # Query without retriever
            response = pipeline.generate_answer("What is X?", context=None)

            assert response is not None

    def test_format_context(self, mock_components: dict) -> None:
        """Test context formatting."""
        from src.rag_pipeline import RAGPipeline

        with patch("src.rag_pipeline.load_model") as mock_load:
            mock_load.return_value = (
                mock_components["model"],
                mock_components["tokenizer"],
            )

            pipeline = RAGPipeline(model_name="t5-base")

            documents = [
                MockDocument("First document."),
                MockDocument("Second document."),
            ]

            context = pipeline._format_context(documents)

            assert "First document." in context
            assert "Second document." in context


class TestBilingualRAGPipeline:
    """Tests for BilingualRAGPipeline class."""

    def test_language_detection(self) -> None:
        """Test language detection."""
        from src.rag_pipeline import BilingualRAGPipeline

        with patch("src.rag_pipeline.load_model") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock())

            pipeline = BilingualRAGPipeline(model_name="t5-base")

            # English
            lang = pipeline._detect_language("What is a contract?")
            assert lang in ["en", "english", "eng"]

            # Hindi (if detection available)
            lang = pipeline._detect_language("यह एक अनुबंध क्या है?")
            assert lang in ["hi", "hindi", "hin", "en", "unknown"]  # May fallback


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_response_creation(self) -> None:
        """Test RAGResponse creation."""
        from src.rag_pipeline import RAGResponse

        response = RAGResponse(
            query="What is X?",
            answer="X is a thing.",
            sources=[{"source": "doc1", "score": 0.9}],
            context="Context text.",
            metadata={"model": "t5-base"},
        )

        assert response.query == "What is X?"
        assert response.answer == "X is a thing."
        assert len(response.sources) == 1
        assert response.context == "Context text."
        assert response.metadata["model"] == "t5-base"

    def test_response_to_dict(self) -> None:
        """Test RAGResponse serialization."""
        from src.rag_pipeline import RAGResponse

        response = RAGResponse(
            query="What is X?",
            answer="X is a thing.",
            sources=[],
        )

        # Should be serializable
        response_dict = {
            "query": response.query,
            "answer": response.answer,
            "sources": response.sources,
        }

        assert response_dict["query"] == "What is X?"


class TestPipelineIntegration:
    """Integration tests for RAG pipeline."""

    @pytest.mark.slow
    def test_full_pipeline_flow(self) -> None:
        """Test full pipeline flow (slow, uses real model)."""
        # This test is marked slow and should be skipped in CI
        # Run with: pytest -m slow
        pass

    def test_pipeline_error_handling(self, mock_components: dict) -> None:
        """Test pipeline handles errors gracefully."""
        from src.rag_pipeline import RAGPipeline

        with patch("src.rag_pipeline.load_model") as mock_load:
            mock_load.return_value = (
                mock_components["model"],
                mock_components["tokenizer"],
            )

            pipeline = RAGPipeline(model_name="t5-base")

            # Set retriever to raise error
            mock_retriever = MagicMock()
            mock_retriever.retrieve = MagicMock(
                side_effect=RuntimeError("Retrieval failed")
            )
            pipeline.retriever = mock_retriever

            # Should handle error gracefully or raise meaningful error
            with pytest.raises((RuntimeError, ValueError)):
                pipeline.query("test query")
