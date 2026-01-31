"""Tests for activation steering module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.activation_steering import (
    ActivationSteering,
    HybridSteering,
    compute_faithfulness_metrics,
)


class TestActivationSteering:
    """Tests for ActivationSteering class."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock T5 model."""
        model = MagicMock()
        model.config.d_model = 768
        model.config.num_decoder_layers = 12

        # Mock decoder layers
        decoder_block = MagicMock()
        model.decoder.block = nn.ModuleList([decoder_block for _ in range(12)])

        return model

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer."""
        tokenizer = MagicMock()

        def encode_side_effect(text: str, return_tensors: str = None, **kwargs):
            result = MagicMock()
            result.input_ids = torch.randint(0, 100, (1, 10))
            result.__getitem__ = lambda self, key: getattr(result, key, torch.randint(0, 100, (1, 10)))
            result.to = lambda device: result
            return result

        tokenizer.side_effect = encode_side_effect
        tokenizer.return_value = encode_side_effect("")

        return tokenizer

    @pytest.fixture
    def steerer(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> ActivationSteering:
        """Create a test steerer."""
        return ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            device="cpu",
        )

    def test_initialization(self, steerer: ActivationSteering) -> None:
        """Test steerer initialization."""
        assert steerer.layer == 6
        assert steerer.device == "cpu"
        assert steerer.steering_vector is None

    def test_set_steering_vector(self, steerer: ActivationSteering) -> None:
        """Test setting steering vector."""
        vector = torch.randn(768)
        steerer.set_steering_vector(vector)

        assert steerer.steering_vector is not None
        assert steerer.steering_vector.shape == (768,)

    def test_set_steering_vector_normalized(self, steerer: ActivationSteering) -> None:
        """Test that steering vector is normalized when requested."""
        vector = torch.randn(768) * 10  # Large magnitude
        steerer.set_steering_vector(vector, normalize=True)

        norm = steerer.steering_vector.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_apply_context_manager(self, steerer: ActivationSteering) -> None:
        """Test apply as context manager."""
        vector = torch.randn(768)
        steerer.set_steering_vector(vector)

        with steerer.apply(multiplier=2.0):
            # Inside context, steering should be active
            assert steerer._hook_handle is not None

        # Outside context, hook should be removed
        assert steerer._hook_handle is None

    def test_apply_without_vector_raises(self, steerer: ActivationSteering) -> None:
        """Test that apply without vector raises error."""
        with pytest.raises((ValueError, RuntimeError)):
            with steerer.apply(multiplier=1.0):
                pass

    def test_save_and_load_vector(self, steerer: ActivationSteering) -> None:
        """Test saving and loading steering vector."""
        vector = torch.randn(768)
        steerer.set_steering_vector(vector)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "steering.pt"
            steerer.save_steering_vector(save_path)

            # Create new steerer and load
            new_steerer = ActivationSteering(
                model=steerer.model,
                tokenizer=steerer.tokenizer,
                layer=6,
                device="cpu",
            )
            new_steerer.load_steering_vector(save_path)

            torch.testing.assert_close(new_steerer.steering_vector, vector)


class TestHybridSteering:
    """Tests for HybridSteering class."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock T5 model."""
        model = MagicMock()
        model.config.d_model = 768
        model.config.num_decoder_layers = 12

        decoder_block = MagicMock()
        model.decoder.block = nn.ModuleList([decoder_block for _ in range(12)])

        return model

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        return tokenizer

    def test_initialization(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test hybrid steerer initialization."""
        from src.reft import ReFTIntervention

        intervention = ReFTIntervention(hidden_size=768, intervention_dim=16)

        hybrid = HybridSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            reft_intervention=intervention,
            reft_layer=6,
            steering_layer=6,
            device="cpu",
        )

        assert hybrid.reft_intervention is intervention
        assert hybrid.reft_layer == 6
        assert hybrid.steerer.layer == 6

    def test_enable_disable_reft(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test enabling/disabling ReFT."""
        from src.reft import ReFTIntervention

        intervention = ReFTIntervention(hidden_size=768, intervention_dim=16)

        hybrid = HybridSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            reft_intervention=intervention,
            reft_layer=6,
            steering_layer=6,
            device="cpu",
        )

        hybrid.enable_reft()
        assert hybrid.reft_enabled

        hybrid.disable_reft()
        assert not hybrid.reft_enabled


class TestFaithfulnessMetrics:
    """Tests for faithfulness metrics computation."""

    def test_basic_metrics(self) -> None:
        """Test basic faithfulness metrics."""
        context = "The contract is valid for 12 months."
        generated = "The contract is valid."

        metrics = compute_faithfulness_metrics(
            context=context,
            generated=generated,
        )

        assert "precision" in metrics
        assert "recall" in metrics
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1

    def test_perfect_recall(self) -> None:
        """Test perfect recall case."""
        context = "Hello world"
        generated = "Hello world extra text"

        metrics = compute_faithfulness_metrics(
            context=context,
            generated=generated,
        )

        assert metrics["recall"] == 1.0

    def test_perfect_precision(self) -> None:
        """Test perfect precision case."""
        context = "Hello world extra text"
        generated = "Hello world"

        metrics = compute_faithfulness_metrics(
            context=context,
            generated=generated,
        )

        assert metrics["precision"] == 1.0

    def test_no_overlap(self) -> None:
        """Test no overlap case."""
        context = "abc"
        generated = "xyz"

        metrics = compute_faithfulness_metrics(
            context=context,
            generated=generated,
        )

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_empty_strings(self) -> None:
        """Test empty string handling."""
        metrics = compute_faithfulness_metrics(
            context="",
            generated="",
        )

        # Should handle gracefully (either 0 or 1 depending on definition)
        assert "precision" in metrics
        assert "recall" in metrics


class TestIntegration:
    """Integration tests for steering components."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_steering_on_gpu(self) -> None:
        """Test steering works on GPU."""
        # This would require actual model on GPU
        pass

    def test_steering_vector_shape_mismatch(self) -> None:
        """Test error on shape mismatch."""
        mock_model = MagicMock()
        mock_model.config.d_model = 768
        mock_model.config.num_decoder_layers = 12

        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=MagicMock(),
            layer=6,
            device="cpu",
        )

        wrong_size_vector = torch.randn(512)  # Wrong size

        with pytest.raises((ValueError, RuntimeError)):
            steerer.set_steering_vector(wrong_size_vector)
