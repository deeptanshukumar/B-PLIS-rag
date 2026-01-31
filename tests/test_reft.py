"""Tests for ReFT intervention module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.reft import ReFTHook, ReFTIntervention, ReFTTrainer


class TestReFTIntervention:
    """Tests for ReFTIntervention class."""

    @pytest.fixture
    def intervention(self) -> ReFTIntervention:
        """Create a test intervention."""
        return ReFTIntervention(
            hidden_size=768,
            intervention_dim=16,
            init_std=0.02,
        )

    def test_initialization(self, intervention: ReFTIntervention) -> None:
        """Test intervention initialization."""
        assert intervention.hidden_size == 768
        assert intervention.intervention_dim == 16
        assert intervention.z.shape == (16,)
        assert intervention.proj.in_features == 16
        assert intervention.proj.out_features == 768

    def test_forward_shape(self, intervention: ReFTIntervention) -> None:
        """Test forward pass output shape."""
        # Batch of hidden states
        hidden = torch.randn(2, 10, 768)  # [batch, seq, hidden]

        output = intervention(hidden)

        assert output.shape == hidden.shape

    def test_forward_with_zero_z(self, intervention: ReFTIntervention) -> None:
        """Test that zero z produces no change."""
        hidden = torch.randn(2, 10, 768)
        intervention.z.data.zero_()

        output = intervention(hidden)

        torch.testing.assert_close(output, hidden, atol=1e-6, rtol=1e-6)

    def test_intervention_adds_delta(self, intervention: ReFTIntervention) -> None:
        """Test that intervention adds a delta to hidden states."""
        hidden = torch.randn(2, 10, 768)
        # Set non-zero z
        intervention.z.data.fill_(1.0)

        output = intervention(hidden)

        # Output should be different
        assert not torch.allclose(output, hidden)

    def test_num_parameters(self, intervention: ReFTIntervention) -> None:
        """Test parameter counting."""
        # z: 16 params
        # proj: 16 * 768 weights + 768 bias = 12288 + 768 = 13056
        # Total: 16 + 13056 = 13072
        num_params = intervention.num_parameters()
        assert num_params == 16 + (16 * 768 + 768)

    def test_gradient_flow(self, intervention: ReFTIntervention) -> None:
        """Test that gradients flow through intervention."""
        hidden = torch.randn(2, 10, 768, requires_grad=True)
        output = intervention(hidden)

        # Backprop
        loss = output.sum()
        loss.backward()

        assert intervention.z.grad is not None
        assert intervention.proj.weight.grad is not None


class TestReFTHook:
    """Tests for ReFTHook class."""

    @pytest.fixture
    def hook(self) -> ReFTHook:
        """Create a test hook."""
        intervention = ReFTIntervention(hidden_size=768, intervention_dim=16)
        return ReFTHook(intervention)

    def test_hook_modifies_output(self, hook: ReFTHook) -> None:
        """Test that hook modifies layer output."""
        # Create dummy module and output
        module = nn.Linear(768, 768)
        input_tensor = torch.randn(2, 10, 768)
        output_tensor = torch.randn(2, 10, 768)

        # Set non-zero z
        hook.intervention.z.data.fill_(1.0)

        # Apply hook
        modified = hook(module, (input_tensor,), output_tensor)

        assert modified is not None
        assert not torch.allclose(modified, output_tensor)

    def test_hook_disabled(self, hook: ReFTHook) -> None:
        """Test that disabled hook doesn't modify output."""
        module = nn.Linear(768, 768)
        input_tensor = torch.randn(2, 10, 768)
        output_tensor = torch.randn(2, 10, 768)

        hook.enabled = False
        modified = hook(module, (input_tensor,), output_tensor)

        assert modified is None or torch.allclose(modified, output_tensor)

    def test_captured_activations(self, hook: ReFTHook) -> None:
        """Test activation capture."""
        module = nn.Linear(768, 768)
        input_tensor = torch.randn(2, 10, 768)
        output_tensor = torch.randn(2, 10, 768)

        hook.capture = True
        hook(module, (input_tensor,), output_tensor)

        assert hook.captured_input is not None
        assert hook.captured_output is not None


class TestReFTTrainer:
    """Tests for ReFTTrainer class."""

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
            result.__getitem__ = lambda self, key: torch.randint(0, 100, (1, 10))
            result.to = lambda device: result
            return result

        tokenizer.side_effect = encode_side_effect
        tokenizer.return_value = encode_side_effect("")

        return tokenizer

    def test_trainer_initialization(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test trainer initialization."""
        intervention = ReFTIntervention(hidden_size=768, intervention_dim=16)

        trainer = ReFTTrainer(
            model=mock_model,
            intervention=intervention,
            tokenizer=mock_tokenizer,
            target_layer=6,
            learning_rate=1e-2,
            num_steps=10,
        )

        assert trainer.target_layer == 6
        assert trainer.learning_rate == 1e-2
        assert trainer.num_steps == 10

    def test_save_and_load(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test saving and loading trainer state."""
        intervention = ReFTIntervention(hidden_size=768, intervention_dim=16)

        trainer = ReFTTrainer(
            model=mock_model,
            intervention=intervention,
            tokenizer=mock_tokenizer,
            target_layer=6,
        )

        # Set some state
        intervention.z.data.fill_(1.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "checkpoint.pt"
            trainer.save(save_path)

            # Create new trainer and load
            new_intervention = ReFTIntervention(hidden_size=768, intervention_dim=16)
            new_trainer = ReFTTrainer(
                model=mock_model,
                intervention=new_intervention,
                tokenizer=mock_tokenizer,
                target_layer=6,
            )
            new_trainer.load(save_path)

            # Check state restored
            torch.testing.assert_close(new_intervention.z, intervention.z)


class TestIntegration:
    """Integration tests for ReFT components."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_intervention_on_gpu(self) -> None:
        """Test intervention works on GPU."""
        intervention = ReFTIntervention(hidden_size=768, intervention_dim=16)
        intervention = intervention.cuda()

        hidden = torch.randn(2, 10, 768, device="cuda")
        output = intervention(hidden)

        assert output.device.type == "cuda"
        assert output.shape == hidden.shape

    def test_multiple_interventions(self) -> None:
        """Test applying multiple interventions."""
        intervention1 = ReFTIntervention(hidden_size=768, intervention_dim=16)
        intervention2 = ReFTIntervention(hidden_size=768, intervention_dim=8)

        hidden = torch.randn(2, 10, 768)

        # Apply sequentially
        output1 = intervention1(hidden)
        output2 = intervention2(output1)

        assert output2.shape == hidden.shape
        assert not torch.allclose(output2, hidden)
