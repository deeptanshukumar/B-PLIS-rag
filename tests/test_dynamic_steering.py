"""
Tests for Dynamic Layer Selection in Activation Steering.

Tests cover:
- Single-layer mode (backward compatibility)
- Multi-layer mode
- Dynamic layer selection logic
- Time-aware decay
- Layer-specific multipliers
- Hook registration/removal (memory safety)
- Checkpoint save/load
- Edge cases
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.activation_steering import ActivationSteering, SteeringConfig
from src.model_loader import load_model


@pytest.fixture
def mock_model():
    """Create a mock T5 model for testing."""
    model = MagicMock()
    
    # Mock decoder blocks (12 layers like Flan-T5-base)
    decoder_blocks = []
    for i in range(12):
        block = MagicMock()
        # Make register_forward_hook return a mock handle
        mock_handle = MagicMock()
        mock_handle.remove = MagicMock()
        block.register_forward_hook.return_value = mock_handle
        decoder_blocks.append(block)
    model.decoder.block = decoder_blocks
    
    # Mock encoder blocks
    encoder_blocks = []
    for i in range(12):
        block = MagicMock()
        mock_handle = MagicMock()
        mock_handle.remove = MagicMock()
        block.register_forward_hook.return_value = mock_handle
        encoder_blocks.append(block)
    model.encoder.block = encoder_blocks
    
    # Mock parameters for device detection
    param = torch.nn.Parameter(torch.tensor([1.0]))
    model.parameters.return_value = [param]
    
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    return tokenizer


@pytest.fixture
def steering_config():
    """Default steering configuration."""
    return SteeringConfig(
        steering_layer=6,
        multiplier=2.0,
        steering_mode="dynamic",
        steering_layer_range=(3, 7),
        steering_max_steps=60,
        layer_multipliers={4: 2.0, 5: 1.5, 6: 1.5, 7: 1.0}
    )


class TestBackwardCompatibility:
    """Test that single-layer mode still works (backward compatibility)."""
    
    def test_single_layer_initialization(self, mock_model, mock_tokenizer):
        """Test initialization in single-layer mode."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="single",
            device=torch.device("cpu")
        )
        
        assert steerer.layer == 6
        assert steerer.steering_mode == "single"
        assert steerer.steering_vector is None
        assert len(steerer.steering_vectors) == 0
    
    def test_single_layer_returns_default(self, mock_model, mock_tokenizer):
        """Test that single mode always returns default layer."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="single",
            device=torch.device("cpu")
        )
        
        # Set a vector
        steerer.steering_vector = torch.randn(768)
        
        # Should always return default layer regardless of runtime state
        layers = steerer.select_layers({"retrieval_score": 0.9})
        assert layers == [6]
        
        layers = steerer.select_layers({"retrieval_score": 0.3})
        assert layers == [6]
    
    def test_set_steering_vector_backward_compat(self, mock_model, mock_tokenizer):
        """Test that set_steering_vector stores in both locations."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            device=torch.device("cpu")
        )
        
        vector = torch.randn(768)
        steerer.set_steering_vector(vector)
        
        # Should be in both locations
        assert steerer.steering_vector is not None
        assert torch.allclose(steerer.steering_vector, vector)
        assert 6 in steerer.steering_vectors
        assert torch.allclose(steerer.steering_vectors[6], vector)


class TestDynamicLayerSelection:
    """Test dynamic layer selection logic."""
    
    def test_high_confidence_selection(self, mock_model, mock_tokenizer):
        """Test layer selection for high retrieval confidence."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            layer_range=(3, 7),
            device=torch.device("cpu")
        )
        
        # Add vectors for all layers
        for layer in [3, 4, 5, 6, 7]:
            steerer.steering_vectors[layer] = torch.randn(768)
        
        # High confidence should select middle layers only (5, 6)
        runtime_state = {"retrieval_score": 0.85}
        layers = steerer.select_layers(runtime_state)
        
        assert set(layers) == {5, 6}
    
    def test_medium_confidence_selection(self, mock_model, mock_tokenizer):
        """Test layer selection for medium retrieval confidence."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            layer_range=(3, 7),
            device=torch.device("cpu")
        )
        
        for layer in [3, 4, 5, 6, 7]:
            steerer.steering_vectors[layer] = torch.randn(768)
        
        # Medium confidence should select middle + early (4, 5, 6)
        runtime_state = {"retrieval_score": 0.65}
        layers = steerer.select_layers(runtime_state)
        
        assert set(layers) == {4, 5, 6}
    
    def test_low_confidence_selection(self, mock_model, mock_tokenizer):
        """Test layer selection for low retrieval confidence."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            layer_range=(3, 7),
            device=torch.device("cpu")
        )
        
        for layer in [3, 4, 5, 6, 7]:
            steerer.steering_vectors[layer] = torch.randn(768)
        
        # Low confidence should select early + middle (3, 4, 5, 6)
        runtime_state = {"retrieval_score": 0.35}
        layers = steerer.select_layers(runtime_state)
        
        assert set(layers) == {3, 4, 5, 6}
    
    def test_late_generation_disables_steering(self, mock_model, mock_tokenizer):
        """Test that steering is disabled after max_steps."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            max_steering_steps=60,
            device=torch.device("cpu")
        )
        
        for layer in [3, 4, 5, 6, 7]:
            steerer.steering_vectors[layer] = torch.randn(768)
        
        # Late generation should return empty list
        runtime_state = {"retrieval_score": 0.9, "generation_step": 70}
        layers = steerer.select_layers(runtime_state)
        
        assert layers == []
    
    def test_mid_generation_reduces_layers(self, mock_model, mock_tokenizer):
        """Test that layers reduce after halfway point."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            max_steering_steps=60,
            device=torch.device("cpu")
        )
        
        for layer in [3, 4, 5, 6, 7]:
            steerer.steering_vectors[layer] = torch.randn(768)
        
        # Mid-generation (after halfway) should keep only central layers
        runtime_state = {"retrieval_score": 0.4, "generation_step": 40}
        layers = steerer.select_layers(runtime_state)
        
        # Should be subset of original, only 5-6
        assert set(layers).issubset({5, 6})
    
    def test_only_returns_layers_with_vectors(self, mock_model, mock_tokenizer):
        """Test that only layers with computed vectors are returned."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            layer_range=(3, 7),
            device=torch.device("cpu")
        )
        
        # Only add vectors for layers 5 and 6
        steerer.steering_vectors[5] = torch.randn(768)
        steerer.steering_vectors[6] = torch.randn(768)
        
        # Even with high range, should only return available layers
        runtime_state = {"retrieval_score": 0.4}  # Would select 3-6
        layers = steerer.select_layers(runtime_state)
        
        assert set(layers) == {5, 6}


class TestTimeAwareDecay:
    """Test time-aware decay functionality."""
    
    def test_early_generation_full_strength(self, mock_model, mock_tokenizer):
        """Test that early generation has full strength."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            max_steering_steps=60,
            device=torch.device("cpu")
        )
        
        # Step 0 should have full multiplier
        mult = steerer.get_layer_multiplier(6, base_multiplier=2.0, generation_step=0)
        assert mult == pytest.approx(2.0, rel=0.01)
    
    def test_mid_generation_reduced_strength(self, mock_model, mock_tokenizer):
        """Test that mid-generation has reduced strength."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            max_steering_steps=60,
            device=torch.device("cpu")
        )
        
        # Step 30 (halfway) should have 75% strength
        mult = steerer.get_layer_multiplier(6, base_multiplier=2.0, generation_step=30)
        expected = 2.0 * 0.75  # 50% decay at halfway
        assert mult == pytest.approx(expected, rel=0.01)
    
    def test_late_generation_minimum_strength(self, mock_model, mock_tokenizer):
        """Test that late generation has minimum 50% strength."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            max_steering_steps=60,
            device=torch.device("cpu")
        )
        
        # Step 60+ should have 50% minimum
        mult = steerer.get_layer_multiplier(6, base_multiplier=2.0, generation_step=60)
        expected = 2.0 * 0.5
        assert mult == pytest.approx(expected, rel=0.01)
    
    def test_layer_multiplier_applied(self, mock_model, mock_tokenizer):
        """Test that layer-specific multipliers are applied."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            max_steering_steps=60,
            layer_multipliers={4: 2.0, 5: 1.5, 6: 1.0},
            device=torch.device("cpu")
        )
        
        # Layer 4 with 2.0 multiplier
        mult = steerer.get_layer_multiplier(4, base_multiplier=2.0, generation_step=0)
        assert mult == pytest.approx(4.0, rel=0.01)  # 2.0 * 2.0
        
        # Layer 5 with 1.5 multiplier
        mult = steerer.get_layer_multiplier(5, base_multiplier=2.0, generation_step=0)
        assert mult == pytest.approx(3.0, rel=0.01)  # 2.0 * 1.5
        
        # Layer 6 with 1.0 multiplier
        mult = steerer.get_layer_multiplier(6, base_multiplier=2.0, generation_step=0)
        assert mult == pytest.approx(2.0, rel=0.01)  # 2.0 * 1.0


class TestHookManagement:
    """Test hook registration and removal (memory safety)."""
    
    def test_hooks_registered_for_active_layers(self, mock_model, mock_tokenizer):
        """Test that hooks are registered only for active layers."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            device=torch.device("cpu")
        )
        
        # Add vectors
        for layer in [4, 5, 6]:
            steerer.steering_vectors[layer] = torch.randn(768)
        
        # Apply with high confidence (should select layers 5, 6)
        runtime_state = {"retrieval_score": 0.9}
        steerer.apply_manual(multiplier=2.0, runtime_state=runtime_state)
        
        # Check that hooks were registered
        assert len(steerer.hook_handles) == 2  # Layers 5 and 6
        assert all(5 <= layer <= 6 for layer in steerer.hook_handles.keys())
        
        # Cleanup
        steerer.remove_manual()
    
    def test_hooks_removed_after_context(self, mock_model, mock_tokenizer):
        """Test that hooks are removed after context manager exits."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="single",
            device=torch.device("cpu")
        )
        
        steerer.steering_vector = torch.randn(768)
        
        # Apply and exit context
        with steerer.apply(multiplier=2.0):
            assert steerer.is_active
            assert len(steerer.hook_handles) > 0
        
        # After context, hooks should be removed
        assert not steerer.is_active
        assert len(steerer.hook_handles) == 0
    
    def test_hooks_removed_on_exception(self, mock_model, mock_tokenizer):
        """Test that hooks are removed even if exception occurs."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            device=torch.device("cpu")
        )
        
        steerer.steering_vector = torch.randn(768)
        
        # Exception during context should still cleanup hooks
        try:
            with steerer.apply(multiplier=2.0):
                assert steerer.is_active
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Hooks should be removed despite exception
        assert not steerer.is_active
        assert len(steerer.hook_handles) == 0
    
    def test_remove_manual_is_idempotent(self, mock_model, mock_tokenizer):
        """Test that remove_manual can be called multiple times safely."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            device=torch.device("cpu")
        )
        
        steerer.steering_vector = torch.randn(768)
        steerer.apply_manual(multiplier=2.0)
        
        # Call remove multiple times
        steerer.remove_manual()
        steerer.remove_manual()  # Should not crash
        steerer.remove_manual()
        
        assert not steerer.is_active
        assert len(steerer.hook_handles) == 0


class TestMultiLayerVectors:
    """Test multi-layer vector computation and management."""
    
    def test_set_steering_vectors(self, mock_model, mock_tokenizer):
        """Test batch setting of steering vectors."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            device=torch.device("cpu")
        )
        
        vectors = {
            4: torch.randn(768),
            5: torch.randn(768),
            6: torch.randn(768)
        }
        
        steerer.set_steering_vectors(vectors)
        
        assert len(steerer.steering_vectors) == 3
        for layer, vector in vectors.items():
            assert layer in steerer.steering_vectors
            assert torch.allclose(steerer.steering_vectors[layer], vector)
    
    def test_set_single_vector_different_layer(self, mock_model, mock_tokenizer):
        """Test setting vector for non-default layer."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            device=torch.device("cpu")
        )
        
        vector = torch.randn(768)
        steerer.set_steering_vector(vector, layer=5)
        
        assert 5 in steerer.steering_vectors
        assert torch.allclose(steerer.steering_vectors[5], vector)
        # Should not affect default steering_vector
        assert steerer.steering_vector is None


class TestCheckpointSaveLoad:
    """Test checkpoint saving and loading."""
    
    def test_save_multi_layer_checkpoint(self, mock_model, mock_tokenizer):
        """Test saving checkpoint with multiple layers."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            layer_range=(3, 7),
            max_steering_steps=60,
            layer_multipliers={4: 2.0, 5: 1.5},
            device=torch.device("cpu")
        )
        
        # Add vectors
        steerer.steering_vectors[4] = torch.randn(768)
        steerer.steering_vectors[5] = torch.randn(768)
        steerer.steering_vectors[6] = torch.randn(768)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            steerer.save(temp_path)
            
            # Load checkpoint and verify
            checkpoint = torch.load(temp_path)
            
            assert 'steering_vectors' in checkpoint
            assert len(checkpoint['steering_vectors']) == 3
            assert 'steering_mode' in checkpoint
            assert checkpoint['steering_mode'] == "dynamic"
            assert checkpoint['layer_range'] == (3, 7)
            assert checkpoint['max_steering_steps'] == 60
        finally:
            Path(temp_path).unlink()
    
    def test_load_multi_layer_checkpoint(self, mock_model, mock_tokenizer):
        """Test loading checkpoint with multiple layers."""
        steerer1 = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            device=torch.device("cpu")
        )
        
        # Create and save vectors
        vectors = {
            4: torch.randn(768),
            5: torch.randn(768),
            6: torch.randn(768)
        }
        steerer1.set_steering_vectors(vectors)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            steerer1.save(temp_path)
            
            # Load into new steerer
            steerer2 = ActivationSteering(
                model=mock_model,
                tokenizer=mock_tokenizer,
                layer=6,
                device=torch.device("cpu")
            )
            steerer2.load(temp_path)
            
            # Verify vectors loaded
            assert len(steerer2.steering_vectors) == 3
            for layer in [4, 5, 6]:
                assert layer in steerer2.steering_vectors
                assert torch.allclose(steerer2.steering_vectors[layer], vectors[layer])
        finally:
            Path(temp_path).unlink()
    
    def test_backward_compat_old_checkpoint(self, mock_model, mock_tokenizer):
        """Test loading old single-layer checkpoint format."""
        # Create old-style checkpoint
        vector = torch.randn(768)
        checkpoint = {
            'steering_vector': vector,
            'layer': 6,
            'component': 'decoder'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            torch.save(checkpoint, temp_path)
        
        try:
            # Load with new code
            steerer = ActivationSteering(
                model=mock_model,
                tokenizer=mock_tokenizer,
                layer=6,
                device=torch.device("cpu")
            )
            steerer.load(temp_path)
            
            # Should load into both locations
            assert steerer.steering_vector is not None
            assert 6 in steerer.steering_vectors
            assert torch.allclose(steerer.steering_vector, vector)
        finally:
            Path(temp_path).unlink()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_apply_without_vectors_raises_error(self, mock_model, mock_tokenizer):
        """Test that apply raises error if no vectors computed."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            device=torch.device("cpu")
        )
        
        with pytest.raises(RuntimeError, match="No steering vectors computed"):
            with steerer.apply(multiplier=2.0):
                pass
    
    def test_out_of_range_layer_handled_gracefully(self, mock_model, mock_tokenizer):
        """Test that out-of-range layers are handled gracefully."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            layer_range=(3, 20),  # 20 is out of range for 12-layer model
            device=torch.device("cpu")
        )
        
        # Add vectors including out-of-range
        steerer.steering_vectors[5] = torch.randn(768)
        steerer.steering_vectors[6] = torch.randn(768)
        steerer.steering_vectors[15] = torch.randn(768)  # Out of range
        
        # Should only return valid layers
        runtime_state = {"retrieval_score": 0.9}
        layers = steerer.select_layers(runtime_state)
        
        assert all(layer < 12 for layer in layers)
    
    def test_empty_layer_selection_no_crash(self, mock_model, mock_tokenizer):
        """Test that empty layer selection doesn't crash."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            max_steering_steps=10,
            device=torch.device("cpu")
        )
        
        steerer.steering_vectors[6] = torch.randn(768)
        
        # Late generation returns empty list
        runtime_state = {"retrieval_score": 0.9, "generation_step": 100}
        
        # Should not crash, just no hooks registered
        steerer.apply_manual(multiplier=2.0, runtime_state=runtime_state)
        assert steerer.is_active
        assert len(steerer.hook_handles) == 0
        
        steerer.remove_manual()
    
    def test_default_runtime_state_values(self, mock_model, mock_tokenizer):
        """Test that missing runtime_state keys use defaults."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            device=torch.device("cpu")
        )
        
        steerer.steering_vectors[6] = torch.randn(768)
        
        # Empty runtime_state should use defaults
        layers = steerer.select_layers({})
        assert isinstance(layers, list)
        
        # None should also work
        layers = steerer.select_layers(None)
        assert isinstance(layers, list)


class TestIntegration:
    """Integration tests with real (small) model."""
    
    @pytest.mark.slow
    def test_real_model_single_layer(self):
        """Test with real T5-small model (single layer mode)."""
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        steerer = ActivationSteering(
            model=model,
            tokenizer=tokenizer,
            layer=3,
            steering_mode="single",
            device=torch.device("cpu")
        )
        
        # Create dummy steering vector
        hidden_size = model.config.d_model
        steerer.steering_vector = torch.randn(hidden_size)
        
        # Test generation with steering
        prompt = "What is the capital of France?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with steerer.apply(multiplier=1.0):
            outputs = model.generate(**inputs, max_new_tokens=10)
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    @pytest.mark.slow
    def test_real_model_multi_layer(self):
        """Test with real T5-small model (multi-layer dynamic mode)."""
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        steerer = ActivationSteering(
            model=model,
            tokenizer=tokenizer,
            layer=3,
            steering_mode="dynamic",
            layer_range=(2, 4),
            device=torch.device("cpu")
        )
        
        # Create dummy steering vectors
        hidden_size = model.config.d_model
        for layer in [2, 3, 4]:
            steerer.steering_vectors[layer] = torch.randn(hidden_size)
        
        # Test generation with dynamic steering
        prompt = "What is the capital of France?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        runtime_state = {"retrieval_score": 0.85}
        with steerer.apply(multiplier=1.0, runtime_state=runtime_state):
            outputs = model.generate(**inputs, max_new_tokens=10)
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestSteeringHook:
    """Test the steering hook function itself."""
    
    def test_hook_applies_steering_to_active_layer(self, mock_model, mock_tokenizer):
        """Test that hook modifies activations for active layers."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            device=torch.device("cpu")
        )
        
        # Setup
        hidden_size = 768
        steering_vector = torch.randn(hidden_size)
        steerer.steering_vectors[6] = steering_vector
        steerer._active_layers = [6]
        steerer.is_active = True
        steerer.multiplier = 2.0
        
        # Create mock hidden states
        batch_size, seq_len = 1, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        original_hidden = hidden_states.clone()
        
        # Get the target module (layer 6)
        target_module = mock_model.decoder.block[6]
        
        # Call hook
        output = (hidden_states,)  # Tuple format
        result = steerer._steering_hook(target_module, (), output)
        
        # Verify steering was applied
        modified_hidden = result[0]
        assert not torch.allclose(modified_hidden, original_hidden)
    
    def test_hook_inactive_no_modification(self, mock_model, mock_tokenizer):
        """Test that inactive hook doesn't modify activations."""
        steerer = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            device=torch.device("cpu")
        )
        
        # Setup but keep inactive
        steerer.is_active = False
        
        hidden_states = torch.randn(1, 10, 768)
        original = hidden_states.clone()
        
        target_module = mock_model.decoder.block[6]
        output = (hidden_states,)
        result = steerer._steering_hook(target_module, (), output)
        
        # Should be unchanged
        assert torch.allclose(result[0], original)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
