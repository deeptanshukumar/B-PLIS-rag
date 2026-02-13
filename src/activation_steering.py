"""
Activation Steering for B-PLIS-RAG.

Implements ContextFocus-inspired activation steering to enhance context
faithfulness in RAG systems. Based on techniques from arXiv:2601.04131.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import get_config
from src.utils import HookManager

logger = logging.getLogger(__name__)


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""
    steering_layer: int = 13  # Layer for steering (equiv to layer 13 in larger models)
    multiplier: float = 2.0  # Steering strength
    component: str = "decoder"  # Model component to steer
    position: str = "last"  # Position to extract activations: "last", "mean", "all"
    # Dynamic layer selection
    steering_mode: str = "single"  # "single" or "dynamic"
    steering_layer_range: Tuple[int, int] = (3, 7)  # Valid range for dynamic selection
    steering_max_steps: int = 60  # Max generation steps to apply steering
    layer_multipliers: Optional[dict] = None  # Per-layer multipliers (override global)


class ActivationSteering:
    """
    Activation steering for context-faithful generation with dynamic layer selection.
    
    Computes steering vectors as mean activation differences between context-aware
    and context-free prompts. Supports:
    - Multi-layer steering (one vector per layer)
    - Dynamic layer selection based on retrieval confidence & generation timestep
    - Time-aware steering (fades over generation steps)
    - Layer-specific multipliers
    
    Based on ContextFocus (arXiv:2601.04131) with dynamic extensions.
    
    Design rationale:
    - Early layers (3-4): Capture low-level context patterns
    - Middle layers (5-6): Core semantic steering - most effective
    - Late layers (7-8): Fine-grained alignment, weaker steering to preserve fluency
    
    Example:
        >>> steerer = ActivationSteering(model, tokenizer, layer_range=(3, 7))
        >>> steerer.compute_steering_vectors(positive_prompts, negative_prompts, layers=[4,5,6])
        >>> with steerer.apply(multiplier=2.0, runtime_state={"retrieval_score": 0.85}):
        ...     output = model.generate(input_ids)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        layer: int = 6,
        component: str = "decoder",
        device: Optional[torch.device] = None,
        # Dynamic steering parameters
        layer_range: Optional[Tuple[int, int]] = None,
        max_steering_steps: int = 60,
        layer_multipliers: Optional[dict] = None,
        steering_mode: str = "single",
    ) -> None:
        """
        Initialize activation steering.
        
        Args:
            model: T5 model.
            tokenizer: Tokenizer for the model.
            layer: Default layer index (used in single mode or as fallback).
            component: Model component ('encoder' or 'decoder'). Use 'decoder'.
            device: Device for computations.
            layer_range: Valid range for dynamic selection, e.g. (3, 7).
            max_steering_steps: Max generation steps before disabling steering.
            layer_multipliers: Dict mapping layer -> multiplier, e.g. {4: 2.0, 5: 1.5}.
            steering_mode: "single" (backward compat) or "dynamic" (adaptive).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer  # Default/fallback layer
        self.component = component
        self.device = device or next(model.parameters()).device
        
        # Dynamic steering configuration
        self.steering_mode = steering_mode
        self.layer_range = layer_range or (3, 7)  # Default: middle layers
        self.max_steering_steps = max_steering_steps
        self.layer_multipliers = layer_multipliers or {}
        
        # Multi-layer steering vectors: layer_idx -> steering_vector
        # Backward compatible: if only one vector exists, behaves like old code
        self.steering_vectors: dict[int, torch.Tensor] = {}
        
        # Legacy single-vector support (for backward compatibility)
        self.steering_vector: Optional[torch.Tensor] = None
        
        # Hook management: layer_idx -> hook_handle
        self.hook_handles: dict[int, torch.utils.hooks.RemovableHandle] = {}
        self.is_active = False
        self.multiplier = 2.0  # Global multiplier
        
        # Runtime state tracking (for time-aware steering)
        self._generation_step = 0
        self._active_layers: List[int] = []
        
        # Get target modules for all potential layers
        if component == "encoder":
            self.target_modules = model.encoder.block
        else:
            self.target_modules = model.decoder.block
        
        # For capturing activations during vector computation
        self._captured_activations: List[torch.Tensor] = []
        
        mode_str = f"mode={steering_mode}, range={self.layer_range}"
        logger.info(f"ActivationSteering initialized at {component} ({mode_str})")
    
    def select_layers(self, runtime_state: Optional[dict] = None) -> List[int]:
        """
        Dynamically select which layers to apply steering to based on runtime signals.
        
        Layer selection strategy:
        - HIGH retrieval confidence (>0.8): Middle layers only (5-6)
          → Strong context signal, focus on semantic core
        - MEDIUM confidence (0.5-0.8): Middle + early layers (4-6)
          → Moderate signal, broader steering
        - LOW confidence (<0.5): Early + middle layers (3-6)
          → Weak signal, compensate with wider range
        - LATE generation (>max_steps): Disable all steering
          → Preserve fluency in final tokens
        
        Args:
            runtime_state: Dict with optional keys:
                - "retrieval_score": float in [0, 1], higher = more confident
                - "retrieval_margin": float, score gap between top and 2nd result
                - "generation_step": int, current decoding timestep
                
        Returns:
            List of layer indices to steer. Empty list disables steering.
        """
        # If in single-layer mode, always return the default layer
        if self.steering_mode == "single":
            has_vectors = (self.steering_vector is not None) or len(self.steering_vectors) > 0
            return [self.layer] if has_vectors else []
        
        # Extract runtime signals
        runtime_state = runtime_state or {}
        retrieval_score = runtime_state.get("retrieval_score", 0.7)  # Default: medium confidence
        generation_step = runtime_state.get("generation_step", 0)
        
        # Time-aware steering: disable after max_steps to preserve fluency
        if generation_step >= self.max_steering_steps:
            return []  # Stop steering in late generation
        
        # Confidence-based layer selection
        min_layer, max_layer = self.layer_range
        
        if retrieval_score >= 0.8:
            # HIGH confidence: steer middle layers only (core semantic layers)
            # Rationale: Strong context signal, focus on semantic alignment
            selected = list(range(5, min(7, max_layer + 1)))
        elif retrieval_score >= 0.5:
            # MEDIUM confidence: middle + early layers
            # Rationale: Moderate signal, use broader range for robustness
            selected = list(range(4, min(7, max_layer + 1)))
        else:
            # LOW confidence: early + middle layers
            # Rationale: Weak signal, compensate with wider steering
            selected = list(range(max(3, min_layer), min(7, max_layer + 1)))
        
        # Filter to only layers with computed steering vectors
        selected = [l for l in selected if l in self.steering_vectors]
        
        # Early generation: full steering; late generation: gradual fade
        # Apply time-based modulation (reduce layers after halfway point)
        if generation_step > self.max_steering_steps // 2:
            # Keep only central layers for late-stage refinement
            selected = [l for l in selected if 5 <= l <= 6]
        
        return selected
    
    def get_layer_multiplier(self, layer: int, base_multiplier: float, generation_step: int = 0) -> float:
        """
        Compute effective multiplier for a specific layer and timestep.
        
        Layer-specific multipliers encode the principle that different layers
        need different steering strengths:
        - Early layers (3-4): Strong steering (2.0+) - shape context representation
        - Middle layers (5-6): Moderate steering (1.5-2.0) - semantic core
        - Late layers (7+): Weak steering (1.0) - preserve fluency
        
        Time decay: Reduce strength as generation progresses to avoid
        over-steering final tokens.
        
        Args:
            layer: Layer index.
            base_multiplier: Global multiplier.
            generation_step: Current generation timestep.
            
        Returns:
            Effective multiplier (float).
        """
        # Layer-specific multiplier (if configured)
        layer_mult = self.layer_multipliers.get(layer, 1.0)
        
        # Base multiplier
        effective = base_multiplier * layer_mult
        
        # Time decay: Linear fade from full strength to 50% over max_steps
        # Rationale: Early tokens need strong steering, late tokens need fluency
        if self.max_steering_steps > 0:
            decay_factor = max(0.5, 1.0 - 0.5 * (generation_step / self.max_steering_steps))
            effective *= decay_factor
        
        return effective
    
    def _capture_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Tuple[torch.Tensor, ...],
    ) -> None:
        """Hook to capture activations for steering vector computation."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Get last token activation
        last_token_activation = hidden_states[:, -1, :].detach().clone()
        self._captured_activations.append(last_token_activation)
    
    def _steering_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Hook to apply steering during generation (multi-layer version).
        
        This hook is registered on each active layer. It:
        1. Checks if steering is active for this layer
        2. Retrieves the layer-specific steering vector
        3. Applies layer-specific multiplier with time decay
        4. Adds steering to hidden states
        
        Args:
            module: The transformer block being hooked.
            input: Module inputs.
            output: Module outputs (hidden_states, ...).
            
        Returns:
            Modified output with steering applied.
        """
        if not self.is_active:
            return output
        
        # Find which layer this module corresponds to
        layer_idx = None
        for idx, target_module in enumerate(self.target_modules):
            if module is target_module:
                layer_idx = idx
                break
        
        # If this layer is not in active layers or has no vector, skip
        if layer_idx is None or layer_idx not in self._active_layers:
            return output
        
        # Get steering vector for this layer (check both dict and legacy single vector)
        steering_vector = self.steering_vectors.get(layer_idx)
        if steering_vector is None and layer_idx == self.layer:
            steering_vector = self.steering_vector
        
        if steering_vector is None:
            return output
        
        # Unpack output
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()
        
        # Compute effective multiplier with time decay
        effective_multiplier = self.get_layer_multiplier(
            layer_idx, 
            self.multiplier, 
            self._generation_step
        )
        
        # Apply steering vector (broadcast across batch and sequence)
        # Shape: [1, 1, hidden_dim] -> broadcasts to [batch, seq, hidden_dim]
        steering = steering_vector.unsqueeze(0).unsqueeze(0)
        steering = steering.to(hidden_states.device, dtype=hidden_states.dtype)
        steering = steering * effective_multiplier
        
        modified = hidden_states + steering
        
        # Increment generation step (approximation - counts hook calls)
        # More accurate tracking would use generate() callbacks, but this is simpler
        self._generation_step += 1
        
        if rest:
            return (modified,) + rest
        return modified
    
    def compute_steering_vector(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute a single steering vector (backward compatible API).
        
        This method maintains backward compatibility by computing steering
        for self.layer and storing in both self.steering_vector (legacy)
        and self.steering_vectors[self.layer] (new multi-layer dict).
        
        Positive prompts include context, negative prompts do not.
        The steering vector is the mean difference of last-token activations.
        
        Args:
            positive_prompts: Prompts with context (e.g., "Use context: X. Question: Y").
            negative_prompts: Prompts without context (e.g., "Question: Y").
            normalize: Whether to L2-normalize the steering vector.
            
        Returns:
            The computed steering vector.
        """
        if len(positive_prompts) != len(negative_prompts):
            raise ValueError("Must have equal number of positive and negative prompts")
        
        logger.info(f"Computing steering vector from {len(positive_prompts)} prompt pairs")
        
        positive_activations = []
        negative_activations = []
        
        # Get target module for the default layer
        target_module = self.target_modules[self.layer]
        
        # Register capture hook
        handle = target_module.register_forward_hook(self._capture_hook)
        
        try:
            self.model.eval()
            
            with torch.no_grad():
                # Collect positive activations
                for prompt in tqdm(positive_prompts, desc="Positive prompts"):
                    self._captured_activations.clear()
                    
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    
                    # For encoder-decoder models, we need to run the encoder
                    # and get decoder activations during generation
                    _ = self.model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                    )
                    
                    if self._captured_activations:
                        positive_activations.append(self._captured_activations[0])
                
                # Collect negative activations
                for prompt in tqdm(negative_prompts, desc="Negative prompts"):
                    self._captured_activations.clear()
                    
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    
                    _ = self.model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                    )
                    
                    if self._captured_activations:
                        negative_activations.append(self._captured_activations[0])
        
        finally:
            handle.remove()
            self._captured_activations.clear()
        
        if not positive_activations or not negative_activations:
            raise RuntimeError("Failed to capture activations")
        
        # Stack and compute mean
        positive_mean = torch.stack(positive_activations).mean(dim=0)  # [batch=1, hidden]
        negative_mean = torch.stack(negative_activations).mean(dim=0)
        
        # Compute steering vector as difference
        steering_vector = positive_mean - negative_mean  # [batch=1, hidden]
        steering_vector = steering_vector.squeeze(0)  # [hidden]
        
        # Optionally normalize
        if normalize:
            steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
        
        # Store in both legacy and new locations
        self.steering_vector = steering_vector
        self.steering_vectors[self.layer] = steering_vector
        
        logger.info(f"Steering vector computed for layer {self.layer}, norm: {steering_vector.norm().item():.4f}")
        
        return steering_vector
    
    def compute_steering_vectors(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layers: Optional[List[int]] = None,
        normalize: bool = True,
    ) -> dict[int, torch.Tensor]:
        """
        Compute steering vectors for multiple layers (new multi-layer API).
        
        This is the recommended method for dynamic steering. It computes
        separate steering vectors for each specified layer, allowing
        layer-specific adaptation.
        
        Args:
            positive_prompts: Prompts with context.
            negative_prompts: Prompts without context.
            layers: List of layer indices to compute vectors for.
                    If None, uses self.layer_range.
            normalize: Whether to L2-normalize each vector.
            
        Returns:
            Dict mapping layer_idx -> steering_vector.
        """
        if len(positive_prompts) != len(negative_prompts):
            raise ValueError("Must have equal number of positive and negative prompts")
        
        if layers is None:
            layers = list(range(self.layer_range[0], self.layer_range[1] + 1))
        
        logger.info(f"Computing steering vectors for layers {layers} from {len(positive_prompts)} pairs")
        
        computed_vectors = {}
        
        for layer_idx in layers:
            if layer_idx >= len(self.target_modules):
                logger.warning(f"Layer {layer_idx} out of range, skipping")
                continue
            
            logger.info(f"Computing steering vector for layer {layer_idx}")
            
            positive_activations = []
            negative_activations = []
            
            target_module = self.target_modules[layer_idx]
            handle = target_module.register_forward_hook(self._capture_hook)
            
            try:
                self.model.eval()
                
                with torch.no_grad():
                    # Positive activations
                    for prompt in tqdm(positive_prompts, desc=f"Layer {layer_idx} positive", leave=False):
                        self._captured_activations.clear()
                        
                        inputs = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512,
                        ).to(self.device)
                        
                        _ = self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
                        
                        if self._captured_activations:
                            positive_activations.append(self._captured_activations[0])
                    
                    # Negative activations
                    for prompt in tqdm(negative_prompts, desc=f"Layer {layer_idx} negative", leave=False):
                        self._captured_activations.clear()
                        
                        inputs = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512,
                        ).to(self.device)
                        
                        _ = self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
                        
                        if self._captured_activations:
                            negative_activations.append(self._captured_activations[0])
            
            finally:
                handle.remove()
                self._captured_activations.clear()
            
            if not positive_activations or not negative_activations:
                logger.warning(f"Failed to capture activations for layer {layer_idx}")
                continue
            
            # Compute steering vector
            positive_mean = torch.stack(positive_activations).mean(dim=0).squeeze(0)
            negative_mean = torch.stack(negative_activations).mean(dim=0).squeeze(0)
            steering_vector = positive_mean - negative_mean
            
            if normalize:
                steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
            
            computed_vectors[layer_idx] = steering_vector
            logger.info(f"  Layer {layer_idx}: norm={steering_vector.norm().item():.4f}")
        
        # Store computed vectors
        self.steering_vectors.update(computed_vectors)
        
        logger.info(f"Computed {len(computed_vectors)} steering vectors")
        return computed_vectors
    
    def compute_from_examples(
        self,
        examples: List[dict],
        context_key: str = "context",
        query_key: str = "query",
        layers: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute steering vector(s) from example dictionaries.
        
        Args:
            examples: List of dicts with context and query keys.
            context_key: Key for context in examples.
            query_key: Key for query in examples.
            layers: List of layers to compute vectors for. If None:
                    - Single mode: computes for self.layer only
                    - Dynamic mode: computes for all layers in layer_range
            
        Returns:
            The computed steering vector for self.layer (backward compat).
        """
        positive_prompts = []
        negative_prompts = []
        
        for ex in examples:
            context = ex[context_key]
            query = ex[query_key]
            
            # Positive: with context
            positive_prompts.append(
                f"Use context: {context}\n\nQuestion: {query}\n\nAnswer:"
            )
            
            # Negative: without context
            negative_prompts.append(
                f"Question: {query}\n\nAnswer:"
            )
        
        # Determine which layers to compute
        if layers is None:
            if self.steering_mode == "dynamic":
                # Compute for all layers in range
                layers = list(range(self.layer_range[0], self.layer_range[1] + 1))
                self.compute_steering_vectors(positive_prompts, negative_prompts, layers)
                return self.steering_vectors[self.layer]  # Return default layer vector
            else:
                # Single mode: compute for default layer only
                return self.compute_steering_vector(positive_prompts, negative_prompts)
        else:
            # User specified layers explicitly
            self.compute_steering_vectors(positive_prompts, negative_prompts, layers)
            return self.steering_vectors[self.layer]
    
    def set_steering_vector(self, vector: torch.Tensor, layer: Optional[int] = None) -> None:
        """
        Set a steering vector directly (backward compatible + multi-layer support).
        
        Args:
            vector: Pre-computed steering vector.
            layer: Layer index. If None, uses self.layer (backward compat).
        """
        layer = layer if layer is not None else self.layer
        vector = vector.to(self.device)
        
        # Store in both locations for backward compatibility
        if layer == self.layer:
            self.steering_vector = vector
        self.steering_vectors[layer] = vector
        
        logger.info(f"Steering vector set for layer {layer}")
    
    def set_steering_vectors(self, vectors: dict[int, torch.Tensor]) -> None:
        """
        Set multiple steering vectors at once.
        
        Args:
            vectors: Dict mapping layer_idx -> steering_vector.
        """
        for layer, vector in vectors.items():
            self.steering_vectors[layer] = vector.to(self.device)
        
        logger.info(f"Set steering vectors for layers: {list(vectors.keys())}")
    
    @contextmanager
    def apply(
        self,
        multiplier: float = 2.0,
        runtime_state: Optional[dict] = None,
    ) -> Generator["ActivationSteering", None, None]:
        """
        Context manager to apply steering during generation.
        
        Supports both single-layer (backward compat) and dynamic multi-layer steering.
        Automatically selects layers based on runtime_state when in dynamic mode.
        
        Args:
            multiplier: Global steering strength (default 2.0).
            runtime_state: Dict with runtime signals for dynamic selection:
                - "retrieval_score": float in [0, 1]
                - "generation_step": int (optional, tracked internally)
                
        Yields:
            Self for method chaining.
            
        Example:
            >>> # Single-layer (backward compatible)
            >>> with steerer.apply(multiplier=2.0):
            ...     output = model.generate(input_ids)
            
            >>> # Dynamic multi-layer
            >>> with steerer.apply(multiplier=2.0, runtime_state={"retrieval_score": 0.85}):
            ...     output = model.generate(input_ids)
        """
        # Check if we have any steering vectors
        has_vectors = (self.steering_vector is not None) or len(self.steering_vectors) > 0
        if not has_vectors:
            raise RuntimeError("No steering vectors computed. Call compute_steering_vector(s) first.")
        
        # Reset generation step counter
        self._generation_step = 0
        
        # Select layers dynamically
        self._active_layers = self.select_layers(runtime_state)
        
        if not self._active_layers:
            logger.warning("No layers selected for steering (may be disabled by select_layers)")
        
        # Set multiplier
        self.multiplier = multiplier
        
        # Register hooks for all active layers
        for layer_idx in self._active_layers:
            # Check if we have a vector for this layer (either in dict or legacy single vector)
            has_layer_vector = (layer_idx in self.steering_vectors) or \
                               (layer_idx == self.layer and self.steering_vector is not None)
            if has_layer_vector:
                target_module = self.target_modules[layer_idx]
                handle = target_module.register_forward_hook(self._steering_hook)
                self.hook_handles[layer_idx] = handle
        
        self.is_active = True
        
        logger.debug(f"Steering activated on layers {self._active_layers} with multiplier {multiplier}")
        
        try:
            yield self
        finally:
            # Remove all hooks
            for handle in self.hook_handles.values():
                handle.remove()
            self.hook_handles.clear()
            self.is_active = False
            self._active_layers = []
    
    def apply_manual(
        self, 
        multiplier: float = 2.0, 
        runtime_state: Optional[dict] = None
    ) -> None:
        """
        Manually activate steering (remember to call remove_manual).
        
        Args:
            multiplier: Strength of steering.
            runtime_state: Runtime signals for dynamic layer selection.
        """
        has_vectors = (self.steering_vector is not None) or len(self.steering_vectors) > 0
        if not has_vectors:
            raise RuntimeError("No steering vectors computed")
        
        # Reset step counter
        self._generation_step = 0
        
        # Select layers
        self._active_layers = self.select_layers(runtime_state)
        self.multiplier = multiplier
        
        # Remove existing hooks
        self.remove_manual()
        
        # Register new hooks
        for layer_idx in self._active_layers:
            # Check if we have a vector for this layer
            has_layer_vector = (layer_idx in self.steering_vectors) or \
                               (layer_idx == self.layer and self.steering_vector is not None)
            if has_layer_vector:
                target_module = self.target_modules[layer_idx]
                handle = target_module.register_forward_hook(self._steering_hook)
                self.hook_handles[layer_idx] = handle
        
        self.is_active = True
        logger.debug(f"Steering manually activated on layers {self._active_layers}")
    
    def remove_manual(self) -> None:
        """Remove manually applied steering (safe, idempotent)."""
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()
        self.is_active = False
        self._active_layers = []
    
    def save(self, path: str) -> None:
        """
        Save steering vector(s) to file.
        
        Supports both single-vector (backward compat) and multi-vector formats.
        """
        if not self.steering_vectors and not self.steering_vector:
            raise RuntimeError("No steering vectors to save")
        
        # Save multi-layer format
        checkpoint = {
            "steering_vectors": {k: v.cpu() for k, v in self.steering_vectors.items()},
            "layer": self.layer,  # Default/fallback layer
            "component": self.component,
            "steering_mode": self.steering_mode,
            "layer_range": self.layer_range,
            "max_steering_steps": self.max_steering_steps,
            "layer_multipliers": self.layer_multipliers,
        }
        
        # For backward compatibility, also save single vector if it exists
        if self.steering_vector is not None:
            checkpoint["steering_vector"] = self.steering_vector.cpu()
        
        torch.save(checkpoint, path)
        logger.info(f"Steering vectors saved to {path} (layers: {list(self.steering_vectors.keys())})")
    
    def load(self, path: str) -> None:
        """
        Load steering vector(s) from file.
        
        Automatically detects single-vector vs multi-vector format.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load multi-vector format (new)
        if "steering_vectors" in checkpoint:
            self.steering_vectors = {
                k: v.to(self.device) for k, v in checkpoint["steering_vectors"].items()
            }
            logger.info(f"Loaded steering vectors for layers: {list(self.steering_vectors.keys())}")
        
        # Load single-vector format (backward compat)
        if "steering_vector" in checkpoint:
            self.steering_vector = checkpoint["steering_vector"].to(self.device)
            # Also store in multi-layer dict
            layer = checkpoint.get("layer", self.layer)
            self.steering_vectors[layer] = self.steering_vector
            logger.info(f"Loaded single steering vector for layer {layer}")
        
        # Load configuration
        self.layer = checkpoint.get("layer", self.layer)
        self.component = checkpoint.get("component", self.component)
        self.steering_mode = checkpoint.get("steering_mode", self.steering_mode)
        self.layer_range = checkpoint.get("layer_range", self.layer_range)
        self.max_steering_steps = checkpoint.get("max_steering_steps", self.max_steering_steps)
        self.layer_multipliers = checkpoint.get("layer_multipliers", self.layer_multipliers) or {}
        
        logger.info(f"Steering configuration loaded from {path}")


class HybridSteering:
    """
    Combines ReFT interventions with activation steering for maximum
    context faithfulness.
    
    ReFT provides learned per-example interventions while activation
    steering provides a global bias toward context usage.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        reft_intervention: Any,  # ReFTIntervention
        reft_layer: int = 6,
        steering_layer: int = 6,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize hybrid steering.
        
        Args:
            model: T5 model.
            tokenizer: Tokenizer.
            reft_intervention: ReFT intervention module.
            reft_layer: Layer for ReFT intervention.
            steering_layer: Layer for activation steering.
            device: Device for computations.
        """
        from src.reft import ReFTHook
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        # ReFT components
        self.reft_intervention = reft_intervention
        self.reft_hook = ReFTHook(model, reft_intervention, reft_layer)
        
        # Activation steering components
        self.activation_steerer = ActivationSteering(
            model, tokenizer, steering_layer, device=self.device
        )
    
    def compute_steering_vector(self, examples: List[dict]) -> None:
        """Compute the activation steering vector from examples."""
        self.activation_steerer.compute_from_examples(examples)
    
    @contextmanager
    def apply(
        self,
        use_reft: bool = True,
        use_steering: bool = True,
        steering_multiplier: float = 2.0,
    ) -> Generator["HybridSteering", None, None]:
        """
        Apply both ReFT and activation steering.
        
        Args:
            use_reft: Whether to apply ReFT intervention.
            use_steering: Whether to apply activation steering.
            steering_multiplier: Strength of activation steering.
            
        Yields:
            Self for method chaining.
        """
        try:
            if use_reft:
                self.reft_hook.register()
            
            if use_steering and self.activation_steerer.steering_vector is not None:
                self.activation_steerer.apply_manual(steering_multiplier)
            
            yield self
            
        finally:
            if use_reft:
                self.reft_hook.remove()
            
            if use_steering:
                self.activation_steerer.remove_manual()


def compute_faithfulness_metrics(
    model: nn.Module,
    tokenizer: Any,
    examples: List[dict],
    steerer: Optional[ActivationSteering] = None,
    multiplier: float = 2.0,
) -> dict:
    """
    Compute faithfulness metrics (p_s, p_o) from ContextFocus paper.
    
    p_s: Ratio of answers that match context-substituted answer
    p_o: Ratio of answers that match original (parametric) answer
    
    Args:
        model: T5 model.
        tokenizer: Tokenizer.
        examples: List of dicts with query, context, context_answer, original_answer.
        steerer: Optional ActivationSteering to apply.
        multiplier: Steering multiplier.
        
    Returns:
        Dictionary with metrics.
    """
    device = next(model.parameters()).device
    
    context_matches = 0
    original_matches = 0
    total = len(examples)
    
    for example in tqdm(examples, desc="Computing metrics"):
        query = example["query"]
        context = example["context"]
        context_answer = example.get("context_answer", example.get("answer", ""))
        original_answer = example.get("original_answer", "")
        
        # Build prompt
        prompt = f"Use context: {context}\n\nQuestion: {query}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate with or without steering
        with torch.no_grad():
            if steerer is not None:
                with steerer.apply(multiplier):
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            else:
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        # Check matches
        if context_answer.lower() in generated or generated in context_answer.lower():
            context_matches += 1
        
        if original_answer and (
            original_answer.lower() in generated or generated in original_answer.lower()
        ):
            original_matches += 1
    
    return {
        "p_s": context_matches / total if total > 0 else 0,  # Substituted (context-faithful)
        "p_o": original_matches / total if total > 0 else 0,  # Original (parametric)
        "total": total,
        "context_matches": context_matches,
        "original_matches": original_matches,
    }
