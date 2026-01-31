"""
Activation Steering for B-PLIS-RAG.

Implements ContextFocus-inspired activation steering to enhance context
faithfulness in RAG systems. Based on techniques from arXiv:2601.04131.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple

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


class ActivationSteering:
    """
    Activation steering for context-faithful generation.
    
    Computes a steering vector as the mean difference between activations
    when processing prompts with context vs. without context. This vector
    is then added to hidden states during generation to bias the model
    toward using the provided context.
    
    Based on ContextFocus (arXiv:2601.04131) methodology.
    
    Example:
        >>> steerer = ActivationSteering(model, tokenizer, layer=13)
        >>> steerer.compute_steering_vector(positive_prompts, negative_prompts)
        >>> with steerer.apply(multiplier=2.0):
        ...     output = model.generate(input_ids)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        layer: int = 6,
        component: str = "decoder",
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize activation steering.
        
        Args:
            model: T5 model.
            tokenizer: Tokenizer for the model.
            layer: Layer index to apply steering.
            component: Model component ('encoder' or 'decoder').
            device: Device for computations.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.component = component
        self.device = device or next(model.parameters()).device
        
        # Steering vector (computed later)
        self.steering_vector: Optional[torch.Tensor] = None
        
        # Hook handles
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.is_active = False
        self.multiplier = 2.0
        
        # Get target layer
        if component == "encoder":
            self.target_module = model.encoder.block[layer]
        else:
            self.target_module = model.decoder.block[layer]
        
        # For capturing activations during vector computation
        self._captured_activations: List[torch.Tensor] = []
        
        logger.info(f"ActivationSteering initialized at {component} layer {layer}")
    
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
        """Hook to apply steering during generation."""
        if not self.is_active or self.steering_vector is None:
            return output
        
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()
        
        # Apply steering vector (broadcast across sequence)
        steering = self.steering_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        steering = steering.to(hidden_states.device, dtype=hidden_states.dtype)
        steering = steering * self.multiplier
        
        modified = hidden_states + steering
        
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
        Compute the steering vector from positive/negative prompt pairs.
        
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
        
        # Register capture hook
        handle = self.target_module.register_forward_hook(self._capture_hook)
        
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
        
        self.steering_vector = steering_vector
        
        logger.info(f"Steering vector computed, norm: {steering_vector.norm().item():.4f}")
        
        return steering_vector
    
    def compute_from_examples(
        self,
        examples: List[dict],
        context_key: str = "context",
        query_key: str = "query",
    ) -> torch.Tensor:
        """
        Compute steering vector from example dictionaries.
        
        Args:
            examples: List of dicts with context and query keys.
            context_key: Key for context in examples.
            query_key: Key for query in examples.
            
        Returns:
            The computed steering vector.
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
        
        return self.compute_steering_vector(positive_prompts, negative_prompts)
    
    def set_steering_vector(self, vector: torch.Tensor) -> None:
        """
        Set the steering vector directly.
        
        Args:
            vector: Pre-computed steering vector.
        """
        self.steering_vector = vector.to(self.device)
    
    @contextmanager
    def apply(
        self,
        multiplier: float = 2.0
    ) -> Generator["ActivationSteering", None, None]:
        """
        Context manager to apply steering during generation.
        
        Args:
            multiplier: Strength of steering (default 2.0).
            
        Yields:
            Self for method chaining.
            
        Example:
            >>> with steerer.apply(multiplier=2.0):
            ...     output = model.generate(input_ids)
        """
        if self.steering_vector is None:
            raise RuntimeError("Steering vector not computed. Call compute_steering_vector first.")
        
        self.multiplier = multiplier
        self.hook_handle = self.target_module.register_forward_hook(self._steering_hook)
        self.is_active = True
        
        try:
            yield self
        finally:
            if self.hook_handle is not None:
                self.hook_handle.remove()
                self.hook_handle = None
            self.is_active = False
    
    def apply_manual(self, multiplier: float = 2.0) -> None:
        """
        Manually activate steering (remember to call remove_manual).
        
        Args:
            multiplier: Strength of steering.
        """
        if self.steering_vector is None:
            raise RuntimeError("Steering vector not computed")
        
        self.multiplier = multiplier
        if self.hook_handle is not None:
            self.hook_handle.remove()
        
        self.hook_handle = self.target_module.register_forward_hook(self._steering_hook)
        self.is_active = True
    
    def remove_manual(self) -> None:
        """Remove manually applied steering."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        self.is_active = False
    
    def save(self, path: str) -> None:
        """Save steering vector to file."""
        if self.steering_vector is None:
            raise RuntimeError("No steering vector to save")
        
        torch.save({
            "steering_vector": self.steering_vector.cpu(),
            "layer": self.layer,
            "component": self.component,
        }, path)
        logger.info(f"Steering vector saved to {path}")
    
    def load(self, path: str) -> None:
        """Load steering vector from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.steering_vector = checkpoint["steering_vector"].to(self.device)
        self.layer = checkpoint["layer"]
        self.component = checkpoint["component"]
        logger.info(f"Steering vector loaded from {path}")


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
