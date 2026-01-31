"""
ReFT (Representation Fine-Tuning) Implementation for B-PLIS-RAG.

Implements low-dimensional latent interventions to steer T5 model toward
context-faithful generation. Based on ReFT methodology with adaptations
for RAG faithfulness steering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.config import get_config
from src.utils import HookManager, clear_memory

logger = logging.getLogger(__name__)


@dataclass
class ReFTConfig:
    """Configuration for ReFT intervention."""
    hidden_size: int = 768  # T5-base hidden size
    intervention_dim: int = 16  # Low-dimensional intervention
    target_layer: int = 6  # Mid-layer for balanced intervention
    init_std: float = 0.02  # Initialization standard deviation
    
    @classmethod
    def for_model(cls, model: nn.Module, intervention_dim: int = 16, target_layer: int = 6) -> "ReFTConfig":
        """Create config based on model architecture."""
        hidden_size = model.config.d_model
        num_layers = len(model.decoder.block)
        
        # Ensure target_layer is valid
        target_layer = min(target_layer, num_layers - 1)
        
        return cls(
            hidden_size=hidden_size,
            intervention_dim=intervention_dim,
            target_layer=target_layer,
        )


class ReFTIntervention(nn.Module):
    """
    ReFT Intervention Module.
    
    Implements a low-dimensional latent intervention that modifies hidden states
    to steer the model toward context-faithful generation. The intervention uses
    a learned latent vector z and a projection matrix to compute a delta that
    is added to the hidden states.
    
    Architecture:
        z (latent) -> proj (Linear) -> delta (added to hidden states)
    
    Args:
        hidden_size: Model hidden dimension (e.g., 768 for T5-base).
        intervention_dim: Dimension of the latent intervention vector.
        init_std: Standard deviation for weight initialization.
        
    Example:
        >>> intervention = ReFTIntervention(hidden_size=768, intervention_dim=16)
        >>> delta = intervention()  # Shape: [hidden_size]
        >>> # Add delta to hidden states during forward pass
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        intervention_dim: int = 16,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intervention_dim = intervention_dim
        self.init_std = init_std
        
        # Learnable latent vector z
        self.z = nn.Parameter(torch.zeros(intervention_dim))
        
        # Projection matrix: intervention_dim -> hidden_size
        self.proj = nn.Linear(intervention_dim, hidden_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"ReFTIntervention initialized: dim={intervention_dim}, "
            f"hidden_size={hidden_size}, params={self.num_parameters()}"
        )
    
    def _init_weights(self) -> None:
        """Initialize weights with normal distribution."""
        nn.init.normal_(self.proj.weight, mean=0.0, std=self.init_std)
        # z starts at zero (no intervention initially)
        nn.init.zeros_(self.z)
    
    def forward(self) -> torch.Tensor:
        """
        Compute the intervention delta.
        
        Returns:
            Delta tensor of shape [hidden_size] to add to hidden states.
        """
        return self.proj(self.z)
    
    def intervention(self) -> torch.Tensor:
        """Alias for forward() for clarity."""
        return self.forward()
    
    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset(self) -> None:
        """Reset the latent vector z to zero."""
        with torch.no_grad():
            self.z.zero_()
    
    def get_z_norm(self) -> float:
        """Get the L2 norm of the latent vector."""
        return self.z.norm().item()
    
    def to_dict(self) -> dict:
        """Serialize intervention state to dictionary."""
        return {
            "hidden_size": self.hidden_size,
            "intervention_dim": self.intervention_dim,
            "init_std": self.init_std,
            "z": self.z.detach().cpu().tolist(),
            "proj_weight": self.proj.weight.detach().cpu().tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ReFTIntervention":
        """Create intervention from dictionary."""
        intervention = cls(
            hidden_size=data["hidden_size"],
            intervention_dim=data["intervention_dim"],
            init_std=data["init_std"],
        )
        intervention.z.data = torch.tensor(data["z"])
        intervention.proj.weight.data = torch.tensor(data["proj_weight"])
        return intervention


class ReFTHook:
    """
    Hook manager for applying ReFT interventions to model layers.
    
    Registers forward hooks on specified layers to add the intervention
    delta to hidden states during forward pass.
    """
    
    def __init__(
        self,
        model: nn.Module,
        intervention: ReFTIntervention,
        target_layer: int = 6,
        component: str = "decoder",
    ) -> None:
        """
        Initialize the hook manager.
        
        Args:
            model: T5 model to hook.
            intervention: ReFT intervention module.
            target_layer: Layer index to intervene at.
            component: Model component ('encoder' or 'decoder').
        """
        self.model = model
        self.intervention = intervention
        self.target_layer = target_layer
        self.component = component
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.is_active = False
        
        # Get the target layer
        if component == "encoder":
            self.target_module = model.encoder.block[target_layer]
        else:
            self.target_module = model.decoder.block[target_layer]
    
    def _hook_fn(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward hook function that adds intervention to hidden states.
        
        The delta is broadcast across batch and sequence dimensions.
        """
        if not self.is_active:
            return output
        
        # Get hidden states (first element of output tuple for T5 blocks)
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()
        
        # Compute and apply intervention
        delta = self.intervention()  # Shape: [hidden_size]
        
        # Broadcast delta: [hidden_size] -> [1, 1, hidden_size] -> [batch, seq, hidden_size]
        delta = delta.unsqueeze(0).unsqueeze(0)
        delta = delta.to(hidden_states.device, dtype=hidden_states.dtype)
        
        # Add intervention
        modified_hidden = hidden_states + delta
        
        if rest:
            return (modified_hidden,) + rest
        return modified_hidden
    
    def register(self) -> None:
        """Register the forward hook."""
        if self.handle is not None:
            self.remove()
        
        self.handle = self.target_module.register_forward_hook(self._hook_fn)
        self.is_active = True
        logger.debug(f"ReFT hook registered at {self.component} layer {self.target_layer}")
    
    def remove(self) -> None:
        """Remove the forward hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.is_active = False
        logger.debug("ReFT hook removed")
    
    def activate(self) -> None:
        """Activate the hook (intervention will be applied)."""
        self.is_active = True
    
    def deactivate(self) -> None:
        """Deactivate the hook (intervention will not be applied)."""
        self.is_active = False
    
    def __enter__(self) -> "ReFTHook":
        self.register()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.remove()


class ReFTTrainer:
    """
    Trainer for ReFT interventions.
    
    Optimizes the intervention's latent vector z to steer the model
    toward context-faithful outputs using contrastive training on
    conflict examples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        intervention: ReFTIntervention,
        tokenizer: Any,
        target_layer: int = 6,
        learning_rate: float = 1e-2,
        num_steps: int = 100,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: T5 model (frozen).
            intervention: ReFT intervention to train.
            tokenizer: Tokenizer for the model.
            target_layer: Layer to apply intervention.
            learning_rate: Learning rate for optimizer.
            num_steps: Number of optimization steps per example.
            device: Device for training.
        """
        self.model = model
        self.intervention = intervention
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.device = device or next(model.parameters()).device
        
        # Move intervention to device
        self.intervention = self.intervention.to(self.device)
        
        # Create optimizer for intervention parameters only
        self.optimizer = Adam(self.intervention.parameters(), lr=learning_rate)
        
        # Create hook
        self.hook = ReFTHook(model, intervention, target_layer)
    
    def train_on_example(
        self,
        query: str,
        context: str,
        target_answer: str,
        verbose: bool = False,
    ) -> dict:
        """
        Train intervention on a single conflict example.
        
        Args:
            query: The query text.
            context: The retrieved context (ground truth source).
            target_answer: The context-faithful answer.
            verbose: Whether to print progress.
            
        Returns:
            Dictionary with training metrics.
        """
        # Prepare input
        prompt = f"Use context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        labels = self.tokenizer(
            target_answer,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).input_ids.to(self.device)
        
        # Disable model caching for training
        self.model.config.use_cache = False
        
        # Reset intervention
        self.intervention.reset()
        
        # Register hook
        self.hook.register()
        
        losses = []
        try:
            for step in range(self.num_steps):
                self.optimizer.zero_grad()
                
                # Forward pass with intervention
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=labels,
                )
                
                loss = outputs.loss
                
                # Backward pass (only updates intervention parameters)
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                
                if verbose and (step + 1) % 10 == 0:
                    logger.info(f"Step {step + 1}/{self.num_steps}, Loss: {loss.item():.4f}")
        
        finally:
            # Always remove hook
            self.hook.remove()
            self.model.config.use_cache = True
        
        return {
            "final_loss": losses[-1] if losses else float("inf"),
            "losses": losses,
            "z_norm": self.intervention.get_z_norm(),
        }
    
    def train(
        self,
        examples: List[dict],
        epochs: int = 1,
        verbose: bool = True,
    ) -> dict:
        """
        Train intervention on multiple examples.
        
        Args:
            examples: List of dicts with 'query', 'context', 'answer' keys.
            epochs: Number of passes through all examples.
            verbose: Whether to print progress.
            
        Returns:
            Dictionary with training metrics.
        """
        all_losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            iterator = tqdm(examples, desc=f"Epoch {epoch + 1}/{epochs}") if verbose else examples
            
            for example in iterator:
                result = self.train_on_example(
                    query=example["query"],
                    context=example["context"],
                    target_answer=example["answer"],
                    verbose=False,
                )
                epoch_losses.append(result["final_loss"])
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            all_losses.extend(epoch_losses)
            
            if verbose:
                logger.info(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")
        
        return {
            "avg_loss": sum(all_losses) / len(all_losses),
            "all_losses": all_losses,
            "z_norm": self.intervention.get_z_norm(),
        }
    
    def save(self, path: str) -> None:
        """Save intervention to file."""
        torch.save({
            "intervention_state": self.intervention.state_dict(),
            "intervention_config": self.intervention.to_dict(),
            "target_layer": self.target_layer,
        }, path)
        logger.info(f"Intervention saved to {path}")
    
    def load(self, path: str) -> None:
        """Load intervention from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.intervention.load_state_dict(checkpoint["intervention_state"])
        self.target_layer = checkpoint["target_layer"]
        logger.info(f"Intervention loaded from {path}")


def verify_intervention(
    model: nn.Module,
    tokenizer: Any,
    intervention: ReFTIntervention,
    target_layer: int,
    test_prompt: str = "What is a contract?",
) -> dict:
    """
    Verify that the intervention actually changes model outputs.
    
    Args:
        model: T5 model.
        tokenizer: Tokenizer.
        intervention: ReFT intervention.
        target_layer: Layer to apply intervention.
        test_prompt: Prompt to test with.
        
    Returns:
        Dictionary with verification results.
    """
    device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    # Generate without intervention
    intervention.reset()
    with torch.no_grad():
        outputs_zero = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )
    text_zero = tokenizer.decode(outputs_zero[0], skip_special_tokens=True)
    
    # Set non-zero intervention
    with torch.no_grad():
        intervention.z.fill_(1.0)
    
    # Generate with intervention
    hook = ReFTHook(model, intervention, target_layer)
    with hook:
        with torch.no_grad():
            outputs_one = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
    text_one = tokenizer.decode(outputs_one[0], skip_special_tokens=True)
    
    # Reset
    intervention.reset()
    
    # Check if outputs differ
    outputs_differ = text_zero != text_one
    
    return {
        "outputs_differ": outputs_differ,
        "text_zero_z": text_zero,
        "text_nonzero_z": text_one,
        "verification_passed": outputs_differ,
    }


def create_reft_intervention(
    model: nn.Module,
    intervention_dim: int = 16,
    target_layer: int = 6,
) -> Tuple[ReFTIntervention, ReFTHook]:
    """
    Create a ReFT intervention and hook for a model.
    
    Args:
        model: T5 model.
        intervention_dim: Dimension of latent intervention.
        target_layer: Layer to apply intervention.
        
    Returns:
        Tuple of (intervention, hook).
    """
    config = ReFTConfig.for_model(model, intervention_dim, target_layer)
    intervention = ReFTIntervention(
        hidden_size=config.hidden_size,
        intervention_dim=config.intervention_dim,
        init_std=config.init_std,
    )
    
    device = next(model.parameters()).device
    intervention = intervention.to(device)
    
    hook = ReFTHook(model, intervention, target_layer)
    
    return intervention, hook
