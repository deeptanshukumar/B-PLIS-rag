"""
Model loading utilities for B-PLIS-RAG.

Handles loading T5 models with optimizations for memory efficiency
and inference performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import nn
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from src.config import get_config, get_device, get_dtype

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    name: str = "t5-base"
    device: str = "auto"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    freeze_params: bool = True
    use_cache: bool = True
    max_length: int = 512
    
    @classmethod
    def from_global_config(cls) -> "ModelConfig":
        """Create ModelConfig from global configuration."""
        config = get_config()
        return cls(
            name=config.model.name,
            device=config.settings.device,
            dtype=config.settings.dtype,
            max_length=config.model.max_length,
        )


def load_model(
    model_name: str = "t5-base",
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    freeze_params: bool = True,
    use_cache: bool = True,
) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """
    Load a T5 model and tokenizer with optimizations.
    
    Args:
        model_name: Name or path of the model to load.
        device: Device to load the model on. If None, uses config default.
        dtype: Data type for the model. If None, uses config default.
        freeze_params: Whether to freeze all parameters.
        use_cache: Whether to enable KV-cache for generation.
        
    Returns:
        Tuple of (model, tokenizer).
        
    Example:
        >>> model, tokenizer = load_model("t5-base", device="cuda", freeze_params=True)
        >>> # Model is ready for inference with frozen parameters
    """
    config = get_config()
    
    # Resolve device and dtype
    if device is None:
        device = get_device(config.settings.device)
    elif isinstance(device, str):
        device = get_device(device)
    
    if dtype is None:
        dtype = get_dtype(config.settings.dtype)
    elif isinstance(dtype, str):
        dtype = get_dtype(dtype)
    
    logger.info(f"Loading model '{model_name}' on {device} with dtype {dtype}")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        model_name,
        model_max_length=config.model.max_length,
        legacy=False,
    )
    
    # Load model with optimizations
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if str(device) == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    # Move to device if not using device_map
    if str(device) != "cuda" or not torch.cuda.is_available():
        model = model.to(device)
    
    # Configure caching
    model.config.use_cache = use_cache
    
    # Freeze parameters for efficiency
    if freeze_params:
        freeze_model_parameters(model)
        logger.info("All model parameters frozen")
    
    # Set to evaluation mode
    model.eval()
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {total_params:,} total params, {trainable_params:,} trainable")
    
    return model, tokenizer


def freeze_model_parameters(model: nn.Module) -> None:
    """
    Freeze all parameters in the model.
    
    Args:
        model: Model to freeze.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model_parameters(model: nn.Module) -> None:
    """
    Unfreeze all parameters in the model.
    
    Args:
        model: Model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True


def get_model_layer(
    model: T5ForConditionalGeneration,
    layer_idx: int,
    component: Literal["encoder", "decoder"] = "decoder"
) -> nn.Module:
    """
    Get a specific layer from the T5 model.
    
    Args:
        model: T5 model.
        layer_idx: Index of the layer (0-indexed).
        component: Which component to get the layer from.
        
    Returns:
        The specified layer module.
        
    Raises:
        IndexError: If layer_idx is out of range.
    """
    if component == "encoder":
        layers = model.encoder.block
    else:
        layers = model.decoder.block
    
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(
            f"Layer index {layer_idx} out of range for {component} "
            f"with {len(layers)} layers"
        )
    
    return layers[layer_idx]


def get_hidden_size(model: T5ForConditionalGeneration) -> int:
    """
    Get the hidden size of the model.
    
    Args:
        model: T5 model.
        
    Returns:
        Hidden size (e.g., 768 for t5-base).
    """
    return model.config.d_model


def get_num_layers(
    model: T5ForConditionalGeneration,
    component: Literal["encoder", "decoder"] = "decoder"
) -> int:
    """
    Get the number of layers in a model component.
    
    Args:
        model: T5 model.
        component: Which component to count layers for.
        
    Returns:
        Number of layers.
    """
    if component == "encoder":
        return len(model.encoder.block)
    else:
        return len(model.decoder.block)


def prepare_for_training(
    model: T5ForConditionalGeneration,
    use_gradient_checkpointing: bool = False,
) -> None:
    """
    Prepare model for training by disabling cache and optionally enabling
    gradient checkpointing.
    
    Args:
        model: Model to prepare.
        use_gradient_checkpointing: Whether to enable gradient checkpointing.
    """
    # Disable cache during training (incompatible with gradient computation)
    model.config.use_cache = False
    
    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")


def prepare_for_inference(model: T5ForConditionalGeneration) -> None:
    """
    Prepare model for inference by enabling cache and disabling dropout.
    
    Args:
        model: Model to prepare.
    """
    model.config.use_cache = True
    model.eval()


@torch.no_grad()
def generate_text(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    num_beams: int = 4,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    **kwargs,
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: T5 model.
        tokenizer: T5 tokenizer.
        prompt: Input prompt text.
        max_new_tokens: Maximum number of tokens to generate.
        num_beams: Number of beams for beam search.
        do_sample: Whether to use sampling.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling.
        **kwargs: Additional generation arguments.
        
    Returns:
        Generated text.
    """
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_length or 512,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        early_stopping=True,
        **kwargs,
    )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def get_model_info(model: T5ForConditionalGeneration) -> dict:
    """
    Get information about the model.
    
    Args:
        model: T5 model.
        
    Returns:
        Dictionary with model information.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "name": model.config.name_or_path,
        "hidden_size": model.config.d_model,
        "num_encoder_layers": len(model.encoder.block),
        "num_decoder_layers": len(model.decoder.block),
        "num_heads": model.config.num_heads,
        "vocab_size": model.config.vocab_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }
