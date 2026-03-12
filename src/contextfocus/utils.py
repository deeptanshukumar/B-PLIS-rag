from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any  # may be AutoTokenizer or AutoProcessor
    device: torch.device


def load_hf_causal_lm(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    device_map: str | Dict[str, int] | None = "auto",
    trust_remote_code: bool = True,
) -> ModelBundle:
    """
    Load a Hugging Face autoregressive generator model plus a tokenizer/processor.

    Notes:
    - For gated models, set HF_TOKEN (or HF_TOK) in env.
    - For Gemma 3 multimodal checkpoints, Hugging Face recommends AutoProcessor and
      Transformers >= 4.50.0.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOK")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    tokenizer = None
    if AutoProcessor is not None and "gemma-3" in model_id:
        try:
            tokenizer = AutoProcessor.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
        except Exception:
            tokenizer = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Try standard causal LM first
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        # Fallback for Gemma 3 conditional generation class
        if "gemma-3" in model_id:
            try:
                from transformers import Gemma3ForConditionalGeneration

                model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_id,
                    token=token,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )
            except Exception:
                raise e
        else:
            raise e

    model.eval()
    # Detect actual device from model parameters (works for CUDA, MPS, CPU)
    device = next(model.parameters()).device
    return ModelBundle(model=model, tokenizer=tokenizer, device=device)


def get_model_hidden_size(model: Any) -> int:
    """
    Get hidden size from model config, handling different architectures.
    
    Different models store hidden dimension size in different attributes:
    - Most models: config.hidden_size
    - Multimodal models (e.g. Gemma3): config.text_config.hidden_size
    - Some transformer models: config.d_model
    - GPT models: config.n_embd
    """
    config = model.config
    
    # Try multimodal config first (Gemma3, etc.)
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
        return config.text_config.hidden_size
    
    # Standard hidden_size attribute
    if hasattr(config, 'hidden_size'):
        return config.hidden_size
    
    # Alternative attributes
    if hasattr(config, 'd_model'):
        return config.d_model
    
    if hasattr(config, 'n_embd'):
        return config.n_embd
    
    raise AttributeError(
        f"Could not find hidden size in config of type {type(config).__name__}. "
        f"Tried: text_config.hidden_size, hidden_size, d_model, n_embd"
    )


def decode(tokenizer_or_processor: Any, token_ids: torch.Tensor) -> str:
    if hasattr(tokenizer_or_processor, "decode"):
        return tokenizer_or_processor.decode(token_ids, skip_special_tokens=True)
    # AutoProcessor sometimes has tokenizer
    tok = getattr(tokenizer_or_processor, "tokenizer", None)
    if tok is None:
        raise ValueError("No decode method found.")
    return tok.decode(token_ids, skip_special_tokens=True)


def get_eos_id(tokenizer_or_processor: Any) -> int:
    for attr in ["eos_token_id", "tokenizer.eos_token_id"]:
        try:
            if attr == "eos_token_id":
                v = getattr(tokenizer_or_processor, attr)
            else:
                v = getattr(getattr(tokenizer_or_processor, "tokenizer"), "eos_token_id")
            if v is not None:
                return int(v)
        except Exception:
            pass
    return 0


def tokenize_text(tokenizer_or_processor: Any, text: str, *, max_length: int | None = None):
    # For processors (like Gemma3), try to use the underlying tokenizer
    if hasattr(tokenizer_or_processor, "tokenizer"):
        tok = tokenizer_or_processor.tokenizer
        return tok(
            text,
            return_tensors="pt",
            truncation=True if max_length is not None else False,
            max_length=max_length,
        )
    
    # Standard tokenizer
    return tokenizer_or_processor(
        text,
        return_tensors="pt",
        truncation=True if max_length is not None else False,
        max_length=max_length,
    )


def tokenize_chat(tokenizer_or_processor: Any, messages: list[dict], *, max_length: int | None = None, add_generation_prompt: bool = True):
    """
    Use the model's chat template if available.
    Works with tokenizers and processors that implement apply_chat_template.
    """
    if hasattr(tokenizer_or_processor, "apply_chat_template"):
        try:
            return tokenizer_or_processor.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            # Gemma3 processor may fail - try using underlying tokenizer
            if hasattr(tokenizer_or_processor, "tokenizer"):
                tok = tokenizer_or_processor.tokenizer
                if hasattr(tok, "apply_chat_template"):
                    try:
                        return tok.apply_chat_template(
                            messages,
                            add_generation_prompt=add_generation_prompt,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                    except Exception:
                        pass
    
    # Fallback: concatenate roles naively
    text = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        text += f"{role.upper()}: {content}\n"
    return tokenize_text(tokenizer_or_processor, text, max_length=max_length)


def get_transformer_blocks(model: Any):
    """
    Best-effort introspection to find the transformer block list (ModuleList).
    Supports common HF architectures: Llama, Mistral, Gemma, GPT-NeoX, etc.
    """
    # Common patterns
    candidates = [
        ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("model.model.layers", lambda m: getattr(getattr(getattr(m, "model", None), "model", None), "layers", None)),
        ("language_model.model.layers", lambda m: getattr(getattr(getattr(m, "language_model", None), "model", None), "layers", None)),
        ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("gpt_neox.layers", lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
    ]
    for _, getter in candidates:
        blocks = getter(model)
        if blocks is not None:
            try:
                n = len(blocks)
                if n > 0:
                    return blocks
            except Exception:
                continue

    # Fallback: search for the largest ModuleList of blocks
    best = None
    best_len = 0
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > best_len:
            best = module
            best_len = len(module)
    if best is None:
        raise ValueError("Could not locate transformer blocks on this model.")
    return best
