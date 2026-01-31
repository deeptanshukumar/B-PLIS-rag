"""
Utility functions for B-PLIS-RAG.

Common utilities for logging, timing, memory management, and text processing.
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


@contextmanager
def timer(name: str = "Operation") -> Generator[None, None, None]:
    """
    Context manager for timing operations.
    
    Args:
        name: Name of the operation being timed.
        
    Yields:
        None
        
    Example:
        >>> with timer("Model inference"):
        ...     output = model(input)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{name} took {elapsed:.3f}s")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for timing function execution.
    
    Args:
        func: Function to time.
        
    Returns:
        Wrapped function that logs execution time.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def clear_memory() -> None:
    """Clear CUDA cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def memory_efficient_context() -> Generator[None, None, None]:
    """
    Context manager for memory-efficient operations.
    
    Clears memory before and after the operation.
    """
    clear_memory()
    try:
        yield
    finally:
        clear_memory()


def get_memory_usage() -> dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in MB.
    """
    import psutil
    
    stats = {
        "ram_used_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "ram_percent": psutil.virtual_memory().percent,
    }
    
    if torch.cuda.is_available():
        stats["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        stats["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
    
    return stats


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "..."
) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to add if truncated.
        
    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean.
        
    Returns:
        Cleaned text.
    """
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk.
        chunk_size: Size of each chunk in characters.
        overlap: Overlap between chunks.
        
    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at word boundary
        if end < text_len:
            last_space = chunk.rfind(' ')
            if last_space > chunk_size // 2:
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= text_len - overlap:
            break
    
    return [c for c in chunks if c]


def compute_char_overlap(
    pred_start: int,
    pred_end: int,
    gold_start: int,
    gold_end: int
) -> tuple[int, int, int]:
    """
    Compute character-level overlap between predicted and gold spans.
    
    Args:
        pred_start: Predicted span start.
        pred_end: Predicted span end.
        gold_start: Gold span start.
        gold_end: Gold span end.
        
    Returns:
        Tuple of (overlap_chars, pred_chars, gold_chars).
    """
    overlap_start = max(pred_start, gold_start)
    overlap_end = min(pred_end, gold_end)
    
    overlap_chars = max(0, overlap_end - overlap_start)
    pred_chars = pred_end - pred_start
    gold_chars = gold_end - gold_start
    
    return overlap_chars, pred_chars, gold_chars


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def torch_inference_mode() -> Generator[None, None, None]:
    """
    Context manager for torch inference mode with no_grad.
    
    More efficient than no_grad for inference-only operations.
    """
    with torch.inference_mode():
        yield


def batch_iterable(
    iterable: list[T],
    batch_size: int
) -> Generator[list[T], None, None]:
    """
    Yield batches from an iterable.
    
    Args:
        iterable: List to batch.
        batch_size: Size of each batch.
        
    Yields:
        Batches of items.
    """
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value.
        denominator: Denominator value.
        default: Default value if division by zero.
        
    Returns:
        Division result or default.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        num_bytes: Number of bytes.
        
    Returns:
        Formatted string (e.g., "1.5 GB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


class HookManager:
    """
    Manager for PyTorch hooks with automatic cleanup.
    
    Example:
        >>> manager = HookManager()
        >>> manager.register(module, hook_fn)
        >>> # ... use hooks ...
        >>> manager.remove_all()
    """
    
    def __init__(self) -> None:
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
    
    def register_forward_hook(
        self,
        module: torch.nn.Module,
        hook: Callable
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook and track the handle."""
        handle = module.register_forward_hook(hook)
        self.handles.append(handle)
        return handle
    
    def register_forward_pre_hook(
        self,
        module: torch.nn.Module,
        hook: Callable
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a forward pre-hook and track the handle."""
        handle = module.register_forward_pre_hook(hook)
        self.handles.append(handle)
        return handle
    
    def remove_all(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def __enter__(self) -> "HookManager":
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.remove_all()
