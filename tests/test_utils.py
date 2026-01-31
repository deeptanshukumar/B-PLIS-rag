"""Tests for utility functions."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.utils import (
    HookManager,
    chunk_text,
    compute_overlap,
    format_prompt,
    get_memory_usage,
    load_json,
    save_json,
    timer,
)


class TestHookManager:
    """Tests for HookManager class."""

    def test_register_hook(self) -> None:
        """Test registering a hook."""
        manager = HookManager()

        # Create a simple module
        module = nn.Linear(10, 10)

        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            pass

        handle = manager.register_hook(module, hook_fn, name="test_hook")

        assert "test_hook" in manager.hooks
        assert len(manager.hooks) == 1

    def test_remove_hook(self) -> None:
        """Test removing a specific hook."""
        manager = HookManager()
        module = nn.Linear(10, 10)

        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            pass

        manager.register_hook(module, hook_fn, name="test_hook")
        manager.remove_hook("test_hook")

        assert "test_hook" not in manager.hooks

    def test_clear_hooks(self) -> None:
        """Test clearing all hooks."""
        manager = HookManager()
        module1 = nn.Linear(10, 10)
        module2 = nn.Linear(10, 10)

        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            pass

        manager.register_hook(module1, hook_fn, name="hook1")
        manager.register_hook(module2, hook_fn, name="hook2")

        assert len(manager.hooks) == 2

        manager.clear()

        assert len(manager.hooks) == 0

    def test_context_manager(self) -> None:
        """Test HookManager as context manager."""
        manager = HookManager()
        module = nn.Linear(10, 10)

        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            pass

        with manager:
            manager.register_hook(module, hook_fn, name="test_hook")
            assert len(manager.hooks) == 1

        # Hooks should be cleared after exiting context
        assert len(manager.hooks) == 0

    def test_hook_actually_called(self) -> None:
        """Test that registered hook is actually called."""
        manager = HookManager()
        module = nn.Linear(10, 10)
        call_count = {"count": 0}

        def hook_fn(
            mod: nn.Module, input: tuple, output: torch.Tensor
        ) -> torch.Tensor:
            call_count["count"] += 1
            return output

        manager.register_hook(module, hook_fn, name="counter_hook")

        # Run forward pass
        x = torch.randn(1, 10)
        _ = module(x)

        assert call_count["count"] == 1

        manager.clear()


class TestChunkText:
    """Tests for chunk_text function."""

    def test_basic_chunking(self) -> None:
        """Test basic text chunking."""
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=30, overlap=10)

        assert len(chunks) > 1
        assert all(len(c) <= 30 for c in chunks)

    def test_small_text(self) -> None:
        """Test chunking text smaller than chunk_size."""
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap(self) -> None:
        """Test that chunks have proper overlap."""
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_text(text, chunk_size=10, overlap=3)

        # Check overlap exists
        for i in range(len(chunks) - 1):
            overlap = compute_overlap(chunks[i], chunks[i + 1])
            assert overlap >= 0

    def test_empty_text(self) -> None:
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=10, overlap=3)
        assert chunks == [] or chunks == [""]


class TestComputeOverlap:
    """Tests for compute_overlap function."""

    def test_full_overlap(self) -> None:
        """Test when strings fully overlap."""
        overlap = compute_overlap("abcdef", "defghi")
        assert overlap == 3  # "def"

    def test_no_overlap(self) -> None:
        """Test when strings don't overlap."""
        overlap = compute_overlap("abc", "xyz")
        assert overlap == 0

    def test_identical_strings(self) -> None:
        """Test with identical strings."""
        overlap = compute_overlap("test", "test")
        assert overlap == 4


class TestTimer:
    """Tests for timer context manager."""

    def test_timer_measures_time(self) -> None:
        """Test that timer measures elapsed time."""
        with timer() as t:
            time.sleep(0.1)

        assert t.elapsed >= 0.1
        assert t.elapsed < 0.2

    def test_timer_name(self) -> None:
        """Test timer with name (for logging)."""
        with timer("test operation") as t:
            pass

        assert t.elapsed >= 0


class TestFormatPrompt:
    """Tests for format_prompt function."""

    def test_basic_prompt(self) -> None:
        """Test basic prompt formatting."""
        prompt = format_prompt(
            query="What is X?",
            context="X is a thing.",
        )

        assert "What is X?" in prompt
        assert "X is a thing." in prompt

    def test_prompt_with_template(self) -> None:
        """Test prompt with custom template."""
        template = "Q: {query}\nC: {context}\nA:"
        prompt = format_prompt(
            query="What is X?",
            context="X is a thing.",
            template=template,
        )

        assert prompt == "Q: What is X?\nC: X is a thing.\nA:"

    def test_prompt_without_context(self) -> None:
        """Test prompt without context."""
        prompt = format_prompt(query="What is X?")

        assert "What is X?" in prompt


class TestJsonIO:
    """Tests for JSON save/load functions."""

    def test_save_and_load_json(self) -> None:
        """Test saving and loading JSON."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = Path(f.name)

        save_json(data, path)
        loaded = load_json(path)

        assert loaded == data

    def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_json(Path("nonexistent.json"))


class TestGetMemoryUsage:
    """Tests for get_memory_usage function."""

    def test_returns_dict(self) -> None:
        """Test that memory usage returns dict."""
        usage = get_memory_usage()

        assert isinstance(usage, dict)
        assert "process_memory_mb" in usage

    def test_has_gpu_info_if_available(self) -> None:
        """Test GPU info included if available."""
        usage = get_memory_usage()

        if torch.cuda.is_available():
            assert "gpu_allocated_mb" in usage
            assert "gpu_cached_mb" in usage
