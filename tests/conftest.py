"""Pytest configuration and fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator
import sys

import pytest

# Import torch - handle potential docstring issue
try:
    import torch
except RuntimeError:
    # If torch is already loaded, get it from sys.modules
    if 'torch' in sys.modules:
        torch = sys.modules['torch']
    else:
        raise


@pytest.fixture(scope="session")
def device() -> str:
    """Get the test device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for testing."""
    return [
        "This Agreement shall be effective for a period of one year.",
        "The Parties agree to maintain confidentiality of all information.",
        "Either party may terminate this agreement with 30 days notice.",
        "All disputes shall be resolved through binding arbitration.",
        "The governing law shall be the laws of the State of California.",
    ]


@pytest.fixture
def sample_qa_pairs() -> list[dict]:
    """Sample Q&A pairs for testing."""
    return [
        {
            "query": "How long is the agreement valid?",
            "context": "This Agreement shall be effective for a period of one year.",
            "answer": "One year",
        },
        {
            "query": "How can the agreement be terminated?",
            "context": "Either party may terminate this agreement with 30 days notice.",
            "answer": "With 30 days notice",
        },
        {
            "query": "Which state's laws govern this agreement?",
            "context": "The governing law shall be the laws of the State of California.",
            "answer": "California",
        },
    ]


# Markers for different test categories
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")


# Skip GPU tests if not available
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests based on markers and environment."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
