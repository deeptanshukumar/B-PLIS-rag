"""Tests for configuration module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.config import (
    Config,
    PathsConfig,
    ReFTConfig,
    RetrieverConfig,  # Changed from RetrievalConfig
    SteeringConfig,
    get_config,
    setup_environment,
)


class TestPathsConfig:
    """Tests for PathsConfig."""

    def test_default_paths(self) -> None:
        """Test that default paths are set correctly."""
        paths = PathsConfig()
        # Paths are absolute, starting from PROJECT_ROOT
        assert paths.data_dir.name == "data"
        assert paths.checkpoints_dir.name == "checkpoints"
        assert paths.cache_dir.name == ".cache"

    def test_path_conversion(self) -> None:
        """Test string to Path conversion."""
        paths = PathsConfig(data_dir="custom/data")  # type: ignore[arg-type]
        assert isinstance(paths.data_dir, Path)
        assert paths.data_dir == Path("custom/data")


class TestReFTConfig:
    """Tests for ReFTConfig."""

    def test_default_values(self) -> None:
        """Test default ReFT configuration."""
        config = ReFTConfig()
        assert config.intervention_dim == 16
        assert config.target_layer == 6
        assert config.learning_rate == 1e-2
        assert config.num_steps == 100

    def test_custom_values(self) -> None:
        """Test custom ReFT configuration."""
        config = ReFTConfig(
            intervention_dim=32,
            target_layer=8,
            learning_rate=1e-3,
            num_steps=200,
        )
        assert config.intervention_dim == 32
        assert config.target_layer == 8
        assert config.learning_rate == 1e-3
        assert config.num_steps == 200


class TestSteeringConfig:
    """Tests for SteeringConfig."""

    def test_default_values(self) -> None:
        """Test default steering configuration."""
        config = SteeringConfig()
        assert config.steering_layer == 13
        assert config.multiplier == 2.0
        assert config.steering_mode == "single"

    def test_custom_values(self) -> None:
        """Test custom steering configuration."""
        config = SteeringConfig(
            steering_layer=10,
            multiplier=1.5,
            steering_mode="dynamic",
        )
        assert config.steering_layer == 10
        assert config.multiplier == 1.5
        assert config.steering_mode == "dynamic"


class TestRetrieverConfig:
    """Tests for RetrieverConfig."""

    def test_default_values(self) -> None:
        """Test default retriever configuration."""
        config = RetrieverConfig()
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.index_type == "IndexFlatL2"
        assert config.top_k == 5


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = Config()
        assert config.model.name == "t5-base"
        assert config.settings.seed == 42
        # Check nested configs exist
        assert config.steering is not None
        assert config.reft is not None

    def test_nested_configs(self) -> None:
        """Test nested configuration objects."""
        config = Config()
        assert isinstance(config.paths, PathsConfig)
        assert isinstance(config.reft, ReFTConfig)
        assert isinstance(config.steering, SteeringConfig)
        assert isinstance(config.retriever, RetrieverConfig)

    def test_config_from_toml(self) -> None:
        """Test loading config from TOML file."""
        toml_content = """
[model]
name = "google/flan-t5-base"
max_length = 1024

[settings]
seed = 123

[reft]
intervention_dim = 32

[steering]
steering_layer = 8
multiplier = 1.5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()

            config = Config.from_toml(Path(f.name))
            assert config.model.name == "google/flan-t5-base"
            assert config.model.max_length == 1024
            assert config.settings.seed == 123
            assert config.reft.intervention_dim == 32
            assert config.steering.steering_layer == 8
            assert config.steering.multiplier == 1.5

    def test_get_config_singleton(self) -> None:
        """Test get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestSetupEnvironment:
    """Tests for setup_environment."""

    def test_setup_sets_seeds(self) -> None:
        """Test that setup_environment sets random seeds."""
        import random

        import numpy as np
        import torch

        config = Config()
        config.settings.seed = 42
        setup_environment(config)

        # Test reproducibility
        random_val1 = random.random()
        np_val1 = np.random.random()
        torch_val1 = torch.rand(1).item()

        config.settings.seed = 42
        setup_environment(config)

        random_val2 = random.random()
        np_val2 = np.random.random()
        torch_val2 = torch.rand(1).item()

        assert random_val1 == random_val2
        assert np_val1 == np_val2
        assert torch_val1 == torch_val2
