"""
Configuration management for B-PLIS-RAG.

Handles loading credentials, settings, and environment configuration
using TOML files and environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass, field

try:
    import tomli
except ImportError:
    import tomllib as tomli  # Python 3.11+

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class HuggingFaceConfig(BaseModel):
    """HuggingFace API configuration."""
    token: str = ""


class OpenAIConfig(BaseModel):
    """OpenAI API configuration (optional)."""
    api_key: str = ""
    organization: str = ""


class SettingsConfig(BaseModel):
    """General settings."""
    device: str = "auto"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    seed: int = 42
    log_level: str = "INFO"


class PathsConfig(BaseModel):
    """Path configuration."""
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    corpus_dir: Path = Field(default=PROJECT_ROOT / "data" / "corpus")
    benchmarks_dir: Path = Field(default=PROJECT_ROOT / "data" / "benchmarks")
    embeddings_dir: Path = Field(default=PROJECT_ROOT / "data" / "embeddings")
    cache_dir: Path = Field(default=PROJECT_ROOT / ".cache")
    checkpoints_dir: Path = Field(default=PROJECT_ROOT / "checkpoints")
    outputs_dir: Path = Field(default=PROJECT_ROOT / "outputs")

    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        for field_name in self.model_fields:
            path = getattr(self, field_name)
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)


class ModelConfigSettings(BaseModel):
    """Model configuration."""
    name: str = "t5-base"
    max_length: int = 512
    max_new_tokens: int = 100


class RetrieverConfig(BaseModel):
    """Retriever configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    index_type: str = "IndexFlatL2"
    top_k: int = 5


class ReFTConfig(BaseModel):
    """ReFT intervention configuration."""
    intervention_dim: int = 16
    target_layer: int = 6
    learning_rate: float = 0.01
    num_steps: int = 100


class SteeringConfig(BaseModel):
    """Activation steering configuration."""
    steering_layer: int = 13
    multiplier: float = 2.0
    # Dynamic steering parameters
    steering_mode: str = "single"  # "single" or "dynamic"
    steering_layer_range: tuple = (3, 7)  # Valid range for dynamic selection
    steering_max_steps: int = 60  # Max generation steps to apply steering
    layer_multipliers: dict = {}  # Per-layer multipliers, e.g., {4: 2.0, 5: 1.5}


class BilingualConfig(BaseModel):
    """Bilingual settings."""
    primary_language: str = "en"
    secondary_language: str = "hi"
    translation_model: str = "Helsinki-NLP/opus-mt-en-hi"


@dataclass
class Config:
    """
    Main configuration class for B-PLIS-RAG.
    
    Loads configuration from TOML files and environment variables.
    """
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    settings: SettingsConfig = field(default_factory=SettingsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfigSettings = field(default_factory=ModelConfigSettings)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reft: ReFTConfig = field(default_factory=ReFTConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    bilingual: BilingualConfig = field(default_factory=BilingualConfig)

    @classmethod
    def from_toml(cls, path: Path | str) -> "Config":
        """Load configuration from a TOML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "rb") as f:
            data = tomli.load(f)
        
        return cls(
            huggingface=HuggingFaceConfig(**data.get("huggingface", {})),
            openai=OpenAIConfig(**data.get("openai", {})),
            settings=SettingsConfig(**data.get("settings", {})),
            paths=PathsConfig(**data.get("paths", {})),
            model=ModelConfigSettings(**data.get("model", {})),
            retriever=RetrieverConfig(**data.get("retriever", {})),
            reft=ReFTConfig(**data.get("reft", {})),
            steering=SteeringConfig(**data.get("steering", {})),
            bilingual=BilingualConfig(**data.get("bilingual", {})),
        )

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables
        if hf_token := os.environ.get("HF_TOKEN"):
            config.huggingface.token = hf_token
        if openai_key := os.environ.get("OPENAI_API_KEY"):
            config.openai.api_key = openai_key
        if data_dir := os.environ.get("BPLIS_DATA_DIR"):
            config.paths.data_dir = Path(data_dir)
        if cache_dir := os.environ.get("BPLIS_CACHE_DIR"):
            config.paths.cache_dir = Path(cache_dir)
        if device := os.environ.get("BPLIS_DEVICE"):
            config.settings.device = device
        
        return config

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.paths.ensure_dirs()


# Global configuration instance
_config: Config | None = None


def get_config(
    config_path: Path | str | None = None,
    reload: bool = False
) -> Config:
    """
    Get or create the global configuration.
    
    Args:
        config_path: Optional path to a TOML configuration file.
        reload: Whether to reload the configuration.
        
    Returns:
        The global Config instance.
    """
    global _config
    
    if _config is None or reload:
        # Try to load from credentials file first
        credentials_path = PROJECT_ROOT / "credentials" / "credentials.toml"
        
        if config_path:
            _config = Config.from_toml(config_path)
        elif credentials_path.exists():
            _config = Config.from_toml(credentials_path)
        else:
            # Fall back to environment variables and defaults
            _config = Config.from_env()
        
        # Ensure directories exist
        _config.ensure_directories()
    
    return _config


def setup_environment(config: Config | None = None) -> None:
    """
    Set up the environment based on configuration.
    
    Args:
        config: Configuration to use. If None, uses global config.
    """
    import logging
    import random
    import numpy as np
    import torch
    
    if config is None:
        config = get_config()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set random seeds for reproducibility
    seed = config.settings.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set HuggingFace token
    if config.huggingface.token:
        os.environ["HF_TOKEN"] = config.huggingface.token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = config.huggingface.token


def get_device(device_config: str = "auto") -> torch.device:
    """
    Get the appropriate torch device based on configuration.
    
    Args:
        device_config: Device configuration string.
        
    Returns:
        torch.device instance.
    """
    import torch
    
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_config)


def get_dtype(dtype_config: str) -> "torch.dtype":
    """
    Get the appropriate torch dtype based on configuration.
    
    Args:
        dtype_config: Dtype configuration string.
        
    Returns:
        torch.dtype instance.
    """
    import torch
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_config, torch.float16)
