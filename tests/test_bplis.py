"""Tests for B-PLIS (Budgeted PLIS / CMA-ES latent vector search).

These tests verify shapes, deterministic seeding, and the zero-budget
graceful-fallback without requiring a real model or CMA-ES convergence.
"""
import numpy as np
import torch

from contextfocus.steering.bplis import BPLISConfig, LatentInterventionSearch


# ---------- helpers ----------


class _DummyConfig:
    """Minimal config object so model.config.vocab_size works."""
    vocab_size = 128


class _DummyModel(torch.nn.Module):
    """Tiny model stub that satisfies LatentInterventionSearch constructor."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden)  # gives parameters()
        self.config = _DummyConfig()

    def forward(self, **kw):
        raise NotImplementedError("Stub — should not be called in shape tests")


# ---------- orthoprojector ----------


def test_orthoprojector_shape():
    hidden_size = 128
    intrinsic = 16
    model = _DummyModel(hidden_size)
    cfg = BPLISConfig(intrinsic_dim=intrinsic)

    lis = LatentInterventionSearch(
        model=model, tokenizer=None, hidden_size=hidden_size, cfg=cfg,
    )
    U = lis.setup_random_orthoprojector(device="cpu")

    assert U.shape == (hidden_size, intrinsic)


def test_orthoprojector_orthonormal_columns():
    """Columns of U should be mutually orthonormal."""
    hidden_size = 64
    intrinsic = 8
    model = _DummyModel(hidden_size)
    cfg = BPLISConfig(intrinsic_dim=intrinsic)

    lis = LatentInterventionSearch(
        model=model, tokenizer=None, hidden_size=hidden_size, cfg=cfg,
    )
    U = lis.setup_random_orthoprojector(device="cpu")

    # U^T U should be identity(intrinsic)
    eye = torch.eye(intrinsic)
    assert torch.allclose(U.T @ U, eye, atol=1e-5)


# ---------- z → Δh mapping ----------


def test_z_to_dh_shape():
    hidden_size = 128
    intrinsic = 16
    model = _DummyModel(hidden_size)
    cfg = BPLISConfig(intrinsic_dim=intrinsic)

    lis = LatentInterventionSearch(
        model=model, tokenizer=None, hidden_size=hidden_size, cfg=cfg,
    )
    lis.setup_random_orthoprojector(device="cpu")

    z = np.zeros(intrinsic, dtype=np.float32)
    dh = lis._z_to_dh(z)
    assert tuple(dh.shape) == (hidden_size,)


def test_z_zero_gives_zero_dh():
    hidden_size = 64
    intrinsic = 8
    model = _DummyModel(hidden_size)
    cfg = BPLISConfig(intrinsic_dim=intrinsic)

    lis = LatentInterventionSearch(
        model=model, tokenizer=None, hidden_size=hidden_size, cfg=cfg,
    )
    lis.setup_random_orthoprojector(device="cpu")

    z = np.zeros(intrinsic, dtype=np.float32)
    dh = lis._z_to_dh(z)
    assert torch.allclose(dh, torch.zeros(hidden_size), atol=1e-7)


# ---------- search graceful fallback ----------


def test_search_zero_generations_returns_zero():
    """When max_generations=0 search must return a zero vector (no crash)."""
    hidden_size = 32
    intrinsic = 4
    model = _DummyModel(hidden_size)
    cfg = BPLISConfig(intrinsic_dim=intrinsic, max_generations=0)

    lis = LatentInterventionSearch(
        model=model, tokenizer=None, hidden_size=hidden_size, cfg=cfg,
    )

    dh = lis.search(
        query_inputs={},
        target_layers=[0],
        context_token_ids=[1, 2, 3],
        max_generations=0,
    )
    assert tuple(dh.shape) == (hidden_size,)
    assert float(dh.abs().sum()) == 0.0


def test_search_zero_popsize_returns_zero():
    """When popsize=0 search must return a zero vector."""
    hidden_size = 32
    intrinsic = 4
    model = _DummyModel(hidden_size)
    cfg = BPLISConfig(intrinsic_dim=intrinsic, popsize=0)

    lis = LatentInterventionSearch(
        model=model, tokenizer=None, hidden_size=hidden_size, cfg=cfg,
    )

    dh = lis.search(
        query_inputs={},
        target_layers=[0],
        context_token_ids=[1, 2],
        popsize=0,
    )
    assert tuple(dh.shape) == (hidden_size,)
    assert float(dh.abs().sum()) == 0.0


# ---------- deterministic seeding ----------


def test_orthoprojector_deterministic():
    """Same seed → same U."""
    hidden = 64
    intrinsic = 8
    cfg = BPLISConfig(intrinsic_dim=intrinsic, seed=42)

    lis1 = LatentInterventionSearch(_DummyModel(hidden), None, hidden, cfg)
    U1 = lis1.setup_random_orthoprojector("cpu")

    # Re-create to reset seeds
    lis2 = LatentInterventionSearch(_DummyModel(hidden), None, hidden, cfg)
    U2 = lis2.setup_random_orthoprojector("cpu")

    assert torch.allclose(U1, U2, atol=1e-6)
