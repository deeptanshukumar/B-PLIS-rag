"""Tests for householder norm-preserving 2-D rotation."""
import math

import pytest
import torch

from contextfocus.steering.householder import householder_rotate_last_token


# ---------- norm preservation ----------


def test_householder_norm_preserve():
    """||h[:,-1,:]|| must be identical before and after rotation (within FP tol)."""
    B, S, H = 2, 8, 128
    h = torch.randn(B, S, H)
    v = torch.randn(H)
    theta = 0.7
    out = householder_rotate_last_token(h, v, theta)

    n_before = h[:, -1, :].norm(dim=-1)
    n_after = out[:, -1, :].norm(dim=-1)
    assert torch.allclose(n_before, n_after, atol=1e-5)


def test_norm_preserve_batch_vectors():
    """Same property when v is [B, H]."""
    B, S, H = 4, 6, 64
    h = torch.randn(B, S, H)
    v = torch.randn(B, H)
    theta = 1.2
    out = householder_rotate_last_token(h, v, theta)

    n_before = h[:, -1, :].norm(dim=-1)
    n_after = out[:, -1, :].norm(dim=-1)
    assert torch.allclose(n_before, n_after, atol=1e-5)


# ---------- non-last tokens unchanged ----------


def test_only_last_token_changed():
    """Positions 0..S-2 must be untouched."""
    B, S, H = 3, 10, 32
    h = torch.randn(B, S, H)
    v = torch.randn(H)
    out = householder_rotate_last_token(h, v, 0.5)
    assert torch.allclose(h[:, :-1, :], out[:, :-1, :], atol=1e-7)


# ---------- theta=0 is identity ----------


def test_theta_zero_identity():
    B, S, H = 2, 4, 64
    h = torch.randn(B, S, H)
    v = torch.randn(H)
    out = householder_rotate_last_token(h, v, theta=0.0)
    assert torch.allclose(h, out, atol=1e-6)


# ---------- theta=pi is non-trivial but still norm-preserving ----------


def test_theta_pi_norm_preserve():
    B, S, H = 1, 3, 128
    h = torch.randn(B, S, H)
    v = torch.randn(H)
    out = householder_rotate_last_token(h, v, theta=math.pi)
    n_before = h[:, -1, :].norm(dim=-1)
    n_after = out[:, -1, :].norm(dim=-1)
    assert torch.allclose(n_before, n_after, atol=1e-5)


# ---------- output shape ----------


def test_output_shape():
    B, S, H = 5, 7, 256
    h = torch.randn(B, S, H)
    v = torch.randn(H)
    out = householder_rotate_last_token(h, v, 0.3)
    assert out.shape == h.shape


# ---------- fallback for parallel h and v ----------


def test_parallel_h_v_does_not_crash():
    """When h_last is nearly parallel to v the fallback branch fires."""
    B, S, H = 1, 2, 16
    v = torch.randn(H)
    h = torch.zeros(B, S, H)
    h[:, -1, :] = v.unsqueeze(0) * 3.0  # exactly parallel
    out = householder_rotate_last_token(h, v, 0.5)
    assert out.shape == h.shape
    n_before = h[:, -1, :].norm(dim=-1)
    n_after = out[:, -1, :].norm(dim=-1)
    assert torch.allclose(n_before, n_after, atol=1e-4)


# ---------- bad v shape ----------


def test_invalid_v_shape_raises():
    h = torch.randn(2, 4, 8)
    v = torch.randn(3, 8)  # batch dim mismatch
    with pytest.raises(ValueError, match="v must be"):
        householder_rotate_last_token(h, v, 0.5)
