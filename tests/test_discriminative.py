"""Tests for sign-based discriminative layer filter."""
import torch

from contextfocus.steering.discriminative import get_discriminative_layers


def test_discriminative_simple():
    """A single layer with opposite-sign projection must be returned."""
    H = 64
    L = 6
    d = torch.randn(H)
    d = d / d.norm()

    mu_pos = [torch.randn(H) for _ in range(L)]
    mu_neg = [m.clone() for m in mu_pos]  # same → no sign flip … except layer 2
    mu_pos[2] = d * 1.0
    mu_neg[2] = -d * 0.8

    layers = get_discriminative_layers(mu_pos, mu_neg, d)
    assert 2 in layers


def test_no_flip_returns_empty():
    """When no layer flips sign, the list should be empty."""
    H = 32
    L = 4
    d = torch.ones(H)
    mu_pos = [torch.ones(H) for _ in range(L)]
    mu_neg = [torch.ones(H) * 0.5 for _ in range(L)]

    layers = get_discriminative_layers(mu_pos, mu_neg, d)
    assert layers == []


def test_all_flip():
    """All layers can flip if constructed that way."""
    H = 16
    L = 5
    d = torch.ones(H)
    mu_pos = [d * 1.0 for _ in range(L)]
    mu_neg = [-d * 0.3 for _ in range(L)]

    layers = get_discriminative_layers(mu_pos, mu_neg, d)
    assert layers == list(range(L))


def test_mismatched_lengths():
    """Handles mu_pos and mu_neg of different lengths gracefully."""
    H = 8
    d = torch.randn(H)
    mu_pos = [torch.randn(H) for _ in range(6)]
    mu_neg = [torch.randn(H) for _ in range(4)]  # shorter

    layers = get_discriminative_layers(mu_pos, mu_neg, d)
    # Should only iterate min(6, 4) = 4 layers
    assert all(l < 4 for l in layers)


def test_d_feat_normalised_internally():
    """A non-unit d_feat should give the same result as a scaled one."""
    H = 32
    L = 4
    d = torch.randn(H)

    mu_pos = [torch.randn(H) for _ in range(L)]
    mu_neg = [torch.randn(H) for _ in range(L)]

    layers_unit = get_discriminative_layers(mu_pos, mu_neg, d / d.norm())
    layers_scaled = get_discriminative_layers(mu_pos, mu_neg, d * 42.0)
    assert layers_unit == layers_scaled
