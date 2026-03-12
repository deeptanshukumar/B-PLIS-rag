"""Sign-based discriminative layer filter.

Selects layers where the projections of the positive and negative mean
activations onto a global feature direction have **opposite signs**.
These are exactly the layers where the steering vector can *flip* the
model's internal representation from "memory mode" to "context mode".

Exported
--------
get_discriminative_layers(mu_pos, mu_neg, d_feat) -> List[int]
"""
from __future__ import annotations

from typing import List, Sequence

import torch


def get_discriminative_layers(
    mu_pos: Sequence[torch.Tensor],
    mu_neg: Sequence[torch.Tensor],
    d_feat: torch.Tensor,
) -> List[int]:
    """Return layer indices where ``sign(mu_pos_k · d) ≠ sign(mu_neg_k · d)``.

    Parameters
    ----------
    mu_pos : sequence of Tensor [H]
        Per-layer mean activation vectors for the *positive* (context-in) condition.
    mu_neg : sequence of Tensor [H]
        Per-layer mean activation vectors for the *negative* (context-out) condition.
    d_feat : Tensor [H]
        Global feature direction (will be L2-normalised internally).

    Returns
    -------
    disc_layers : list of int
        Layer indices where the sign of the projection flips between conditions.
    """
    d = d_feat.to(dtype=torch.float32)
    d = d / (d.norm() + 1e-12)

    n_layers = min(len(mu_pos), len(mu_neg))
    disc_layers: List[int] = []

    for k in range(n_layers):
        mp = mu_pos[k].to(dtype=torch.float32)
        mn = mu_neg[k].to(dtype=torch.float32)
        proj_pos = float(torch.dot(mp, d))
        proj_neg = float(torch.dot(mn, d))
        if proj_pos * proj_neg < 0:
            disc_layers.append(k)

    return disc_layers
