"""Norm-preserving 2-D rotation of hidden-state *h* in the plane spanned by
steering vector *v* and the orthogonal component of *h*.

The core idea: instead of *adding* ``multiplier * v`` (which changes ||h||),
we rotate *h* towards *v* by angle ``theta`` while keeping the norm fixed.
This avoids activation-magnitude drift that can destabilise decoding.

Exported
--------
householder_rotate_last_token(h, v, theta, eps) -> Tensor
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def _ensure_shape_v(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Broadcast *v* to ``[B, H]`` matching ``h[:, -1, :]``."""
    if v.dim() == 1:
        return v.unsqueeze(0).expand(h.size(0), -1)
    elif v.dim() == 2 and v.size(0) == h.size(0):
        return v
    else:
        raise ValueError(
            f"v must be [hidden] or [batch, hidden] matching h batch size "
            f"(got v.shape={tuple(v.shape)}, h batch={h.size(0)})"
        )


def householder_rotate_last_token(
    h: torch.Tensor,
    v: torch.Tensor,
    theta: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Norm-preserving 2-D rotation of the last-token hidden vector per batch.

    Parameters
    ----------
    h : Tensor [B, S, H]
        Full hidden-state tensor.  Only the last sequence position is touched.
    v : Tensor [H] or [B, H]
        Steering vector (direction to rotate toward).
    theta : float
        Rotation angle in radians.  ``pi/6`` ≈ 30° is a good starting point.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    h_out : Tensor [B, S, H]
        Copy of *h* with only ``h[:, -1, :]`` rotated.  ``||h[:, -1, :]||``
        is preserved within floating-point tolerance.
    """
    B, S, H = h.shape
    device = h.device
    dtype = h.dtype

    # ---- work on last token ----
    h_last = h[:, -1, :].to(dtype=dtype)          # [B, H]
    v_b = _ensure_shape_v(v.to(dtype=dtype, device=device), h)  # [B, H]

    # b1 = unit vector along v
    v_norm = v_b.norm(dim=-1, keepdim=True).clamp_min(eps)      # [B, 1]
    b1 = v_b / v_norm                                           # [B, H]

    # component of h along b1
    alpha = torch.sum(h_last * b1, dim=-1, keepdim=True)        # [B, 1]

    # orthogonal residual → b2
    h_orth = h_last - alpha * b1                                # [B, H]
    h_orth_norm = h_orth.norm(dim=-1, keepdim=True)             # [B, 1]

    # If h is (nearly) parallel to v, build a deterministic fallback b2
    # Use conservative threshold to avoid numerical instability
    parallel_threshold = eps * 1000  # 1e-5 for default eps=1e-8
    small_mask = (h_orth_norm < parallel_threshold).squeeze(-1) # [B]
    
    # Start with normal computation for all cases
    b2 = h_orth / h_orth_norm.clamp_min(eps)                    # [B, H]
    
    if small_mask.any():
        # Build orthogonal fallback for parallel cases
        fallback = torch.roll(b1, shifts=1, dims=-1)            # [B, H]
        fallback = fallback - (fallback * b1).sum(-1, keepdim=True) * b1
        fallback_norm = fallback.norm(dim=-1, keepdim=True).clamp_min(eps)
        fallback = fallback / fallback_norm                     # [B, H]
        # Replace b2 rows where mask is True
        b2[small_mask] = fallback[small_mask]

    # in-plane coefficient along b2
    beta = torch.sum(h_last * b2, dim=-1, keepdim=True)        # [B, 1]

    # rotation in the (b1, b2) plane
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    alpha_p = cos_t * alpha - sin_t * beta
    beta_p = sin_t * alpha + cos_t * beta

    proj_new = alpha_p * b1 + beta_p * b2
    proj_old = alpha  * b1 + beta  * b2

    h_last_steered = h_last - proj_old + proj_new               # [B, H]

    # ---- assemble output (no in-place mutation) ----
    h_out = h.clone()
    h_out[:, -1, :] = h_last_steered
    return h_out
