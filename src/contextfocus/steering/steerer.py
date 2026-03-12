from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from contextfocus.steering.householder import householder_rotate_last_token
from contextfocus.utils import get_transformer_blocks


@dataclass
class SteeringConfig:
    layer: int
    multiplier: float = 2.0
    apply_to_prompt_last_token: bool = True  # steer the last token in the prompt forward pass
    steer_all_positions: bool = False        # if True, add to all positions, else last token only


class ActivationSteerer:
    """
    Adds a steering vector to residual stream activations at a chosen transformer block.

    Implementation note:
    - We hook the *output* of the selected transformer block.
    - We modify the hidden states by adding m * v either to the last position or all positions.
    """

    def __init__(self, model: Any, vector: torch.Tensor, cfg: SteeringConfig):
        self.model = model
        self.vector = vector.to(device=model.device, dtype=torch.float16 if model.dtype == torch.float16 else torch.bfloat16)
        self.cfg = cfg
        self._handle = None

    def _hook(self, module, inputs, output):
        # output may be tensor or tuple. We assume first item is hidden states.
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = None

        if hs is None:
            return output

        # hs: [B, T, D]
        hs = hs.clone()
        if self.cfg.steer_all_positions:
            hs = hs + (self.cfg.multiplier * self.vector)[None, None, :]
        else:
            hs[:, -1, :] = hs[:, -1, :] + (self.cfg.multiplier * self.vector)[None, :]

        if rest is None:
            return hs
        return (hs,) + rest

    def __enter__(self):
        blocks = get_transformer_blocks(self.model)
        if self.cfg.layer < 0 or self.cfg.layer >= len(blocks):
            raise ValueError(f"Layer {self.cfg.layer} out of range for {len(blocks)} layers.")
        self._handle = blocks[self.cfg.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def load_vector(vectors_dir: str | Path, layer: int) -> torch.Tensor:
    p = Path(vectors_dir) / f"layer_{layer:03d}.pt"
    return torch.load(p, map_location="cpu")


# ---------------------------------------------------------------------------
# Norm-preserving rotation steerer (Householder / 2-D rotation variant)
# ---------------------------------------------------------------------------


class HouseholderSteerer:
    """Context manager that rotates the last-token hidden state toward *vector*.

    Unlike :class:`ActivationSteerer` (which *adds* ``m * v``),
    ``HouseholderSteerer`` performs a norm-preserving 2-D rotation in the
    plane spanned by the steering vector and the orthogonal component of the
    hidden state.  This avoids activation-magnitude drift.

    Parameters
    ----------
    model : Any
        HuggingFace causal LM.
    vector : Tensor [H] or [B, H]
        Steering direction (static or query-specific Δh).
    layer : int
        Transformer block index.
    theta : float
        Rotation angle in radians (default ``π/6 ≈ 0.524``).
    apply_to_prompt_last_token : bool
        Currently unused — kept for API symmetry with
        :class:`ActivationSteerer`.
    """

    def __init__(
        self,
        model: Any,
        vector: torch.Tensor,
        layer: int,
        theta: float = math.pi / 6,
        apply_to_prompt_last_token: bool = True,
    ):
        self.model = model
        self.vector = vector  # may be on CPU; moved lazily
        self.layer = layer
        self.theta = float(theta)
        self.apply_to_prompt_last_token = apply_to_prompt_last_token
        self._handle: Optional[Any] = None
        self._prompt_length: Optional[int] = None

    # ---- forward hook --------------------------------------------------

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = None

        if hs is None:
            return output

        # Track prompt length on first call
        if self._prompt_length is None:
            self._prompt_length = hs.size(1)  # sequence length
            
        # Only apply rotation during prompt phase (not generation)
        current_seq_len = hs.size(1)
        if current_seq_len != self._prompt_length:
            # We're in generation phase, don't apply steering
            return output

        # Lazy device / dtype alignment
        self.vector = self.vector.to(device=hs.device, dtype=hs.dtype)

        hs_steered = householder_rotate_last_token(hs, self.vector, self.theta)

        if rest is None:
            return hs_steered
        return (hs_steered,) + rest

    # ---- context manager -----------------------------------------------

    def __enter__(self):
        blocks = get_transformer_blocks(self.model)
        if self.layer < 0 or self.layer >= len(blocks):
            raise ValueError(
                f"Layer {self.layer} out of range for {len(blocks)} layers."
            )
        # Reset prompt length tracking for each context
        self._prompt_length = None
        self._handle = blocks[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
