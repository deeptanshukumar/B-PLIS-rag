from __future__ import annotations

"""Dynamic layer selection strategies.

Why the original heuristic can fail
---------------------------------
The classic dynamic heuristic ranks layers by alignment between

    Δh_l = h_l(context-in) - h_l(context-out)

and the learned steering vector v_l. That measures *representation change* on the prompt.
But steering effectiveness depends on *causal sensitivity of the logits* to a residual
intervention at layer l.

This module keeps the original alignment selector (for ablations) and adds a new,
logit-causal selector that typically correlates much better with generation-time steering.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import torch

from contextfocus.steering.discriminative import get_discriminative_layers
from contextfocus.steering.steerer import load_vector
from contextfocus.utils import get_transformer_blocks


# -----------------------------------------------------------------------------
# (A) Original alignment-based selector (kept for ablations)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentSelectConfig:
    top_k: int = 6
    score_mode: str = "cosine_times_norm"  # or "cosine", "norm"
    layer_band: Tuple[int, int] | None = None  # restrict to [lo, hi) if set


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return (a * b).sum()


class AlignmentLayerSelector:
    """Ranks layers by alignment between Δh_l and v_l."""

    def __init__(self, model: Any, vectors_dir: str | Path, cfg: AlignmentSelectConfig = AlignmentSelectConfig()):
        self.model = model
        self.vectors_dir = Path(vectors_dir)
        self.cfg = cfg
        self.n_layers = len(get_transformer_blocks(model))
        self._vectors: Optional[List[torch.Tensor]] = None

    def _ensure_vectors(self):
        if self._vectors is not None:
            return
        self._vectors = [load_vector(self.vectors_dir, l).float() for l in range(self.n_layers)]

    def rank_layers(self, pos_hidden_states: Sequence[torch.Tensor], neg_hidden_states: Sequence[torch.Tensor]) -> List[dict]:
        """pos/neg_hidden_states are HF outputs.hidden_states tuples."""
        self._ensure_vectors()

        lo, hi = 0, self.n_layers
        if self.cfg.layer_band is not None:
            lo, hi = self.cfg.layer_band

        scores: List[dict] = []
        for l in range(lo, hi):
            h_pos = pos_hidden_states[l + 1][0, -1, :].detach().float().cpu()
            h_neg = neg_hidden_states[l + 1][0, -1, :].detach().float().cpu()
            delta = h_pos - h_neg
            v = self._vectors[l].cpu()

            cos = float(_cosine(delta, v))
            nrm = float(delta.norm(p=2))

            if self.cfg.score_mode == "cosine":
                s = cos
            elif self.cfg.score_mode == "norm":
                s = nrm
            else:
                s = cos * nrm
            scores.append({"layer": l, "score": float(s), "cos": cos, "delta_norm": nrm})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[: self.cfg.top_k]


# -----------------------------------------------------------------------------
# (B) New selector: Bayesian Causal Influence Layer Selection (BCILS)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class InfluenceSelectConfig:
    """Config for logit-causal dynamic selection.

    token_top_k: how many vocab tokens define the context-vs-memory contrast set.
    prior_weight: how strongly to bias toward layers that are globally effective (from a sweep).
    fallback_to_best: if true, fall back to best_static_layer when confidence is low.
    """

    token_top_k: int = 64
    prior_weight: float = 1.5
    return_top_k: int = 6
    best_static_layer: int = 14
    fallback_to_best: bool = True
    confidence_margin: float = 0.02  # if top score - runner up < margin, use fallback
    layer_band: Tuple[int, int] | None = None


def _read_layer_prior(layer_sweep_path: Optional[str | Path], n_layers: int) -> Optional[torch.Tensor]:
    """Reads a prior over layers from a layer sweep json.

    Expected format: list of {"layer": int, "ps": float, ...}
    We convert ps into logit(ps) as a stable additive prior.
    """
    if layer_sweep_path is None:
        return None
    p = Path(layer_sweep_path)
    if not p.exists():
        return None

    try:
        data = json.loads(p.read_text())
        prior = torch.zeros(n_layers, dtype=torch.float32)
        eps = 1e-4
        for row in data:
            l = int(row["layer"])
            ps = float(row.get("ps", 0.0))
            ps = min(max(ps, eps), 1.0 - eps)
            prior[l] = float(torch.log(torch.tensor(ps / (1.0 - ps))))
        return prior
    except Exception:
        return None


class InfluenceLayerSelector:
    """Dynamic layer selection using a causal, logit-sensitivity criterion."""

    def __init__(
        self,
        model: Any,
        vectors_dir: str | Path,
        cfg: InfluenceSelectConfig = InfluenceSelectConfig(),
        layer_sweep_path: Optional[str | Path] = None,
    ):
        self.model = model
        self.vectors_dir = Path(vectors_dir)
        self.cfg = cfg
        self.blocks = get_transformer_blocks(model)
        self.n_layers = len(self.blocks)
        self._vectors: Optional[List[torch.Tensor]] = None
        self._prior = _read_layer_prior(layer_sweep_path, self.n_layers)

    def _ensure_vectors(self):
        if self._vectors is not None:
            return
        self._vectors = [load_vector(self.vectors_dir, l).float() for l in range(self.n_layers)]

    def choose_layer(self, *, in_inputs: Dict[str, torch.Tensor], out_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Returns {chosen_layer, ranked_layers}. Inputs must already be on model.device."""
        self._ensure_vectors()
        lo, hi = 0, self.n_layers
        if self.cfg.layer_band is not None:
            lo, hi = self.cfg.layer_band

        # 1) context-out logits (no grad)
        with torch.no_grad():
            out_logits = self.model(**out_inputs, use_cache=False).logits[:, -1, :].float()

        # 2) forward with grad for context-in; capture per-layer last-token residuals
        saved_last: List[Optional[torch.Tensor]] = [None for _ in range(self.n_layers)]
        handles = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                if hs is None:
                    return output
                last = hs[:, -1, :]
                last.retain_grad()
                saved_last[layer_idx] = last
                return output

            return hook

        for l in range(lo, hi):
            handles.append(self.blocks[l].register_forward_hook(make_hook(l)))

        try:
            self.model.zero_grad(set_to_none=True)
            in_out = self.model(**in_inputs, use_cache=False)
            in_logits = in_out.logits[:, -1, :].float()

            # 3) context-vs-memory token sets via logsoftmax diff
            with torch.no_grad():
                lsm_in = torch.log_softmax(in_logits.detach(), dim=-1)
                lsm_out = torch.log_softmax(out_logits.detach(), dim=-1)
                diff = lsm_in - lsm_out
                k = min(self.cfg.token_top_k, diff.shape[-1])
                ctx_tokens = torch.topk(diff, k=k, dim=-1).indices[0]
                mem_tokens = torch.topk(-diff, k=k, dim=-1).indices[0]

            # 4) utility U and backward
            U = torch.logsumexp(in_logits[0, ctx_tokens], dim=0) - torch.logsumexp(in_logits[0, mem_tokens], dim=0)
            U.backward()

            # 5) influence per layer + Bayesian prior
            ranked: List[dict] = []
            for l in range(lo, hi):
                last = saved_last[l]
                if last is None or last.grad is None:
                    continue
                grad = last.grad.detach()[0].float().to("cpu")
                v = self._vectors[l]
                influence = float(torch.dot(grad, v) / (v.norm(p=2) + 1e-8))
                score = influence
                if self._prior is not None:
                    score += float(self.cfg.prior_weight * self._prior[l].item())
                ranked.append({"layer": l, "score": float(score), "influence": influence})

            ranked.sort(key=lambda x: x["score"], reverse=True)

            # 5b) ----- Discriminative sign-based layer filter -----
            # Build per-layer mean vectors from the saved residuals.
            mu_pos_list = []
            mu_neg_list = []
            for l in range(lo, hi):
                last_pos = saved_last[l]
                if last_pos is not None:
                    mu_pos_list.append(last_pos.detach()[0].float().cpu())
                else:
                    mu_pos_list.append(torch.zeros_like(self._vectors[l]))
            # For negative side, re-run a no-grad forward to collect residuals
            # (cheaper than another grad pass).  We already have out_logits, so
            # we grab them from the saved hook outputs of the *first* (context-out)
            # forward.  However those hooks were not active then.  Instead, we
            # approximate mu_neg as mu_pos minus the direction of the steering
            # vector (the vectors *are* the population-level pos-neg diff).
            for l in range(lo, hi):
                v_l = self._vectors[l]
                mu_neg_list.append(mu_pos_list[l - lo] - v_l)

            # Global feature direction = mean(mu_pos) - mean(mu_neg)
            _device_cpu = torch.device("cpu")
            mu_pos_stack = torch.stack([m.to(_device_cpu) for m in mu_pos_list])
            mu_neg_stack = torch.stack([m.to(_device_cpu) for m in mu_neg_list])
            d_feat = mu_pos_stack.mean(dim=0) - mu_neg_stack.mean(dim=0)
            d_feat = d_feat / (d_feat.norm() + 1e-12)

            disc_layers = get_discriminative_layers(mu_pos_list, mu_neg_list, d_feat)

            # Restrict ranked list to discriminative layers when available
            ranked_layer_indices = [r["layer"] for r in ranked]
            if disc_layers:
                filtered = [r for r in ranked if r["layer"] in disc_layers]
                if filtered:
                    ranked = filtered
                # else: keep original ranked (defence)

            # 6) confidence-aware fallback to best static layer (layer 14)
            chosen = ranked[0]["layer"] if ranked else self.cfg.best_static_layer
            if self.cfg.fallback_to_best and len(ranked) >= 2:
                if (ranked[0]["score"] - ranked[1]["score"]) < self.cfg.confidence_margin:
                    chosen = self.cfg.best_static_layer

            return {
                "chosen_layer": int(chosen),
                "ranked_layers": ranked[: self.cfg.return_top_k],
                "discriminative_layers": disc_layers,
            }
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
