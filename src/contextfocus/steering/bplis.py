"""B-PLIS — Budgeted PLIS (CMA-ES latent vector search).

Synthesises a *query-specific* perturbation Δh by searching in a
low-dimensional intrinsic subspace using CMA-ES.  The search optimises
a grounding proxy (probability mass on context tokens) through short
probe generations.

Exported
--------
LatentInterventionSearch   – main search class
BPLISConfig                – configuration dataclass
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Lazy import so the module can still be imported if cma is absent at
# parse time — the error will be raised at .search() call time instead.
try:
    import cma as _cma  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _cma = None


@dataclass(frozen=True)
class BPLISConfig:
    """Tunables for Latent Intervention Search."""

    intrinsic_dim: int = 64
    max_generations: int = 10
    popsize: int = 8
    sigma0: float = 0.5
    seed: int = 7
    householder_theta: float = 0.6  # radians (≈ 34°)


class LatentInterventionSearch:
    """CMA-ES search in a low-rank random subspace for a query-specific Δh.

    Parameters
    ----------
    model : torch.nn.Module
        HuggingFace causal LM (or any ``torch.nn.Module`` with ``.generate``).
    tokenizer : Any
        Corresponding tokenizer / processor.
    hidden_size : int
        Dimensionality of the model's residual stream.
    cfg : BPLISConfig
        Search hyper-parameters.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        hidden_size: int,
        cfg: BPLISConfig = BPLISConfig(),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.U: Optional[torch.Tensor] = None  # [hidden_size, intrinsic_dim]
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Random orthonormal projector
    # ------------------------------------------------------------------

    def setup_random_orthoprojector(
        self, device: Optional[torch.device | str] = None
    ) -> torch.Tensor:
        """Build an orthonormal projection matrix ``U`` via QR decomposition.

        Returns
        -------
        U : Tensor [hidden_size, intrinsic_dim]
        """
        device = device or self.device
        # QR decomposition not supported on MPS; use CPU then move to target device
        A = torch.randn(
            self.hidden_size, self.cfg.intrinsic_dim,
            device="cpu", dtype=torch.float32,
        )
        Q, _R = torch.linalg.qr(A)              # Q: [H, d]
        self.U = Q.to(device)
        return self.U

    # ------------------------------------------------------------------
    # Low-rank → full-rank mapping
    # ------------------------------------------------------------------

    def _z_to_dh(self, z: np.ndarray) -> torch.Tensor:
        """Map intrinsic vector *z* ``(d,)`` to full Δh ``(H,)``."""
        z_t = torch.from_numpy(z.astype(np.float32)).to(self.U.device)
        return (self.U @ z_t).to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        dh: torch.Tensor,
        query_inputs: Dict[str, torch.Tensor],
        target_layers: List[int],
        probe_tokens: int,
        context_token_ids: List[int],
    ) -> float:
        """Score a candidate Δh by grounding proxy over a short probe.

        Applies ``dh`` via :class:`HouseholderSteerer` at each
        ``target_layer``, generates ``probe_tokens`` tokens greedily,
        and returns ``grounding + 0.1 * normalised_entropy``.
        """
        # Import here to avoid circular import at module level
        from contextfocus.steering.steerer import HouseholderSteerer
        from contextfocus.utils import get_eos_id

        device = self.device
        dh = dh.detach().to(device)

        # Enter one steerer per target layer
        steerers: List[HouseholderSteerer] = []
        for layer in target_layers:
            s = HouseholderSteerer(
                self.model,
                vector=dh,
                layer=layer,
                theta=self.cfg.householder_theta,
            )
            s.__enter__()
            steerers.append(s)

        try:
            with torch.no_grad():
                inputs_cuda = {k: v.to(device) for k, v in query_inputs.items()}
                outputs = self.model.generate(
                    **inputs_cuda,
                    max_new_tokens=probe_tokens,
                    do_sample=False,
                    pad_token_id=get_eos_id(self.tokenizer),
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                scores = outputs.scores  # list[Tensor [B, V]]

                total_mass = 0.0
                mean_entropy = 0.0
                count = max(len(scores), 1)

                ctx_ids = torch.tensor(context_token_ids, dtype=torch.long, device=device)
                for step_logits in scores:
                    probs = F.softmax(step_logits[0].float(), dim=-1)
                    total_mass += float(probs[ctx_ids].sum())
                    mean_entropy += float(-(probs * torch.log(probs + 1e-12)).sum())

                grounding = total_mass / count
                norm_entropy = mean_entropy / count / math.log(
                    getattr(self.model.config, "vocab_size", 32000) + 1
                )
                reward = grounding + 0.1 * norm_entropy
        finally:
            for s in steerers:
                s.__exit__(None, None, None)

        return float(reward)

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------

    def search(
        self,
        query_inputs: Dict[str, torch.Tensor],
        target_layers: List[int],
        context_token_ids: List[int],
        max_generations: Optional[int] = None,
        popsize: Optional[int] = None,
        probe_tokens: int = 24,
    ) -> torch.Tensor:
        """Run CMA-ES search and return the best Δh ``[hidden_size]``.

        Parameters
        ----------
        query_inputs : dict
            Tokenised prompt (keys ``input_ids``, ``attention_mask``).
        target_layers : list of int
            Layers at which the candidate Δh will be injected.
        context_token_ids : list of int
            Unique token IDs from the context string (grounding proxy).
        max_generations : int, optional
            Override ``cfg.max_generations``.
        popsize : int, optional
            Override ``cfg.popsize``.
        probe_tokens : int
            Number of tokens in each probe generation.

        Returns
        -------
        dh_best : Tensor [hidden_size]
            Query-specific perturbation vector.
        """
        if _cma is None:
            raise ImportError(
                "cma is required for B-PLIS search.  Install via: pip install cma>=3.1.0"
            )

        if self.U is None:
            self.setup_random_orthoprojector(self.device)

        gens = max_generations if max_generations is not None else self.cfg.max_generations
        pop = popsize if popsize is not None else self.cfg.popsize

        # Graceful fallback: if gens or pop is 0, return zero vector
        if gens <= 0 or pop <= 0:
            return torch.zeros(self.hidden_size, dtype=torch.float32, device=self.device)

        es = _cma.CMAEvolutionStrategy(
            np.zeros(self.cfg.intrinsic_dim, dtype=np.float64),
            self.cfg.sigma0,
            {
                "popsize": pop,
                "seed": self.cfg.seed,
                "verb_log": 0,
                "verbose": -9,
                "maxiter": gens,
            },
        )

        for _gen_idx in range(gens):
            candidate_zs = es.ask()
            fitnesses = []
            for z in candidate_zs:
                dh = self._z_to_dh(z)
                reward = self._evaluate_candidate(
                    dh, query_inputs, target_layers, probe_tokens, context_token_ids,
                )
                fitnesses.append(-reward)  # CMA-ES minimises
            es.tell(candidate_zs, fitnesses)

        best_z = es.result.xbest
        return self._z_to_dh(best_z)
