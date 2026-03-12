from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from contextfocus.inference.conflict_detector import ConflictDetectConfig, detect_conflict
from contextfocus.prompting.templates import (
    PromptParts,
    build_openended_messages,
    build_openended_prompt,
    can_use_chat_template,
)
from contextfocus.steering.bplis import BPLISConfig, LatentInterventionSearch
from contextfocus.steering.dynamic_selector import InfluenceLayerSelector, InfluenceSelectConfig
from contextfocus.steering.steerer import ActivationSteerer, HouseholderSteerer, SteeringConfig, load_vector
from contextfocus.utils import ModelBundle, decode, get_eos_id, get_model_hidden_size, tokenize_chat, tokenize_text


@dataclass(frozen=True)
class SearchConfig:
    budget: int = 6  # number of candidate configs evaluated
    probe_tokens: int = 24
    final_tokens: int = 64
    multipliers: Tuple[float, ...] = (1.0, 2.0, 3.0)
    top_k_layers: int = 6
    js_threshold: float = 0.08
    max_length: int = 1024
    # ---- B-PLIS options (latent vector search) ----
    use_bplis: bool = False
    bplis_generations: int = 10
    bplis_popsize: int = 8
    bplis_intrinsic_dim: int = 64
    householder_theta: float = 0.15


def _context_token_ids(tokenizer_or_processor, context: str) -> torch.Tensor:
    # Unwrap processor → inner tokenizer (Gemma 3 uses AutoProcessor, not AutoTokenizer)
    inner = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    ids = inner(context, add_special_tokens=False).input_ids
    uniq = sorted(set(i for i in ids if i is not None))
    return torch.tensor(uniq, dtype=torch.long)


def _grounding_mass(scores: List[torch.Tensor], context_ids: torch.Tensor) -> float:
    if not scores:
        return 0.0
    ctx = context_ids
    total = 0.0
    for step_logits in scores:
        p = F.softmax(step_logits.float(), dim=-1)
        total += float(p[0, ctx].sum().detach().cpu())
    return total / len(scores)


def _repetition_rate(text: str) -> float:
    toks = text.lower().split()
    if len(toks) < 8:
        return 0.0
    uniq = len(set(toks))
    return 1.0 - (uniq / len(toks))


def _encode_prompt(bundle: ModelBundle, *, question: str, context: str, oi_prompt: bool, max_length: int):
    tok = bundle.tokenizer
    model = bundle.model
    parts = PromptParts(system="", context=context, question=question)
    if can_use_chat_template(tok):
        msgs = build_openended_messages(parts, oi_prompt=oi_prompt)
        return tokenize_chat(tok, msgs, max_length=max_length).to(model.device)
    prompt = build_openended_prompt(parts, oi_prompt=oi_prompt)
    return tokenize_text(tok, prompt, max_length=max_length).to(model.device)


def budgeted_latent_activation_search(
    bundle: ModelBundle,
    *,
    vectors_dir: str,
    question: str,
    context: str,
    cfg: SearchConfig = SearchConfig(),
    oi_prompt: bool = False,
) -> dict:
    """
    Per-query inference-time method:

    1) Conflict detection using context-in vs context-out divergence.
       If no conflict, generate normally (no steering).
    2) If conflict, rank candidate layers dynamically using the same probe forward passes.
    3) Evaluate up to `budget` candidate (layer, multiplier) configs with short probe generations.
    4) Pick the best config by an approximate grounding proxy.
    """
    model = bundle.model
    tok = bundle.tokenizer

    det = detect_conflict(
        model,
        tok,
        question=question,
        context=context,
        oi_prompt=oi_prompt,
        cfg=ConflictDetectConfig(js_threshold=cfg.js_threshold, max_length=cfg.max_length),
    )

    inputs = _encode_prompt(bundle, question=question, context=context, oi_prompt=oi_prompt, max_length=cfg.max_length)
    input_len = int(inputs["input_ids"].shape[-1])

    if not det["is_conflict"]:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.final_tokens,
                do_sample=False,
                pad_token_id=get_eos_id(tok),
            )
        return {
            "used_search": False,
            "divergence": det["divergence"],
            "layer": None,
            "multiplier": 0.0,
            "text": decode(tok, out[0, input_len:]),
        }

    # Causal dynamic selection using logit-sensitivity
    out_inputs = _encode_prompt(bundle, question=question, context="", oi_prompt=oi_prompt, max_length=cfg.max_length)
    selector = InfluenceLayerSelector(
        model,
        vectors_dir=vectors_dir,
        cfg=InfluenceSelectConfig(return_top_k=cfg.top_k_layers),
    )
    with torch.enable_grad():
        sel = selector.choose_layer(in_inputs=inputs, out_inputs=out_inputs)
    ranked = sel["ranked_layers"]
    
    # Build layers list: chosen + ranked + best_static_layer (14) as safety net
    layers = []
    seen = set()
    for l in [sel["chosen_layer"], *[r["layer"] for r in ranked], selector.cfg.best_static_layer]:
        if l not in seen:
            layers.append(l)
            seen.add(l)

    candidates = []
    for l in layers:
        for m in cfg.multipliers:
            candidates.append((l, m))
    candidates = candidates[: cfg.budget]

    ctx_ids = _context_token_ids(tok, context).to(model.device)

    best: Dict[str, Any] = {"score": -1.0, "layer": None, "multiplier": None, "dh": None, "source": "static"}
    traces = []

    # ---- (A) Static-vector candidate evaluation (original path) ----
    for (layer, mult) in candidates:
        v = load_vector(vectors_dir, layer=layer)
        steer_cfg = SteeringConfig(layer=layer, multiplier=mult)
        with ActivationSteerer(model, v, steer_cfg):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg.probe_tokens,
                    do_sample=False,
                    pad_token_id=get_eos_id(tok),
                    return_dict_in_generate=True,
                    output_scores=True,
                )

        grounding = _grounding_mass(out.scores, ctx_ids)
        probe_text = decode(tok, out.sequences[0, input_len:])
        rep = _repetition_rate(probe_text)
        score = grounding - 0.05 * rep

        traces.append(
            {
                "layer": layer,
                "multiplier": mult,
                "grounding": grounding,
                "repetition": rep,
                "score": score,
                "source": "static",
            }
        )
        if score > best["score"]:
            best = {"score": score, "layer": layer, "multiplier": mult, "dh": None, "source": "static"}

    # ---- (B) B-PLIS latent vector search (optional) ----
    bplis_dh = None
    if cfg.use_bplis:
        hidden_size = get_model_hidden_size(model)
        bplis_cfg = BPLISConfig(
            intrinsic_dim=cfg.bplis_intrinsic_dim,
            max_generations=cfg.bplis_generations,
            popsize=cfg.bplis_popsize,
            householder_theta=cfg.householder_theta,
        )
        searcher = LatentInterventionSearch(
            model=model,
            tokenizer=tok,
            hidden_size=hidden_size,
            cfg=bplis_cfg,
        )
        # Use discriminative layers from selector if available, else top layers
        bplis_target = sel.get("discriminative_layers", layers[:3]) or layers[:3]
        ctx_id_list = ctx_ids.cpu().tolist()
        bplis_dh = searcher.search(
            query_inputs=inputs,
            target_layers=bplis_target[:3],  # limit for budget
            context_token_ids=ctx_id_list,
            max_generations=cfg.bplis_generations,
            popsize=cfg.bplis_popsize,
            probe_tokens=cfg.probe_tokens,
        )

        # Evaluate the B-PLIS Δh on target layers via probe
        steerers_ctx = []
        for tl in bplis_target[:3]:
            s = HouseholderSteerer(model, vector=bplis_dh, layer=tl, theta=cfg.householder_theta)
            s.__enter__()
            steerers_ctx.append(s)
        try:
            with torch.no_grad():
                out_bplis = model.generate(
                    **inputs,
                    max_new_tokens=cfg.probe_tokens,
                    do_sample=False,
                    pad_token_id=get_eos_id(tok),
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        finally:
            for s in steerers_ctx:
                s.__exit__(None, None, None)

        grounding_b = _grounding_mass(out_bplis.scores, ctx_ids)
        probe_text_b = decode(tok, out_bplis.sequences[0, input_len:])
        rep_b = _repetition_rate(probe_text_b)
        score_b = grounding_b - 0.05 * rep_b

        traces.append(
            {
                "layer": bplis_target[:3],
                "multiplier": None,
                "grounding": grounding_b,
                "repetition": rep_b,
                "score": score_b,
                "source": "bplis",
            }
        )
        if score_b > best["score"]:
            best = {
                "score": score_b,
                "layer": bplis_target[:3],
                "multiplier": None,
                "dh": bplis_dh,
                "source": "bplis",
            }

    # ---- Final generation with best candidate ----
    if best["source"] == "bplis" and best["dh"] is not None:
        # Use HouseholderSteerer with synthesised Δh
        final_steerers = []
        target_layers = best["layer"] if isinstance(best["layer"], list) else [best["layer"]]
        for tl in target_layers:
            s = HouseholderSteerer(model, vector=best["dh"], layer=tl, theta=cfg.householder_theta)
            s.__enter__()
            final_steerers.append(s)
        try:
            with torch.no_grad():
                final = model.generate(
                    **inputs,
                    max_new_tokens=cfg.final_tokens,
                    do_sample=False,
                    pad_token_id=get_eos_id(tok),
                )
        finally:
            for s in final_steerers:
                s.__exit__(None, None, None)
    else:
        # Original static-vector path
        final_v = load_vector(vectors_dir, layer=best["layer"])
        with ActivationSteerer(model, final_v, SteeringConfig(layer=best["layer"], multiplier=best["multiplier"])):
            with torch.no_grad():
                final = model.generate(
                    **inputs,
                    max_new_tokens=cfg.final_tokens,
                    do_sample=False,
                    pad_token_id=get_eos_id(tok),
                )

    return {
        "used_search": True,
        "divergence": det["divergence"],
        "ranked_layers": ranked,
        "candidates": traces,
        "layer": best["layer"],
        "multiplier": best["multiplier"],
        "source": best["source"],
        "text": decode(tok, final[0, input_len:]),
    }
