from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from tqdm import tqdm

from contextfocus.data.confiaq import ConFiQAExample
from contextfocus.eval.metrics import score_batch
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
class EvalConfig:
    max_new_tokens: int = 64
    multiplier: float = 2.0
    oi_prompt: bool = False
    dynamic_layers: bool = False
    top_k_layers: int = 6
    max_length: int = 1024
    # Optional: path to layer_sweep.json for a prior over layers (recommended).
    layer_sweep_path: str | None = None
    # ---- B-PLIS options (latent vector search) ----
    use_bplis: bool = False
    bplis_generations: int = 10
    bplis_popsize: int = 8
    bplis_intrinsic_dim: int = 64
    householder_theta: float = 0.15  # radians (≈ 9°)


def _encode_openended(bundle: ModelBundle, *, context: str, question: str, oi_prompt: bool, max_length: int):
    tok = bundle.tokenizer
    model = bundle.model
    parts = PromptParts(system="", context=context, question=question)

    if can_use_chat_template(tok):
        msgs = build_openended_messages(parts, oi_prompt=oi_prompt)
        return tokenize_chat(tok, msgs, max_length=max_length).to(model.device)
    prompt = build_openended_prompt(parts, oi_prompt=oi_prompt)
    return tokenize_text(tok, prompt, max_length=max_length).to(model.device)


@torch.no_grad()
def _generate(bundle: ModelBundle, prompt_inputs, *, max_new_tokens: int) -> str:
    model = bundle.model
    tok = bundle.tokenizer
    input_len = int(prompt_inputs["input_ids"].shape[-1])
    out = model.generate(
        **prompt_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=get_eos_id(tok),
    )
    return decode(tok, out[0, input_len:])


def evaluate_confiaq(
    bundle: ModelBundle,
    examples: Iterable[ConFiQAExample],
    *,
    vectors_dir: str | Path,
    layer: Optional[int] = None,
    cfg: EvalConfig = EvalConfig(),
) -> dict:
    model = bundle.model
    tok = bundle.tokenizer

    gens: List[str] = []
    originals: List[str] = []
    substituted: List[str] = []
    chosen_layers: List[int] = []

    selector = None
    if cfg.dynamic_layers:
        selector = InfluenceLayerSelector(
            model,
            vectors_dir=vectors_dir,
            cfg=InfluenceSelectConfig(return_top_k=cfg.top_k_layers),
            layer_sweep_path=cfg.layer_sweep_path,
        )

    examples_list = list(examples)
    subset_name = getattr(examples_list[0], 'subset', '') if examples_list else ''
    
    # Initialize B-PLIS searcher if enabled
    bplis_searcher = None
    if cfg.use_bplis:
        hidden_size = get_model_hidden_size(model)
        bplis_cfg = BPLISConfig(
            intrinsic_dim=cfg.bplis_intrinsic_dim,
            max_generations=cfg.bplis_generations,
            popsize=cfg.bplis_popsize,
            householder_theta=cfg.householder_theta,
        )
        bplis_searcher = LatentInterventionSearch(
            model=model,
            tokenizer=tok,
            hidden_size=hidden_size,
            cfg=bplis_cfg,
        )
    
    for ex in tqdm(examples_list, desc=f"Eval {subset_name}"):
        in_inputs = _encode_openended(bundle, context=ex.context, question=ex.question, oi_prompt=cfg.oi_prompt, max_length=cfg.max_length)

        if cfg.use_bplis:
            # B-PLIS: search for optimal latent vector
            # Get context token IDs for grounding
            # Unwrap processor → inner tokenizer (Gemma 3 uses AutoProcessor)
            _inner_tok = getattr(tok, "tokenizer", tok)
            ctx_ids = _inner_tok(ex.context, add_special_tokens=False).input_ids
            ctx_id_list = sorted(set(i for i in ctx_ids if i is not None))
            
            # Determine target layers
            if cfg.dynamic_layers:
                out_inputs = _encode_openended(bundle, context="", question=ex.question, oi_prompt=cfg.oi_prompt, max_length=cfg.max_length)
                with torch.enable_grad():
                    sel = selector.choose_layer(in_inputs=in_inputs, out_inputs=out_inputs)
                chosen = int(sel["chosen_layer"])
                chosen_layers.append(chosen)
                target_layers = [chosen]
            elif layer is not None:
                target_layers = [layer]
                chosen_layers.append(layer)
            else:
                # Use top layers from selector or default to middle layers
                target_layers = list(range(model.config.num_hidden_layers // 2, model.config.num_hidden_layers // 2 + 3))
                chosen_layers.append(target_layers[0])
            
            # Search for optimal delta-h
            dh = bplis_searcher.search(
                query_inputs=in_inputs,
                target_layers=target_layers[:1],  # Use primary layer
                context_token_ids=ctx_id_list,
                max_generations=cfg.bplis_generations,
                popsize=cfg.bplis_popsize,
                probe_tokens=cfg.max_new_tokens,
            )
            
            # Apply using Householder steering
            with HouseholderSteerer(model, vector=dh, layer=target_layers[0], theta=cfg.householder_theta):
                gens.append(_generate(bundle, in_inputs, max_new_tokens=cfg.max_new_tokens))
        
        elif cfg.dynamic_layers:
            # Causal dynamic selection uses logits sensitivity rather than prompt-space alignment.
            out_inputs = _encode_openended(bundle, context="", question=ex.question, oi_prompt=cfg.oi_prompt, max_length=cfg.max_length)
            with torch.enable_grad():
                sel = selector.choose_layer(in_inputs=in_inputs, out_inputs=out_inputs)
            chosen = int(sel["chosen_layer"])
            chosen_layers.append(chosen)

            v = load_vector(vectors_dir, chosen)
            with ActivationSteerer(model, v, SteeringConfig(layer=chosen, multiplier=cfg.multiplier)):
                gens.append(_generate(bundle, in_inputs, max_new_tokens=cfg.max_new_tokens))
        else:
            if layer is None:
                gens.append(_generate(bundle, in_inputs, max_new_tokens=cfg.max_new_tokens))
            else:
                v = load_vector(vectors_dir, layer)
                with ActivationSteerer(model, v, SteeringConfig(layer=layer, multiplier=cfg.multiplier)):
                    gens.append(_generate(bundle, in_inputs, max_new_tokens=cfg.max_new_tokens))

        originals.append(ex.original_answer)
        substituted.append(ex.substituted_answer)

    counts = score_batch(gens, original_answers=originals, substituted_answers=substituted)
    result = {
        "n": counts.n,
        "ps": counts.ps,
        "po": counts.po,
        "ps_rate": counts.ps_rate,
        "po_rate": counts.po_rate,
        "mr": counts.mr,
    }
    if chosen_layers:
        result["chosen_layers"] = chosen_layers
    return result
