#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from contextfocus.inference.budgeted_search import SearchConfig, budgeted_latent_activation_search
from contextfocus.inference.conflict_detector import ConflictDetectConfig, detect_conflict
from contextfocus.prompting.templates import PromptParts, build_openended_messages, build_openended_prompt, can_use_chat_template
from contextfocus.steering.bplis import BPLISConfig, LatentInterventionSearch
from contextfocus.steering.dynamic_selector import InfluenceLayerSelector, InfluenceSelectConfig
from contextfocus.steering.steerer import ActivationSteerer, HouseholderSteerer, SteeringConfig, load_vector
from contextfocus.utils import ModelBundle, decode, get_eos_id, get_model_hidden_size, load_hf_causal_lm, tokenize_chat, tokenize_text
import torch


def _encode_prompt(bundle: ModelBundle, *, question: str, context: str, oi_prompt: bool, max_length: int):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--vectors_dir", type=str, required=True)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--context", type=str, required=True)
    ap.add_argument("--layer", type=int, default=None, help="Fixed layer for static steering")
    ap.add_argument("--multiplier", type=float, default=2.0, help="Steering multiplier")
    ap.add_argument("--dynamic_layers", type=str, default="true", help="Enable dynamic layer selection")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--probe_tokens", type=int, default=24)
    ap.add_argument("--final_tokens", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    # ---- B-PLIS options ----
    ap.add_argument("--use_bplis", type=str, default="false", help="Enable B-PLIS latent vector search")
    ap.add_argument("--bplis_generations", type=int, default=10, help="B-PLIS CMA-ES generations")
    ap.add_argument("--bplis_popsize", type=int, default=8, help="B-PLIS CMA-ES population size")
    ap.add_argument("--bplis_intrinsic_dim", type=int, default=64, help="B-PLIS intrinsic dimension")
    ap.add_argument("--householder_theta", type=float, default=0.15, help="Householder rotation angle (radians)")
    args = ap.parse_args()

    bundle = load_hf_causal_lm(args.model_id, dtype=args.dtype)

    dynamic_layers = args.dynamic_layers.lower() in ["1", "true", "yes", "y"]
    use_bplis = args.use_bplis.lower() in ["1", "true", "yes", "y"]
    
    # Encode the prompt
    inputs = _encode_prompt(bundle, question=args.question, context=args.context, oi_prompt=False, max_length=1024)
    
    if use_bplis:
        # B-PLIS: search for optimal latent vector using CMA-ES
        hidden_size = get_model_hidden_size(bundle.model)
        bplis_cfg = BPLISConfig(
            intrinsic_dim=args.bplis_intrinsic_dim,
            max_generations=args.bplis_generations,
            popsize=args.bplis_popsize,
            householder_theta=args.householder_theta,
        )
        bplis_searcher = LatentInterventionSearch(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            hidden_size=hidden_size,
            cfg=bplis_cfg,
        )
        
        # Get context token IDs for grounding
        _inner_tok = getattr(bundle.tokenizer, "tokenizer", bundle.tokenizer)
        ctx_ids = _inner_tok(args.context, add_special_tokens=False).input_ids
        ctx_id_list = sorted(set(i for i in ctx_ids if i is not None))
        
        # Determine target layer
        if dynamic_layers:
            out_inputs = _encode_prompt(bundle, question=args.question, context="", oi_prompt=False, max_length=1024)
            selector = InfluenceLayerSelector(
                bundle.model,
                vectors_dir=args.vectors_dir,
                cfg=InfluenceSelectConfig(return_top_k=6),
            )
            with torch.enable_grad():
                sel = selector.choose_layer(in_inputs=inputs, out_inputs=out_inputs)
            target_layer = int(sel["chosen_layer"])
        else:
            target_layer = args.layer if args.layer is not None else 14
            
        # Search for optimal delta-h
        dh = bplis_searcher.search(
            query_inputs=inputs,
            target_layers=[target_layer],
            context_token_ids=ctx_id_list,
            max_generations=args.bplis_generations,
            popsize=args.bplis_popsize,
            probe_tokens=args.final_tokens,
        )
        
        # Generate with Householder steering
        with HouseholderSteerer(bundle.model, vector=dh, layer=target_layer, theta=args.householder_theta):
            text = _generate(bundle, inputs, max_new_tokens=args.final_tokens)
            
        result = {
            "method": "bplis",
            "layer": target_layer,
            "multiplier": args.multiplier,  # Not used in BPLIS but included for consistency
            "householder_theta": args.householder_theta,
            "text": text
        }
        
    elif dynamic_layers:
        # Dynamic layer selection using budgeted search
        config_kwargs = {
            "budget": args.budget,
            "probe_tokens": args.probe_tokens, 
            "final_tokens": args.final_tokens,
            "use_bplis": False,
            "multipliers": (args.multiplier,),  # Use only the specified multiplier
        }
        cfg = SearchConfig(**config_kwargs)
        result = budgeted_latent_activation_search(
            bundle,
            vectors_dir=args.vectors_dir,
            question=args.question,
            context=args.context,
            cfg=cfg,
        )
        result["method"] = "dynamic"
        
    else:
        # Fixed layer steering
        if args.layer is None:
            # No steering
            text = _generate(bundle, inputs, max_new_tokens=args.final_tokens)
            result = {
                "method": "none",
                "layer": None,
                "multiplier": 0.0,
                "text": text
            }
        else:
            # Fixed layer with specified multiplier
            v = load_vector(args.vectors_dir, args.layer)
            with ActivationSteerer(bundle.model, v, SteeringConfig(layer=args.layer, multiplier=args.multiplier)):
                text = _generate(bundle, inputs, max_new_tokens=args.final_tokens)
            result = {
                "method": "static",
                "layer": args.layer,
                "multiplier": args.multiplier,
                "text": text
            }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
