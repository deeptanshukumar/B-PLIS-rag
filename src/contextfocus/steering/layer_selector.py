from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from tqdm import tqdm

from contextfocus.data.nqswap import NQSwapExample
from contextfocus.eval.metrics import score_batch
from contextfocus.prompting.templates import (
    PromptParts,
    build_openended_messages,
    build_openended_prompt,
    can_use_chat_template,
)
from contextfocus.steering.steerer import ActivationSteerer, SteeringConfig, load_vector
from contextfocus.utils import ModelBundle, decode, get_eos_id, get_transformer_blocks, tokenize_chat, tokenize_text


@dataclass(frozen=True)
class LayerSelectConfig:
    n_eval: int = 200
    max_new_tokens: int = 64
    multiplier: float = 2.0
    oi_prompt: bool = False
    seed: int = 7
    max_length: int = 1024


@torch.no_grad()
def _generate_one(bundle: ModelBundle, prompt_inputs, *, max_new_tokens: int) -> str:
    model = bundle.model
    tok = bundle.tokenizer
    # Important: Hugging Face `generate` returns the full sequence (prompt + completion).
    # For scoring we must decode only the completion, otherwise the prompt (which contains
    # the retrieved context) will trivially include the substituted answer.
    input_len = int(prompt_inputs["input_ids"].shape[-1])
    out = model.generate(
        **prompt_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=get_eos_id(tok),
    )
    return decode(tok, out[0, input_len:])


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
def select_best_layer(
    bundle: ModelBundle,
    examples: Iterable[NQSwapExample],
    *,
    vectors_dir: str | Path,
    out_dir: str | Path,
    cfg: LayerSelectConfig = LayerSelectConfig(),
) -> dict:
    model = bundle.model
    blocks = get_transformer_blocks(model)
    n_layers = len(blocks)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_ex = []
    for ex in examples:
        eval_ex.append(ex)
        if len(eval_ex) >= cfg.n_eval:
            break
    if not eval_ex:
        raise ValueError("No eval examples were loaded for layer selection.")

    # Baseline unsteered
    base_gens: List[str] = []
    for ex in tqdm(eval_ex, desc="Baseline (m=0)"):
        inputs = _encode_openended(bundle, context=ex.substituted_context, question=ex.question, oi_prompt=cfg.oi_prompt, max_length=cfg.max_length)
        base_gens.append(_generate_one(bundle, inputs, max_new_tokens=cfg.max_new_tokens))

    base_counts = score_batch(
        base_gens,
        original_answers=[(ex.original_answers[0] if ex.original_answers else "") for ex in eval_ex],
        substituted_answers=[(ex.substituted_answers[0] if ex.substituted_answers else "") for ex in eval_ex],
    )
    base_ps = base_counts.ps_rate

    best = {"layer": None, "ps": -1.0, "base_ps": base_ps}
    per_layer = []

    for layer in tqdm(range(n_layers), desc="Sweeping layers"):
        v = load_vector(vectors_dir, layer=layer)
        steer_cfg = SteeringConfig(layer=layer, multiplier=cfg.multiplier)
        gens: List[str] = []
        with ActivationSteerer(model, v, steer_cfg):
            for ex in eval_ex:
                inputs = _encode_openended(bundle, context=ex.substituted_context, question=ex.question, oi_prompt=cfg.oi_prompt, max_length=cfg.max_length)
                gens.append(_generate_one(bundle, inputs, max_new_tokens=cfg.max_new_tokens))

        counts = score_batch(
            gens,
            original_answers=[(ex.original_answers[0] if ex.original_answers else "") for ex in eval_ex],
            substituted_answers=[(ex.substituted_answers[0] if ex.substituted_answers else "") for ex in eval_ex],
        )
        ps = counts.ps_rate
        per_layer.append({"layer": layer, "ps": ps, "po": counts.po_rate, "mr": counts.mr})

        if ps > best["ps"]:
            best = {"layer": layer, "ps": ps, "po": counts.po_rate, "mr": counts.mr, "base_ps": base_ps}

    (out_dir / "layer_sweep.json").write_text(json.dumps(per_layer, indent=2), encoding="utf-8")
    (out_dir / "best_layer.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best
