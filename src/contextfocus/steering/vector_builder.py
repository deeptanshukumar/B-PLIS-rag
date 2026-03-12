from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from tqdm import tqdm

from contextfocus.data.nqswap import NQSwapExample
from contextfocus.prompting.templates import (
    POS_SYSTEM_VARIANTS,
    PromptParts,
    build_vector_messages,
    build_vector_prompts,
    can_use_chat_template,
)
from contextfocus.utils import ModelBundle, get_model_hidden_size, get_transformer_blocks, tokenize_chat, tokenize_text


@dataclass(frozen=True)
class VectorBuildConfig:
    n_examples: int = 1501
    seed: int = 7
    max_length: int = 1024
    system_variants: int = 20  # use the first N variants


def _last_token_hidden_states(outputs, n_layers: int) -> List[torch.Tensor]:
    """
    Returns per-layer last-token hidden states as a list of tensors shape [d].
    We align "layer index" with transformer block index, so:
      - hidden_states[0] is embedding output (ignored)
      - hidden_states[i+1] is output after block i
    """
    hs = outputs.hidden_states
    if hs is None:
        raise ValueError("Model did not return hidden states. Set output_hidden_states=True.")

    out: List[torch.Tensor] = []
    for i in range(n_layers):
        h = hs[i + 1]  # [B, T, D]
        out.append(h[0, -1, :].detach().float().cpu())
    return out


@torch.no_grad()
def build_contextfocus_vectors(
    bundle: ModelBundle,
    examples: Iterable[NQSwapExample],
    out_dir: str | Path,
    *,
    cfg: VectorBuildConfig = VectorBuildConfig(),
) -> Path:
    """
    Build ContextFocus steering vectors for every layer:

      v_l = mean_i( h_l(pos_i) - h_l(neg_i) )

    Where pos has system instruction + context + question, and neg has question only.
    """
    out_dir = Path(out_dir)
    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    model = bundle.model
    tok = bundle.tokenizer

    blocks = get_transformer_blocks(model)
    n_layers = len(blocks)
    d_model = get_model_hidden_size(model)

    sums = [torch.zeros(d_model, dtype=torch.float32) for _ in range(n_layers)]
    count = 0

    variants = POS_SYSTEM_VARIANTS[: cfg.system_variants]
    use_chat = can_use_chat_template(tok)

    pbar = tqdm(examples, desc="Building vectors", total=cfg.n_examples)
    for ex in pbar:
        if count >= cfg.n_examples:
            break

        system = variants[count % len(variants)]
        parts = PromptParts(system=system, context=ex.substituted_context, question=ex.question)

        if use_chat:
            pos_msgs, neg_msgs = build_vector_messages(parts, system_variant=system)
            pos_in = tokenize_chat(tok, pos_msgs).to(model.device)
            neg_in = tokenize_chat(tok, neg_msgs).to(model.device)
        else:
            pos, neg = build_vector_prompts(parts, system_variant=system)
            pos_in = tokenize_text(tok, pos, max_length=cfg.max_length).to(model.device)
            neg_in = tokenize_text(tok, neg, max_length=cfg.max_length).to(model.device)

        pos_out = model(**pos_in, output_hidden_states=True, use_cache=False)
        neg_out = model(**neg_in, output_hidden_states=True, use_cache=False)

        pos_h = _last_token_hidden_states(pos_out, n_layers=n_layers)
        neg_h = _last_token_hidden_states(neg_out, n_layers=n_layers)

        for l in range(n_layers):
            sums[l] += (pos_h[l] - neg_h[l])

        count += 1

    if count == 0:
        raise ValueError("No examples were processed. Check dataset loading.")

    for l in range(n_layers):
        v = (sums[l] / count).contiguous()
        torch.save(v, vectors_dir / f"layer_{l:03d}.pt")

    meta = {
        "method": "ContextFocus vcomb",
        "model_id": getattr(model.config, "_name_or_path", None),
        "n_layers": n_layers,
        "n_examples": count,
        "system_variants_used": len(variants),
        "max_length": cfg.max_length,
        "used_chat_template": bool(use_chat),
        "note": "Vectors are saved as float32 CPU tensors, one per transformer block index.",
    }
    (out_dir / "vectors_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return vectors_dir
