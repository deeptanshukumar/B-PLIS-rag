from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from contextfocus.prompting.templates import (
    PromptParts,
    build_openended_messages,
    build_openended_prompt,
    can_use_chat_template,
)
from contextfocus.utils import tokenize_chat, tokenize_text


@dataclass(frozen=True)
class ConflictDetectConfig:
    max_length: int = 1024
    js_threshold: float = 0.08  # tune on a held-out set
    use_js: bool = True


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(m.log(), p, reduction="sum") + F.kl_div(m.log(), q, reduction="sum"))


@torch.no_grad()
def detect_conflict(
    model: Any,
    tokenizer_or_processor: Any,
    *,
    question: str,
    context: str,
    oi_prompt: bool = False,
    cfg: ConflictDetectConfig = ConflictDetectConfig(),
) -> dict:
    """
    Two-pass probe:
      - context-in: system + context + question
      - context-out: question only

    We compute a divergence between next-token distributions. Low divergence means context does not
    change the model's immediate behavior and we can skip costly interventions.
    """
    use_chat = can_use_chat_template(tokenizer_or_processor)

    if use_chat:
        in_msgs = build_openended_messages(PromptParts(system="", context=context, question=question), oi_prompt=oi_prompt)
        out_msgs = build_openended_messages(PromptParts(system="", context="", question=question), oi_prompt=oi_prompt)
        in_toks = tokenize_chat(tokenizer_or_processor, in_msgs).to(model.device)
        out_toks = tokenize_chat(tokenizer_or_processor, out_msgs).to(model.device)
        in_prompt = str(in_msgs)
        out_prompt = str(out_msgs)
    else:
        in_prompt = build_openended_prompt(PromptParts(system="", context=context, question=question), oi_prompt=oi_prompt)
        out_prompt = build_openended_prompt(PromptParts(system="", context="", question=question), oi_prompt=oi_prompt)
        in_toks = tokenize_text(tokenizer_or_processor, in_prompt, max_length=cfg.max_length).to(model.device)
        out_toks = tokenize_text(tokenizer_or_processor, out_prompt, max_length=cfg.max_length).to(model.device)

    in_out = model(**in_toks, output_hidden_states=True, use_cache=False)
    out_out = model(**out_toks, output_hidden_states=True, use_cache=False)

    in_logits = in_out.logits[0, -1, :].float()
    out_logits = out_out.logits[0, -1, :].float()
    p = F.softmax(in_logits, dim=-1)
    q = F.softmax(out_logits, dim=-1)

    if cfg.use_js:
        div = float(_js_divergence(p, q).detach().cpu())
    else:
        div = float(F.kl_div(q.log(), p, reduction="sum").detach().cpu())

    return {
        "divergence": div,
        "is_conflict": div >= cfg.js_threshold,
        "in_hidden_states": in_out.hidden_states,
        "out_hidden_states": out_out.hidden_states,
        "in_prompt": in_prompt,
        "out_prompt": out_prompt,
    }
