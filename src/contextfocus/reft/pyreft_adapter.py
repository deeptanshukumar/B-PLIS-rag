from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import torch
import transformers

import pyreft

from contextfocus.data.nqswap import NQSwapExample
from contextfocus.prompting.templates import POS_SYSTEM_VARIANTS, PromptParts, build_openended_prompt, build_vector_prompts
from contextfocus.utils import get_model_hidden_size


@dataclass(frozen=True)
class ReFTConfig:
    layer: int = 15
    low_rank_dimension: int = 4
    component: str = "block_output"
    lr: float = 4e-3
    epochs: float = 5.0
    batch_size: int = 4
    output_dir: str = "artifacts/reft"


def build_reft_model(model: Any, *, cfg: ReFTConfig) -> Any:
    """
    Create a pyreft ReFT model that intervenes on the residual stream output ("block_output")
    at a single layer and last prompt position.

    This follows the official pyreft README pattern.
    """
    reft_config = pyreft.ReftConfig(
        representations={
            "layer": cfg.layer,
            "component": cfg.component,
            "low_rank_dimension": cfg.low_rank_dimension,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=get_model_hidden_size(model),
                low_rank_dimension=cfg.low_rank_dimension,
            ),
        }
    )
    reft_model = pyreft.get_reft_model(model, reft_config)
    return reft_model


def _build_supervised_pairs(ex: NQSwapExample, system_variant: str) -> tuple[str, str]:
    """
    Supervised pair for context faithfulness:
    - prompt includes substituted context + question
    - target is substituted answer (first variant)
    """
    parts = PromptParts(system="", context=ex.substituted_context, question=ex.question)
    prompt = build_openended_prompt(parts, oi_prompt=False)
    target = ex.substituted_answers[0] if ex.substituted_answers else ""
    return prompt, target


def train_reft_on_nqswap(
    model: Any,
    tokenizer: Any,
    examples: Iterable[NQSwapExample],
    *,
    cfg: ReFTConfig = ReFTConfig(),
    n_train: int = 256,
    seed: int = 7,
) -> Any:
    """
    Trains a ReFT intervention using pyreft's last-position supervised trainer.
    This is a baseline/novelty comparison, separate from activation steering.

    Note: For better reproduction, match the paper's prompt formatting and dataset splits.
    """
    torch.manual_seed(seed)

    reft_model = build_reft_model(model, cfg=cfg)
    reft_model.set_device(str(model.device))
    reft_model.print_trainable_parameters()

    prompts: List[str] = []
    targets: List[str] = []
    variants = POS_SYSTEM_VARIANTS

    for i, ex in enumerate(examples):
        if i >= n_train:
            break
        system_variant = variants[i % len(variants)]
        p, t = _build_supervised_pairs(ex, system_variant=system_variant)
        prompts.append(p)
        targets.append(t)

    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer, model, prompts, targets
    )

    args = transformers.TrainingArguments(
        num_train_epochs=cfg.epochs,
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        logging_steps=10,
        save_steps=200,
        report_to=[],
    )
    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model, tokenizer=tokenizer, args=args, **data_module
    )
    trainer.train()
    return reft_model
