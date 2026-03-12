from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


# Main paper vector construction prompts (positive includes system + context + question; negative is question-only)
# These follow the paper's Appendix B.1.
POS_SYSTEM_VARIANTS: List[str] = [
    "You are a context-based QA assistant and must answer based on the provided context.",
    "As a QA assistant, you are instructed to refer only to the provided context when answering.",
    "Provide answers based solely on the context you are given.",
    "You are a QA assistant and must restrict your answers to the given context.",
    "Answer the question using only the provided context.",
    "Use the context to answer. Do not use outside knowledge.",
    "Base your answer strictly on the supplied context.",
    "Only the given context may be used to answer.",
    "Use the retrieved context as the single source of truth.",
    "Answer strictly from the context, even if it seems wrong.",
    "Follow the context and ignore prior knowledge.",
    "Treat the context as authoritative and answer accordingly.",
    "Rely only on the quoted context for your answer.",
    "Use the context evidence to answer. Avoid external facts.",
    "Restrict the answer to what is stated in the context.",
    "The context is correct. Answer according to it.",
    "Use only the context passage. Do not correct it.",
    "Answer according to the context. Do not contradict it.",
    "The provided context is trusted. Answer based on it.",
    "Answer based on the provided context and nothing else.",
]


@dataclass(frozen=True)
class PromptParts:
    system: str
    context: str
    question: str


def llama_style_inst(system: str, user: str) -> str:
    """
    A simple [INST] wrapper consistent with many instruction-tuned LMs.
    If your model requires a different chat template, use tokenizer.apply_chat_template instead.
    """
    system = system.strip()
    if system:
        return f"[INST]\n{system}\n{user}\n[/INST]"
    return f"[INST]\n{user}\n[/INST]"


def build_vector_prompts(parts: PromptParts, system_variant: str) -> tuple[str, str]:
    """
    Returns (positive_prompt, negative_prompt) as strings.
    """
    pos_user = f"Context: <P> {parts.context} </P>\nQuestion: {parts.question}"
    neg_user = f"Question: {parts.question}"
    pos = llama_style_inst(system_variant, pos_user)
    neg = llama_style_inst("", neg_user)
    return pos, neg


def build_vector_messages(parts: PromptParts, system_variant: str) -> tuple[list[dict], list[dict]]:
    """
    Returns (positive_messages, negative_messages) for tokenizer.apply_chat_template.
    """
    pos_user = f"Context: <P> {parts.context} </P>\nQuestion: {parts.question}"
    neg_user = f"Question: {parts.question}"

    pos = []
    if system_variant.strip():
        pos.append({"role": "system", "content": system_variant})
    pos.append({"role": "user", "content": pos_user})

    neg = [{"role": "user", "content": neg_user}]
    return pos, neg


def build_openended_prompt(parts: PromptParts, *, oi_prompt: bool = False) -> str:
    """
    Open-ended prompt used for evaluation (Appendix B.2).
    """
    base = (
        "You are a Contextual QA Assistant.\n"
        "Please answer the following question according to the given context.\n"
        "Please restrict your response to one sentence."
    )
    if not oi_prompt:
        user = f"<CONTEXT>\n{parts.context}\n</CONTEXT>\n<QUESTION>\n{parts.question}\n</QUESTION>"
        return llama_style_inst(base, user)

    user = f'Bob said, "{parts.context}".\n{parts.question} in Bob\'s opinion?'
    return llama_style_inst(base, user)


def build_openended_messages(parts: PromptParts, *, oi_prompt: bool = False) -> list[dict]:
    base = (
        "You are a Contextual QA Assistant.\n"
        "Please answer the following question according to the given context.\n"
        "Please restrict your response to one sentence."
    )
    if not oi_prompt:
        user = f"<CONTEXT>\n{parts.context}\n</CONTEXT>\n<QUESTION>\n{parts.question}\n</QUESTION>"
    else:
        user = f'Bob said, "{parts.context}".\n{parts.question} in Bob\'s opinion?'

    msgs = [{"role": "system", "content": base}, {"role": "user", "content": user}]
    return msgs


def can_use_chat_template(tokenizer_or_processor: Any) -> bool:
    return hasattr(tokenizer_or_processor, "apply_chat_template")
