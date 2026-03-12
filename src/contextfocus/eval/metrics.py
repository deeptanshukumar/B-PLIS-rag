from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


_NEGATIONS = [
    "not",
    "no",
    "never",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "didn't",
    "didnt",
]


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_negated_span(text_norm: str, answer_norm: str, window: int = 3) -> bool:
    """
    Heuristic: if a negation token appears within a small window before the answer span, treat as negated.
    This approximates the "exclude explicit negations" rule described in the paper.
    """
    tokens = text_norm.split()
    ans_tokens = answer_norm.split()
    if not ans_tokens:
        return False

    for i in range(0, len(tokens) - len(ans_tokens) + 1):
        if tokens[i : i + len(ans_tokens)] == ans_tokens:
            start = max(0, i - window)
            prefix = tokens[start:i]
            if any(t in _NEGATIONS for t in prefix):
                return True
    return False


def contains_answer(
    generated: str,
    answer: str,
    *,
    exclude_negated: bool = False,
) -> bool:
    if not answer:
        return False
    g = normalize(generated)
    a = normalize(answer)
    if a and a in g:
        if exclude_negated and _has_negated_span(g, a):
            return False
        return True
    return False


@dataclass
class FaithfulnessCounts:
    n: int
    ps: int
    po: int

    @property
    def ps_rate(self) -> float:
        return self.ps / self.n if self.n else 0.0

    @property
    def po_rate(self) -> float:
        return self.po / self.n if self.n else 0.0

    @property
    def mr(self) -> float:
        denom = (self.po + self.ps)
        return self.po / denom if denom else 0.0


def score_batch(
    generations: Iterable[str],
    original_answers: Iterable[str],
    substituted_answers: Iterable[str],
    *,
    exclude_negated_ps: bool = True,
) -> FaithfulnessCounts:
    n = 0
    ps = 0
    po = 0
    for g, o, s in zip(generations, original_answers, substituted_answers):
        n += 1
        if contains_answer(g, s, exclude_negated=exclude_negated_ps):
            ps += 1
        if contains_answer(g, o, exclude_negated=False):
            po += 1
    return FaithfulnessCounts(n=n, ps=ps, po=po)
