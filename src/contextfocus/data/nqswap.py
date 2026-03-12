from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from datasets import load_dataset


@dataclass(frozen=True)
class NQSwapExample:
    id: str
    question: str
    substituted_context: str
    substituted_answers: List[str]
    original_answers: List[str]


def load_nqswap(
    *,
    dataset_name: Optional[str] = None,
    split: str = "dev",
    streaming: bool = False,
) -> Iterable[NQSwapExample]:
    """
    Loads NQ-SWAP from HF datasets.
    The paper uses NQ-SWAP as a knowledge-conflict dataset for vector construction and layer selection.

    Default dataset candidates:
      - pminervini/NQ-Swap
      - younanna/NQ-Swap
    """
    name = dataset_name or os.environ.get("NQSWAP_DATASET") or "pminervini/NQ-Swap"
    ds = load_dataset(name, split=split, streaming=streaming)
    for row in ds:
        yield NQSwapExample(
            id=str(row.get("id", "")),
            question=row["question"],
            substituted_context=row.get("substituted_context") or row.get("sub_context") or row.get("context") or "",
            substituted_answers=list(row.get("substituted_answers") or row.get("sub_answer") or []),
            original_answers=list(row.get("original_answers") or row.get("org_answer") or row.get("orig_answer") or []),
        )
