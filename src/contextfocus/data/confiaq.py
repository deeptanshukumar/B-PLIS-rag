from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ConFiQAExample:
    id: str
    subset: str  # QA, MR, MC
    question: str
    context: str
    original_answer: str
    substituted_answer: str


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "data" in obj:
        obj = obj["data"]
    if not isinstance(obj, list):
        raise ValueError(f"Unexpected JSON format in {path}")
    return obj


def _iter_candidate_files(root: Path, subset: str, split: str) -> List[Path]:
    """
    ConFiQA may be stored in different layouts depending on the repo version.
    We search for common patterns.
    """
    subset = subset.upper()
    patterns = [
        f"**/{subset}/{split}.jsonl",
        f"**/{subset}/{split}.json",
        f"**/{subset}_{split}.jsonl",
        f"**/{subset}_{split}.json",
        f"**/{subset.lower()}/{split}.jsonl",
        f"**/{subset.lower()}/{split}.json",
        f"**/ConFiQA-{subset}.json",
        f"**/ConFiQA-{subset}.jsonl",
        f"ConFiQA-{subset}.json",
        f"ConFiQA-{subset}.jsonl",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(root.glob(pat))
    # Deduplicate
    uniq = []
    seen = set()
    for p in hits:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def load_confiaq(
    confiaq_root: str | Path,
    *,
    subset: str,
    split: str = "test",
    limit: Optional[int] = None,
) -> Iterable[ConFiQAExample]:
    """
    Loads ConFiQA subset (QA, MR, or MC).

    Expects to be pointed at the Context-DPO repo data directory or any extracted ConFiQA root.
    """
    root = Path(confiaq_root)
    if not root.exists():
        raise FileNotFoundError(f"ConFiQA root not found: {root}")

    files = _iter_candidate_files(root, subset=subset, split=split)
    if not files:
        raise FileNotFoundError(
            f"Could not find ConFiQA files for subset={subset}, split={split} under {root}. "
            f"Try setting CONFIQA_ROOT to the correct directory."
        )

    # Prefer jsonl
    files = sorted(files, key=lambda p: (p.suffix != ".jsonl", len(str(p))))
    path = files[0]

    if path.suffix == ".jsonl":
        rows = _read_jsonl(path)
    else:
        rows = _read_json(path)

    n = 0
    for i, row in enumerate(rows):
        ex = ConFiQAExample(
            id=str(row.get("id", i)),
            subset=subset.upper(),
            question=row.get("question") or row.get("query") or "",
            context=row.get("cf_context") or row.get("context") or row.get("passage") or row.get("retrieved_context") or "",
            original_answer=str(row.get("orig_answer") or row.get("original_answer") or row.get("answer") or ""),
            substituted_answer=str(row.get("cf_answer") or row.get("substituted_answer") or row.get("sub_answer") or row.get("counterfactual_answer") or ""),
        )
        yield ex
        n += 1
        if limit is not None and n >= limit:
            break
