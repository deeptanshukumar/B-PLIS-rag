#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from contextfocus.data.nqswap import load_nqswap
from contextfocus.steering.vector_builder import VectorBuildConfig, build_contextfocus_vectors
from contextfocus.utils import load_hf_causal_lm, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_examples", type=int, default=1501)
    ap.add_argument("--split", type=str, default="dev")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    args = ap.parse_args()

    set_seed(7)
    bundle = load_hf_causal_lm(args.model_id, dtype=args.dtype)

    ds = load_nqswap(split=args.split)
    cfg = VectorBuildConfig(n_examples=args.n_examples, max_length=args.max_length)

    vectors_dir = build_contextfocus_vectors(bundle, ds, out_dir=args.out_dir, cfg=cfg)
    print(f"Saved vectors to: {vectors_dir}")


if __name__ == "__main__":
    main()
