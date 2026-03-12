#!/usr/bin/env python
from __future__ import annotations

import argparse

from contextfocus.data.nqswap import load_nqswap
from contextfocus.steering.layer_selector import LayerSelectConfig, select_best_layer
from contextfocus.utils import load_hf_causal_lm, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--vectors_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--multiplier", type=float, default=2.0)
    ap.add_argument("--split", type=str, default="dev")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    args = ap.parse_args()

    set_seed(7)
    bundle = load_hf_causal_lm(args.model_id, dtype=args.dtype)
    ds = load_nqswap(split=args.split)

    cfg = LayerSelectConfig(n_eval=args.n_eval, multiplier=args.multiplier)
    best = select_best_layer(bundle, ds, vectors_dir=args.vectors_dir, out_dir=args.out_dir, cfg=cfg)
    print("Best layer:", best)


if __name__ == "__main__":
    main()
