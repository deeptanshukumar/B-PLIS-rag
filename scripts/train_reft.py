#!/usr/bin/env python
from __future__ import annotations

import argparse

from contextfocus.data.nqswap import load_nqswap
from contextfocus.reft.pyreft_adapter import ReFTConfig, train_reft_on_nqswap
from contextfocus.utils import load_hf_causal_lm, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--output_dir", type=str, default="artifacts/reft")
    ap.add_argument("--split", type=str, default="dev")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    args = ap.parse_args()

    set_seed(7)
    bundle = load_hf_causal_lm(args.model_id, dtype=args.dtype)

    ds = load_nqswap(split=args.split)
    cfg = ReFTConfig(layer=args.layer, low_rank_dimension=args.rank, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, output_dir=args.output_dir)
    _ = train_reft_on_nqswap(bundle.model, bundle.tokenizer, ds, cfg=cfg, n_train=args.n_train)
    print("ReFT training complete.")


if __name__ == "__main__":
    main()
