#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextfocus.data.confiaq import load_confiaq
from contextfocus.eval.evaluator import EvalConfig, evaluate_confiaq
from contextfocus.utils import load_hf_causal_lm, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--confiaq_root", type=str, required=True)
    ap.add_argument("--vectors_dir", type=str, required=True)
    ap.add_argument("--layer_json", type=str, default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--multiplier", type=float, default=2.0)
    ap.add_argument("--dynamic_layers", type=str, default="false")
    ap.add_argument(
        "--layer_sweep_path",
        type=str,
        default=None,
        help="Optional path to layer_sweep.json (used as a prior for causal dynamic selection)",
    )
    ap.add_argument("--n_per_subset", type=int, default=1500)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    # ---- B-PLIS options ----
    ap.add_argument("--use_bplis", type=str, default="false", help="Enable B-PLIS latent vector search")
    ap.add_argument("--bplis_generations", type=int, default=10, help="B-PLIS CMA-ES generations")
    ap.add_argument("--bplis_popsize", type=int, default=8, help="B-PLIS CMA-ES population size")
    ap.add_argument("--bplis_intrinsic_dim", type=int, default=64, help="B-PLIS intrinsic dimension")
    ap.add_argument("--householder_theta", type=float, default=0.15, help="Householder rotation angle (radians)")
    args = ap.parse_args()

    set_seed(7)
    bundle = load_hf_causal_lm(args.model_id, dtype=args.dtype)

    layer = args.layer
    if layer is None and args.layer_json:
        best = json.loads(Path(args.layer_json).read_text(encoding="utf-8"))
        layer = best.get("layer")

    dynamic_layers = args.dynamic_layers.lower() in ["1", "true", "yes", "y"]
    use_bplis = args.use_bplis.lower() in ["1", "true", "yes", "y"]

    sweep_path = args.layer_sweep_path
    if sweep_path is None and args.layer_json:
        candidate = Path(args.layer_json).with_name("layer_sweep.json")
        if candidate.exists():
            sweep_path = str(candidate)
    
    cfg = EvalConfig(
        multiplier=args.multiplier,
        dynamic_layers=dynamic_layers,
        layer_sweep_path=sweep_path,
        use_bplis=use_bplis,
        bplis_generations=args.bplis_generations,
        bplis_popsize=args.bplis_popsize,
        bplis_intrinsic_dim=args.bplis_intrinsic_dim,
        householder_theta=args.householder_theta,
    )

    out = {
        "model_id": args.model_id,
        "dynamic_layers": dynamic_layers,
        "use_bplis": use_bplis,
        "layer": layer,
        "multiplier": args.multiplier,
        "subsets": {},
    }

    for subset in ["QA", "MR", "MC"]:
        ds = load_confiaq(args.confiaq_root, subset=subset, split="test", limit=args.n_per_subset)
        res = evaluate_confiaq(bundle, ds, vectors_dir=args.vectors_dir, layer=layer, cfg=cfg)
        out["subsets"][subset] = res
        print(subset, res)

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/confiaq_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved artifacts/confiaq_results.json")


if __name__ == "__main__":
    main()
