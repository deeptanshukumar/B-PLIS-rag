#!/usr/bin/env python
"""Bar-chart comparison of ConFiQA evaluation results.

Compares three steering configurations side-by-side for each ConFiQA subset
(QA, MR, MC) across three metrics (ps_rate, po_rate, mr).

Configurations
--------------
1. Static Layer 14           — ``artifacts/confiaq_results_static.json``
2. Dynamic BCILS (old)       — ``artifacts/confiaq_results_olddyna.json``
3. Dynamic BCILS + CMA-ES    — ``artifacts/confiaq_results.json``
   (Discriminative layer filtering + Householder norm-preserving steering)

Usage
-----
    python scripts/plot_confiaq_results.py              # saves PNG + shows plot
    python scripts/plot_confiaq_results.py --no-show    # saves PNG only (CI)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------- matplotlib setup (Agg fallback for headless) ----------
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "matplotlib is required for this script.\n"
        "Install it with:  pip install matplotlib"
    )


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

RESULT_FILES = {
    "Static (Layer 14)": ARTIFACTS / "confiaq_results_static.json",
    "Dynamic BCILS\n(old alignment)": ARTIFACTS / "confiaq_results_olddyna.json",
    "Dynamic BCILS + CMA-ES\n(Discrim. + Householder)": ARTIFACTS / "confiaq_results.json",
}

SUBSETS = ["QA", "MR", "MC"]
METRICS = ["ps_rate", "po_rate", "mr"]
METRIC_LABELS = {
    "ps_rate": "PS Rate (↑ better)",
    "po_rate": "PO Rate (↓ better)",
    "mr": "Memory Rate (↓ better)",
}


def load_results() -> dict[str, dict[str, dict[str, float]]]:
    """Return ``{method: {subset: {metric: value}}}``."""
    data: dict[str, dict[str, dict[str, float]]] = {}
    for label, path in RESULT_FILES.items():
        if not path.exists():
            print(f"WARNING: {path} not found — skipping '{label}'")
            continue
        raw = path.read_text(encoding="utf-8")
        # confiaq_results.json starts with terminal output; find the JSON
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # File may have terminal preamble; find the first '{' line
            lines = raw.splitlines()
            json_start = next(
                (i for i, ln in enumerate(lines) if ln.strip().startswith("{")),
                None,
            )
            if json_start is None:
                print(f"WARNING: Could not parse JSON from {path}")
                continue
            obj = json.loads("\n".join(lines[json_start:]))
        subsets_data: dict[str, dict[str, float]] = {}
        for subset in SUBSETS:
            s = obj.get("subsets", {}).get(subset)
            if s is None:
                continue
            subsets_data[subset] = {m: s[m] for m in METRICS}
        data[label] = subsets_data
    return data


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

# Colour palette (colour-blind friendly)
COLORS = {
    "Static (Layer 14)": "#4C72B0",
    "Dynamic BCILS\n(old alignment)": "#DD8452",
    "Dynamic BCILS + CMA-ES\n(Discrim. + Householder)": "#55A868",
}


def plot(data: dict, save_path: Path, show: bool = True) -> None:
    methods = list(data.keys())
    n_methods = len(methods)
    n_subsets = len(SUBSETS)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    fig.suptitle(
        "ConFiQA Evaluation — Gemma 3 4B-IT (n=1500 per subset)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    bar_width = 0.22
    x = np.arange(n_subsets)

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        for m_idx, method in enumerate(methods):
            vals = [data[method].get(s, {}).get(metric, 0.0) for s in SUBSETS]
            offset = (m_idx - (n_methods - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                vals,
                bar_width,
                label=method,
                color=COLORS.get(method, f"C{m_idx}"),
                edgecolor="white",
                linewidth=0.5,
            )
            # Value labels on bars
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                )

        ax.set_xlabel("ConFiQA Subset", fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SUBSETS, fontsize=10)
        ax.set_ylim(0, max(
            max(data[m].get(s, {}).get(metric, 0) for s in SUBSETS for m in methods) * 1.25,
            0.05,
        ))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Single legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=n_methods,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved chart → {save_path}")
    if show:
        plt.show()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Save chart without opening a window (useful in CI / SSH)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(ARTIFACTS / "confiaq_comparison_chart.png"),
        help="Output file path (default: artifacts/confiaq_comparison_chart.png)",
    )
    args = ap.parse_args()

    data = load_results()
    if not data:
        raise SystemExit("No result files found. Run eval_confiaq.py first.")

    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot(data, save_path, show=not args.no_show)


if __name__ == "__main__":
    main()
