"""
results/plot_paper_figures.py
─────────────────────────────
Reproduces Figure 2 (per-domain TLB JGA gain over SLM-only) and Figure 3
(cost–quality trade-off: LLM share vs. DST JGA) of the CADRE paper from the
result JSON files written by `python cadre.py eval`.

Expected inputs (one per configuration, default names produced by cadre.py):
    results/slm_only_204turns.json
    results/llm_only_204turns.json
    results/base_knn_router_204turns.json
    results/knn_router+domain_aware_204turns.json
    results/knn_router+semantic_exemplars_204turns.json
    results/mrl_classifier_204turns.json

Any file that is missing falls back to the numbers reported in the paper
(Tables 1–3), so the figures can be regenerated even before all runs exist;
such series are marked "(paper)" in the legend.

Usage:
    python results/plot_paper_figures.py [--results_dir results] [--turns 204]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ─── Paper-reported values (fallbacks) ────────────────────────────────────────

PAPER = {
    "slm_only": {
        "label": "SLM only", "tlb_jga": 72.06, "dst_jga": 25.49, "avg_slot_f1": 75.20,
        "llm_assignment_ratio": 0.0,
        "per_domain_tlb_jga": {"restaurant": 69.23, "train": 70.27, "attraction": 55.56, "hotel": 53.33, "taxi": 11.11},
    },
    "llm_only": {
        "label": "LLM only", "tlb_jga": 77.94, "dst_jga": 40.20, "avg_slot_f1": 73.03,
        "llm_assignment_ratio": 100.0,
        "per_domain_tlb_jga": {"restaurant": 80.77, "train": 73.17, "attraction": 61.54, "hotel": 62.22, "taxi": 55.56},
    },
    "base_knn_router": {
        "label": "Base KNN router", "tlb_jga": 75.00, "dst_jga": 38.24, "avg_slot_f1": 74.86,
        "llm_assignment_ratio": 35.3,
        "per_domain_tlb_jga": {"restaurant": 73.08, "train": 73.68, "attraction": 65.38, "hotel": 52.27, "taxi": 25.00},
    },
    "knn_router+domain_aware": {
        "label": "CADRE (full)", "tlb_jga": 79.41, "dst_jga": 48.04, "avg_slot_f1": 79.44,
        "llm_assignment_ratio": 40.2,
        "per_domain_tlb_jga": {"restaurant": 84.62, "train": 81.58, "attraction": 62.96, "hotel": 55.56, "taxi": 50.00},
    },
    "knn_router+semantic_exemplars": {
        "label": "+ Semantic ex.", "tlb_jga": 79.90, "dst_jga": 47.06, "avg_slot_f1": 80.77,
        "llm_assignment_ratio": 40.2,
        "per_domain_tlb_jga": None,
    },
    "mrl_classifier": {
        "label": "MRL classifier", "tlb_jga": 75.98, "dst_jga": 36.27, "avg_slot_f1": 74.57,
        "llm_assignment_ratio": 54.9,
        "per_domain_tlb_jga": None,
    },
}

DOMAIN_ORDER = ["taxi", "hotel", "attraction", "train", "restaurant"]
COLORS = {
    "slm_only": "#7f7f7f", "llm_only": "#d9534f", "base_knn_router": "#f0a500",
    "knn_router+domain_aware": "#3266ad", "knn_router+semantic_exemplars": "#5cb85c",
    "mrl_classifier": "#8e5ea2",
}


def load_results(results_dir: Path, turns: int) -> dict[str, dict]:
    """Loads each configuration's JSON if present, else the paper's numbers."""
    out = {}
    for key, fallback in PAPER.items():
        path = results_dir / f"{key}_{turns}turns.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            r["label"] = fallback["label"]
            r["source"] = "run"
            r.setdefault("per_domain_tlb_jga", r.get("per_domain_tlb_accuracy"))
            print(f"[figures] {key:32s} ← {path}")
        else:
            r = dict(fallback)
            r["source"] = "paper"
            print(f"[figures] {key:32s} ← paper values (no {path.name})")
        out[key] = r
    return out


def _label(r: dict) -> str:
    return r["label"] + (" (paper)" if r["source"] == "paper" else "")


# ─── Figure 2: per-domain gain over SLM-only ─────────────────────────────────

def plot_figure2(res: dict[str, dict], out: Path):
    base = res["slm_only"]["per_domain_tlb_jga"]
    series = ["llm_only", "base_knn_router", "knn_router+domain_aware"]
    series = [s for s in series if res[s].get("per_domain_tlb_jga")]

    x = np.arange(len(DOMAIN_ORDER))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, key in enumerate(series):
        pd = res[key]["per_domain_tlb_jga"]
        gains = [pd.get(d, np.nan) - base.get(d, np.nan) for d in DOMAIN_ORDER]
        ax.bar(x + (i - (len(series) - 1) / 2) * width, gains, width,
               label=_label(res[key]), color=COLORS[key])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_ORDER)
    ax.set_ylabel("Δ TLB JGA over SLM-only (pp)")
    ax.set_title("Figure 2: Per-domain TLB JGA gain over the SLM-only baseline")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[figures] Saved → {out}")


# ─── Figure 3: cost–quality trade-off ────────────────────────────────────────

def plot_figure3(res: dict[str, dict], out: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key, r in res.items():
        x, y = r["llm_assignment_ratio"], r["dst_jga"]
        ax.scatter(x, y, s=140 if key == "knn_router+domain_aware" else 80,
                   color=COLORS[key], edgecolor="black", zorder=3,
                   marker="*" if key == "knn_router+domain_aware" else "o")
        ax.annotate(_label(r), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    # Empirical Pareto front: no other point has ≤ LLM share and ≥ DST JGA.
    pts = sorted((r["llm_assignment_ratio"], r["dst_jga"], k) for k, r in res.items())
    front, best = [], -1.0
    for x, y, k in pts:
        if y > best:
            front.append((x, y))
            best = y
    if len(front) > 1:
        ax.plot(*zip(*front), linestyle="--", color="gray", linewidth=1, label="Pareto front", zorder=2)
        ax.legend(loc="lower right")

    ax.set_xlabel("LLM share of traffic (%)  — proxy for cost / latency")
    ax.set_ylabel("DST JGA (%)")
    ax.set_xlim(-5, 105)
    ax.set_title("Figure 3: Cost–quality trade-off across configurations")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[figures] Saved → {out}")


# ─── Table 1 / 3 as markdown ─────────────────────────────────────────────────

def print_tables(res: dict[str, dict]):
    print("\n| Configuration | TLB JGA | DST JGA | Slot F1 | % SLM | % LLM |")
    print("|---|---|---|---|---|---|")
    for key, r in res.items():
        llm = r["llm_assignment_ratio"]
        print(f"| {_label(r)} | {r['tlb_jga']:.2f} | {r['dst_jga']:.2f} | "
              f"{r.get('avg_slot_f1', float('nan')):.2f} | {100 - llm:.1f} | {llm:.1f} |")


def main():
    parser = argparse.ArgumentParser(description="Plot CADRE paper Figures 2 and 3")
    parser.add_argument("--results_dir", default=str(Path(__file__).parent))
    parser.add_argument("--turns", type=int, default=204)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    res = load_results(results_dir, args.turns)
    plot_figure2(res, results_dir / f"figure2_per_domain_gain_{args.turns}turns.png")
    plot_figure3(res, results_dir / f"figure3_cost_quality_{args.turns}turns.png")
    print_tables(res)


if __name__ == "__main__":
    main()
