#!/usr/bin/env python3
"""Create the single summary figure used in the three-page TFM/SSL brief."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/reports/bar_ilan_tfm_ssl_final/tables"
OUT = ROOT / "results/reports/bar_ilan_tfm_ssl_final/three_page_brief"
OUT.mkdir(parents=True, exist_ok=True)

DISPLAY = {
    "tabiclv2": "TabICLv2",
    "tabiclv2_self_training": "TabICLv2 + ST",
    "tabpfn3": "TabPFN-3",
    "tabpfn3_self_training": "TabPFN-3 + ST",
    "unlabeled_attention_ssl": "Unlabeled attention",
    "laplacian_ssl": "Laplacian SSL",
    "geometric_attention_ssl": "Combined model",
    "embedding_alignment_ssl": "Embedding alignment",
}

COLORS = {
    "tabiclv2": "#35618f",
    "tabiclv2_self_training": "#4f8a5b",
    "tabpfn3": "#2b78b8",
    "tabpfn3_self_training": "#2ca25f",
    "unlabeled_attention_ssl": "#ed8b2c",
    "laplacian_ssl": "#8064a2",
    "geometric_attention_ssl": "#c83e4d",
    "embedding_alignment_ssl": "#d76ab4",
}

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "savefig.dpi": 300,
    }
)


def main() -> None:
    ranks = pd.read_csv(TABLES / "new_method_failure_aware_ranking.csv")
    self_effects = pd.read_csv(TABLES / "self_training_paired_effects.csv")
    components = pd.read_csv(TABLES / "combined_vs_components_paired_effects.csv")

    fig = plt.figure(figsize=(12.4, 4.05))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.48)

    # A. Failure-aware mean rank across all requested methods.
    ax = fig.add_subplot(grid[0, 0])
    r = ranks.sort_values("mean_cell_rank", ascending=False)
    y = np.arange(len(r))
    ax.barh(y, r["mean_cell_rank"], color=[COLORS[m] for m in r["method"]])
    ax.set_yticks(y, [DISPLAY[m] for m in r["method"]])
    ax.set_xlabel("Mean rank across 40 dataset × budget cells\n(lower is better)")
    ax.set_title("A  Complete-grid ranking", loc="left", fontweight="bold")
    ax.set_xlim(0, 8)
    for i, value in enumerate(r["mean_cell_rank"]):
        ax.text(value + 0.10, i, f"{value:.2f}", va="center", fontsize=7)

    # B. Exact paired self-training effects by budget.
    ax = fig.add_subplot(grid[0, 1])
    for method, marker, offset in [
        ("tabpfn3_self_training", "o", -4),
        ("tabiclv2_self_training", "s", 4),
    ]:
        d = self_effects[
            (self_effects["method"] == method)
            & self_effects["scope"].str.startswith("budget_")
        ].copy()
        d["budget"] = d["scope"].str.replace("budget_", "", regex=False).astype(int)
        d = d.sort_values("budget")
        yerr = np.vstack(
            [
                d["delta_mean"] - d["delta_ci95_low"],
                d["delta_ci95_high"] - d["delta_mean"],
            ]
        )
        ax.errorbar(
            d["budget"] + offset,
            d["delta_mean"],
            yerr=yerr,
            marker=marker,
            capsize=3,
            linewidth=1.7,
            markersize=5,
            color=COLORS[method],
            label=DISPLAY[method],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([50, 100, 250, 500])
    ax.set_xlabel("Label budget")
    ax.set_ylabel("Δ balanced accuracy\n(self-training − frozen)")
    ax.set_title("B  Matched self-training effect", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.3, loc="upper right")

    # C. Combined model versus each independently trained component.
    ax = fig.add_subplot(grid[0, 2])
    order = ["unlabeled_attention_ssl", "laplacian_ssl", "embedding_alignment_ssl"]
    c = components.set_index("component_method").loc[order].reset_index()
    y = np.arange(len(c))
    xerr = np.vstack(
        [
            c["delta_mean"] - c["delta_ci95_low"],
            c["delta_ci95_high"] - c["delta_mean"],
        ]
    )
    ax.errorbar(
        c["delta_mean"],
        y,
        xerr=xerr,
        fmt="o",
        color="#333333",
        ecolor="#666666",
        capsize=4,
        markersize=5,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    short_components = {
        "unlabeled_attention_ssl": "Attention",
        "laplacian_ssl": "Laplacian",
        "embedding_alignment_ssl": "Alignment",
    }
    ax.set_yticks(y, [short_components[m] for m in c["component_method"]])
    ax.set_xlabel("Paired Δ balanced accuracy\n(combined − component)")
    ax.set_title("C  Does combination help?", loc="left", fontweight="bold")
    ax.set_xlim(-0.10, 0.055)

    fig.suptitle(
        "Foundation-model ranking and the incremental value of unlabeled-data mechanisms",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(OUT / "summary_figure.pdf", bbox_inches="tight")
    fig.savefig(OUT / "summary_figure.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
