"""Generate presentation-ready figures for the supervisor brief.

Reads only existing result files (no model runs).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw/mini_wave_all_methods_plus_vime.csv"
AGG = ROOT / "results/aggregated/mini_wave_all_methods_plus_vime"
OUT = ROOT / "results/supervisor_brief/figures"
OUT.mkdir(parents=True, exist_ok=True)

BAL = "metric_balanced_accuracy_mean"
DATASETS = ["phoneme", "spambase", "letter"]
BUDGETS = [50, 100, 250, 500]

FAMILY = {
    "logistic_regression": "supervised",
    "random_forest": "supervised",
    "xgboost": "supervised",
    "lightgbm": "supervised",
    "catboost": "supervised",
    "mlp": "supervised",
    "label_spreading": "graph_ssl",
    "label_propagation": "graph_ssl",
    "self_training_lr": "self_training",
    "self_training_xgboost": "self_training",
    "self_training_lightgbm": "self_training",
    "self_training_catboost": "self_training",
    "rpl_lr": "rpl",
    "rpl_lite_xgboost": "rpl",
    "sslae": "neural_ssl",
    "scarf": "neural_ssl",
    "vime": "neural_ssl",
    "vime_lite": "neural_ssl",
}
NEURAL = {m for m, f in FAMILY.items() if f == "neural_ssl"}

FAMILY_COLORS = {
    "supervised": "#1f77b4",
    "graph_ssl": "#9467bd",
    "self_training": "#2ca02c",
    "rpl": "#ff7f0e",
    "neural_ssl": "#d62728",
}

METHOD_COLORS = {
    "logistic_regression": "#1f77b4",
    "random_forest": "#17becf",
    "xgboost": "#2ca02c",
    "lightgbm": "#bcbd22",
    "catboost": "#006400",
    "mlp": "#7f7f7f",
    "label_spreading": "#9467bd",
    "label_propagation": "#c5b0d5",
    "self_training_lr": "#ff7f0e",
    "self_training_xgboost": "#ffbb78",
    "self_training_lightgbm": "#8c564b",
    "self_training_catboost": "#e377c2",
    "rpl_lr": "#bf00ff",
    "rpl_lite_xgboost": "#aec7e8",
    "sslae": "#d62728",
    "scarf": "#000000",
    "vime": "#e60000",
    "vime_lite": "#ff9896",
}

# Manual label offsets for runtime plot (dx, dy in points) to reduce overlap.
RT_LABEL_OFFSETS = {
    "random_forest": (8, 6),
    "catboost": (8, -10),
    "xgboost": (8, 4),
    "rpl_lite_xgboost": (-72, 8),
    "self_training_xgboost": (8, -12),
    "self_training_catboost": (8, 6),
    "vime": (8, 6),
    "vime_lite": (-58, -10),
    "scarf": (8, 6),
    "sslae": (-42, 8),
    "self_training_lightgbm": (8, -14),
    "logistic_regression": (8, 6),
    "self_training_lr": (8, -10),
    "lightgbm": (8, 6),
    "rpl_lr": (8, 6),
    "label_spreading": (8, -10),
    "mlp": (8, 6),
    "label_propagation": (8, -12),
}

summary = pd.read_csv(AGG / "summary_by_dataset_method_budget.csv")
comp = summary[summary["is_complete"] == True].copy()  # noqa: E712
raw = pd.read_csv(RAW)


def cell(df, dataset, budget, methods=None):
    sub = df[(df.dataset == dataset) & (df.n_labeled == budget)]
    if methods is not None:
        sub = sub[sub.method.isin(methods)]
    return sub.dropna(subset=[BAL])


def best_in(df, dataset, budget, methods):
    sub = cell(df, dataset, budget, methods)
    if sub.empty:
        return None
    return sub.loc[sub[BAL].idxmax()]


def _collect_top_methods():
    """Methods that appear in top-3 at any budget, for consistent legend across panels."""
    keep = set()
    for ds in DATASETS:
        for b in BUDGETS:
            sub = cell(comp, ds, b).sort_values(BAL, ascending=False).head(3)
            keep.update(sub.method.tolist())
    return sorted(keep)


# ---------------------------------------------------------------- Figure A ---
def fig_best_methods():
    keep = _collect_top_methods()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), sharey=False)
    line_handles = []
    for m in keep:
        (h,) = axes[0].plot([], [], marker="o", linewidth=2.2, markersize=7,
                            color=METHOD_COLORS.get(m, "#333333"), label=m)
        line_handles.append(h)
    for ax, ds in zip(axes, DATASETS):
        for m in keep:
            xs, ys = [], []
            for b in BUDGETS:
                c = cell(comp, ds, b, [m])
                if not c.empty:
                    xs.append(b)
                    ys.append(float(c[BAL].iloc[0]))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=7,
                        color=METHOD_COLORS.get(m, "#333333"))
        ax.set_title(ds, fontsize=13, fontweight="bold")
        ax.set_xlabel("labeled budget")
        ax.set_xticks(BUDGETS)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("balanced accuracy (complete-seed mean)")
    fig.suptitle("A. Best-performing methods per dataset across label budgets",
                 fontsize=15, fontweight="bold", y=0.98)
    ncol = 4
    fig.legend(line_handles, keep, loc="lower center", ncol=ncol, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    fig.savefig(OUT / "best_methods_by_dataset_budget.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure B ---
def fig_family():
    families = ["supervised", "graph_ssl", "self_training", "rpl", "neural_ssl"]
    family_labels = {
        "supervised": "supervised",
        "graph_ssl": "graph SSL",
        "self_training": "self-training",
        "rpl": "RPL",
        "neural_ssl": "neural SSL",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    handles = []
    for fam in families:
        (h,) = axes[0].plot([], [], marker="o", linewidth=2.6, markersize=7,
                            color=FAMILY_COLORS[fam], label=family_labels[fam])
        handles.append(h)
    for ax, ds in zip(axes, DATASETS):
        for fam in families:
            members = [m for m, f in FAMILY.items() if f == fam]
            xs, ys = [], []
            for b in BUDGETS:
                c = cell(comp, ds, b, members)
                if not c.empty:
                    xs.append(b)
                    ys.append(float(c[BAL].max()))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2.6, markersize=7,
                        color=FAMILY_COLORS[fam])
        ax.set_title(ds, fontsize=13, fontweight="bold")
        ax.set_xlabel("labeled budget")
        ax.set_xticks(BUDGETS)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("best balanced accuracy in family")
    fig.suptitle("B. Best balanced accuracy per method family across budgets",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.legend(handles, [family_labels[f] for f in families], loc="lower center",
               ncol=5, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    fig.savefig(OUT / "method_family_vs_budget.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure C ---
def fig_neural_delta():
    classical = [m for m in FAMILY if m not in NEURAL]
    labels, deltas, colors = [], [], []
    ds_color = {"phoneme": "#1f77b4", "spambase": "#2ca02c", "letter": "#d62728"}
    rows = []
    for ds in DATASETS:
        for b in BUDGETS:
            bn = best_in(comp, ds, b, NEURAL)
            bc = best_in(comp, ds, b, classical)
            if bn is None or bc is None:
                continue
            d = float(bn[BAL] - bc[BAL])
            labels.append(f"{ds}\n@{b}")
            deltas.append(d)
            colors.append(ds_color[ds])
            rows.append((ds, b, bn.method, float(bn[BAL]), bc.method, float(bc[BAL]), d))
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    bars = ax.bar(range(len(deltas)), deltas, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Δ balanced accuracy")
    ax.set_title(
        "C. Best neural SSL minus best classical\n"
        "(including VIME-lite ablation; positive = neural wins)",
        fontsize=13, fontweight="bold",
    )
    for b, d in zip(bars, deltas):
        ax.text(
            b.get_x() + b.get_width() / 2,
            d + (0.004 if d >= 0 else -0.004),
            f"{d:+.3f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=8,
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=ds_color[d]) for d in DATASETS]
    ax.legend(handles, DATASETS, title="dataset", fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "neural_vs_classical_delta.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return rows


# ---------------------------------------------------------------- Figure D ---
def fig_runtime():
    succ = raw[raw.status == "success"].copy()
    g = succ.groupby("method").agg(
        rt=("runtime_seconds", "mean"),
        bal=("metric_balanced_accuracy", "mean"),
    ).reset_index()

    highlight = {
        "random_forest", "catboost", "vime", "vime_lite", "sslae", "scarf",
        "rpl_lite_xgboost", "self_training_lightgbm",
    }

    fig, ax = plt.subplots(figsize=(14, 8.5))
    for _, r in g.iterrows():
        m = r.method
        size = 160 if m in highlight else 90
        lw = 1.0 if m in highlight else 0.5
        ax.scatter(r.rt, r.bal, s=size, color=METHOD_COLORS.get(m, "#333"),
                   edgecolor="black", linewidth=lw, zorder=4 if m in highlight else 2)
        if m in highlight:
            dx, dy = RT_LABEL_OFFSETS.get(m, (6, 4))
            ax.annotate(
                m, (r.rt, r.bal), fontsize=9, fontweight="bold",
                xytext=(dx, dy), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
            )
    ax.set_xscale("log")
    ax.set_xlabel("avg runtime per run (s, log scale)")
    ax.set_ylabel("avg balanced accuracy (all datasets/budgets)")
    ax.set_title("D. Runtime vs. performance (averaged over the whole grid)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "runtime_vs_performance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return g.sort_values("bal", ascending=False)


# ---------------------------------------------------------------- Figure E ---
def fig_vime():
    vime_sp50 = float(cell(comp, "spambase", 50, ["vime"])[BAL].iloc[0])
    vl_sp50 = float(cell(comp, "spambase", 50, ["vime_lite"])[BAL].iloc[0])
    delta = vl_sp50 - vime_sp50

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    line_handles = []
    for m, color, lab in [
        ("vime", "#e60000", "vime (faithful-core)"),
        ("vime_lite", "#1f77b4", "vime_lite (ablation)"),
    ]:
        (h,) = axes[0].plot([], [], marker="o", linewidth=2.6, markersize=8, color=color, label=lab)
        line_handles.append(h)
    for ax, ds in zip(axes, DATASETS):
        for m, color in [("vime", "#e60000"), ("vime_lite", "#1f77b4")]:
            xs, ys = [], []
            for b in BUDGETS:
                c = cell(comp, ds, b, [m])
                if not c.empty:
                    xs.append(b)
                    ys.append(float(c[BAL].iloc[0]))
            ax.plot(xs, ys, marker="o", linewidth=2.6, markersize=8, color=color)
        ax.set_title(ds, fontsize=13, fontweight="bold")
        ax.set_xlabel("labeled budget")
        ax.set_xticks(BUDGETS)
        ax.grid(True, alpha=0.3)
        if ds == "spambase":
            ax.annotate(
                f"@50: lite − faithful = +{delta:.3f}\n(lite {vl_sp50:.3f}, vime {vime_sp50:.3f})",
                xy=(50, vl_sp50),
                xytext=(120, vl_sp50 + 0.04),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff8dc", ec="#888888"),
            )
    axes[0].set_ylabel("balanced accuracy")
    fig.suptitle(
        "E. Faithful-core VIME vs. VIME-lite ablation\n"
        "(ablation comparison — not two independent paper baselines)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.legend(line_handles, [h.get_label() for h in line_handles],
               loc="lower center", ncol=2, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(OUT / "vime_vs_vime_lite.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


fig_best_methods()
fig_family()
delta_rows = fig_neural_delta()
rt = fig_runtime()
fig_vime()

print("=== FIGURES WRITTEN ===")
for p in sorted(OUT.glob("*.png")):
    print(" ", p.name)
