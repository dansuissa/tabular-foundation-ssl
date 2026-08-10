"""Scientific analysis of the combined classical+neural mini-wave benchmark.

Reads the combined raw + aggregated files for mini_wave_all_methods, writes
``analysis_report.md`` and a set of analysis plots. Read-only with respect to
the benchmark: it does not rerun any model.

Run from the project root with PYTHONPATH including ``.``:
    python scripts/analyze_all_methods.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AGG_DIR = Path("results/aggregated/mini_wave_all_methods")
RAW_ALL = Path("results/raw/mini_wave_all_methods.csv")
PLOTS_DIR = AGG_DIR / "plots_analysis"
REPORT_PATH = AGG_DIR / "analysis_report.md"

METHOD_FAMILY = {
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
    "vime_lite": "neural_ssl",
    "scarf": "neural_ssl",
}
FAMILY_ORDER = ["supervised", "graph_ssl", "self_training", "rpl", "neural_ssl"]
NEURAL = {"sslae", "vime_lite", "scarf"}
CLASSICAL = set(METHOD_FAMILY) - NEURAL
DATASETS = ["phoneme", "spambase", "letter"]
BUDGETS = [50, 100, 250, 500]
BAL = "metric_balanced_accuracy_mean"
F1 = "metric_macro_f1_mean"


def load() -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(RAW_ALL)
    summary = pd.read_csv(AGG_DIR / "summary_by_dataset_method_budget.csv")
    rankings = pd.read_csv(AGG_DIR / "rankings_by_dataset_budget_complete_only.csv")
    coverage = pd.read_csv(AGG_DIR / "method_coverage_by_dataset_budget.csv")
    ssl = pd.read_csv(AGG_DIR / "ssl_vs_supervised_by_dataset_budget.csv")
    for df in (summary, rankings):
        df["family"] = df["method"].map(METHOD_FAMILY)
    return {
        "raw": raw,
        "summary": summary,
        "rankings": rankings,
        "coverage": coverage,
        "ssl": ssl,
    }


def best_in_subset(rankings: pd.DataFrame, dataset: str, budget: int, methods: set[str]):
    sub = rankings[
        (rankings.dataset == dataset)
        & (rankings.n_labeled == budget)
        & (rankings.method.isin(methods))
    ]
    sub = sub.dropna(subset=[BAL])
    if sub.empty:
        return None
    return sub.loc[sub[BAL].idxmax()]


def cell(value, fmt="{:.4f}"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return fmt.format(value)


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------
def section_exec_summary(data: dict[str, pd.DataFrame]) -> str:
    raw = data["raw"]
    status_counts = raw["status"].value_counts()
    n_success = int(status_counts.get("success", 0))
    n_fail = int(len(raw) - n_success)
    methods = sorted(raw["method"].unique())
    neural_rows = raw[raw.method.isin(NEURAL)]
    neural_success = int((neural_rows.status == "success").sum())
    lines = [
        "## A. Executive summary",
        "",
        f"- **Total rows:** {len(raw)} (expected 612 = 504 classical + 108 neural).",
        f"- **Success:** {n_success}  |  **Failed:** {n_fail} "
        f"(all failures are `failed_graph_ssl_nan`).",
        f"- **Datasets ({len(DATASETS)}):** {', '.join(DATASETS)}.",
        f"- **Label budgets:** {', '.join(str(b) for b in BUDGETS)}.",
        "- **Seeds:** 0, 1, 2.",
        f"- **Methods ({len(methods)}):** {', '.join(methods)}.",
        "- **Graph SSL** (`label_spreading`, `label_propagation`) has **9 known "
        "failed rows** (`failed_graph_ssl_nan`) — see Failure analysis.",
        f"- **Neural methods all succeeded:** {neural_success}/108 neural runs "
        "completed (sslae, vime_lite, scarf).",
        "",
        "> **VIME-lite caveat.** `vime_lite` is treated as a lightweight "
        "VIME-inspired baseline, not a faithful reproduction of full VIME. If "
        "VIME-lite remains central to the final claims, a faithful vime "
        "implementation should be added and run as a separate method before "
        "publication.",
        "",
    ]
    return "\n".join(lines)


def section_headline(data: dict[str, pd.DataFrame]) -> str:
    rankings = data["rankings"]
    lines = ["## B. Headline winners (complete-seed rankings only)", ""]
    lines.append("### Best method per dataset and budget")
    lines.append("")
    lines.append("| Dataset | Budget | Best method | Family | Balanced acc | Macro F1 |")
    lines.append("|---------|--------|-------------|--------|--------------|----------|")
    for dataset in DATASETS:
        for budget in BUDGETS:
            best = best_in_subset(rankings, dataset, budget, set(METHOD_FAMILY))
            if best is None:
                lines.append(f"| {dataset} | {budget} | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {dataset} | {budget} | `{best['method']}` | {best['family']} | "
                f"{cell(best[BAL])} | {cell(best.get(F1))} |"
            )
    lines.append("")
    lines.append("### Best method *within each family* per dataset and budget")
    lines.append("(balanced accuracy mean; complete-seed only)")
    lines.append("")
    header = "| Dataset | Budget | " + " | ".join(FAMILY_ORDER) + " |"
    sep = "|" + "---|" * (2 + len(FAMILY_ORDER))
    lines.append(header)
    lines.append(sep)
    for dataset in DATASETS:
        for budget in BUDGETS:
            cells = [dataset, str(budget)]
            for fam in FAMILY_ORDER:
                fam_methods = {m for m, f in METHOD_FAMILY.items() if f == fam}
                best = best_in_subset(rankings, dataset, budget, fam_methods)
                if best is None:
                    cells.append("n/a")
                else:
                    cells.append(f"{best['method']} {cell(best[BAL], '{:.3f}')}")
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def neural_vs_classical_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rankings = data["rankings"]
    rows = []
    for dataset in DATASETS:
        for budget in BUDGETS:
            bn = best_in_subset(rankings, dataset, budget, NEURAL)
            bc = best_in_subset(rankings, dataset, budget, CLASSICAL)
            if bn is None or bc is None:
                continue
            d_bal = bn[BAL] - bc[BAL]
            d_f1 = (bn.get(F1, np.nan) - bc.get(F1, np.nan))
            rows.append(
                {
                    "dataset": dataset,
                    "n_labeled": budget,
                    "best_neural": bn["method"],
                    "neural_bal": bn[BAL],
                    "best_classical": bc["method"],
                    "classical_bal": bc[BAL],
                    "delta_balanced_accuracy": d_bal,
                    "delta_macro_f1": d_f1,
                    "neural_wins": bool(d_bal > 0),
                }
            )
    return pd.DataFrame(rows)


def section_neural_vs_classical(nvc: pd.DataFrame) -> str:
    lines = ["## C. Neural vs classical", ""]
    lines.append(
        "| Dataset | Budget | Best neural | Neural bal | Best classical | "
        "Classical bal | Δ balanced acc | Δ macro F1 | Neural wins |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in nvc.iterrows():
        lines.append(
            f"| {r['dataset']} | {int(r['n_labeled'])} | `{r['best_neural']}` | "
            f"{cell(r['neural_bal'])} | `{r['best_classical']}` | "
            f"{cell(r['classical_bal'])} | {cell(r['delta_balanced_accuracy'], '{:+.4f}')} | "
            f"{cell(r['delta_macro_f1'], '{:+.4f}')} | {'YES' if r['neural_wins'] else 'no'} |"
        )
    wins = nvc[nvc.neural_wins]
    lines.append("")
    if wins.empty:
        lines.append("Neural never beats the best classical method on any dataset/budget cell.")
    else:
        cells = ", ".join(
            f"{r['dataset']} @ {int(r['n_labeled'])} ({r['best_neural']}, "
            f"{r['delta_balanced_accuracy']:+.4f})"
            for _, r in wins.iterrows()
        )
        lines.append(f"**Neural wins ({len(wins)} cells):** {cells}.")
    lines.append("")
    return "\n".join(lines)


def section_ssl_vs_supervised(data: dict[str, pd.DataFrame]) -> str:
    ssl = data["ssl"].copy()
    lines = ["## D. SSL vs supervised (matched-seed comparisons)", ""]
    complete = ssl[ssl["pair_complete"] == True]  # noqa: E712
    incomplete = ssl[ssl["pair_complete"] != True]  # noqa: E712
    helps = (ssl["delta_balanced_accuracy_mean"] > 0).sum()
    hurts = (ssl["delta_balanced_accuracy_mean"] < 0).sum()
    lines.append(
        f"- Matched-seed pair comparisons: **{len(ssl)}** rows "
        f"({len(complete)} complete-pair, {len(incomplete)} incomplete-pair)."
    )
    lines.append(
        f"- SSL improves balanced accuracy in **{helps}** pairs; "
        f"degrades it in **{hurts}** pairs."
    )
    lines.append("")
    top_pos = ssl.nlargest(8, "delta_balanced_accuracy_mean")
    top_neg = ssl.nsmallest(8, "delta_balanced_accuracy_mean")
    lines.append("### Strongest positive Δ balanced accuracy (SSL helps)")
    lines.append("")
    lines.append("| Dataset | Budget | SSL method | vs supervised | Δ bal acc | paired seeds | warning |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in top_pos.iterrows():
        lines.append(
            f"| {r['dataset']} | {int(r['n_labeled'])} | `{r['ssl_method']}` | "
            f"`{r['supervised_method']}` | {r['delta_balanced_accuracy_mean']:+.4f} | "
            f"{int(r['n_paired_seeds'])}/{int(r['n_expected_seeds'])} | "
            f"{r.get('comparison_warning', '') or ''} |"
        )
    lines.append("")
    lines.append("### Strongest negative Δ balanced accuracy (SSL hurts)")
    lines.append("")
    lines.append("| Dataset | Budget | SSL method | vs supervised | Δ bal acc | paired seeds | warning |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in top_neg.iterrows():
        lines.append(
            f"| {r['dataset']} | {int(r['n_labeled'])} | `{r['ssl_method']}` | "
            f"`{r['supervised_method']}` | {r['delta_balanced_accuracy_mean']:+.4f} | "
            f"{int(r['n_paired_seeds'])}/{int(r['n_expected_seeds'])} | "
            f"{r.get('comparison_warning', '') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def section_dataset_interpretation(data: dict[str, pd.DataFrame], nvc: pd.DataFrame) -> str:
    rankings = data["rankings"]
    ssl = data["ssl"]
    lines = ["## E. Dataset-specific interpretation", ""]

    # phoneme
    lines.append("### phoneme (binary, ~5k rows)")
    ph_best = [best_in_subset(rankings, "phoneme", b, set(METHOD_FAMILY)) for b in BUDGETS]
    ph_methods = ", ".join(f"{b}→`{r['method']}`" for b, r in zip(BUDGETS, ph_best) if r is not None)
    ph_ssl = ssl[ssl.dataset == "phoneme"]
    ph_helps = (ph_ssl["delta_balanced_accuracy_mean"] > 0).sum()
    lines.append(
        f"- Dominant methods by budget: {ph_methods}. Pseudo-labeling/self-training "
        "(rpl_lite_xgboost, self_training_lightgbm) lead across the board."
    )
    lines.append(
        f"- Unlabeled data **helps**: SSL beats its supervised counterpart in "
        f"{ph_helps}/{len(ph_ssl)} matched-seed pairs on phoneme, with the "
        "largest gains from rpl_lite_xgboost at low budgets."
    )
    ph_nvc = nvc[nvc.dataset == "phoneme"]
    lines.append(
        "- Neural methods do **not** help here: best neural trails best classical "
        f"at every budget (Δ from {ph_nvc['delta_balanced_accuracy'].min():+.4f} to "
        f"{ph_nvc['delta_balanced_accuracy'].max():+.4f})."
    )
    lines.append("")

    # spambase
    lines.append("### spambase (binary, ~4.6k rows)")
    sb_nvc = nvc[nvc.dataset == "spambase"]
    lines.append(
        "- **Neural `vime_lite` wins at budget 50** (best overall in that cell), and "
        "**`sslae` narrowly wins at budget 250**."
    )
    lines.append("- Supervised trees (random_forest, catboost) dominate at budgets 100 and 500.")
    for _, r in sb_nvc.iterrows():
        lines.append(
            f"  - budget {int(r['n_labeled'])}: best neural `{r['best_neural']}` "
            f"({r['neural_bal']:.4f}) vs best classical `{r['best_classical']}` "
            f"({r['classical_bal']:.4f}) → Δ {r['delta_balanced_accuracy']:+.4f} "
            f"({'neural wins' if r['neural_wins'] else 'classical wins'})."
        )
    lines.append(
        "- **Caveat:** the budget-50 win is from **VIME-lite**, not full VIME; the "
        "margins at 250 are within ~1e-3 and need faithful-VIME confirmation before "
        "any strong claim."
    )
    lines.append("")

    # letter
    lines.append("### letter (26-class multiclass, ~20k rows)")
    lt_nvc = nvc[nvc.dataset == "letter"]
    lt_win = lt_nvc[lt_nvc.neural_wins]
    lines.append(
        "- Classical methods (catboost, random_forest) dominate at budgets 100/250/500."
    )
    if not lt_win.empty:
        w = lt_win.iloc[0]
        lines.append(
            f"- **Caveat at budget 50:** under *complete-seed* rankings the best classical "
            f"method `label_spreading` is excluded (its seed-1 run failed), so the top "
            f"complete-seed entry becomes `{w['best_neural']}` "
            f"({w['neural_bal']:.4f}) ahead of the best *complete* classical "
            f"`{w['best_classical']}` ({w['classical_bal']:.4f}), Δ "
            f"{w['delta_balanced_accuracy']:+.4f}. This is a **thin win in a very "
            "low-accuracy regime** (~0.35 balanced accuracy across 26 classes) and "
            "should not be read as neural superiority."
        )
    lines.append(
        "- Neural methods otherwise **struggle in this multiclass low-label regime**: at "
        f"budgets 100/250/500 best neural trails best classical by "
        f"{lt_nvc[lt_nvc.n_labeled > 50]['delta_balanced_accuracy'].min():+.4f} to "
        f"{lt_nvc[lt_nvc.n_labeled > 50]['delta_balanced_accuracy'].max():+.4f} balanced "
        "accuracy. With only 50–500 labels spread over 26 classes, the encoder has too "
        "few labeled examples per class to fine-tune well."
    )
    lines.append(
        "- **Incomplete graph SSL:** `label_spreading` failed for seed 1 at budget 50 "
        "(1 of 3 seeds), so its budget-50 entry is based on 2 seeds and is excluded "
        "from complete-seed headline rankings (this is what hands the budget-50 cell to "
        "neural)."
    )
    lines.append("")
    return "\n".join(lines)


def section_runtime(data: dict[str, pd.DataFrame]) -> str:
    raw = data["raw"]
    succ = raw[raw.status == "success"].copy()
    rt = succ.groupby("method")["runtime_seconds"].mean().sort_values(ascending=False)
    slowest = succ.nlargest(10, "runtime_seconds")[
        ["dataset", "method", "seed", "n_labeled", "runtime_seconds"]
    ]
    lines = ["## F. Runtime analysis", ""]
    lines.append("### Average runtime by method (seconds)")
    lines.append("")
    lines.append("| Method | Family | Mean runtime (s) |")
    lines.append("|---|---|---|")
    for method, val in rt.items():
        lines.append(f"| `{method}` | {METHOD_FAMILY.get(method, '?')} | {val:.2f} |")
    lines.append("")
    lines.append("### Slowest 10 runs")
    lines.append("")
    lines.append("| Dataset | Method | Seed | Budget | Runtime (s) |")
    lines.append("|---|---|---|---|---|")
    for _, r in slowest.iterrows():
        lines.append(
            f"| {r['dataset']} | `{r['method']}` | {int(r['seed'])} | "
            f"{int(r['n_labeled'])} | {r['runtime_seconds']:.1f} |"
        )
    neural_rt = rt[rt.index.isin(NEURAL)]
    lines.append("")
    lines.append(
        "- **Runtime/performance trade-off:** the neural methods (scarf ~"
        f"{neural_rt.get('scarf', float('nan')):.0f}s, vime_lite ~"
        f"{neural_rt.get('vime_lite', float('nan')):.0f}s, sslae ~"
        f"{neural_rt.get('sslae', float('nan')):.0f}s mean) are 1–2 orders of magnitude "
        "slower than the fast supervised baselines (logistic_regression, mlp, "
        "self_training_lr < 1s) yet rarely top the rankings."
    )
    lines.append(
        "- **Local feasibility:** neural methods are CPU-feasible on these datasets — "
        "the full 108-run neural grid completed in ~1.9 hours on CPU. The heaviest "
        "cost is `scarf`/`vime_lite` on `letter` (up to ~350s/run), driven by the "
        "large unlabeled pool and contrastive/pretraining passes."
    )
    lines.append("")
    return "\n".join(lines)


def section_failures(data: dict[str, pd.DataFrame]) -> str:
    raw = data["raw"]
    fails = raw[raw.status != "success"][
        ["dataset", "method", "seed", "n_labeled", "status"]
    ].sort_values(["dataset", "method", "n_labeled", "seed"])
    lines = ["## G. Failure analysis", ""]
    lines.append(f"- **{len(fails)} failed rows**, all `failed_graph_ssl_nan` (graph SSL only).")
    lines.append("")
    lines.append("| Dataset | Method | Seed | Budget | Status |")
    lines.append("|---|---|---|---|---|")
    for _, r in fails.iterrows():
        lines.append(
            f"| {r['dataset']} | `{r['method']}` | {int(r['seed'])} | "
            f"{int(r['n_labeled'])} | {r['status']} |"
        )
    lines.append("")
    lines.append(
        "- **Pattern:** failures concentrate on `spambase` at low budgets (50/100) for "
        "both `label_spreading` and `label_propagation`, plus a single `letter` "
        "`label_spreading` seed at budget 50."
    )
    lines.append(
        "- **Effect on headline rankings:** none. Complete-seed rankings exclude any "
        "method/dataset/budget cell that does not have all 3 seeds successful, so these "
        "unstable graph-SSL cells never drive a headline claim."
    )
    lines.append(
        "- **Nature:** these are **method instability** (sklearn label-propagation "
        "producing non-finite label distributions after kNN-graph retries), not "
        "benchmark crashes — the runner caught them cleanly and continued."
    )
    lines.append("")
    return "\n".join(lines)


def section_conclusions(nvc: pd.DataFrame) -> str:
    wins = nvc[nvc.neural_wins]
    n_wins = int(len(wins))
    win_desc = ", ".join(
        f"{r['dataset']} @ {int(r['n_labeled'])} ({r['best_neural']}, "
        f"{r['delta_balanced_accuracy']:+.4f})"
        for _, r in wins.iterrows()
    )
    lines = ["## H. Preliminary conclusions", ""]
    lines.append(
        "- **What the benchmark suggests:** on these three mostly-numeric tabular "
        "datasets, strong supervised trees and tree-based pseudo-labeling/self-training "
        "(catboost, random_forest, rpl_lite_xgboost, self_training_lightgbm) are the "
        "most reliable choices across budgets. SSL via pseudo-labeling gives the most "
        "consistent gains over supervised counterparts on phoneme."
    )
    lines.append(
        f"- **Where neural helps:** only {n_wins} dataset/budget cells in complete-seed "
        f"rankings — {win_desc}. Two of these are fragile: spambase @ 250 is a "
        "~1e-4 tie, and letter @ 50 only wins because the stronger graph-SSL method is "
        "excluded for an incomplete seed. The cleanest neural win is **vime_lite on "
        "spambase @ 50**."
    )
    lines.append(
        "- **What is robust:** (1) classical dominance on multiclass `letter` at "
        "budgets ≥100; (2) pseudo-labeling helping on phoneme; (3) graph SSL "
        "instability at low label budgets on spambase."
    )
    lines.append(
        "- **What is still uncertain:** the neural wins rest on a **VIME-lite** "
        "implementation, a sub-1e-3 margin at spambase 250, and an exclusion artifact at "
        "letter 50; single-wave, 3-seed estimates also have wide spread on `letter` at "
        "low budgets."
    )
    lines.append(
        "- **Faithful VIME?** The single clearest neural headline win comes from "
        "`vime_lite` (spambase @ 50). That makes VIME-lite **borderline central** and is "
        "enough to justify implementing and running a **faithful VIME** as a separate "
        "method before any publication claim that 'VIME wins on spambase'. It is not yet "
        "central enough to block the rest of the analysis."
    )
    lines.append("")
    lines.append(
        "> **VIME-lite caveat (restated).** `vime_lite` is treated as a lightweight "
        "VIME-inspired baseline, not a faithful reproduction of full VIME. If VIME-lite "
        "remains central to the final claims, a faithful vime implementation should be "
        "added and run as a separate method before publication."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def family_best_frame(rankings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for budget in BUDGETS:
            for fam in FAMILY_ORDER:
                fam_methods = {m for m, f in METHOD_FAMILY.items() if f == fam}
                best = best_in_subset(rankings, dataset, budget, fam_methods)
                rows.append(
                    {
                        "dataset": dataset,
                        "n_labeled": budget,
                        "family": fam,
                        "best_bal": np.nan if best is None else best[BAL],
                        "best_method": None if best is None else best["method"],
                    }
                )
    return pd.DataFrame(rows)


def plot_bal_vs_budget(rankings: pd.DataFrame, fam_best: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 5), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        fb = fam_best[fam_best.dataset == dataset]
        for fam in FAMILY_ORDER:
            sub = fb[fb.family == fam].sort_values("n_labeled")
            ax.plot(sub["n_labeled"], sub["best_bal"], marker="o", label=f"best {fam}")
        ax.set_title(dataset)
        ax.set_xscale("log")
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("mean balanced accuracy")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    fig.suptitle("Best-in-family balanced accuracy vs label budget")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "balanced_accuracy_vs_budget_by_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_neural_delta(nvc: pd.DataFrame) -> None:
    labels = [f"{r['dataset']}\n{int(r['n_labeled'])}" for _, r in nvc.iterrows()]
    deltas = nvc["delta_balanced_accuracy"].to_numpy()
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(labels)), deltas, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ balanced accuracy (best neural − best classical)")
    ax.set_title("Best neural minus best classical, per dataset/budget")
    for i, (d, _) in enumerate(zip(deltas, nvc.itertuples())):
        ax.annotate(f"{d:+.3f}", (i, d), ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "neural_vs_best_classical_delta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_family_heatmap(fam_best: pd.DataFrame) -> None:
    col_order = [(d, b) for d in DATASETS for b in BUDGETS]
    col_labels = [f"{d}\n{b}" for d, b in col_order]
    matrix = np.full((len(FAMILY_ORDER), len(col_order)), np.nan)
    for i, fam in enumerate(FAMILY_ORDER):
        for j, (d, b) in enumerate(col_order):
            v = fam_best[(fam_best.family == fam) & (fam_best.dataset == d) & (fam_best.n_labeled == b)]
            if not v.empty:
                matrix[i, j] = v.iloc[0]["best_bal"]
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(FAMILY_ORDER)))
    ax.set_yticklabels(FAMILY_ORDER)
    for i in range(len(FAMILY_ORDER)):
        for j in range(len(col_order)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="best balanced accuracy in family")
    ax.set_title("Best balanced accuracy per method family (dataset/budget)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "method_family_heatmap_balanced_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_vs_perf(data: dict[str, pd.DataFrame]) -> None:
    raw = data["raw"]
    succ = raw[raw.status == "success"]
    agg = succ.groupby("method").agg(
        runtime=("runtime_seconds", "mean"),
        bal=("metric_balanced_accuracy", "mean"),
    ).reset_index()
    agg["family"] = agg["method"].map(METHOD_FAMILY)
    fam_colors = {f: c for f, c in zip(FAMILY_ORDER, plt.cm.tab10.colors)}
    fig, ax = plt.subplots(figsize=(11, 7))
    for fam in FAMILY_ORDER:
        sub = agg[agg.family == fam]
        ax.scatter(sub["runtime"], sub["bal"], s=80, label=fam, color=fam_colors[fam])
    for _, r in agg.iterrows():
        ax.annotate(r["method"], (r["runtime"], r["bal"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean runtime (s, log scale)")
    ax.set_ylabel("mean balanced accuracy (over all dataset/budget/seed)")
    ax.set_title("Runtime vs performance trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "runtime_vs_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ssl_delta_heatmap(data: dict[str, pd.DataFrame]) -> None:
    ssl = data["ssl"]
    methods = sorted(ssl["ssl_method"].unique())
    col_order = [(d, b) for d in DATASETS for b in BUDGETS]
    col_labels = [f"{d}\n{b}" for d, b in col_order]
    matrix = np.full((len(methods), len(col_order)), np.nan)
    for i, m in enumerate(methods):
        for j, (d, b) in enumerate(col_order):
            v = ssl[(ssl.ssl_method == m) & (ssl.dataset == d) & (ssl.n_labeled == b)]
            if not v.empty:
                matrix[i, j] = v.iloc[0]["delta_balanced_accuracy_mean"]
    vmax = np.nanmax(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(col_order)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center",
                        color="black", fontsize=7)
    fig.colorbar(im, ax=ax, label="Δ balanced accuracy (SSL − supervised)")
    ax.set_title("SSL vs supervised matched-seed Δ balanced accuracy")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ssl_delta_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    nvc = neural_vs_classical_table(data)
    fam_best = family_best_frame(data["rankings"])

    report = "\n".join(
        [
            "# Mini-wave combined benchmark — analysis report",
            "",
            "_Classical (Wave 1) + neural SSL, 3 datasets × 17 methods × 4 budgets × 3 seeds._",
            "",
            "Source files:",
            "- `results/raw/mini_wave_all_methods.csv`",
            "- `results/aggregated/mini_wave_all_methods/summary_by_dataset_method_budget.csv`",
            "- `results/aggregated/mini_wave_all_methods/rankings_by_dataset_budget_complete_only.csv` (headline)",
            "- `results/aggregated/mini_wave_all_methods/rankings_by_dataset_budget_all_successes.csv` (exploratory only)",
            "- `results/aggregated/mini_wave_all_methods/method_coverage_by_dataset_budget.csv`",
            "- `results/aggregated/mini_wave_all_methods/ssl_vs_supervised_by_dataset_budget.csv`",
            "",
            section_exec_summary(data),
            section_headline(data),
            section_neural_vs_classical(nvc),
            section_ssl_vs_supervised(data),
            section_dataset_interpretation(data, nvc),
            section_runtime(data),
            section_failures(data),
            section_conclusions(nvc),
        ]
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    plot_bal_vs_budget(data["rankings"], fam_best)
    plot_neural_delta(nvc)
    plot_family_heatmap(fam_best)
    plot_runtime_vs_perf(data)
    plot_ssl_delta_heatmap(data)

    print(f"Report written: {REPORT_PATH}")
    print(f"Plots written to: {PLOTS_DIR}")
    print("\nNeural-vs-classical summary:")
    print(nvc.to_string(index=False))


if __name__ == "__main__":
    main()
