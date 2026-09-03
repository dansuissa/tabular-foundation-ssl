"""Build the historical assets for the canonical main report.

Read-only w.r.t. results/raw and results/aggregated. Writes ONLY into
results/reports/main_report/ (figures and machine-readable summaries).

Primary metric: balanced accuracy (robust to class imbalance). Failures are
excluded from metric means (they carry no metric); rankings are computed per
(dataset, budget) cell over methods that produced >=1 successful seed, so methods
with recorded graph-SSL failures are neither rewarded nor hard-penalized.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RAW = Path("results/raw/low_class_wave_paper_methods.csv")
OUT = Path("results/reports/main_report")
TABLES = OUT / "tables" / "historical"
FIG = OUT / "figures" / "historical"
TABLES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PRIMARY = "metric_balanced_accuracy"

FAMILIES = {
    "logistic_regression": "Supervised",
    "mlp": "Supervised",
    "random_forest": "Supervised",
    "xgboost": "Supervised",
    "lightgbm": "Supervised",
    "catboost": "Supervised",
    "label_propagation": "Graph SSL",
    "label_spreading": "Graph SSL",
    "self_training_lr": "Self-training",
    "self_training_xgboost": "Self-training",
    "self_training_lightgbm": "Self-training",
    "self_training_catboost": "Self-training",
    "rpl_lr": "RPL/pseudo-label",
    "rpl_lite_xgboost": "RPL/pseudo-label",
    "vime": "Neural SSL",
    "scarf": "Neural SSL",
    "sslae": "Neural SSL",
}
FAMILY_ORDER = ["Supervised", "Graph SSL", "Self-training", "RPL/pseudo-label", "Neural SSL"]
FAMILY_COLORS = {
    "Supervised": "#4C72B0",
    "Graph SSL": "#DD8452",
    "Self-training": "#55A868",
    "RPL/pseudo-label": "#C44E52",
    "Neural SSL": "#8172B3",
}
BUDGETS = [50, 100, 250, 500]
CANONICAL_FIGURES = {
    "fig1_overall_ranking",
    "fig2_method_dataset_heatmap",
    "fig3_budget_curves_by_family",
    "fig4b_ssl_delta_heatmap",
    "fig5_best_method_tiles",
    "fig6b_perf_vs_runtime",
    "fig7_failures_heatmap",
}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
})


def save(fig, name):
    if name in CANONICAL_FIGURES:
        fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(RAW)
    report = {}

    # ---------------- Integrity verification ----------------
    datasets = sorted(df["dataset"].unique())
    methods = sorted(df["method"].unique())
    dup = int(df.duplicated(subset=["dataset", "method", "seed", "n_labeled"]).sum())
    per_ds = df["dataset"].value_counts().to_dict()
    integrity = {
        "row_count": int(len(df)),
        "row_count_ok": len(df) == 2040,
        "n_datasets": len(datasets),
        "datasets_ok": len(datasets) == 10,
        "n_methods": len(methods),
        "methods_ok": len(methods) == 17,
        "no_letter": "letter" not in datasets,
        "no_vime_lite": "vime_lite" not in methods,
        "rows_per_dataset": per_ds,
        "rows_per_dataset_ok": all(v == 204 for v in per_ds.values()),
        "duplicate_keys": dup,
        "duplicate_keys_ok": dup == 0,
        "status_counts": df["status"].value_counts().to_dict(),
    }
    report["integrity"] = integrity

    ok = df[df["status"] == "success"].copy()
    ok["family"] = ok["method"].map(FAMILIES)

    # ---------------- Dataset table ----------------
    ds_rows = []
    for d, g in df.groupby("dataset"):
        r0 = g.iloc[0]
        # total available rows ~ largest (train_labeled + unlabeled + val + test) proxy
        n_classes = int(r0["n_classes"])
        feats = int(r0["n_features_before_preprocessing"])
        feats_after = int(r0["n_features_after_preprocessing"])
        # approximate full size using max seen sums
        approx_n = int((g["n_unlabeled"] + g["train_labeled_size"] + g["val_size"] + g["test_size"]).max())
        ds_rows.append({
            "dataset": d,
            "dataset_id": int(r0["dataset_id"]),
            "approx_rows_used": approx_n,
            "n_features_raw": feats,
            "n_features_processed": feats_after,
            "n_classes": n_classes,
            "task": "binary" if n_classes == 2 else "multiclass",
            "test_size": int(r0["test_size"]),
        })
    ds_table = pd.DataFrame(ds_rows).sort_values("approx_rows_used").reset_index(drop=True)
    # size bucket
    def bucket(n):
        if n < 3000:
            return "small"
        if n < 15000:
            return "medium"
        return "large"
    ds_table["size_bucket"] = ds_table["approx_rows_used"].map(bucket)
    ds_table.to_csv(TABLES / "dataset_table.csv", index=False)
    report["dataset_table"] = ds_table.to_dict(orient="records")

    # ---------------- Per (dataset, method, budget) mean primary metric ----------------
    cell = (ok.groupby(["dataset", "n_labeled", "method"])[PRIMARY]
            .agg(["mean", "std", "count"]).reset_index())

    # ---------------- Failure-aware overall ranking (per-cell rank) ----------------
    # Within each (dataset, budget), rank methods by mean primary metric (1=best).
    ranks = []
    for (d, b), g in cell.groupby(["dataset", "n_labeled"]):
        g = g.copy()
        g["rank"] = g["mean"].rank(ascending=False, method="average")
        ranks.append(g)
    ranked = pd.concat(ranks)
    n_cells = ranked.groupby("method").size()
    mean_rank = ranked.groupby("method")["rank"].mean().sort_values()
    overall = pd.DataFrame({
        "method": mean_rank.index,
        "mean_rank": mean_rank.values,
        "n_cells_ranked": n_cells.reindex(mean_rank.index).values,
    })
    overall["family"] = overall["method"].map(FAMILIES)
    # overall mean primary metric over successful runs (context only)
    ov_metric = ok.groupby("method")[PRIMARY].mean()
    overall["mean_balanced_acc_success"] = overall["method"].map(ov_metric).values
    overall = overall.reset_index(drop=True)
    overall.to_csv(TABLES / "overall_method_ranking.csv", index=False)
    report["overall_ranking"] = overall.to_dict(orient="records")

    # complete-case count: cells where all 17 methods have >=1 success
    cell_counts = cell.groupby(["dataset", "n_labeled"])['method'].nunique()
    report["n_complete_cells_all17"] = int((cell_counts == 17).sum())
    report["n_total_cells"] = int(len(cell_counts))

    # ---------------- FIGURE 1: overall ranking (failure-aware) ----------------
    fig, ax = plt.subplots(figsize=(10, 8))
    o = overall.sort_values("mean_rank", ascending=True)
    colors = [FAMILY_COLORS[f] for f in o["family"]]
    # variability: std of per-cell rank
    rank_std = ranked.groupby("method")["rank"].std().reindex(o["method"]).values
    ax.barh(o["method"], o["mean_rank"], color=colors, xerr=rank_std,
            error_kw={"elinewidth": 1, "alpha": 0.4, "capsize": 3})
    ax.invert_yaxis()
    ax.set_xlabel("Mean rank across (dataset x budget) cells  (1 = best; lower is better)")
    ax.set_title("Overall method ranking — balanced accuracy\n(failure-aware: ranked per cell over successful seeds)")
    for i, (m, v, nc) in enumerate(zip(o["method"], o["mean_rank"], o["n_cells_ranked"])):
        ax.text(v + 0.1, i, f"{v:.1f}" + ("" if nc == 40 else f"  ({nc}/40)"),
                va="center", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILY_ORDER]
    ax.legend(handles, FAMILY_ORDER, title="Family", loc="lower right", fontsize=10)
    ax.set_xlim(0, max(o["mean_rank"]) + 3)
    save(fig, "fig1_overall_ranking")

    # ---------------- FIGURE 2: method x dataset heatmap (mean over budgets) ----------------
    hm = (ok.groupby(["method", "dataset"])[PRIMARY].mean()
          .unstack("dataset"))
    # order methods by family then mean rank
    m_order = (overall.assign(fam_ord=overall["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}))
               .sort_values(["fam_ord", "mean_rank"])['method'].tolist())
    hm = hm.reindex(index=m_order, columns=ds_table["dataset"].tolist())
    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = LinearSegmentedColormap.from_list("ba", ["#b2182b", "#f7f7f7", "#2166ac"])
    im = ax.imshow(hm.values, aspect="auto", cmap=cmap, vmin=0.4, vmax=0.95)
    ax.set_xticks(range(len(hm.columns)))
    ax.set_xticklabels(hm.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels(hm.index)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.55 < v < 0.85 else "white", fontsize=8)
    # separators between families
    fam_seq = [FAMILIES[m] for m in hm.index]
    for i in range(1, len(fam_seq)):
        if fam_seq[i] != fam_seq[i - 1]:
            ax.axhline(i - 0.5, color="black", lw=1.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean balanced accuracy (over 4 budgets x 3 seeds, successes)")
    ax.set_title("Method x dataset — mean balanced accuracy\n(datasets ordered small -> large; failures excluded)")
    ax.set_axisbelow(True)
    ax.grid(False)
    save(fig, "fig2_method_dataset_heatmap")

    # ---------------- FIGURE 3: budget curves by family ----------------
    fig, ax = plt.subplots(figsize=(11, 7))
    fam_budget = (ok.groupby(["family", "n_labeled"])[PRIMARY].mean().reset_index()
                  .rename(columns={PRIMARY: "mean"}))
    for fam in FAMILY_ORDER:
        sub = fam_budget[fam_budget["family"] == fam].sort_values("n_labeled")
        ax.plot(sub["n_labeled"], sub["mean"], marker="o", lw=2.5, ms=8,
                color=FAMILY_COLORS[fam], label=fam)
    ax.set_xticks(BUDGETS)
    ax.set_xlabel("Label budget (number of labeled examples)")
    ax.set_ylabel("Mean balanced accuracy (pooled over 10 datasets x 3 seeds)")
    ax.set_title("Balanced accuracy vs label budget, by method family\n(successes only)")
    ax.legend(title="Family", fontsize=11)
    save(fig, "fig3_budget_curves_by_family")

    # ---------------- FIGURE 3b: top methods budget curves ----------------
    top_methods = overall.sort_values("mean_rank")["method"].head(6).tolist()
    fig, ax = plt.subplots(figsize=(11, 7))
    mb = (ok.groupby(["method", "n_labeled"])[PRIMARY].mean().reset_index()
          .rename(columns={PRIMARY: "mean"}))
    for m in top_methods:
        sub = mb[mb["method"] == m].sort_values("n_labeled")
        ax.plot(sub["n_labeled"], sub["mean"], marker="o", lw=2.2, ms=7,
                color=FAMILY_COLORS[FAMILIES[m]], label=m)
    ax.set_xticks(BUDGETS)
    ax.set_xlabel("Label budget")
    ax.set_ylabel("Mean balanced accuracy (pooled datasets x seeds)")
    ax.set_title("Top-6 methods (by overall rank) — balanced accuracy vs budget")
    ax.legend(fontsize=10)
    save(fig, "fig3b_top_methods_budget_curves")

    # ---------------- FIGURE 4: SSL vs supervised deltas ----------------
    ssl = pd.read_csv("results/aggregated/low_class_wave_paper_methods/ssl_vs_supervised_by_dataset_budget.csv")
    # aggregate mean delta per ssl_method (matched-seed), across dataset+budget
    d_by_method = (ssl.groupby("ssl_method")["delta_balanced_accuracy_mean"]
                   .agg(["mean", "std", "count"]).reset_index()
                   .sort_values("mean"))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = [FAMILY_COLORS[FAMILIES[m]] for m in d_by_method["ssl_method"]]
    ax.barh(d_by_method["ssl_method"], d_by_method["mean"], color=colors,
            xerr=d_by_method["std"], error_kw={"elinewidth": 1, "alpha": 0.4, "capsize": 3})
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Mean Δ balanced accuracy (SSL − matched supervised baseline)")
    ax.set_title("SSL vs matched supervised baseline\n(matched seeds; averaged over datasets x budgets)")
    for i, (m, v) in enumerate(zip(d_by_method["ssl_method"], d_by_method["mean"])):
        ax.text(v + (0.001 if v >= 0 else -0.001), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=10)
    save(fig, "fig4_ssl_vs_supervised_delta")

    # 4b: SSL delta heatmap dataset x budget for the pseudo-label families pooled
    pivot = (ssl.groupby(["dataset", "n_labeled"])["delta_balanced_accuracy_mean"]
             .mean().unstack("n_labeled").reindex(ds_table["dataset"].tolist()))
    fig, ax = plt.subplots(figsize=(8.5, 8))
    vmax = np.nanmax(np.abs(pivot.values))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=9,
                        color="black" if abs(v) < vmax * 0.6 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Δ balanced accuracy (all SSL-vs-sup pairs)")
    ax.set_xlabel("Label budget")
    ax.set_title("SSL benefit by dataset x budget\n(mean over all matched SSL-vs-supervised pairs)")
    ax.grid(False)
    save(fig, "fig4b_ssl_delta_heatmap")

    report["ssl_delta_by_method"] = d_by_method.round(4).to_dict(orient="records")
    report["ssl_delta_by_dataset_budget"] = pivot.round(4).reset_index().to_dict(orient="records")

    # ---------------- FIGURE 5: best method by dataset x budget (tile) ----------------
    best = cell.sort_values("mean").groupby(["dataset", "n_labeled"]).tail(1)
    best_pivot = best.pivot(index="dataset", columns="n_labeled", values="method").reindex(ds_table["dataset"].tolist())
    best_val = best.pivot(index="dataset", columns="n_labeled", values="mean").reindex(ds_table["dataset"].tolist())
    # color tiles by family
    fam_idx = {f: i for i, f in enumerate(FAMILY_ORDER)}
    color_mat = np.vectorize(lambda m: fam_idx[FAMILIES[m]] if isinstance(m, str) else np.nan)(best_pivot.values)
    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = matplotlib.colors.ListedColormap([FAMILY_COLORS[f] for f in FAMILY_ORDER])
    ax.imshow(color_mat, aspect="auto", cmap=cmap, vmin=0, vmax=len(FAMILY_ORDER) - 1)
    ax.set_xticks(range(len(best_pivot.columns)))
    ax.set_xticklabels(best_pivot.columns)
    ax.set_yticks(range(len(best_pivot.index)))
    ax.set_yticklabels(best_pivot.index)
    for i in range(best_pivot.shape[0]):
        for j in range(best_pivot.shape[1]):
            m = best_pivot.values[i, j]
            v = best_val.values[i, j]
            ax.text(j, i, f"{m}\n{v:.2f}", ha="center", va="center", fontsize=8, color="white")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILY_ORDER]
    ax.legend(handles, FAMILY_ORDER, title="Winner family", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_xlabel("Label budget")
    ax.set_title("Best method per dataset x budget\n(highest mean balanced accuracy; tile colored by family)")
    ax.grid(False)
    save(fig, "fig5_best_method_tiles")

    best_out = best[["dataset", "n_labeled", "method", "mean"]].copy()
    best_out["family"] = best_out["method"].map(FAMILIES)
    best_out.to_csv(TABLES / "best_method_by_dataset_budget.csv", index=False)
    report["best_method_family_counts"] = best_out["family"].value_counts().to_dict()
    report["best_method_counts"] = best_out["method"].value_counts().to_dict()

    # ---------------- FIGURE 6: runtime by method ----------------
    rt = ok.groupby("method")["runtime_seconds"].agg(["mean", "median", "std"]).reset_index()
    rt["family"] = rt["method"].map(FAMILIES)
    rt = rt.sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [FAMILY_COLORS[f] for f in rt["family"]]
    ax.barh(rt["method"], rt["mean"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime per run (seconds, log scale)")
    ax.set_title("Runtime by method (mean over all successful runs)")
    for i, (m, v) in enumerate(zip(rt["method"], rt["mean"])):
        ax.text(v * 1.1, i, f"{v:.1f}s", va="center", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILY_ORDER]
    ax.legend(handles, FAMILY_ORDER, title="Family", loc="lower right", fontsize=10)
    save(fig, "fig6_runtime_by_method")
    report["runtime_by_method"] = rt.round(2).to_dict(orient="records")

    # 6b: accuracy vs runtime tradeoff scatter
    fig, ax = plt.subplots(figsize=(11, 7.5))
    perf = ok.groupby("method")[PRIMARY].mean()
    rtm = ok.groupby("method")["runtime_seconds"].mean()
    for m in methods:
        ax.scatter(rtm[m], perf[m], s=120, color=FAMILY_COLORS[FAMILIES[m]], zorder=3,
                   edgecolor="black", linewidth=0.5)
        ax.annotate(m, (rtm[m], perf[m]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime per run (s, log scale)")
    ax.set_ylabel("Mean balanced accuracy (successes, all datasets/budgets)")
    ax.set_title("Performance vs runtime trade-off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILY_ORDER]
    ax.legend(handles, FAMILY_ORDER, title="Family", fontsize=10, loc="lower right")
    save(fig, "fig6b_perf_vs_runtime")

    # ---------------- FIGURE 7: failures heatmap (graph SSL) ----------------
    fail = df[df["status"] != "success"].copy()
    fmethods = sorted(fail["method"].unique())
    # build (dataset,budget) x method count of failed seeds
    idx = [(d, b) for d in datasets for b in BUDGETS]
    fmat = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(idx, names=["dataset", "n_labeled"]),
                        columns=fmethods)
    for (d, b, m), g in fail.groupby(["dataset", "n_labeled", "method"]):
        fmat.loc[(d, b), m] = len(g)
    fmat = fmat.loc[(fmat.sum(axis=1) > 0)]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fmat.values, aspect="auto", cmap="Reds", vmin=0, vmax=3)
    ax.set_xticks(range(len(fmat.columns)))
    ax.set_xticklabels(fmat.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(fmat.index)))
    ax.set_yticklabels([f"{d} @ {b}" for d, b in fmat.index])
    for i in range(fmat.shape[0]):
        for j in range(fmat.shape[1]):
            v = fmat.values[i, j]
            if v > 0:
                ax.text(j, i, f"{v}/3", ha="center", va="center",
                        color="white" if v >= 2 else "black", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Failed seeds (out of 3)")
    ax.set_title("Graph-SSL failures (failed_graph_ssl_nan)\nby dataset x budget x method")
    ax.grid(False)
    save(fig, "fig7_failures_heatmap")
    report["failures_detail"] = (fail.groupby(["dataset", "method", "n_labeled"]).size()
                                 .reset_index(name="failed_seeds").to_dict(orient="records"))

    # ---------------- Binary vs multiclass, small vs large summaries ----------------
    ok2 = ok.merge(ds_table[["dataset", "task", "size_bucket"]], on="dataset", how="left")
    report["family_by_task"] = (ok2.groupby(["family", "task"])[PRIMARY].mean()
                                .unstack("task").round(4).reset_index().to_dict(orient="records"))
    report["family_by_size"] = (ok2.groupby(["family", "size_bucket"])[PRIMARY].mean()
                                .unstack("size_bucket").round(4).reset_index().to_dict(orient="records"))
    report["family_by_budget"] = (ok2.groupby(["family", "n_labeled"])[PRIMARY].mean()
                                  .unstack("n_labeled").round(4).reset_index().to_dict(orient="records"))

    # SSL delta by task and budget (from ssl table)
    ssl2 = ssl.merge(ds_table[["dataset", "task", "size_bucket"]], on="dataset", how="left")
    report["ssl_delta_by_task"] = (ssl2.groupby("task")["delta_balanced_accuracy_mean"]
                                   .agg(["mean", "std", "count"]).round(4).reset_index().to_dict(orient="records"))
    report["ssl_delta_by_budget"] = (ssl2.groupby("n_labeled")["delta_balanced_accuracy_mean"]
                                     .agg(["mean", "std", "count"]).round(4).reset_index().to_dict(orient="records"))
    report["ssl_delta_by_method_budget"] = (ssl2.groupby(["ssl_method", "n_labeled"])["delta_balanced_accuracy_mean"]
                                            .mean().round(4).reset_index().to_dict(orient="records"))

    # fraction of positive deltas overall
    report["ssl_pairs_total"] = int(len(ssl))
    report["ssl_pairs_positive"] = int((ssl["delta_balanced_accuracy_mean"] > 0).sum())
    report["ssl_pairs_negative"] = int((ssl["delta_balanced_accuracy_mean"] < 0).sum())

    # ---------------- write JSON ----------------
    with open(TABLES / "report_summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ---------------- console dump for authoring ----------------
    print("=== INTEGRITY ===")
    print(json.dumps(integrity, indent=2, default=str))
    print("\n=== DATASET TABLE ===")
    print(ds_table.to_string(index=False))
    print("\n=== OVERALL RANKING (failure-aware, mean per-cell rank) ===")
    print(overall.sort_values("mean_rank").to_string(index=False))
    print(f"\ncomplete cells (all 17 methods present): {report['n_complete_cells_all17']}/{report['n_total_cells']}")
    print("\n=== SSL DELTA BY METHOD ===")
    print(d_by_method.round(4).to_string(index=False))
    print("\n=== SSL DELTA BY BUDGET ===")
    print(pd.DataFrame(report["ssl_delta_by_budget"]).to_string(index=False))
    print("\n=== SSL DELTA BY TASK ===")
    print(pd.DataFrame(report["ssl_delta_by_task"]).to_string(index=False))
    print("\n=== SSL pairs pos/neg ===", report["ssl_pairs_positive"], "/", report["ssl_pairs_negative"], "of", report["ssl_pairs_total"])
    print("\n=== BEST METHOD FAMILY COUNTS ===")
    print(report["best_method_family_counts"])
    print("\n=== BEST METHOD COUNTS ===")
    print(report["best_method_counts"])
    print("\n=== FAMILY BY BUDGET (balanced acc) ===")
    print(pd.DataFrame(report["family_by_budget"]).to_string(index=False))
    print("\n=== FAMILY BY TASK ===")
    print(pd.DataFrame(report["family_by_task"]).to_string(index=False))
    print("\n=== FAMILY BY SIZE ===")
    print(pd.DataFrame(report["family_by_size"]).to_string(index=False))
    print("\n=== RUNTIME BY METHOD ===")
    print(rt.round(2).to_string(index=False))
    print("\nDONE. Historical figures and tables written to", OUT)


if __name__ == "__main__":
    main()
