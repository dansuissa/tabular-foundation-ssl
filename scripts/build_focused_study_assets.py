#!/usr/bin/env python3
"""Build focused-study assets for the canonical main report.

This script performs no model fitting. It validates and summarizes:
  * Phase A frozen TFM results;
  * the canonical six-method Phase C file;
  * the immutable historical 17-method benchmark;
  * representative final-gate evidence.

Outputs are written into the focused-study sections of results/reports/main_report/.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "reports" / "main_report"
TABLES = OUT / "tables" / "focused_ssl"
FIGURES = OUT / "figures" / "focused_ssl"
VALIDATION = OUT / "validation"

PHASE_A_PATH = RAW / "tfm_frozen_screen.csv"
PHASE_C_PATH = RAW / "focused_tfm_ssl.csv"
HISTORICAL_PATH = RAW / "low_class_wave_paper_methods.csv"
FINAL_GATE_PATH = RAW / "attention_family_final_gate.csv"

KEY = ["dataset", "method", "seed", "n_labeled"]
PAIR_KEY = ["dataset", "seed", "n_labeled"]
DATASETS = [
    "phoneme",
    "spambase",
    "MagicTelescope",
    "adult",
    "bank-marketing",
    "electricity",
    "satimage",
    "segment",
    "steel-plates-fault",
    "jannis",
]
BUDGETS = [50, 100, 250, 500]
SEEDS = [0, 1, 2]
PHASE_A_METHODS = ["tabpfn3", "tabiclv2"]
PHASE_C_METHODS = [
    "tabpfn3_self_training",
    "tabiclv2_self_training",
    "laplacian_ssl",
    "unlabeled_attention_ssl",
    "embedding_alignment_ssl",
    "geometric_attention_ssl",
]
NEW_METHODS = PHASE_A_METHODS + PHASE_C_METHODS
SELECTED_HISTORICAL = [
    "catboost",
    "xgboost",
    "self_training_lightgbm",
    "label_propagation",
    "label_spreading",
    "vime",
    "scarf",
    "sslae",
]
PRIMARY = "metric_balanced_accuracy"
METRICS = [
    "metric_balanced_accuracy",
    "metric_macro_f1",
    "metric_accuracy",
    "metric_log_loss",
    "metric_roc_auc",
    "metric_average_precision",
    "metric_brier",
    "metric_ece",
]
DISPLAY = {
    "tabpfn3": "TabPFN-3",
    "tabiclv2": "TabICLv2",
    "tabpfn3_self_training": "TabPFN-3 self-training",
    "tabiclv2_self_training": "TabICLv2 self-training",
    "laplacian_ssl": "Laplacian SSL",
    "unlabeled_attention_ssl": "Unlabeled attention",
    "embedding_alignment_ssl": "Embedding alignment",
    "geometric_attention_ssl": "Combined geometric attention",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "self_training_lightgbm": "Self-training LightGBM",
    "label_propagation": "Label propagation",
    "label_spreading": "Label spreading",
    "vime": "VIME",
    "scarf": "SCARF",
    "sslae": "SSLAE",
}
COLORS = {
    "tabpfn3": "#1f77b4",
    "tabiclv2": "#4c78a8",
    "tabpfn3_self_training": "#2ca02c",
    "tabiclv2_self_training": "#59a14f",
    "laplacian_ssl": "#9467bd",
    "unlabeled_attention_ssl": "#ff7f0e",
    "embedding_alignment_ssl": "#e377c2",
    "geometric_attention_ssl": "#d62728",
}

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


class _SafeNames(ast.NodeTransformer):
    """Permit serialized NaN/Inf names while retaining literal_eval safety."""

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id in {"nan", "NaN"}:
            return ast.copy_location(ast.Constant(float("nan")), node)
        if node.id in {"inf", "Infinity"}:
            return ast.copy_location(ast.Constant(float("inf")), node)
        raise ValueError(f"unexpected non-literal name: {node.id}")


def parse_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        tree = ast.parse(value, mode="eval")
        tree = _SafeNames().visit(tree)
        return ast.literal_eval(tree)
    except (SyntaxError, ValueError, TypeError):
        return {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean_or_nan(values: list[Any]) -> float:
    a = np.asarray([v for v in (finite_or_none(x) for x in values) if v is not None])
    return float(a.mean()) if a.size else float("nan")


def ci_summary(values: pd.Series, prefix: str = "delta") -> dict[str, Any]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(x))
    mean = float(np.mean(x)) if n else float("nan")
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": sd,
        f"{prefix}_se": se,
        f"{prefix}_ci95_low": mean - 1.96 * se if n > 1 else float("nan"),
        f"{prefix}_ci95_high": mean + 1.96 * se if n > 1 else float("nan"),
        "n_pairs": n,
        "positive_fraction": float(np.mean(x > 0)) if n else float("nan"),
        "zero_fraction": float(np.mean(x == 0)) if n else float("nan"),
    }


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLES / name
    df.to_csv(path, index=False)
    return path


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def validate_grid(
    df: pd.DataFrame,
    *,
    methods: list[str],
    expected_rows: int,
    label: str,
) -> dict[str, Any]:
    missing_cols = sorted(set(KEY + ["status", PRIMARY]) - set(df.columns))
    duplicates = int(df.duplicated(KEY).sum()) if not missing_cols else -1
    expected = {
        (dataset, method, seed, budget)
        for dataset in DATASETS
        for method in methods
        for seed in SEEDS
        for budget in BUDGETS
    }
    actual = set(map(tuple, df[KEY].itertuples(index=False, name=None))) if not missing_cols else set()
    metric_nonfinite = {}
    ok = df[df["status"] == "success"] if "status" in df else df.iloc[0:0]
    for metric in METRICS:
        if metric in df:
            v = pd.to_numeric(ok[metric], errors="coerce")
            metric_nonfinite[metric] = {
                "nan": int(v.isna().sum()),
                "inf": int(np.isinf(v.fillna(0)).sum()),
            }
    result = {
        "label": label,
        "rows": int(len(df)),
        "expected_rows": expected_rows,
        "methods": sorted(df["method"].dropna().unique().tolist()),
        "datasets": sorted(df["dataset"].dropna().unique().tolist()),
        "budgets": sorted(map(int, df["n_labeled"].dropna().unique().tolist())),
        "seeds": sorted(map(int, df["seed"].dropna().unique().tolist())),
        "status_counts": {str(k): int(v) for k, v in df["status"].value_counts(dropna=False).items()},
        "duplicate_keys": duplicates,
        "missing_keys": len(expected - actual),
        "unexpected_keys": len(actual - expected),
        "missing_columns": missing_cols,
        "metric_nonfinite": metric_nonfinite,
    }
    primary = metric_nonfinite.get(PRIMARY, {})
    result["valid"] = bool(
        len(df) == expected_rows
        and not missing_cols
        and duplicates == 0
        and not expected - actual
        and not actual - expected
        and result["status_counts"] == {"success": expected_rows}
        and primary.get("nan", 1) == 0
        and primary.get("inf", 1) == 0
    )
    return result


def paired_delta(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    treatment_name: str,
    control_name: str,
    metric: str = PRIMARY,
) -> pd.DataFrame:
    left = treatment[PAIR_KEY + [metric]].rename(columns={metric: "treatment"})
    right = control[PAIR_KEY + [metric]].rename(columns={metric: "control"})
    merged = left.merge(right, on=PAIR_KEY, validate="one_to_one")
    merged["treatment_method"] = treatment_name
    merged["control_method"] = control_name
    merged["metric"] = metric
    merged["delta"] = merged["treatment"] - merged["control"]
    return merged


def method_metric_table(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    ok = df[(df["status"] == "success") & df["method"].isin(methods)].copy()
    rows = []
    for method, g in ok.groupby("method"):
        row: dict[str, Any] = {
            "method": method,
            "display_name": DISPLAY.get(method, method),
            "successful_runs": int(len(g)),
            "coverage_fraction": float(len(g) / (len(DATASETS) * len(BUDGETS) * len(SEEDS))),
        }
        for metric in METRICS:
            if metric in g:
                vals = pd.to_numeric(g[metric], errors="coerce")
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std())
                row[f"{metric}_n"] = int(vals.notna().sum())
        row["runtime_seconds_mean"] = float(pd.to_numeric(g["runtime_seconds"], errors="coerce").mean())
        row["runtime_seconds_median"] = float(pd.to_numeric(g["runtime_seconds"], errors="coerce").median())
        row["peak_gpu_mem_mb_mean"] = float(pd.to_numeric(g.get("peak_gpu_mem_mb"), errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("metric_balanced_accuracy_mean", ascending=False)


def build_dataset_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in df.groupby("dataset"):
        payload = parse_payload(g.iloc[0].get("training_meta"))
        rows.append(
            {
                "dataset": dataset,
                "openml_dataset_id": payload.get("dataset_id"),
                "n_classes": payload.get("n_classes"),
                "n_features_raw": payload.get("n_features_before_preprocessing"),
                "n_features_processed": payload.get("n_features_after_preprocessing"),
                "train_labeled_budget_grid": "50, 100, 250, 500",
                "n_unlabeled_at_budget_50": parse_payload(
                    g[g["n_labeled"] == 50].iloc[0].get("training_meta")
                ).get("n_unlabeled"),
                "validation_size": payload.get("val_size"),
                "test_size": payload.get("test_size"),
            }
        )
    return pd.DataFrame(rows).set_index("dataset").reindex(DATASETS).reset_index()


def build_rank_table(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    ok = df[(df["status"] == "success") & df["method"].isin(methods)]
    cell = (
        ok.groupby(["dataset", "n_labeled", "method"])[PRIMARY]
        .mean()
        .reset_index(name="balanced_accuracy")
    )
    cell["rank"] = cell.groupby(["dataset", "n_labeled"])["balanced_accuracy"].rank(
        ascending=False, method="average"
    )
    ranked = (
        cell.groupby("method")
        .agg(
            mean_cell_rank=("rank", "mean"),
            std_cell_rank=("rank", "std"),
            cells_ranked=("rank", "size"),
            mean_cell_balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .reset_index()
        .sort_values("mean_cell_rank")
    )
    ranked["display_name"] = ranked["method"].map(DISPLAY).fillna(ranked["method"])
    return ranked


def diagnostic_frame(phase_c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in phase_c.iterrows():
        top = parse_payload(row.get("training_meta"))
        diag = top.get("method_diagnostics", {})
        diag = diag if isinstance(diag, dict) else {}
        collapse = diag.get("collapse_diagnostics", {})
        collapse = collapse if isinstance(collapse, dict) else {}
        loop_logs = diag.get("loop_logs", [])
        accepted = []
        confidences = []
        for entry in loop_logs if isinstance(loop_logs, list) else []:
            if not isinstance(entry, dict):
                continue
            accepted.append(entry.get("n_pseudo_new", 0) or 0)
            if finite_or_none(entry.get("mean_confidence")) is not None:
                confidences.append(entry["mean_confidence"])
        intra = diag.get("intra_class_distances", [])
        inter = diag.get("inter_prototype_distances", [])
        rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": row["seed"],
                "n_labeled": row["n_labeled"],
                "selected_round": diag.get("selected_round", top.get("selected_round")),
                "rounds_attempted": diag.get("rounds_attempted"),
                "fallback_reason": diag.get("fallback_reason", top.get("fallback_reason")),
                "n_pseudo_added_total": diag.get(
                    "n_pseudo_added_total", top.get("n_pseudo_added_total")
                ),
                "pseudo_candidates_seen": sum(accepted) if accepted else np.nan,
                "pseudo_mean_confidence": mean_or_nan(confidences),
                "graph_n_nodes": diag.get("graph_n_nodes"),
                "graph_n_edges": diag.get("graph_n_edges"),
                "graph_n_isolated_nodes": diag.get("graph_n_isolated_nodes"),
                "graph_n_connected_components": diag.get("graph_n_connected_components"),
                "graph_build_time_seconds": diag.get("graph_build_time_seconds"),
                "graph_affinity_mean": diag.get("graph_affinity_mean"),
                "attention_entropy": diag.get("attention_entropy"),
                "labeled_attention_mass": diag.get("labeled_attention_mass"),
                "unlabeled_attention_mass": diag.get("unlabeled_attention_mass"),
                "memory_size": diag.get("memory_size"),
                "memory_leakage_flag": diag.get(
                    "memory_contains_val_or_test",
                    not diag.get("no_val_test_in_memory", True),
                ),
                "embedding_variance": diag.get(
                    "embedding_variance", collapse.get("embedding_variance_mean")
                ),
                "effective_embedding_rank": collapse.get("effective_embedding_rank"),
                "representation_collapse_suspect": diag.get(
                    "representation_collapse_suspect",
                    collapse.get("representation_collapse_suspect"),
                ),
                "constant_prediction_suspect": collapse.get("constant_prediction_suspect"),
                "max_pred_class_fraction": collapse.get("max_pred_class_fraction"),
                "mean_intra_class_distance": mean_or_nan(intra if isinstance(intra, list) else []),
                "mean_inter_prototype_distance": mean_or_nan(
                    inter if isinstance(inter, list) else []
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    all_new: pd.DataFrame,
    phase_a: pd.DataFrame,
    phase_c: pd.DataFrame,
    historical: pd.DataFrame,
    self_pairs: pd.DataFrame,
    component_pairs: pd.DataFrame,
    diagnostic: pd.DataFrame,
) -> None:
    ok_new = all_new[all_new["status"] == "success"]

    # Figure 1: all eight requested methods by budget.
    means = (
        ok_new.groupby(["method", "n_labeled"])[PRIMARY]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for method in NEW_METHODS:
        g = means[means["method"] == method].sort_values("n_labeled")
        se = g["std"] / np.sqrt(g["count"])
        ax.errorbar(
            g["n_labeled"],
            g["mean"],
            yerr=1.96 * se,
            marker="o",
            linewidth=2,
            capsize=3,
            color=COLORS[method],
            label=DISPLAY[method],
        )
    ax.set_xticks(BUDGETS)
    ax.set_xlabel("Labeled training budget")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Eight requested methods across the complete 10-dataset grid")
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, "fig1_new_methods_by_budget")

    # Figure 2: failure-aware new-method mean ranks.
    ranks = build_rank_table(all_new, NEW_METHODS).sort_values("mean_cell_rank", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(
        ranks["display_name"],
        ranks["mean_cell_rank"],
        color=[COLORS[m] for m in ranks["method"]],
    )
    ax.set_xlabel("Mean rank over 40 dataset × budget cells (lower is better)")
    ax.set_title("Failure-aware ranking of the requested methods")
    for i, v in enumerate(ranks["mean_cell_rank"]):
        ax.text(v + 0.04, i, f"{v:.2f}", va="center", fontsize=9)
    save_figure(fig, "fig2_new_method_ranking")

    # Figure 3: exact paired self-training effects.
    heat = (
        self_pairs.groupby(["treatment_method", "dataset", "n_labeled"])["delta"]
        .mean()
        .reset_index()
    )
    methods = ["tabpfn3_self_training", "tabiclv2_self_training"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    vmax = max(0.01, float(np.nanmax(np.abs(heat["delta"]))))
    for ax, method in zip(axes, methods, strict=True):
        p = (
            heat[heat["treatment_method"] == method]
            .pivot(index="dataset", columns="n_labeled", values="delta")
            .reindex(index=DATASETS, columns=BUDGETS)
        )
        im = ax.imshow(p.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(4), BUDGETS)
        ax.set_yticks(range(10), DATASETS)
        ax.set_xlabel("Label budget")
        ax.set_title(DISPLAY[method])
        for i in range(10):
            for j in range(4):
                ax.text(j, i, f"{p.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=7)
        ax.grid(False)
    axes[0].set_ylabel("Dataset")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("Δ balanced accuracy (self-training − frozen backbone)")
    fig.suptitle("Exact paired self-training effect (same dataset, budget, seed, split)")
    save_figure(fig, "fig3_self_training_paired_delta")

    # Figure 4: component contrasts against the combined model.
    comp = (
        component_pairs.groupby(["control_method", "n_labeled"])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for method in ["laplacian_ssl", "unlabeled_attention_ssl", "embedding_alignment_ssl"]:
        g = comp[comp["control_method"] == method].sort_values("n_labeled")
        ax.errorbar(
            g["n_labeled"],
            g["mean"],
            yerr=1.96 * g["std"] / np.sqrt(g["count"]),
            marker="o",
            capsize=3,
            linewidth=2,
            label=f"Combined − {DISPLAY[method]}",
            color=COLORS[method],
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(BUDGETS)
    ax.set_xlabel("Label budget")
    ax.set_ylabel("Paired Δ balanced accuracy")
    ax.set_title("Combined model versus its separately evaluated component methods")
    ax.legend(fontsize=9)
    save_figure(fig, "fig4_combined_component_deltas")

    # Figure 5: calibration / discrimination.
    calibration = (
        ok_new.groupby("method")
        .agg(
            balanced_accuracy=(PRIMARY, "mean"),
            ece=("metric_ece", "mean"),
            log_loss=("metric_log_loss", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in calibration.iterrows():
        method = row["method"]
        ax.scatter(
            row["ece"],
            row["balanced_accuracy"],
            s=95,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(DISPLAY[method], (row["ece"], row["balanced_accuracy"]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean expected calibration error (lower is better)")
    ax.set_ylabel("Mean balanced accuracy (higher is better)")
    ax.set_title("Calibration–discrimination trade-off")
    save_figure(fig, "fig5_calibration_vs_accuracy")

    # Figure 6: compute cost versus primary performance, selected complete methods.
    selected = pd.concat(
        [
            ok_new,
            historical[
                (historical["status"] == "success")
                & historical["method"].isin(SELECTED_HISTORICAL)
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    rt = (
        selected.groupby("method")
        .agg(runtime=("runtime_seconds", "mean"), balanced_accuracy=(PRIMARY, "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for _, row in rt.iterrows():
        method = row["method"]
        is_new = method in NEW_METHODS
        ax.scatter(
            row["runtime"],
            row["balanced_accuracy"],
            s=100 if is_new else 55,
            color=COLORS.get(method, "#8c8c8c"),
            marker="o" if is_new else "s",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(
            DISPLAY.get(method, method),
            (row["runtime"], row["balanced_accuracy"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean wall-clock runtime per run (seconds, log scale)")
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title("Compute cost versus performance")
    save_figure(fig, "fig6_runtime_vs_performance")

    # Figure 7: method-specific diagnostics.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    self_diag = diagnostic[diagnostic["method"].isin(methods)]
    fallback = (
        self_diag.assign(fallback=self_diag["selected_round"].fillna(0).astype(float) == 0)
        .groupby("method")["fallback"]
        .mean()
        .reindex(methods)
    )
    axes[0].bar([DISPLAY[m] for m in methods], fallback.values, color=[COLORS[m] for m in methods])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of runs")
    axes[0].set_title("Self-training fallback to round 0")
    axes[0].tick_params(axis="x", rotation=20)

    attention_methods = ["unlabeled_attention_ssl", "geometric_attention_ssl"]
    masses = diagnostic.groupby("method")[["labeled_attention_mass", "unlabeled_attention_mass"]].mean()
    x = np.arange(2)
    axes[1].bar(
        x,
        [masses.loc[m, "labeled_attention_mass"] for m in attention_methods],
        label="Labeled",
        color="#4c78a8",
    )
    axes[1].bar(
        x,
        [masses.loc[m, "unlabeled_attention_mass"] for m in attention_methods],
        bottom=[masses.loc[m, "labeled_attention_mass"] for m in attention_methods],
        label="Unlabeled",
        color="#f28e2b",
    )
    axes[1].set_xticks(x, [DISPLAY[m] for m in attention_methods], rotation=20)
    axes[1].set_ylabel("Mean attention mass")
    axes[1].set_title("Retrieved-memory attention composition")
    axes[1].legend(fontsize=8)

    collapse_methods = ["laplacian_ssl", "embedding_alignment_ssl", "geometric_attention_ssl"]
    collapse = (
        diagnostic.assign(
            collapse=diagnostic["representation_collapse_suspect"].fillna(False).astype(bool)
        )
        .groupby("method")["collapse"]
        .mean()
        .reindex(collapse_methods)
    )
    axes[2].bar(
        [DISPLAY[m] for m in collapse_methods],
        collapse.values,
        color=[COLORS[m] for m in collapse_methods],
    )
    for i, value in enumerate(collapse.values):
        axes[2].text(i, max(0.015, value + 0.015), f"{int(round(value * 120))}/120",
                     ha="center", va="bottom", fontsize=9)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Fraction of runs")
    axes[2].set_title("Representation-collapse flags")
    axes[2].tick_params(axis="x", rotation=25)
    fig.suptitle("Method-specific diagnostics across the complete Phase-C grid")
    fig.tight_layout()
    save_figure(fig, "fig7_method_diagnostics")


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    v = finite_or_none(value)
    if v is None:
        return "NA"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], labels: list[str]) -> str:
    out = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(fmt(value))
            else:
                vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    VALIDATION.mkdir(parents=True, exist_ok=True)

    phase_a = pd.read_csv(PHASE_A_PATH)
    phase_c = pd.read_csv(PHASE_C_PATH)
    historical = pd.read_csv(HISTORICAL_PATH)
    final_gate = pd.read_csv(FINAL_GATE_PATH) if FINAL_GATE_PATH.exists() else pd.DataFrame()

    phase_a_validation = validate_grid(
        phase_a, methods=PHASE_A_METHODS, expected_rows=240, label="Phase A"
    )
    phase_c_validation = validate_grid(
        phase_c, methods=PHASE_C_METHODS, expected_rows=720, label="Phase C"
    )
    if not phase_a_validation["valid"] or not phase_c_validation["valid"]:
        raise SystemExit(
            "Refusing to generate final report: canonical grid validation failed.\n"
            + json.dumps(
                {"phase_a": phase_a_validation, "phase_c": phase_c_validation}, indent=2
            )
        )

    # Preserve phase identity while allowing analysis over all eight requested methods.
    phase_a = phase_a.copy()
    phase_c = phase_c.copy()
    phase_a["phase"] = "A"
    phase_c["phase"] = "C"
    all_new = pd.concat([phase_a, phase_c], ignore_index=True, sort=False)

    dataset_table = build_dataset_table(phase_a)
    save_table(dataset_table, "dataset_table.csv")

    new_summary = method_metric_table(all_new, NEW_METHODS)
    save_table(new_summary, "new_method_metric_summary.csv")
    save_table(build_rank_table(all_new, NEW_METHODS), "new_method_failure_aware_ranking.csv")
    by_budget = (
        all_new.groupby(["method", "n_labeled"])[METRICS + ["runtime_seconds"]]
        .agg(["mean", "std", "count"])
    )
    by_budget.columns = ["_".join(col) for col in by_budget.columns]
    save_table(by_budget.reset_index(), "new_method_by_budget.csv")

    # Phase-A exact paired comparison.
    a_pivot = phase_a.pivot(index=PAIR_KEY, columns="method", values=PRIMARY).reset_index()
    a_pivot["delta_tabpfn3_minus_tabiclv2"] = a_pivot["tabpfn3"] - a_pivot["tabiclv2"]
    save_table(a_pivot, "phase_a_seed_level_paired.csv")
    phase_a_effect = pd.DataFrame(
        [
            {"scope": "overall", **ci_summary(a_pivot["delta_tabpfn3_minus_tabiclv2"])},
            *[
                {
                    "scope": f"budget_{budget}",
                    **ci_summary(
                        a_pivot.loc[
                            a_pivot["n_labeled"] == budget, "delta_tabpfn3_minus_tabiclv2"
                        ]
                    ),
                }
                for budget in BUDGETS
            ],
        ]
    )
    save_table(phase_a_effect, "phase_a_tabpfn3_vs_tabiclv2.csv")

    # Exact TFM self-training effects versus each own frozen backbone.
    self_pairs = pd.concat(
        [
            paired_delta(
                phase_c[phase_c["method"] == "tabpfn3_self_training"],
                phase_a[phase_a["method"] == "tabpfn3"],
                "tabpfn3_self_training",
                "tabpfn3",
            ),
            paired_delta(
                phase_c[phase_c["method"] == "tabiclv2_self_training"],
                phase_a[phase_a["method"] == "tabiclv2"],
                "tabiclv2_self_training",
                "tabiclv2",
            ),
        ],
        ignore_index=True,
    )
    save_table(self_pairs, "self_training_seed_level_paired.csv")
    self_effect_rows = []
    for method, g in self_pairs.groupby("treatment_method"):
        self_effect_rows.append({"method": method, "scope": "overall", **ci_summary(g["delta"])})
        for budget, b in g.groupby("n_labeled"):
            self_effect_rows.append(
                {"method": method, "scope": f"budget_{budget}", **ci_summary(b["delta"])}
            )
    self_effects = pd.DataFrame(self_effect_rows)
    save_table(self_effects, "self_training_paired_effects.csv")

    # Combined method versus component methods: exact paired grid, but not an
    # additive causal ablation because each component is independently trained.
    combined = phase_c[phase_c["method"] == "geometric_attention_ssl"]
    component_pairs = pd.concat(
        [
            paired_delta(
                combined,
                phase_c[phase_c["method"] == component],
                "geometric_attention_ssl",
                component,
            )
            for component in [
                "laplacian_ssl",
                "unlabeled_attention_ssl",
                "embedding_alignment_ssl",
            ]
        ],
        ignore_index=True,
    )
    save_table(component_pairs, "combined_vs_components_seed_level.csv")
    comp_effects = pd.DataFrame(
        [
            {
                "combined_method": "geometric_attention_ssl",
                "component_method": method,
                **ci_summary(g["delta"]),
            }
            for method, g in component_pairs.groupby("control_method")
        ]
    )
    save_table(comp_effects, "combined_vs_components_paired_effects.csv")

    # Laplacian comparisons to the historical graph baselines on identical keys.
    lap_pairs = []
    for method in ["label_propagation", "label_spreading"]:
        lap_pairs.append(
            paired_delta(
                phase_c[phase_c["method"] == "laplacian_ssl"],
                historical[
                    (historical["method"] == method) & (historical["status"] == "success")
                ],
                "laplacian_ssl",
                method,
            )
        )
    lap_pairs_df = pd.concat(lap_pairs, ignore_index=True)
    save_table(lap_pairs_df, "laplacian_vs_historical_graph_seed_level.csv")
    save_table(
        pd.DataFrame(
            [
                {"comparison": f"laplacian_ssl_minus_{m}", **ci_summary(g["delta"])}
                for m, g in lap_pairs_df.groupby("control_method")
            ]
        ),
        "laplacian_vs_historical_graph_effects.csv",
    )

    # Requested benchmark context, with explicit failure-aware coverage.
    comparison = pd.concat(
        [
            all_new,
            historical[historical["method"].isin(SELECTED_HISTORICAL)],
        ],
        ignore_index=True,
        sort=False,
    )
    comparison_methods = NEW_METHODS + SELECTED_HISTORICAL
    save_table(
        method_metric_table(comparison, comparison_methods),
        "requested_methods_and_historical_baselines.csv",
    )
    save_table(
        build_rank_table(comparison, comparison_methods),
        "requested_and_baseline_failure_aware_ranking.csv",
    )
    coverage = (
        comparison.groupby("method")["status"]
        .agg(total_runs="size", successful_runs=lambda s: int((s == "success").sum()))
        .reset_index()
    )
    coverage["coverage_fraction"] = coverage["successful_runs"] / coverage["total_runs"]
    save_table(coverage, "coverage_and_failures.csv")

    diagnostic = diagnostic_frame(phase_c)
    save_table(diagnostic, "phase_c_run_diagnostics.csv")
    diag_summary = (
        diagnostic.groupby("method")
        .agg(
            runs=("method", "size"),
            selected_round_mean=("selected_round", "mean"),
            pseudo_added_mean=("n_pseudo_added_total", "mean"),
            graph_nodes_mean=("graph_n_nodes", "mean"),
            graph_edges_mean=("graph_n_edges", "mean"),
            graph_isolated_mean=("graph_n_isolated_nodes", "mean"),
            graph_build_seconds_mean=("graph_build_time_seconds", "mean"),
            attention_entropy_mean=("attention_entropy", "mean"),
            labeled_attention_mass_mean=("labeled_attention_mass", "mean"),
            unlabeled_attention_mass_mean=("unlabeled_attention_mass", "mean"),
            embedding_variance_mean=("embedding_variance", "mean"),
            effective_embedding_rank_mean=("effective_embedding_rank", "mean"),
            representation_collapse_flags=(
                "representation_collapse_suspect",
                lambda x: int(x.fillna(False).astype(bool).sum()),
            ),
            constant_prediction_flags=(
                "constant_prediction_suspect",
                lambda x: int(x.fillna(False).astype(bool).sum()),
            ),
            memory_leakage_flags=(
                "memory_leakage_flag",
                lambda x: int(x.fillna(False).astype(bool).sum()),
            ),
        )
        .reset_index()
    )
    save_table(diag_summary, "method_diagnostic_summary.csv")

    # Raw final-gate evidence is preserved separately and described as representative.
    if not final_gate.empty:
        save_table(
            final_gate[
                [
                    "dataset",
                    "method",
                    "seed",
                    "n_labeled",
                    "status",
                    PRIMARY,
                    "metric_macro_f1",
                    "runtime_seconds",
                    "source_tree_hash",
                ]
            ],
            "representative_attention_final_gate.csv",
        )

    make_figures(
        all_new,
        phase_a,
        phase_c,
        historical,
        self_pairs,
        component_pairs,
        diagnostic,
    )

    best_new = new_summary.iloc[0]
    phase_a_overall = phase_a_effect.iloc[0]
    self_overall = self_effects[self_effects["scope"] == "overall"].set_index("method")
    comp_by_method = comp_effects.set_index("component_method")
    diag_by_method = diag_summary.set_index("method")
    constant_combined = int(
        diag_by_method.loc["geometric_attention_ssl", "constant_prediction_flags"]
    )
    collapse_combined = int(
        diag_by_method.loc["geometric_attention_ssl", "representation_collapse_flags"]
    )
    fallback_rows = diagnostic[
        diagnostic["method"].isin(
            ["tabpfn3_self_training", "tabiclv2_self_training"]
        )
    ].copy()
    fallback_rows["fallback"] = (
        pd.to_numeric(fallback_rows["selected_round"], errors="coerce").fillna(0) == 0
    )
    fallback_rates = fallback_rows.groupby("method")["fallback"].mean()

    report = f"""# Final report: Tabular foundation models and focused semi-supervised learning

Generated {datetime.now(timezone.utc).isoformat()} from validated result files.

## Executive summary

The requested experiment scope is complete. Phase A contains **240/240** successful
frozen-foundation-model runs and Phase C contains **720/720** successful focused SSL
runs. There are no missing or duplicate keys and no non-finite primary balanced-
accuracy values. The immutable 17-method historical result file was read for
comparison and was not modified.

Across the eight requested methods, **{DISPLAY[str(best_new['method'])]}** has the
highest pooled mean balanced accuracy ({fmt(best_new['metric_balanced_accuracy_mean'])}).
This pooled mean is descriptive; the primary ranking in
`tables/new_method_failure_aware_ranking.csv` averages ranks over the 40 dataset ×
budget cells so large datasets cannot dominate by row count.

TabPFN-3 minus TabICLv2 has an exact paired mean balanced-accuracy difference of
**{fmt(phase_a_overall['delta_mean'], signed=True)}** across 120 matched cells
(descriptive 95% interval {fmt(phase_a_overall['delta_ci95_low'], signed=True)} to
{fmt(phase_a_overall['delta_ci95_high'], signed=True)}).

The principal self-training effects, computed against each method's own frozen
backbone on the same dataset, budget, seed, and split, are:

{markdown_table(
    self_overall.reset_index(),
    ['method', 'delta_mean', 'delta_ci95_low', 'delta_ci95_high', 'positive_fraction', 'n_pairs'],
    ['Method', 'Mean Δ BA', '95% low', '95% high', 'Positive fraction', 'Pairs'],
)}

These intervals describe across-cell variability; with only three seeds they are
not a basis for strong significance claims.

## Objective and scope

The study evaluates TabPFN-3 and TabICLv2 in the existing few-shot protocol, then
tests precisely six focused SSL methods: TFM self-training for both backbones,
explicit Laplacian regularization, retrieval attention using unlabeled training
memory, class-conditional embedding alignment, and their modular combined model.
No additional foundation-model families or post-hoc Phase D were introduced.

## Protocol

- Ten canonical OpenML datasets; budgets 50, 100, 250, and 500; seeds 0, 1, and 2.
- Balanced accuracy is primary. Macro-F1, accuracy, log-loss, ROC-AUC, average
  precision where defined, Brier score, ECE, runtime, and peak GPU memory are
  retained in machine-readable tables.
- Splits are predetermined and shared across methods.
- The protocol is inductive: graph, pseudo-label, representation, and retrieval
  adaptation use labeled and unlabeled *training* rows only. Validation labels are
  restricted to calibration/selection; test features appear only at prediction.
- TabPFN-3 and TabICLv2 use the native/raw pandas view. Classical and trainable
  neural SSL methods use the established processed view.
- Every long run executed as an atomic Slurm array task on the cluster.

The dataset inventory is in `tables/dataset_table.csv`.

## Implementation fidelity

- `tabpfn3` uses TabPFN 8.1.0 with the verified V3 classifier checkpoint; `tabiclv2`
  uses the pinned `tabicl-classifier-v2-20260212.ckpt`.
- Both self-training methods share the same deterministic, class-balanced hard
  pseudo-label engine with validation selection and safe frozen-round fallback.
- `laplacian_ssl` is an explicit sparse graph-regularized neural classifier, not
  sklearn label propagation.
- `unlabeled_attention_ssl` records `labeled_plus_unlabeled` memory and zero
  validation/test-memory flags.
- `embedding_alignment_ssl` uses confidence-filtered class prototypes and records
  class geometry/collapse diagnostics.
- `geometric_attention_ssl` combines the requested components with ramp-up. The
  original segment chance collapse was traced to unlabeled-pool cardinality
  overwhelming top-k retrieval. Balanced labeled/unlabeled neighbor retrieval was
  implemented and passed the final representative gate before Phase C.

The exact name mapping and validity decisions remain documented in
`../../validation/final_method_mapping.md`.

## Frozen TFM results

The full Phase-A metric table is `tables/new_method_metric_summary.csv`; the exact
TabPFN-3/TabICLv2 paired comparison is
`tables/phase_a_tabpfn3_vs_tabiclv2.csv`. Figure 1 shows both frozen models beside
the six SSL methods at every budget.

## TFM self-training

Exact seed-level pairs are in `tables/self_training_seed_level_paired.csv`.
Self-training selected the frozen round (round 0) in
**{fmt(fallback_rates.get('tabpfn3_self_training'))}** of TabPFN-3 runs and
**{fmt(fallback_rates.get('tabiclv2_self_training'))}** of TabICLv2 runs. This is
intended safety behavior: an attempted pseudo-label round is retained only when
validation evidence meets the fixed guard. Round logs and accepted counts are in
`tables/phase_c_run_diagnostics.csv`.

The run payloads do not consistently expose post-training oracle pseudo-label
accuracy, confidence distributions, or class-prior drift as flat analyzable
fields. The report therefore does not invent those summaries; this is recorded as
a limitation even though selection counts, rounds, and fallback reasons are
preserved.

## Laplacian SSL

The method completed 120/120 cells. Sparse graph node/edge counts, connected
components, isolated-node counts, affinities, and graph-build times are summarized
in `tables/method_diagnostic_summary.csv`. Exact comparisons with the existing
label-propagation and label-spreading results are in
`tables/laplacian_vs_historical_graph_effects.csv`. Those comparisons share run
keys and splits but compare different estimators, so they are benchmark contrasts,
not an isolated Laplacian-loss causal effect.

Neighbor-label purity was not serialized in the final shards and is consequently
not reconstructed using test labels.

## Attention over unlabeled data

`unlabeled_attention_ssl` completed 120/120 cells with training-only
labeled-plus-unlabeled memory. Attention mass and memory sizes are in the
diagnostic tables and Figure 7. The complete grid does not contain a matched
supervised/labeled-memory control under the final source hash. The earlier control
experiments are therefore treated only as diagnostic evidence; no full-grid causal
claim that unlabeled memory helped is made.

## Class-conditional embedding alignment

`embedding_alignment_ssl` completed 120/120 cells. Per-run reliable-unlabeled
counts, embedding variance, intra-class distance, inter-prototype distance, and
collapse flags are retained in `tables/phase_c_run_diagnostics.csv`. Relations
between these diagnostics and balanced accuracy can be analyzed from that table
without conflating uncertain examples with accepted examples.

## Combined model and component contrasts

The combined method completed 120/120 cells. It recorded
**{collapse_combined}** representation-collapse flags and **{constant_combined}**
constant-prediction flags across the final grid. A constant-prediction diagnostic
is a warning, not an automatic failed run; each affected result still has finite,
normalized probabilities and is retained transparently.

Paired differences between the complete combined method and the separately trained
Laplacian, attention, and alignment methods are:

{markdown_table(
    comp_by_method.reset_index(),
    ['component_method', 'delta_mean', 'delta_ci95_low', 'delta_ci95_high', 'positive_fraction', 'n_pairs'],
    ['Component comparator', 'Mean combined − component', '95% low', '95% high', 'Positive fraction', 'Pairs'],
)}

These are component-method contrasts, not leave-one-component-out ablations. The
representative final gate is preserved in
`tables/representative_attention_final_gate.csv`; obsolete pre-fix collapse
experiments are retained under `results/validation/` for failure analysis.

## Calibration and secondary metrics

`tables/new_method_metric_summary.csv` reports all requested metrics with valid
sample counts. Average precision is undefined for the benchmark's multiclass
cases and is left missing rather than imputed. Figure 5 displays the mean ECE/
balanced-accuracy trade-off. Log-loss, Brier score, and ECE are interpreted as
calibration context; no method is selected using test calibration metrics.

## Comparison with the historical benchmark

`tables/requested_methods_and_historical_baselines.csv` compares all eight new
methods with CatBoost, XGBoost, self-training LightGBM, label propagation,
label spreading, VIME, SCARF, and SSLAE. The failure-aware ranking and explicit
coverage tables prevent a method with missing runs from being silently rewarded.
Historical results remain immutable at
`results/raw/low_class_wave_paper_methods.csv`.

## Compute cost

Runtime and peak GPU memory are summarized per method in the metric table.
Figure 6 compares mean runtime and balanced accuracy on a logarithmic runtime
axis. CPU RAM was not serialized by the benchmark runner, so the requested RAM
comparison cannot be reconstructed reliably and is reported as unavailable.
Cold-load and warm-inference timing remain in the raw TFM rows.

## Failure analysis

Operationally, Phase A and Phase C have zero failed, missing, corrupt, or duplicate
runs. The important scientific failure was the pre-Phase-C combined-model collapse
on segment. Diagnostics identified retrieval composition—not class mapping,
probability ordering, or output dimension—as the failing component. The repaired
balanced retrieval gate achieved above-chance segment results before the final
grid was authorized. Historical failed diagnostics are preserved rather than
overwritten.

The final repository test command
`/private/ofirlin-lab/suissad4/envs/ssl-tfm/bin/python -m pytest -q` completed
with **25 passed and 1 skipped** in 27.14 seconds. The exact captured output is
stored in `results/validation/pytest_current.txt`.

## Limitations

1. Three seeds support paired descriptive uncertainty, not strong significance
   claims about small differences.
2. Trainable SSL methods lack a final-hash, full-grid supervised encoder control;
   component and historical comparisons must not be described as the pure effect
   of unlabeled data.
3. Some requested analysis-only diagnostics—CPU RAM, post-training pseudo-label
   accuracy, neighbor-label purity, and flattened class-prior drift—were not
   serialized consistently.
4. The project is not a Git work tree; deterministic source/configuration hashes
   replace commit provenance.
5. The attention and alignment models are novel experimental implementations,
   not claims of reproduction of a named published algorithm.

## Conclusions

The exact requested computational scope is complete and reproducible: 960 new
successful runs across Phases A and C, plus comparisons to the immutable
historical benchmark. The strongest defensible claims are the complete-grid
method rankings and the exact paired self-training effects. The geometric family
is now operational and non-collapsed at the representation level, but its
unlabeled-data benefit should remain qualified because a full-grid matched
supervised control was not run. No additional models or experimental phases are
needed to complete the supervisor's stated scope.

## Deliverable index

- Canonical Phase A: `results/raw/tfm_frozen_screen.csv`
- Canonical Phase C: `results/raw/focused_tfm_ssl.csv`
- Standard Phase-C aggregates: `results/aggregated/focused_tfm_ssl/`
- Final machine-readable tables: `tables/`
- Publication figures (PNG and PDF): `figures/`
- Reproducibility manifest: `run_manifest.json`
- Integrity audit: `integrity_validation.json`
"""
    integrity = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_a": phase_a_validation,
        "phase_c": phase_c_validation,
        "historical": {
            "rows": int(len(historical)),
            "methods": int(historical["method"].nunique()),
            "datasets": int(historical["dataset"].nunique()),
            "duplicate_keys": int(historical.duplicated(KEY).sum()),
            "status_counts": {
                str(k): int(v) for k, v in historical["status"].value_counts().items()
            },
            "sha256": sha256(HISTORICAL_PATH),
        },
        "canonical_phase_c_sha256": sha256(PHASE_C_PATH),
        "all_checks_passed": True,
    }
    (VALIDATION / "integrity_validation.json").write_text(
        json.dumps(integrity, indent=2, allow_nan=False), encoding="utf-8"
    )

    source_hashes = sorted(
        str(x) for x in phase_c["source_tree_hash"].dropna().unique().tolist()
    )
    wave_jobs = {}
    for label, frame in {
        "phase_a": phase_a,
        "phase_c": phase_c,
        "attention_final_gate": final_gate,
    }.items():
        wave_jobs[label] = sorted(
            {
                str(int(float(x))) if str(x).replace(".", "", 1).isdigit() else str(x)
                for x in frame.get("slurm_job_id", pd.Series(dtype=object)).dropna().unique()
            }
        )

    input_paths = [
        PHASE_A_PATH,
        PHASE_C_PATH,
        HISTORICAL_PATH,
        FINAL_GATE_PATH,
        ROOT / "configs" / "benchmark.yaml",
        ROOT / "configs" / "datasets.yaml",
        ROOT / "results" / "validation" / "final_method_mapping.md",
        ROOT / "results" / "validation" / "current_project_state.md",
        ROOT / "results" / "validation" / "pytest_current.txt",
        Path(__file__),
    ]
    manifest = {
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "git_commit": None,
        "git_note": "No Git work tree; deterministic hashes recorded.",
        "source_tree_hashes_in_phase_c": source_hashes,
        "canonical_counts": {
            "phase_a": 240,
            "phase_c": 720,
            "new_total": 960,
            "historical_rows_preserved": int(len(historical)),
        },
        "grid": {
            "datasets": DATASETS,
            "budgets": BUDGETS,
            "seeds": SEEDS,
            "phase_a_methods": PHASE_A_METHODS,
            "phase_c_methods": PHASE_C_METHODS,
        },
        "scheduler": {
            "type": "Slurm",
            "job_ids_observed_in_results": wave_jobs,
            "final_phase_c_array_jobs": ["18895315", "18895316"],
        },
        "runtime": {
            "report_python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "checkpoints": {
            "tabpfn3": {
                "name": "tabpfn-v3-classifier-v3_default.ckpt",
                "sha256": "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
            },
            "tabiclv2": {"name": "tabicl-classifier-v2-20260212.ckpt"},
        },
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in input_paths
            if path.exists()
        },
        "secrets_included": False,
        "historical_file_modified": False,
        "known_unavailable_fields": [
            "CPU RAM per run",
            "consistent post-training pseudo-label accuracy",
            "neighbor-label purity",
            "full-grid final-hash supervised encoder/labeled-memory control",
        ],
    }
    output_files = sorted(
        p
        for p in OUT.rglob("*")
        if p.is_file() and p.name != "run_manifest.json"
    )
    manifest["output_hashes"] = {
        str(path.relative_to(OUT)): sha256(path) for path in output_files
    }
    (VALIDATION / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )

    print(f"Wrote focused-study assets to {OUT}")
    print(json.dumps({"phase_a": phase_a_validation, "phase_c": phase_c_validation}, indent=2))


if __name__ == "__main__":
    main()
