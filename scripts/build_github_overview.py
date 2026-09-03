#!/usr/bin/env python3
"""Build the repository-level summary of all canonical benchmark results.

The project was executed in three validated waves.  This script deliberately
reads their canonical, immutable CSVs instead of scraping report tables.  It
combines the 17 historical baselines, two frozen tabular foundation models,
and six focused SSL methods into one 25-method view suitable for the README.

Run from the repository root with the ``ssl-tfm`` environment::

    python scripts/build_github_overview.py

Generated tables and figures are deterministic given the canonical inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
REPORT_DIR = ROOT / "results" / "reports" / "main_report"
OUTPUT_DIR = REPORT_DIR / "tables" / "overview"
FIGURE_DIR = REPORT_DIR / "figures" / "overview"

CANONICAL_INPUTS = {
    "historical_baselines": RAW_DIR / "low_class_wave_paper_methods.csv",
    "frozen_foundation_models": RAW_DIR / "tfm_frozen_screen.csv",
    "focused_ssl": RAW_DIR / "focused_tfm_ssl.csv",
}

EXPECTED_ROWS = {
    "historical_baselines": 2040,
    "frozen_foundation_models": 240,
    "focused_ssl": 720,
}

EXPECTED_METHOD_COUNTS = {
    "historical_baselines": 17,
    "frozen_foundation_models": 2,
    "focused_ssl": 6,
}

# All three canonical waves share this grid.  Keeping the list explicit makes
# accidental comparisons against a future partial wave fail visibly instead
# of silently changing the denominator.
COMMON_COMPARISON_BUDGETS = (50, 100, 250, 500)

DATASET_ORDER = (
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
)

DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "MLP",
    "label_spreading": "Label spreading",
    "label_propagation": "Label propagation",
    "self_training_lr": "Self-training LR",
    "self_training_xgboost": "Self-training XGBoost",
    "self_training_lightgbm": "Self-training LightGBM",
    "self_training_catboost": "Self-training CatBoost",
    "rpl_lr": "RPL LR",
    "rpl_lite_xgboost": "RPL-lite XGBoost",
    "sslae": "SSLAE",
    "vime": "VIME",
    "scarf": "SCARF",
    "tabpfn3": "TabPFN-3",
    "tabiclv2": "TabICL v2",
    "tabpfn3_self_training": "TabPFN-3 + self-training",
    "tabiclv2_self_training": "TabICL v2 + self-training",
    "laplacian_ssl": "Laplacian SSL",
    "unlabeled_attention_ssl": "Unlabeled attention SSL",
    "embedding_alignment_ssl": "Embedding alignment SSL",
    "geometric_attention_ssl": "Combined geometric + attention SSL",
}

FAMILIES = {
    "logistic_regression": "supervised",
    "random_forest": "supervised",
    "xgboost": "supervised",
    "lightgbm": "supervised",
    "catboost": "supervised",
    "mlp": "supervised",
    "label_spreading": "graph SSL",
    "label_propagation": "graph SSL",
    "self_training_lr": "classical self-training",
    "self_training_xgboost": "classical self-training",
    "self_training_lightgbm": "classical self-training",
    "self_training_catboost": "classical self-training",
    "rpl_lr": "robust pseudo-labeling",
    "rpl_lite_xgboost": "robust pseudo-labeling",
    "sslae": "neural SSL",
    "vime": "neural SSL",
    "scarf": "neural SSL",
    "tabpfn3": "frozen TFM",
    "tabiclv2": "frozen TFM",
    "tabpfn3_self_training": "TFM self-training",
    "tabiclv2_self_training": "TFM self-training",
    "laplacian_ssl": "geometric SSL",
    "unlabeled_attention_ssl": "geometric SSL",
    "embedding_alignment_ssl": "geometric SSL",
    "geometric_attention_ssl": "geometric SSL",
}

FAMILY_COLORS = {
    "supervised": "#6B7280",
    "graph SSL": "#9CA3AF",
    "classical self-training": "#A78BFA",
    "robust pseudo-labeling": "#C084FC",
    "neural SSL": "#F59E0B",
    "frozen TFM": "#2563EB",
    "TFM self-training": "#0EA5E9",
    "geometric SSL": "#10B981",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help="Figure destination; defaults to the canonical main-report figure directory.",
    )
    return parser.parse_args()


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all three waves and enforce the frozen benchmark invariants."""
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for wave, path in CANONICAL_INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical input: {path}")

        frame = pd.read_csv(path)
        if len(frame) != EXPECTED_ROWS[wave]:
            raise ValueError(
                f"{wave}: expected {EXPECTED_ROWS[wave]} rows, found {len(frame)}"
            )
        if frame["method"].nunique() != EXPECTED_METHOD_COUNTS[wave]:
            raise ValueError(
                f"{wave}: expected {EXPECTED_METHOD_COUNTS[wave]} methods, "
                f"found {frame['method'].nunique()}"
            )

        required = {
            "dataset",
            "method",
            "seed",
            "n_labeled",
            "status",
            "runtime_seconds",
            "metric_balanced_accuracy",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{wave}: missing columns {sorted(missing)}")

        duplicated = frame.duplicated(
            ["dataset", "method", "seed", "n_labeled"], keep=False
        )
        if duplicated.any():
            raise ValueError(f"{wave}: duplicate experimental keys found")

        frame = frame.copy()
        frame["wave"] = wave
        success = frame["status"].eq("success")
        coverage_rows.append(
            {
                "wave": wave,
                "source_file": path.relative_to(ROOT).as_posix(),
                "rows": len(frame),
                "methods": frame["method"].nunique(),
                "datasets": frame["dataset"].nunique(),
                "label_budgets": frame["n_labeled"].nunique(),
                "seeds": frame["seed"].nunique(),
                "successful_rows": int(success.sum()),
                "preserved_failures": int((~success).sum()),
            }
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    unexpected = set(combined["method"]) - set(DISPLAY_NAMES)
    missing = set(DISPLAY_NAMES) - set(combined["method"])
    if unexpected or missing:
        raise ValueError(
            f"Method registry mismatch: unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )

    coverage = pd.DataFrame(coverage_rows)
    return combined, coverage


def build_method_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Create a failure-aware summary with equal-weight benchmark-cell ranks."""
    successful = combined.loc[
        combined["status"].eq("success")
        & combined["metric_balanced_accuracy"].notna()
        & combined["n_labeled"].isin(COMMON_COMPARISON_BUDGETS)
    ].copy()

    cell_means = (
        successful.groupby(["dataset", "n_labeled", "method"], as_index=False)
        .agg(
            balanced_accuracy=("metric_balanced_accuracy", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
            successful_seeds=("seed", "nunique"),
        )
    )
    # Reindex to the complete 40-cell × 25-method design before ranking.  A
    # failed/missing cell receives one rank below the worst possible successful
    # rank.  This prevents incomplete methods from improving their mean rank by
    # disappearing on difficult cells, while their accuracy mean still reports
    # only actual successful observations.
    full_index = pd.MultiIndex.from_product(
        [
            sorted(combined["dataset"].unique()),
            list(COMMON_COMPARISON_BUDGETS),
            sorted(DISPLAY_NAMES),
        ],
        names=["dataset", "n_labeled", "method"],
    )
    cell_means = (
        cell_means.set_index(["dataset", "n_labeled", "method"])
        .reindex(full_index)
        .reset_index()
    )
    cell_means["cell_rank"] = cell_means.groupby(
        ["dataset", "n_labeled"]
    )["balanced_accuracy"].rank(method="average", ascending=False)
    cell_means["cell_rank"] = cell_means["cell_rank"].fillna(
        len(DISPLAY_NAMES) + 1
    )

    summary = (
        cell_means.groupby("method", as_index=False)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy_across_cells=("balanced_accuracy", "std"),
            mean_cell_rank=("cell_rank", "mean"),
            median_cell_rank=("cell_rank", "median"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            evaluated_cells=("balanced_accuracy", "count"),
            ranked_cells=("cell_rank", "size"),
        )
    )

    attempted = combined.groupby("method").size().rename("attempted_runs")
    succeeded = (
        combined.loc[combined["status"].eq("success")]
        .groupby("method")
        .size()
        .rename("successful_runs")
    )
    summary = summary.join(attempted, on="method").join(succeeded, on="method")
    summary["successful_runs"] = summary["successful_runs"].fillna(0).astype(int)
    summary["failed_runs"] = summary["attempted_runs"] - summary["successful_runs"]
    summary["coverage_fraction"] = (
        summary["successful_runs"] / summary["attempted_runs"]
    )
    summary["display_name"] = summary["method"].map(DISPLAY_NAMES)
    summary["family"] = summary["method"].map(FAMILIES)
    summary["overall_rank"] = summary["mean_cell_rank"].rank(
        method="min", ascending=True
    ).astype(int)

    columns = [
        "overall_rank",
        "method",
        "display_name",
        "family",
        "mean_cell_rank",
        "median_cell_rank",
        "mean_balanced_accuracy",
        "std_balanced_accuracy_across_cells",
        "mean_runtime_seconds",
        "evaluated_cells",
        "ranked_cells",
        "attempted_runs",
        "successful_runs",
        "failed_runs",
        "coverage_fraction",
    ]
    return summary[columns].sort_values(
        ["mean_cell_rank", "mean_balanced_accuracy"], ascending=[True, False]
    )


def selected_budget_summary(combined: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "tabiclv2_self_training",
        "tabiclv2",
        "tabpfn3_self_training",
        "tabpfn3",
        "catboost",
        "xgboost",
        "sslae",
        "unlabeled_attention_ssl",
    ]
    successful = combined.loc[
        combined["status"].eq("success")
        & combined["method"].isin(selected)
        & combined["n_labeled"].isin(COMMON_COMPARISON_BUDGETS)
    ].copy()
    budget = (
        successful.groupby(["method", "n_labeled"], as_index=False)
        .agg(
            mean_balanced_accuracy=("metric_balanced_accuracy", "mean"),
            standard_error=(
                "metric_balanced_accuracy",
                lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()),
            ),
            runs=("metric_balanced_accuracy", "count"),
        )
    )
    budget["display_name"] = budget["method"].map(DISPLAY_NAMES)
    return budget


def complete_comparison_tables(
    combined: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build readable long-form sources for the two all-method heatmaps."""
    method_order = summary.sort_values("overall_rank")["method"].tolist()
    successful = combined.loc[
        combined["status"].eq("success")
        & combined["metric_balanced_accuracy"].notna()
    ].copy()

    by_budget = (
        successful.groupby(["method", "n_labeled"], as_index=False)
        .agg(
            mean_balanced_accuracy=("metric_balanced_accuracy", "mean"),
            successful_runs=("metric_balanced_accuracy", "count"),
        )
    )
    attempts_budget = (
        combined.groupby(["method", "n_labeled"])
        .size()
        .rename("attempted_runs")
    )
    by_budget = by_budget.join(
        attempts_budget,
        on=["method", "n_labeled"],
    )
    by_budget["display_name"] = by_budget["method"].map(DISPLAY_NAMES)
    by_budget["family"] = by_budget["method"].map(FAMILIES)
    by_budget["method_order"] = pd.Categorical(
        by_budget["method"], categories=method_order, ordered=True
    )
    by_budget = by_budget.sort_values(["method_order", "n_labeled"]).drop(
        columns="method_order"
    )

    cell = (
        successful.groupby(["method", "dataset", "n_labeled"], as_index=False)
        .agg(
            balanced_accuracy=("metric_balanced_accuracy", "mean"),
            successful_runs=("metric_balanced_accuracy", "count"),
        )
    )
    full_index = pd.MultiIndex.from_product(
        [method_order, DATASET_ORDER, COMMON_COMPARISON_BUDGETS],
        names=["method", "dataset", "n_labeled"],
    )
    cell = (
        cell.set_index(["method", "dataset", "n_labeled"])
        .reindex(full_index)
        .reset_index()
    )
    cell["cell_rank"] = cell.groupby(["dataset", "n_labeled"])[
        "balanced_accuracy"
    ].rank(method="average", ascending=False)
    cell["cell_rank"] = cell["cell_rank"].fillna(len(method_order) + 1)
    by_dataset = (
        cell.groupby(["method", "dataset"], as_index=False)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_cell_rank=("cell_rank", "mean"),
            evaluated_budget_cells=("balanced_accuracy", "count"),
        )
    )
    by_dataset["display_name"] = by_dataset["method"].map(DISPLAY_NAMES)
    by_dataset["family"] = by_dataset["method"].map(FAMILIES)
    by_dataset["method_order"] = pd.Categorical(
        by_dataset["method"], categories=method_order, ordered=True
    )
    by_dataset["dataset_order"] = pd.Categorical(
        by_dataset["dataset"], categories=DATASET_ORDER, ordered=True
    )
    by_dataset = by_dataset.sort_values(["method_order", "dataset_order"]).drop(
        columns=["method_order", "dataset_order"]
    )
    return by_budget, by_dataset


def plot_overview(
    summary: pd.DataFrame,
    budget: pd.DataFrame,
    combined: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Render one readable composite rather than a gallery of redundant plots."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(15, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1, 1])
    ax_rank = fig.add_subplot(grid[:, 0])
    ax_budget = fig.add_subplot(grid[0, 1])
    ax_tradeoff = fig.add_subplot(grid[1, 1])

    ranked = summary.sort_values("mean_cell_rank", ascending=False)
    colors = [FAMILY_COLORS[f] for f in ranked["family"]]
    ax_rank.barh(ranked["display_name"], ranked["mean_cell_rank"], color=colors)
    ax_rank.set_xlabel("Mean rank within dataset × label-budget cell (lower is better)")
    ax_rank.set_title("A  All 25 canonical methods")
    ax_rank.grid(axis="x", alpha=0.2)
    ax_rank.set_xlim(left=0)

    line_colors = {
        "tabiclv2_self_training": "#0369A1",
        "tabiclv2": "#2563EB",
        "tabpfn3_self_training": "#0891B2",
        "tabpfn3": "#60A5FA",
        "catboost": "#4B5563",
        "xgboost": "#9CA3AF",
        "sslae": "#D97706",
        "unlabeled_attention_ssl": "#059669",
    }
    for method, frame in budget.groupby("method"):
        frame = frame.sort_values("n_labeled")
        ax_budget.plot(
            frame["n_labeled"],
            frame["mean_balanced_accuracy"],
            marker="o",
            linewidth=1.8,
            color=line_colors[method],
            label=DISPLAY_NAMES[method],
        )
    ax_budget.set_xscale("log")
    ax_budget.set_xticks(
        list(COMMON_COMPARISON_BUDGETS),
        labels=[str(value) for value in COMMON_COMPARISON_BUDGETS],
    )
    ax_budget.xaxis.set_minor_formatter(NullFormatter())
    ax_budget.set_xlabel("Labeled examples")
    ax_budget.set_ylabel("Mean balanced accuracy")
    ax_budget.set_title("B  Performance at four annotation budgets")
    ax_budget.grid(alpha=0.2)
    ax_budget.legend(fontsize=7.2, ncol=2, frameon=False, loc="lower right")

    selected_methods = set(line_colors)
    cell_level = (
        combined.loc[
            combined["status"].eq("success")
            & combined["method"].isin(selected_methods)
            & combined["n_labeled"].isin(COMMON_COMPARISON_BUDGETS)
        ]
        .groupby(["method", "dataset", "n_labeled"], as_index=False)
        .agg(
            balanced_accuracy=("metric_balanced_accuracy", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
        )
        .groupby("method", as_index=False)
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
        )
    )
    for row in cell_level.itertuples(index=False):
        ax_tradeoff.scatter(
            row.runtime_seconds,
            row.balanced_accuracy,
            s=70,
            color=line_colors[row.method],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax_tradeoff.annotate(
            DISPLAY_NAMES[row.method],
            (row.runtime_seconds, row.balanced_accuracy),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.1,
        )
    ax_tradeoff.set_xscale("log")
    ax_tradeoff.set_xlabel("Mean runtime per run (seconds, log scale)")
    ax_tradeoff.set_ylabel("Mean balanced accuracy")
    ax_tradeoff.set_title("C  Accuracy–runtime trade-off (selected methods)")
    ax_tradeoff.grid(alpha=0.2)

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=7, label=family)
        for family, color in FAMILY_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        title="Method family",
    )
    fig.suptitle(
        "Tabular SSL benchmark: complete canonical result set",
        fontsize=16,
        fontweight="bold",
    )

    png_path = output_dir / "project_overview.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def plot_complete_method_matrix(
    summary: pd.DataFrame,
    by_budget: pd.DataFrame,
    by_dataset: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot every canonical method across every budget and every dataset."""
    method_order = summary.sort_values("overall_rank")["method"].tolist()
    labels = [DISPLAY_NAMES[method] for method in method_order]

    budget_matrix = (
        by_budget.pivot(
            index="method",
            columns="n_labeled",
            values="mean_balanced_accuracy",
        )
        .reindex(index=method_order, columns=COMMON_COMPARISON_BUDGETS)
    )
    budget_success = (
        by_budget.pivot(index="method", columns="n_labeled", values="successful_runs")
        .reindex(index=method_order, columns=COMMON_COMPARISON_BUDGETS)
    )
    budget_attempts = (
        by_budget.pivot(index="method", columns="n_labeled", values="attempted_runs")
        .reindex(index=method_order, columns=COMMON_COMPARISON_BUDGETS)
    )
    dataset_rank_matrix = (
        by_dataset.pivot(index="method", columns="dataset", values="mean_cell_rank")
        .reindex(index=method_order, columns=DATASET_ORDER)
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
        }
    )
    fig, (ax_budget, ax_dataset) = plt.subplots(
        1,
        2,
        figsize=(18, 13),
        gridspec_kw={"width_ratios": [0.72, 1.55]},
        sharey=True,
        constrained_layout=True,
    )

    budget_image = ax_budget.imshow(
        budget_matrix.to_numpy(),
        cmap="Blues",
        vmin=0.45,
        vmax=0.80,
        aspect="auto",
    )
    ax_budget.set_xticks(
        range(len(COMMON_COMPARISON_BUDGETS)),
        labels=[str(value) for value in COMMON_COMPARISON_BUDGETS],
    )
    ax_budget.set_yticks(range(len(labels)), labels=labels)
    ax_budget.set_xlabel("Labeled examples")
    ax_budget.set_title("A  Mean balanced accuracy by label budget")
    for row in range(budget_matrix.shape[0]):
        for column in range(budget_matrix.shape[1]):
            value = budget_matrix.iat[row, column]
            incomplete = budget_success.iat[row, column] < budget_attempts.iat[row, column]
            suffix = "†" if incomplete else ""
            color = "white" if value >= 0.66 else "#1F2937"
            ax_budget.text(
                column,
                row,
                f"{value:.3f}{suffix}",
                ha="center",
                va="center",
                fontsize=7.1,
                color=color,
            )
    budget_colorbar = fig.colorbar(
        budget_image,
        ax=ax_budget,
        fraction=0.045,
        pad=0.025,
    )
    budget_colorbar.set_label("Balanced accuracy")

    dataset_image = ax_dataset.imshow(
        dataset_rank_matrix.to_numpy(),
        cmap="viridis",
        vmin=1,
        vmax=len(method_order) + 1,
        aspect="auto",
    )
    ax_dataset.set_xticks(
        range(len(DATASET_ORDER)),
        labels=DATASET_ORDER,
        rotation=36,
        ha="right",
        rotation_mode="anchor",
    )
    ax_dataset.tick_params(axis="y", labelleft=False)
    ax_dataset.set_xlabel("Dataset")
    ax_dataset.set_title("B  Mean within-cell rank by dataset (lower is better)")
    for row in range(dataset_rank_matrix.shape[0]):
        for column in range(dataset_rank_matrix.shape[1]):
            value = dataset_rank_matrix.iat[row, column]
            color = "white" if value <= 10 else "#111827"
            ax_dataset.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color=color,
            )
    dataset_colorbar = fig.colorbar(
        dataset_image,
        ax=ax_dataset,
        fraction=0.025,
        pad=0.02,
    )
    dataset_colorbar.set_label("Mean cell rank; failures receive rank 26")

    for axis in (ax_budget, ax_dataset):
        axis.set_xticks(np.arange(-0.5, len(axis.get_xticks()), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.8)
        axis.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(
        "Complete comparison: all 25 methods, 4 budgets, and 10 datasets",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "† Incomplete historical graph-SSL coverage; metric means use successful runs, while ranks penalize failed cells.",
        fontsize=8,
        color="#374151",
    )

    png_path = output_dir / "complete_method_matrix.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figure_dir = (
        args.figure_dir.resolve()
        if args.figure_dir is not None
        else (FIGURE_DIR.resolve() if output_dir == OUTPUT_DIR.resolve() else output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    combined, coverage = load_and_validate()
    summary = build_method_summary(combined)
    budget = selected_budget_summary(combined)
    all_budget, all_dataset = complete_comparison_tables(combined, summary)

    coverage.to_csv(output_dir / "benchmark_coverage.csv", index=False)
    summary.to_csv(output_dir / "all_method_summary.csv", index=False)
    budget.to_csv(output_dir / "selected_methods_by_budget.csv", index=False)
    all_budget.to_csv(output_dir / "all_methods_by_budget.csv", index=False)
    all_dataset.to_csv(output_dir / "all_methods_by_dataset.csv", index=False)

    png_path = plot_overview(summary, budget, combined, figure_dir)
    matrix_png = plot_complete_method_matrix(
        summary,
        all_budget,
        all_dataset,
        figure_dir,
    )
    top = summary.iloc[0]
    manifest = {
        "canonical_rows": int(len(combined)),
        "canonical_methods": int(combined["method"].nunique()),
        "datasets": int(combined["dataset"].nunique()),
        "label_budgets": sorted(int(v) for v in combined["n_labeled"].unique()),
        "common_comparison_label_budgets": list(COMMON_COMPARISON_BUDGETS),
        "successful_runs": int(combined["status"].eq("success").sum()),
        "preserved_failures": int((~combined["status"].eq("success")).sum()),
        "top_mean_cell_rank_method": str(top["method"]),
        "top_mean_cell_rank": float(top["mean_cell_rank"]),
        "inputs": {
            wave: path.relative_to(ROOT).as_posix()
            for wave, path in CANONICAL_INPUTS.items()
        },
        "outputs": [
            "benchmark_coverage.csv",
            "all_method_summary.csv",
            "selected_methods_by_budget.csv",
            "all_methods_by_budget.csv",
            "all_methods_by_dataset.csv",
            png_path.name,
            matrix_png.name,
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
