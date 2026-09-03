"""Aggregate per-run CSV records into coverage, metric, rank, and plot outputs.

Only successful rows contribute metric values; failure counts remain separate so
incomplete methods are visible to downstream report builders.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir, setup_logging

AGG_METRICS = [
    "metric_accuracy",
    "metric_balanced_accuracy",
    "metric_macro_f1",
    "metric_f1_macro",
    "metric_roc_auc",
    "metric_average_precision",
    "metric_log_loss",
    "runtime_seconds",
    "n_pseudo_added_total",
]
RANK_METRIC = "metric_balanced_accuracy_mean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate raw benchmark CSV files into summary tables and plots."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single raw benchmark CSV file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing raw benchmark CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=Path("results/aggregated"),
        help="Directory for aggregated outputs.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern when using --input-dir.",
    )
    return parser.parse_args()


def load_raw_results(args: argparse.Namespace) -> pd.DataFrame:
    if args.input is not None:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        return pd.read_csv(args.input)

    input_dir = args.input_dir or Path("results/raw")
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir} with pattern {args.pattern}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def _metric_columns(raw: pd.DataFrame) -> list[str]:
    cols = [col for col in AGG_METRICS if col in raw.columns and col.startswith("metric_")]
    if not cols:
        cols = [col for col in raw.columns if col.startswith("metric_")]
    return cols


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "metric_f1_macro" in out.columns and "metric_macro_f1" not in out.columns:
        out["metric_macro_f1"] = out["metric_f1_macro"]
    elif "metric_macro_f1" in out.columns and "metric_f1_macro" not in out.columns:
        out["metric_f1_macro"] = out["metric_macro_f1"]
    return out


def aggregate_mean_std(
    success: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    if success.empty:
        return pd.DataFrame(columns=group_cols)

    available = [col for col in value_cols if col in success.columns]
    grouped = (
        success.groupby(group_cols, dropna=False)[available]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([part for part in col if part]).strip("_")
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


COVERAGE_KEYS = ["dataset", "method", "n_labeled"]
COVERAGE_COLUMNS = [
    "n_expected_seeds",
    "n_success_seeds",
    "n_failed_seeds",
    "success_rate",
    "is_complete",
]
_MESSAGE_SHORT_LEN = 120


def _expected_seed_count(raw: pd.DataFrame) -> int:
    if "seed" not in raw.columns or raw.empty:
        return 0
    return int(raw["seed"].dropna().nunique())


def compute_seed_coverage(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, method, n_labeled) honest seed coverage.

    n_expected_seeds is the number of distinct seeds present anywhere in the
    raw frame (the seed dimension of the planned grid). n_success_seeds counts
    distinct seeds whose status is success for that group.
    """
    if raw.empty or not set(COVERAGE_KEYS).issubset(raw.columns):
        return pd.DataFrame(columns=COVERAGE_KEYS + COVERAGE_COLUMNS)

    expected = _expected_seed_count(raw)
    rows: list[dict] = []
    for keys, group in raw.groupby(COVERAGE_KEYS, dropna=False):
        dataset, method, n_labeled = keys
        success_seeds = int(
            group.loc[group["status"] == "success", "seed"].dropna().nunique()
        )
        n_failed = max(expected - success_seeds, 0)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_labeled": n_labeled,
                "n_expected_seeds": expected,
                "n_success_seeds": success_seeds,
                "n_failed_seeds": n_failed,
                "success_rate": (success_seeds / expected) if expected else np.nan,
                "is_complete": bool(expected > 0 and success_seeds == expected),
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_KEYS + COVERAGE_COLUMNS)


def _shorten(message: object) -> str:
    text = str(message).replace("\n", " ").strip()
    if len(text) > _MESSAGE_SHORT_LEN:
        return text[: _MESSAGE_SHORT_LEN - 3] + "..."
    return text


def build_method_coverage(raw: pd.DataFrame) -> pd.DataFrame:
    """Coverage table including failure statuses and short messages."""
    coverage = compute_seed_coverage(raw)
    if coverage.empty:
        return pd.DataFrame(
            columns=COVERAGE_KEYS
            + [
                "n_expected_seeds",
                "n_success_seeds",
                "n_failed_seeds",
                "success_rate",
                "failure_statuses",
                "failure_messages_short",
            ]
        )

    failures = raw[raw["status"] != "success"].copy()
    fail_info: dict[tuple, dict[str, str]] = {}
    if not failures.empty:
        for keys, group in failures.groupby(COVERAGE_KEYS, dropna=False):
            statuses = sorted({str(s) for s in group["status"].dropna().unique()})
            messages = sorted(
                {_shorten(m) for m in group.get("error_message", pd.Series()).dropna().unique()}
            )
            fail_info[keys] = {
                "failure_statuses": "; ".join(statuses),
                "failure_messages_short": " | ".join(messages),
            }

    out = coverage.drop(columns=["is_complete"]).copy()
    out["failure_statuses"] = out.apply(
        lambda r: fail_info.get((r["dataset"], r["method"], r["n_labeled"]), {}).get(
            "failure_statuses", ""
        ),
        axis=1,
    )
    out["failure_messages_short"] = out.apply(
        lambda r: fail_info.get((r["dataset"], r["method"], r["n_labeled"]), {}).get(
            "failure_messages_short", ""
        ),
        axis=1,
    )
    return out.sort_values(COVERAGE_KEYS)


def merge_coverage(summary: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or coverage.empty:
        return summary
    return summary.merge(coverage, on=COVERAGE_KEYS, how="left")


def build_rankings(summary_by_dataset_method_budget: pd.DataFrame) -> pd.DataFrame:
    if summary_by_dataset_method_budget.empty:
        return pd.DataFrame()

    rank_col = RANK_METRIC if RANK_METRIC in summary_by_dataset_method_budget.columns else None
    if rank_col is None:
        candidates = [
            col
            for col in summary_by_dataset_method_budget.columns
            if col.startswith("metric_balanced_accuracy")
        ]
        rank_col = candidates[0] if candidates else None
    if rank_col is None:
        return pd.DataFrame()

    ranked = summary_by_dataset_method_budget.copy()
    ranked["rank_balanced_accuracy"] = ranked.groupby(["dataset", "n_labeled"])[rank_col].rank(
        ascending=False,
        method="average",
    )
    return ranked.sort_values(["dataset", "n_labeled", "rank_balanced_accuracy"])


def build_average_ranks(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty or "rank_balanced_accuracy" not in rankings.columns:
        return pd.DataFrame()
    return (
        rankings.groupby(["method", "n_labeled"], dropna=False)["rank_balanced_accuracy"]
        .mean()
        .reset_index(name="average_rank")
        .sort_values(["n_labeled", "average_rank"])
    )


def build_rankings_all_successes(summary_with_coverage: pd.DataFrame) -> pd.DataFrame:
    """Rank using successful rows only, but flag incomplete seed coverage."""
    ranked = build_rankings(summary_with_coverage)
    if ranked.empty:
        return ranked
    if "is_complete" in ranked.columns:
        ranked["ranking_warning"] = np.where(
            ranked["is_complete"].fillna(False).astype(bool),
            "",
            "incomplete_seed_coverage",
        )
    else:
        ranked["ranking_warning"] = "incomplete_seed_coverage"
    return ranked


def build_rankings_complete_only(summary_with_coverage: pd.DataFrame) -> pd.DataFrame:
    """Headline ranking: only methods with complete seed coverage."""
    if summary_with_coverage.empty or "is_complete" not in summary_with_coverage.columns:
        return pd.DataFrame()
    complete = summary_with_coverage[
        summary_with_coverage["is_complete"].fillna(False).astype(bool)
    ].copy()
    return build_rankings(complete)


def build_best_method(complete_rankings: pd.DataFrame) -> pd.DataFrame:
    """Best method per dataset/budget from the complete-only rankings."""
    if complete_rankings.empty or "rank_balanced_accuracy" not in complete_rankings.columns:
        return pd.DataFrame()
    best = complete_rankings[complete_rankings["rank_balanced_accuracy"] == 1.0].copy()
    return best.sort_values(["dataset", "n_labeled"])


def plot_metric_by_budget(
    summary: pd.DataFrame,
    dataset: str,
    metric_col: str,
    ylabel: str,
    output_path: Path,
) -> None:
    data = summary[summary["dataset"] == dataset].copy()
    if data.empty or metric_col not in data.columns:
        return

    plt.figure(figsize=(10, 6))
    for method, group in data.groupby("method"):
        group = group.sort_values("n_labeled")
        plt.plot(group["n_labeled"], group[metric_col], marker="o", label=method)

    plt.xscale("log")
    plt.xlabel("Label budget (n_labeled)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs label budget — {dataset}")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_average_rank(average_ranks: pd.DataFrame, output_path: Path) -> None:
    if average_ranks.empty:
        return

    plt.figure(figsize=(10, 6))
    for method, group in average_ranks.groupby("method"):
        group = group.sort_values("n_labeled")
        plt.plot(group["n_labeled"], group["average_rank"], marker="o", label=method)

    plt.xscale("log")
    plt.gca().invert_yaxis()
    plt.xlabel("Label budget (n_labeled)")
    plt.ylabel("Average rank (balanced accuracy)")
    plt.title("Average rank across datasets")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_runtime_by_method(runtime_summary: pd.DataFrame, output_path: Path) -> None:
    if runtime_summary.empty:
        return

    data = runtime_summary.sort_values("runtime_seconds_mean", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(data["method"], data["runtime_seconds_mean"])
    plt.xlabel("Mean runtime (seconds)")
    plt.title("Mean runtime by method")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_plots(
    summary_by_dataset_method_budget: pd.DataFrame,
    rankings: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    plots_dir: Path,
) -> None:
    ensure_dir(plots_dir)
    datasets = sorted(summary_by_dataset_method_budget["dataset"].dropna().unique())

    bal_col = "metric_balanced_accuracy_mean"
    f1_col = "metric_macro_f1_mean"
    if f1_col not in summary_by_dataset_method_budget.columns:
        f1_col = "metric_f1_macro_mean"

    for dataset in datasets:
        plot_metric_by_budget(
            summary_by_dataset_method_budget,
            dataset,
            bal_col,
            "Balanced accuracy",
            plots_dir / f"{dataset}_balanced_accuracy_vs_budget.png",
        )
        if f1_col in summary_by_dataset_method_budget.columns:
            plot_metric_by_budget(
                summary_by_dataset_method_budget,
                dataset,
                f1_col,
                "Macro F1",
                plots_dir / f"{dataset}_macro_f1_vs_budget.png",
            )

    average_ranks = build_average_ranks(rankings)
    plot_average_rank(average_ranks, plots_dir / "average_rank_vs_budget.png")
    plot_runtime_by_method(runtime_summary, plots_dir / "runtime_by_method.png")


def aggregate_results(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw = _normalize_metric_columns(raw)
    metric_cols = _metric_columns(raw)
    value_cols = metric_cols + [
        "runtime_seconds",
        "n_pseudo_added_total",
        "pseudo_label_fraction",
    ]
    value_cols = [col for col in value_cols if col in raw.columns]

    success = raw[raw["status"] == "success"].copy()
    failures = raw[raw["status"].astype(str).str.startswith("failed")].copy()
    skipped = raw[raw["status"] == "skipped_invalid_budget"].copy()
    graph_nan = raw[raw["status"] == "failed_graph_ssl_nan"].copy()

    coverage = compute_seed_coverage(raw)
    method_coverage = build_method_coverage(raw)

    dataset_group_cols = ["dataset"]
    if "dataset_id" in success.columns:
        dataset_group_cols.append("dataset_id")
    dataset_group_cols += ["method", "n_labeled"]
    summary_by_dataset_method_budget = aggregate_mean_std(
        success,
        group_cols=dataset_group_cols,
        value_cols=value_cols,
    )
    summary_by_dataset_method_budget = merge_coverage(
        summary_by_dataset_method_budget, coverage
    )
    summary_by_method_budget = aggregate_mean_std(
        success,
        group_cols=["method", "n_labeled"],
        value_cols=value_cols,
    )
    # Headline rankings consider only complete seed coverage; the all-successes
    # variant is exploratory and carries an explicit ranking_warning.
    rankings_complete_only = build_rankings_complete_only(summary_by_dataset_method_budget)
    rankings_all_successes = build_rankings_all_successes(summary_by_dataset_method_budget)
    best_method = build_best_method(rankings_complete_only)
    runtime_by_method = aggregate_mean_std(
        success,
        group_cols=["method"],
        value_cols=["runtime_seconds"],
    )
    failure_summary = _failure_summary(failures)
    skipped_summary = _failure_summary(skipped) if not skipped.empty else pd.DataFrame()
    graph_nan_summary = _failure_summary(graph_nan) if not graph_nan.empty else pd.DataFrame()
    pseudo_label_summary = _pseudo_label_summary(success)
    ssl_vs_supervised = build_ssl_vs_supervised(success, expected_seeds=_expected_seed_count(raw))

    return {
        "summary_by_dataset_method_budget": summary_by_dataset_method_budget,
        "summary_by_method_budget": summary_by_method_budget,
        # Backward-compatible name kept as the complete-only headline ranking.
        "rankings_by_dataset_budget": rankings_complete_only,
        "rankings_by_dataset_budget_complete_only": rankings_complete_only,
        "rankings_by_dataset_budget_all_successes": rankings_all_successes,
        "best_method_by_dataset_budget": best_method,
        "method_coverage_by_dataset_budget": method_coverage,
        "runtime_by_method": runtime_by_method,
        "failures": failure_summary,
        "skipped_invalid_budget": skipped_summary,
        "failed_graph_ssl_nan": graph_nan_summary,
        "pseudo_label_summary": pseudo_label_summary,
        "ssl_vs_supervised_by_dataset_budget": ssl_vs_supervised,
    }


def _failure_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["dataset", "method", "n_labeled", "status", "error_message", "count"]
        )
    return (
        frame.groupby(
            ["dataset", "method", "n_labeled", "status", "error_message"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "method", "count"], ascending=[True, True, False])
    )


def _pseudo_label_summary(success: pd.DataFrame) -> pd.DataFrame:
    if success.empty or "pseudo_label_fraction" not in success.columns:
        return pd.DataFrame()
    ssl_methods = [
        m
        for m in success["method"].unique()
        if str(m).startswith(("self_training_", "rpl_"))
    ]
    subset = success[success["method"].isin(ssl_methods)].copy()
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(["dataset", "method", "n_labeled"], dropna=False)[
            ["pseudo_label_fraction", "n_pseudo_added_total", "runtime_seconds"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


SSL_PAIRS = [
    ("self_training_lr", "logistic_regression"),
    ("rpl_lr", "logistic_regression"),
    ("self_training_xgboost", "xgboost"),
    ("rpl_lite_xgboost", "xgboost"),
    ("self_training_lightgbm", "lightgbm"),
    ("self_training_catboost", "catboost"),
]


def build_ssl_vs_supervised(success: pd.DataFrame, expected_seeds: int = 0) -> pd.DataFrame:
    if success.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for ssl_method, sup_method in SSL_PAIRS:
        ssl = success[success["method"] == ssl_method]
        sup = success[success["method"] == sup_method]
        if ssl.empty or sup.empty:
            continue
        # Inner-merge on seed keeps only seeds where BOTH methods succeeded, so
        # comparisons are never made across mismatched seed sets.
        merged = ssl.merge(
            sup,
            on=["dataset", "n_labeled", "seed"],
            suffixes=("_ssl", "_sup"),
            how="inner",
        )
        if merged.empty:
            continue
        for _, row in merged.iterrows():
            runtime_ratio = (
                row["runtime_seconds_ssl"] / row["runtime_seconds_sup"]
                if row["runtime_seconds_sup"] not in (0, None, np.nan)
                else np.nan
            )
            rows.append(
                {
                    "dataset": row["dataset"],
                    "n_labeled": row["n_labeled"],
                    "seed": row["seed"],
                    "ssl_method": ssl_method,
                    "supervised_method": sup_method,
                    "delta_balanced_accuracy": row.get("metric_balanced_accuracy_ssl", np.nan)
                    - row.get("metric_balanced_accuracy_sup", np.nan),
                    "delta_macro_f1": row.get("metric_macro_f1_ssl", np.nan)
                    - row.get("metric_macro_f1_sup", np.nan),
                    "delta_roc_auc": row.get("metric_roc_auc_ssl", np.nan)
                    - row.get("metric_roc_auc_sup", np.nan),
                    "runtime_ratio_ssl_over_sup": runtime_ratio,
                    "pseudo_label_fraction": row.get("pseudo_label_fraction_ssl", np.nan),
                }
            )
    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["dataset", "ssl_method", "supervised_method", "n_labeled"], dropna=False)
        .agg(
            {
                "delta_balanced_accuracy": ["mean", "std"],
                "delta_macro_f1": ["mean", "std"],
                "delta_roc_auc": ["mean", "std"],
                "runtime_ratio_ssl_over_sup": ["mean"],
                "pseudo_label_fraction": ["mean"],
                "seed": ["nunique"],
            }
        )
        .reset_index()
    )
    summary.columns = [
        "_".join([part for part in col if part]).strip("_")
        for col in summary.columns.to_flat_index()
    ]
    summary = summary.rename(columns={"seed_nunique": "n_paired_seeds"})
    summary["n_expected_seeds"] = expected_seeds
    summary["pair_complete"] = (
        (expected_seeds > 0) & (summary["n_paired_seeds"] == expected_seeds)
    )
    summary["comparison_warning"] = np.where(
        summary["pair_complete"], "", "incomplete_pair_coverage"
    )
    return summary


def main() -> None:
    setup_logging()
    args = parse_args()
    ensure_dir(args.output_dir)
    plots_dir = args.output_dir / "plots"
    ensure_dir(plots_dir)

    raw = load_raw_results(args)
    outputs = aggregate_results(raw)

    paths = {
        "summary_by_dataset_method_budget.csv": outputs["summary_by_dataset_method_budget"],
        "summary_by_method_budget.csv": outputs["summary_by_method_budget"],
        "rankings_by_dataset_budget.csv": outputs["rankings_by_dataset_budget"],
        "rankings_by_dataset_budget_complete_only.csv": outputs[
            "rankings_by_dataset_budget_complete_only"
        ],
        "rankings_by_dataset_budget_all_successes.csv": outputs[
            "rankings_by_dataset_budget_all_successes"
        ],
        "best_method_by_dataset_budget.csv": outputs["best_method_by_dataset_budget"],
        "method_coverage_by_dataset_budget.csv": outputs[
            "method_coverage_by_dataset_budget"
        ],
        "runtime_by_method.csv": outputs["runtime_by_method"],
        "failures.csv": outputs["failures"],
        "skipped_invalid_budget.csv": outputs["skipped_invalid_budget"],
        "failed_graph_ssl_nan.csv": outputs["failed_graph_ssl_nan"],
        "pseudo_label_summary.csv": outputs["pseudo_label_summary"],
        "ssl_vs_supervised_by_dataset_budget.csv": outputs[
            "ssl_vs_supervised_by_dataset_budget"
        ],
    }
    for filename, frame in paths.items():
        out_path = args.output_dir / filename
        frame.to_csv(out_path, index=False)
        logging.info("Wrote %s", out_path)

    create_plots(
        outputs["summary_by_dataset_method_budget"],
        outputs["rankings_by_dataset_budget"],
        outputs["runtime_by_method"],
        plots_dir,
    )
    logging.info("Wrote plots to %s", plots_dir)


if __name__ == "__main__":
    main()
