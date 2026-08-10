"""Command-line health summary for a raw benchmark result CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark run CSV health.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw benchmark CSV file.",
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    setup_logging()
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    print_section("Overview")
    print(f"rows: {len(df)}")
    print("status counts:")
    print(df["status"].value_counts(dropna=False).to_string())

    print_section("Validation strategy counts")
    if "validation_strategy" in df.columns:
        print(df["validation_strategy"].value_counts(dropna=False).to_string())
    else:
        print("column not present")

    print_section("Duplicate run keys (dataset, method, seed, n_labeled)")
    key_cols = ["dataset", "method", "seed", "n_labeled"]
    if all(col in df.columns for col in key_cols):
        dup = (
            df.groupby(key_cols, dropna=False)
            .size()
            .reset_index(name="count")
            .query("count > 1")
            .sort_values("count", ascending=False)
        )
        if dup.empty:
            print("none")
        else:
            print(dup.to_string(index=False))
    else:
        print("key columns not present")

    print_section("Failures by dataset / method / error")
    failures = df[df["status"].astype(str).str.startswith("failed")]
    if failures.empty:
        print("none")
    else:
        grouped = (
            failures.groupby(["dataset", "method", "error_message"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print(grouped.head(30).to_string(index=False))

    print_section("Average runtime by method")
    if "runtime_seconds" in df.columns:
        print(
            df.groupby("method")["runtime_seconds"]
            .mean()
            .sort_values(ascending=False)
            .to_string()
        )

    print_section("Slowest runs")
    if "runtime_seconds" in df.columns:
        cols = [
            c
            for c in [
                "dataset",
                "method",
                "seed",
                "n_labeled",
                "runtime_seconds",
                "status",
            ]
            if c in df.columns
        ]
        print(
            df.sort_values("runtime_seconds", ascending=False)[cols]
            .head(15)
            .to_string(index=False)
        )

    print_section("Pseudo-label fraction by SSL method")
    if "pseudo_label_fraction" in df.columns:
        ssl = df[df["method"].astype(str).str.startswith(("self_training_", "rpl_"))]
        if ssl.empty:
            print("no SSL rows")
        else:
            print(
                ssl.groupby("method")["pseudo_label_fraction"]
                .agg(["mean", "max", "count"])
                .sort_values("mean", ascending=False)
                .to_string()
            )

    print_section("Graph SSL neighbors used")
    if "graph_n_neighbors_used" in df.columns:
        graph = df[df["method"].isin(["label_spreading", "label_propagation"])].copy()
        if graph.empty:
            print("no graph rows")
        else:
            cols = ["dataset", "method", "seed", "n_labeled", "status", "graph_n_neighbors_used", "graph_retry_count", "graph_n_rows", "graph_n_unlabeled_used"]
            cols = [c for c in cols if c in graph.columns]
            print(graph[cols].to_string(index=False))
    else:
        print("column not present")


if __name__ == "__main__":
    main()
