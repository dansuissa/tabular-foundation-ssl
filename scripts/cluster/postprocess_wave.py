#!/usr/bin/env python3
"""Post-process a completed wave: validate, merge CSV, aggregate, report scaffold."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FORBIDDEN = "low_class_wave_paper_methods.csv"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wave", required=True)
    p.add_argument("--expected", type=int, default=None)
    args = p.parse_args()

    if args.wave == "low_class_wave_paper_methods" or FORBIDDEN in args.wave:
        raise SystemExit("Refusing to touch historical canonical wave")

    project_raw = ROOT / "results" / "raw" / f"{args.wave}.csv"
    agg_dir = ROOT / "results" / "aggregated" / args.wave
    report_dir = ROOT / "results" / "reports" / args.wave
    agg_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Collect
    collect = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "cluster" / "collect_wave.py"),
            "--wave",
            args.wave,
            "--output",
            str(project_raw),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    print(collect.stdout)
    if collect.returncode != 0:
        print(collect.stderr, file=sys.stderr)
        return collect.returncode

    df = pd.read_csv(project_raw)
    key = ["dataset", "method", "seed", "n_labeled"]
    n_dup = int(df.duplicated(subset=key).sum())
    status_counts = df["status"].value_counts(dropna=False).to_dict() if "status" in df else {}
    expected = args.expected
    task_map = Path(os.environ.get("SSL_SHARD_ROOT", "/private/ofirlin-lab/suissad4/results/raw_shards")) / args.wave / "task_map.json"
    if expected is None and task_map.exists():
        expected = len(json.loads(task_map.read_text()))

    # NaN/inf metric scan
    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    nan_inf = {}
    for c in metric_cols:
        vals = pd.to_numeric(df.loc[df["status"] == "success", c], errors="coerce")
        nan_inf[c] = {"nan": int(vals.isna().sum()), "inf": int(np.isinf(vals.fillna(0)).sum())}

    coverage = (
        df.groupby(["dataset", "method", "n_labeled"])["seed"].nunique().reset_index(name="n_seeds")
        if set(["dataset", "method", "n_labeled", "seed"]).issubset(df.columns)
        else pd.DataFrame()
    )
    coverage.to_csv(agg_dir / "coverage_by_cell.csv", index=False)

    summary = {
        "wave": args.wave,
        "n_rows": int(len(df)),
        "expected": expected,
        "duplicate_keys": n_dup,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "nan_inf_metrics": nan_inf,
        "methods": sorted(df["method"].unique().tolist()) if "method" in df else [],
        "datasets": sorted(df["dataset"].unique().tolist()) if "dataset" in df else [],
    }
    (report_dir / "wave_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Aggregate via existing tool when successful rows exist
    if (df.get("status") == "success").any() if "status" in df else len(df):
        aggregate = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.aggregate_results",
                "--input",
                str(project_raw),
                "--output_dir",
                str(agg_dir),
            ],
            cwd=str(ROOT),
            check=False,
        )
        if aggregate.returncode != 0:
            print(
                f"aggregation failed with exit code {aggregate.returncode}",
                file=sys.stderr,
            )
            return aggregate.returncode

    # Ranking tables
    if "metric_balanced_accuracy" in df.columns:
        ok = df[df["status"] == "success"] if "status" in df else df
        rank = (
            ok.groupby("method")["metric_balanced_accuracy"]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False)
        )
        rank.to_csv(report_dir / "balanced_accuracy_ranking.csv")
        if "metric_macro_f1" in ok.columns:
            ok.groupby("method")["metric_macro_f1"].mean().sort_values(ascending=False).to_csv(
                report_dir / "macro_f1_ranking.csv"
            )
        if "runtime_seconds" in ok.columns:
            ok.groupby("method")["runtime_seconds"].agg(["mean", "max"]).to_csv(
                report_dir / "runtime_cost.csv"
            )

    print(json.dumps(summary, indent=2))
    print("wrote", project_raw)
    print("wrote", report_dir)
    return 0 if n_dup == 0 and (expected is None or len(df) == expected or len(df) <= expected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
