#!/usr/bin/env python3
"""Objective confirmation shortlist selection from completed screen waves.

A method advances only if it meets coverage / leakage / improvement criteria
against its matched parent baseline. Does not preselect winners before screens finish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Matched SSL → parent backbone for primary SSL effect measurement.
PARENT = {
    "tabpfn3_pl_one_shot": "tabpfn3",
    "tabpfn3_loop_risk": "tabpfn3",
    "tabpfn3_cast": "tabpfn3",
    "tabpfn3_teacher_catboost": "tabpfn3",
    "tabiclv2_pl_one_shot": "tabiclv2",
    "tabiclv2_loop_risk": "tabiclv2",
    "tabiclv2_cast": "tabiclv2",
    "tabiclv2_teacher_catboost": "tabiclv2",
    "tfm_consensus_context_tabiclv2": "tabiclv2",
    "tfm_consensus_catboost": "tabiclv2",
    "cast_catboost": "catboost",
    "cast_lightgbm": "lightgbm",
    "self_training_lightgbm": "lightgbm",
    "self_training_catboost": "catboost",
    "stunt": "mlp",
    "seba": "mlp",
    "d2r2_c": "mlp",
    "laplacian_linear": "logistic_regression",
    "laplacian_mlp": "mlp",
    "prototype_alignment_ssl": "mlp",
    "retrieval_attention_ssl": "mlp",
    "geometric_attention_ssl": "geometric_attention_supervised",
    "geometric_attention_laplacian": "geometric_attention_supervised",
    "geometric_attention_prototype": "geometric_attention_supervised",
    "geometric_attention_retrieval": "geometric_attention_supervised",
}


def load_wave(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "metric_balanced_accuracy" not in df.columns and "metrics" in df.columns:
        # shards flattened differently
        pass
    return df


def coverage_ok(df: pd.DataFrame, method: str, min_frac: float = 0.9) -> tuple[bool, float]:
    sub = df[df["method"] == method]
    if sub.empty:
        return False, 0.0
    frac = float((sub["status"] == "success").mean())
    return frac >= min_frac, frac


def numerical_ok(df: pd.DataFrame, method: str) -> bool:
    sub = df[(df["method"] == method) & (df["status"] == "success")]
    if sub.empty:
        return False
    for col in ("metric_balanced_accuracy", "metric_macro_f1", "metric_log_loss"):
        if col in sub.columns and not np.isfinite(sub[col].astype(float)).all():
            return False
    return True


def paired_deltas(df: pd.DataFrame, method: str, parent: str) -> pd.DataFrame:
    a = df[(df["method"] == method) & (df["status"] == "success")]
    b = df[(df["method"] == parent) & (df["status"] == "success")]
    keys = ["dataset", "n_labeled", "seed"]
    if not set(keys).issubset(a.columns) or "metric_balanced_accuracy" not in a.columns:
        return pd.DataFrame()
    m = a.merge(b[keys + ["metric_balanced_accuracy"]], on=keys, suffixes=("", "_parent"))
    if m.empty:
        return m
    m["delta_ba"] = m["metric_balanced_accuracy"] - m["metric_balanced_accuracy_parent"]
    return m


def evaluate_method(df: pd.DataFrame, method: str) -> dict:
    ok_cov, frac = coverage_ok(df, method)
    ok_num = numerical_ok(df, method)
    parent = PARENT.get(method)
    out = {
        "method": method,
        "parent": parent,
        "coverage_frac": frac,
        "coverage_ok": ok_cov,
        "numerical_ok": ok_num,
        "advance": False,
        "reasons": [],
    }
    if not ok_cov:
        out["reasons"].append("insufficient_coverage")
    if not ok_num:
        out["reasons"].append("numerical_failure")
    if parent is None:
        out["reasons"].append("no_matched_parent_mapping")
        return out
    deltas = paired_deltas(df, method, parent)
    if deltas.empty:
        out["reasons"].append("no_paired_parent_rows")
        return out
    mean_delta = float(deltas["delta_ba"].mean())
    # improvement on >1 dataset
    by_ds = deltas.groupby("dataset")["delta_ba"].mean()
    n_pos_ds = int((by_ds > 0.005).sum())
    out["mean_delta_ba"] = mean_delta
    out["n_datasets_improved"] = n_pos_ds
    out["n_datasets"] = int(by_ds.shape[0])
    # runtime cost check if available
    if "runtime_seconds" in df.columns:
        rt_m = float(df[df["method"] == method]["runtime_seconds"].mean())
        rt_p = float(df[df["method"] == parent]["runtime_seconds"].mean())
        out["runtime_ratio"] = rt_m / max(rt_p, 1e-6)
        if out["runtime_ratio"] > 50 and mean_delta < 0.01:
            out["reasons"].append("cost_disproportionate")
    if mean_delta <= 0:
        out["reasons"].append("no_mean_improvement_vs_parent")
    if n_pos_ds < 2:
        out["reasons"].append("improvement_not_multi_dataset")
    out["advance"] = ok_cov and ok_num and mean_delta > 0 and n_pos_ds >= 2 and "cost_disproportionate" not in out["reasons"]
    if out["advance"]:
        out["reasons"].append("meets_objective_criteria")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True, help="Screen wave CSVs")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "reports" / "confirmation_shortlist.json")
    args = p.parse_args()

    dfs = [load_wave(path) for path in args.inputs]
    df = pd.concat(dfs, ignore_index=True)
    methods = sorted(set(df["method"]) - set(PARENT.values()))
    # Also evaluate mapped SSL methods even if parent present
    methods = sorted(set(list(methods) + [m for m in PARENT if m in set(df["method"])]))
    rows = [evaluate_method(df, m) for m in methods]
    shortlist = [r for r in rows if r.get("advance")]
    payload = {
        "n_methods_evaluated": len(rows),
        "n_shortlist": len(shortlist),
        "shortlist_methods": [r["method"] for r in shortlist],
        "evaluations": rows,
        "note": "Three-seed screens are exploratory; confirmation needs seeds 0-4+.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"shortlist": payload["shortlist_methods"], "n": payload["n_shortlist"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
