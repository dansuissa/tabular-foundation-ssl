#!/usr/bin/env python3
"""Profile representative TFM/SSL jobs for resource planning.

Runs a small set of profile cells (separate jobs or local) and writes:
  results/validation/resource_profile.csv
  docs/tfm_ssl_resource_plan.md

Does not guess after measurements exist.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROFILE_CELLS = [
    {"dataset": "phoneme", "method": "logistic_regression", "seed": 0, "budget": 50, "hint_profile": "cpu_light"},
    {"dataset": "phoneme", "method": "sslae", "seed": 0, "budget": 50, "hint_profile": "gpu_representation"},
    {"dataset": "phoneme", "method": "tabiclv2", "seed": 0, "budget": 50, "hint_profile": "gpu_tfm_small"},
    {"dataset": "phoneme", "method": "tabpfn3", "seed": 0, "budget": 50, "hint_profile": "gpu_tfm_small"},
    {"dataset": "segment", "method": "laplacian_mlp", "seed": 0, "budget": 50, "hint_profile": "gpu_representation"},
    {"dataset": "segment", "method": "geometric_attention_ssl", "seed": 0, "budget": 50, "hint_profile": "gpu_representation"},
    {"dataset": "adult", "method": "tabiclv2", "seed": 0, "budget": 100, "hint_profile": "gpu_tfm_large"},
    {"dataset": "jannis", "method": "tabiclv2", "seed": 0, "budget": 100, "hint_profile": "gpu_tfm_large"},
    {"dataset": "jannis", "method": "stunt", "seed": 0, "budget": 50, "hint_profile": "gpu_representation"},
]


def peak_rss_mb() -> float:
    # Linux: ru_maxrss is kilobytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def profile_one(cell: dict) -> dict:
    from src.data import dataset_from_config, load_dataset
    from src.run_benchmark import resolve_neural_params, run_single_experiment
    from src.utils import load_yaml, set_seed
    from src.models.neural_ssl import NEURAL_SSL_METHODS
    from src.models.registry_ext import EXTENDED_METHODS

    row = {
        **cell,
        "status": "pending",
        "load_s": None,
        "train_s": None,
        "infer_s": None,
        "total_s": None,
        "peak_rss_mb": None,
        "peak_vram_mb": None,
        "gpu_name": None,
        "error": None,
        "output_bytes": None,
    }
    try:
        import torch

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            row["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        use_cuda = False

    t0 = time.time()
    set_seed(cell["seed"])
    datasets_cfg = load_yaml(ROOT / "configs" / "datasets.yaml")
    benchmark_cfg = load_yaml(ROOT / "configs" / "benchmark.yaml")
    entry = next(e for e in datasets_cfg["datasets"] if e["name"] == cell["dataset"])
    t_load0 = time.time()
    dataset = load_dataset(dataset_from_config(entry))
    row["load_s"] = round(time.time() - t_load0, 3)

    neural_params = (
        resolve_neural_params(benchmark_cfg, cell["method"])
        if cell["method"] in NEURAL_SSL_METHODS
        else None
    )
    method_params = (
        dict(benchmark_cfg.get(cell["method"]) or {})
        if cell["method"] in EXTENDED_METHODS or cell["method"] in NEURAL_SSL_METHODS
        else None
    )

    t_train0 = time.time()
    result = run_single_experiment(
        dataset=dataset,
        method=cell["method"],
        seed=cell["seed"],
        label_budget=cell["budget"],
        test_size=benchmark_cfg["test_size"],
        val_size_from_labeled=benchmark_cfg["val_size_from_labeled"],
        metric_names=benchmark_cfg["metrics"],
        neural_params=neural_params,
        method_params=method_params,
    )
    row["train_s"] = round(time.time() - t_train0, 3)
    row["total_s"] = round(time.time() - t0, 3)
    row["status"] = result.get("status")
    row["error"] = result.get("error_message")
    row["runtime_seconds"] = result.get("runtime_seconds")
    row["peak_rss_mb"] = round(peak_rss_mb(), 1)
    if use_cuda:
        import torch

        row["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    # rough serialize size
    blob = json.dumps(result, default=str)
    row["output_bytes"] = len(blob.encode())
    return row


def choose_profile(row: dict) -> dict:
    """Map measured usage → scheduler request with safety margin."""
    total = float(row.get("total_s") or 0)
    vram = float(row.get("peak_vram_mb") or 0)
    rss = float(row.get("peak_rss_mb") or 0)
    method = row.get("method", "")

    # walltime: 3× observed + 10 min floor, capped reasonably
    wall_s = max(600, int(total * 3) + 600)
    hours = max(1, (wall_s + 3599) // 3600)
    time_str = f"{hours}:00:00"

    mem_gb = max(16, int((rss * 2.5) / 1024) + 8)
    need_gpu = vram > 0 or method.startswith(("tabpfn", "tabicl", "tfm_", "stunt", "seba", "d2r2", "geometric", "laplacian", "vime", "scarf", "sslae", "prototype", "retrieval"))

    if not need_gpu and "tab" not in method:
        name = "cpu_tree_ssl" if "catboost" in method or "lightgbm" in method or "xgboost" in method or "self_training" in method else "cpu_light"
        return {
            "profile": name,
            "partition": "cpu192G-48h",
            "gres": "",
            "cpus": 8 if name == "cpu_tree_ssl" else 4,
            "mem": f"{mem_gb}G",
            "time": time_str,
        }

    if vram >= 20000 or row.get("dataset") in {"adult", "jannis", "electricity"}:
        name = "gpu_tfm_large"
        mem_gb = max(mem_gb, 96)
        hours = max(hours, 4)
    elif method.startswith(("tabpfn", "tabicl", "tfm_")):
        name = "gpu_tfm_small"
        mem_gb = max(mem_gb, 64)
    else:
        name = "gpu_representation"
        mem_gb = max(mem_gb, 64)

    return {
        "profile": name,
        "partition": "p_uriofir",
        "gres": "gpu:1",
        "cpus": 8,
        "mem": f"{mem_gb}G",
        "time": f"{hours}:00:00",
    }


def write_plan(rows: list[dict], path: Path) -> None:
    lines = [
        "# TFM-SSL Resource Plan",
        "",
        "Derived from measured profile runs (not guesses). Safety margin ≈ 2.5× RAM, 3× wall time.",
        "",
        "## Measured cells",
        "",
        "| dataset | method | budget | status | total_s | peak_rss_mb | peak_vram_mb | suggested_profile | mem | time |",
        "|---|---|---:|---|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        sug = r.get("suggested") or {}
        lines.append(
            f"| {r.get('dataset')} | {r.get('method')} | {r.get('budget')} | {r.get('status')} | "
            f"{r.get('total_s')} | {r.get('peak_rss_mb')} | {r.get('peak_vram_mb')} | "
            f"{sug.get('profile')} | {sug.get('mem')} | {sug.get('time')} |"
        )
    lines += [
        "",
        "## Profile definitions",
        "",
        "| profile | partition | gres | default cpus | role |",
        "|---|---|---|---:|---|",
        "| cpu_light | cpu192G-48h | — | 4 | LR / tiny classical |",
        "| cpu_tree_ssl | cpu192G-48h | — | 8 | GBDT + self-training / CAST trees |",
        "| gpu_tfm_small | p_uriofir | gpu:1 | 8 | TabPFN/TabICL on small/medium data |",
        "| gpu_tfm_large | p_uriofir | gpu:1 | 8 | adult/jannis-scale TFM |",
        "| gpu_representation | p_uriofir | gpu:1 | 8 | STUNT/SeBA/Laplacian/geometric |",
        "| gpu_diffusion | p_uriofir | gpu:1 | 8 | reserved for heavy diffusion-like methods |",
        "",
        "## Concurrency",
        "",
        "`p_uriofir` has 2×A100. Default array concurrency: **2** for GPU waves.",
        "CPU waves may use higher concurrency on `cpu192G-48h`.",
        "",
        "## Notes",
        "",
        "- Pre-cache models/datasets before array submission.",
        "- Do not submit multiple jobs that race the same first-time checkpoint download.",
        "- Update this file when new profile measurements supersede prior ones.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--methods-filter", nargs="+", default=None)
    p.add_argument("--skip-missing-auth", action="store_true", default=True)
    args = p.parse_args()

    out_csv = ROOT / "results" / "validation" / "resource_profile.csv"
    out_md = ROOT / "docs" / "tfm_ssl_resource_plan.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell in PROFILE_CELLS:
        if args.methods_filter and cell["method"] not in args.methods_filter:
            continue
        if cell["method"].startswith("tabpfn") and not os.environ.get("TABPFN_TOKEN"):
            row = {**cell, "status": "blocked_missing_TABPFN_TOKEN"}
            row["suggested"] = choose_profile({**cell, "total_s": 600, "peak_vram_mb": 8000, "peak_rss_mb": 8000})
            rows.append(row)
            print("blocked", cell)
            continue
        print("profiling", cell, flush=True)
        try:
            row = profile_one(cell)
        except Exception as exc:  # noqa: BLE001
            row = {**cell, "status": "profile_exception", "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
        row["suggested"] = choose_profile(row)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("dataset", "method", "status", "total_s", "peak_vram_mb", "suggested")}, indent=2))

    fieldnames = [
        "dataset",
        "method",
        "seed",
        "budget",
        "hint_profile",
        "status",
        "load_s",
        "train_s",
        "total_s",
        "peak_rss_mb",
        "peak_vram_mb",
        "gpu_name",
        "output_bytes",
        "error",
        "suggested_profile",
        "suggested_partition",
        "suggested_gres",
        "suggested_cpus",
        "suggested_mem",
        "suggested_time",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            sug = r.get("suggested") or {}
            w.writerow(
                {
                    "dataset": r.get("dataset"),
                    "method": r.get("method"),
                    "seed": r.get("seed"),
                    "budget": r.get("budget"),
                    "hint_profile": r.get("hint_profile"),
                    "status": r.get("status"),
                    "load_s": r.get("load_s"),
                    "train_s": r.get("train_s"),
                    "total_s": r.get("total_s"),
                    "peak_rss_mb": r.get("peak_rss_mb"),
                    "peak_vram_mb": r.get("peak_vram_mb"),
                    "gpu_name": r.get("gpu_name"),
                    "output_bytes": r.get("output_bytes"),
                    "error": r.get("error"),
                    "suggested_profile": sug.get("profile"),
                    "suggested_partition": sug.get("partition"),
                    "suggested_gres": sug.get("gres"),
                    "suggested_cpus": sug.get("cpus"),
                    "suggested_mem": sug.get("mem"),
                    "suggested_time": sug.get("time"),
                }
            )
    write_plan(rows, out_md)
    print("wrote", out_csv)
    print("wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
