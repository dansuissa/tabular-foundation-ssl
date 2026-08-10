#!/usr/bin/env python3
"""Ablate geometric_attention_ssl components after retrieval was repaired.

Identifies which SSL add-on (unlabeled memory / PL / lap / consistency / proto)
collapses BA on segment@50. Writes geometric_attention_ssl_ablation.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "validation"


def run(method: str, **extra):
    from src.data import dataset_from_config, load_dataset
    from src.run_benchmark import run_single_experiment
    from src.utils import load_yaml, set_seed

    bc = load_yaml(ROOT / "configs" / "benchmark.yaml")
    dc = load_yaml(ROOT / "configs" / "datasets.yaml")
    entry = next(e for e in dc["datasets"] if e["name"] == "segment")
    ds = load_dataset(dataset_from_config(entry))
    set_seed(0)
    params = dict(bc.get(method) or {})
    params.update(extra)
    row = run_single_experiment(
        dataset=ds,
        method=method,
        seed=0,
        label_budget=50,
        test_size=bc["test_size"],
        val_size_from_labeled=bc["val_size_from_labeled"],
        metric_names=bc["metrics"],
        method_params=params,
    )
    ba = (
        row.get("balanced_accuracy")
        or row.get("metric_balanced_accuracy")
        or row.get("test_balanced_accuracy")
    )
    # flatten_metrics may keep the raw metric name
    if ba is None:
        for k, v in row.items():
            if "balanced_accuracy" in str(k) and v is not None:
                ba = v
                break
    return {
        "method": method,
        "extra": extra,
        "status": row.get("status"),
        "ba": ba,
        "error": row.get("error_message"),
        "row_keys_sample": sorted(row.keys())[:40],
        "loss_components": (row.get("extra") or {}).get("loss_components")
        if isinstance(row.get("extra"), dict)
        else None,
        "labeled_attention_mass": row.get("labeled_attention_mass"),
        "unlabeled_attention_mass": row.get("unlabeled_attention_mass"),
        "best_epoch": row.get("best_epoch"),
        "epochs_trained": row.get("epochs_trained"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [
        ("retrieval_baseline", "geometric_attention_retrieval", {}),
        # Pin labeled_plus_unlabeled so diagnosis is not confounded by the
        # production default (labeled_only) that avoids unlabeled-bank collapse.
        (
            "ssl_full_unlab_mem",
            "geometric_attention_ssl",
            {"attention_memory": "labeled_plus_unlabeled"},
        ),
        (
            "ssl_labeled_memory",
            "geometric_attention_ssl",
            {"attention_memory": "labeled_only"},
        ),
        (
            "ssl_no_pseudo",
            "geometric_attention_ssl",
            {
                "use_pseudo": False,
                "lambda_pl": 0.0,
                "attention_memory": "labeled_plus_unlabeled",
            },
        ),
        (
            "ssl_no_lap",
            "geometric_attention_ssl",
            {
                "use_laplacian": False,
                "lambda_lap": 0.0,
                "attention_memory": "labeled_plus_unlabeled",
            },
        ),
        (
            "ssl_no_consistency",
            "geometric_attention_ssl",
            {
                "use_consistency": False,
                "lambda_consistency": 0.0,
                "attention_memory": "labeled_plus_unlabeled",
            },
        ),
        (
            "ssl_no_proto_margin",
            "geometric_attention_ssl",
            {
                "use_prototype": False,
                "use_margin": False,
                "lambda_proto": 0.0,
                "lambda_margin": 0.0,
                "attention_memory": "labeled_plus_unlabeled",
            },
        ),
        (
            "ssl_production_default",
            "geometric_attention_ssl",
            {},
        ),
        (
            "ssl_retrieval_only_unlab_mem",
            "geometric_attention_ssl",
            {
                "use_laplacian": False,
                "use_prototype": False,
                "use_pseudo": False,
                "use_consistency": False,
                "use_margin": False,
                "attention_memory": "labeled_plus_unlabeled",
            },
        ),
        (
            "ssl_retrieval_pl_labeled_mem",
            "geometric_attention_ssl",
            {
                "use_laplacian": False,
                "use_prototype": False,
                "use_consistency": False,
                "use_margin": False,
                "use_pseudo": True,
                "attention_memory": "labeled_only",
            },
        ),
        (
            "ssl_all_regs_labeled_mem",
            "geometric_attention_ssl",
            {"attention_memory": "labeled_only"},
        ),
    ]
    results = []
    for name, method, extra in cases:
        print("running", name, flush=True)
        try:
            r = run(method, **extra)
            r["case"] = name
            results.append(r)
            print(name, r["status"], r["ba"], flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "case": name,
                    "method": method,
                    "extra": extra,
                    "status": "exception",
                    "ba": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(name, "EXCEPTION", type(exc).__name__, flush=True)

    path = OUT / "geometric_attention_ssl_ablation.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("wrote", path)
    # Summarize collapses
    for r in results:
        ba = r.get("ba")
        flag = "COLLAPSE" if ba is not None and float(ba) < 0.25 else "ok"
        print(f"SUMMARY {r['case']}: ba={ba} {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
