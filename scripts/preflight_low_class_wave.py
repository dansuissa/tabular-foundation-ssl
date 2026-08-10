"""Preflight validation for the low-class wave (no model runs).

Validates:
- dataset group and method group definitions
- OpenML IDs resolve and targets are single-label classification
- n_classes in [2, 9] (and steel-plates-fault is exactly 7)
- budgets are feasible under strict per-class minimum labeling rules
- expected reuse / new / final row counts for the planned grid

This script downloads dataset metadata/data via sklearn.fetch_openml as needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data import DatasetSpec, load_dataset
from src.splits import InvalidBudgetError, make_ssl_split
from src.utils import load_yaml


LOW_CLASS_GROUP = "low_class_wave"
METHOD_GROUP = "paper_methods_no_vime_lite"

REUSE_INPUT = Path("results/raw/mini_wave_all_methods_plus_vime.csv")
REUSE_DATASETS = {"phoneme", "spambase"}

EXPECTED = {
    "n_methods": 17,
    "n_budgets": 4,
    "n_seeds": 3,
    "n_reuse_datasets": 2,
    "n_new_datasets": 8,
    "n_total_datasets": 10,
    "expected_reused_rows": 408,
    "expected_new_rows": 1632,
    "expected_final_rows": 2040,
}


def _die(msg: str) -> None:
    raise SystemExit(msg)


def _dataset_entry_map(datasets_cfg: dict) -> dict[str, dict]:
    entries = datasets_cfg.get("datasets", []) or []
    return {e["name"]: e for e in entries}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    datasets_cfg = load_yaml(root / "configs/datasets.yaml")
    bench_cfg = load_yaml(root / "configs/benchmark.yaml")

    # --- validate groups -------------------------------------------------
    groups = datasets_cfg.get("dataset_groups", {}) or {}
    if LOW_CLASS_GROUP not in groups:
        _die(f"Missing dataset group '{LOW_CLASS_GROUP}' in configs/datasets.yaml")
    ds_names = list(groups[LOW_CLASS_GROUP])
    if "letter" in set(ds_names):
        _die("Invalid: 'letter' is present in low_class_wave dataset group.")
    if len(ds_names) != EXPECTED["n_total_datasets"]:
        _die(f"low_class_wave must have exactly 10 datasets, got {len(ds_names)}")

    if METHOD_GROUP not in bench_cfg:
        _die(f"Missing method group '{METHOD_GROUP}' in configs/benchmark.yaml")
    methods = list(bench_cfg[METHOD_GROUP])
    if "vime_lite" in set(methods):
        _die("Invalid: 'vime_lite' is present in paper_methods_no_vime_lite.")
    if len(methods) != EXPECTED["n_methods"]:
        _die(f"paper_methods_no_vime_lite must have 17 methods, got {len(methods)}")

    print("OK groups:")
    print("  datasets:", ds_names)
    print("  methods:", methods)

    # --- dataset metadata checks ----------------------------------------
    entry_map = _dataset_entry_map(datasets_cfg)
    missing = [d for d in ds_names if d not in entry_map]
    if missing:
        _die(f"Datasets missing from configs/datasets.yaml 'datasets' list: {missing}")

    budgets = [50, 100, 250, 500]
    seeds = [0, 1, 2]
    test_size = float(bench_cfg["test_size"])
    val_size_from_labeled = float(bench_cfg["val_size_from_labeled"])

    print("\nDataset details (loaded from OpenML):")
    for name in ds_names:
        entry = entry_map[name]
        spec = DatasetSpec(
            name=name,
            openml_id=int(entry["openml_id"]),
            target_column=entry.get("target_column", "class"),
            max_rows=entry.get("max_rows"),
        )
        ds = load_dataset(spec)
        n_classes = int(ds.n_classes)
        if not (2 <= n_classes <= 9):
            _die(f"{name}: invalid n_classes={n_classes} (expected 2..9)")
        if name == "steel-plates-fault" and n_classes != 7:
            _die(f"steel-plates-fault: expected 7 classes, got {n_classes}")
        print(
            f"  - {name:18s} id={ds.openml_id} rows={len(ds.X)} "
            f"features={ds.X.shape[1]} classes={n_classes}"
        )

        # --- feasibility of strict labeled budgets ----------------------
        for b in budgets:
            for seed in seeds:
                try:
                    _ = make_ssl_split(
                        ds.X,
                        ds.y,
                        n_labeled=b,
                        test_size=test_size,
                        val_size_from_labeled=val_size_from_labeled,
                        seed=seed,
                    )
                except InvalidBudgetError as exc:
                    _die(f"{name}: budget={b} seed={seed} infeasible: {exc}")

    # --- reuse row count check ------------------------------------------
    if not (root / REUSE_INPUT).exists():
        _die(f"Missing reuse input file: {REUSE_INPUT}")
    df = pd.read_csv(root / REUSE_INPUT)
    df = df[df["dataset"].isin(REUSE_DATASETS)].copy()
    df = df[df["method"].isin(methods)].copy()
    df = df[df["n_labeled"].isin(budgets)].copy()
    df = df[df["seed"].isin(seeds)].copy()
    # keep failures; exclude vime_lite already removed by method filter
    reuse_rows = int(len(df))
    if reuse_rows != EXPECTED["expected_reused_rows"]:
        _die(
            f"Reuse rows mismatch: expected {EXPECTED['expected_reused_rows']}, got {reuse_rows} "
            f"(check methods/datasets/budgets/seeds filters)."
        )
    print(f"\nOK reuse rows from {REUSE_INPUT}: {reuse_rows} (expected {EXPECTED['expected_reused_rows']})")

    new_rows = EXPECTED["n_new_datasets"] * EXPECTED["n_methods"] * EXPECTED["n_budgets"] * EXPECTED["n_seeds"]
    final_rows = EXPECTED["n_total_datasets"] * EXPECTED["n_methods"] * EXPECTED["n_budgets"] * EXPECTED["n_seeds"]
    if new_rows != EXPECTED["expected_new_rows"] or final_rows != EXPECTED["expected_final_rows"]:
        _die("Internal expected row math mismatch.")
    print(f"Expected new-run rows: {new_rows}")
    print(f"Expected final combined rows: {final_rows}")

    print("\nPREFLIGHT OK")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _die("Interrupted.")
    except Exception as exc:  # noqa: BLE001
        _die(f"Preflight failed: {type(exc).__name__}: {exc}")

