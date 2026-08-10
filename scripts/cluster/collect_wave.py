#!/usr/bin/env python3
"""Collect validated shards into a combined CSV (never touches historical waves)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.results_io.shards import list_shards, read_shard

KEY = ["dataset", "method", "seed", "n_labeled"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wave", required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Combined CSV path (default under SSL_COMBINED_ROOT)",
    )
    args = p.parse_args()

    rows = []
    for path in list_shards(args.wave):
        if path.name in {"task_map.json"} or path.suffix != ".json":
            continue
        if path.name.endswith(".sbatch"):
            continue
        try:
            payload = read_shard(path)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Unreadable shard {path}: {exc}")
        if "dataset" not in payload or "method" not in payload:
            # skip non-shard json helpers
            continue
        rows.append(payload)

    if not rows:
        raise SystemExit(f"No shards found for wave={args.wave}")

    df = pd.DataFrame(rows)
    missing = [c for c in KEY if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing key columns {missing}")
    dup = df.duplicated(subset=KEY, keep=False)
    if bool(dup.any()):
        raise SystemExit(
            "Duplicate run keys among shards; refusing to combine:\n"
            + df.loc[dup, KEY].value_counts().to_string()
        )

    out = args.output
    if out is None:
        base = Path(os.environ.get("SSL_COMBINED_ROOT", "results/combined"))
        base.mkdir(parents=True, exist_ok=True)
        out = base / f"{args.wave}.csv"

    # Historical immutability guard
    forbidden = {
        "results/raw/low_class_wave_paper_methods.csv",
        "low_class_wave_paper_methods.csv",
    }
    if out.name in forbidden or str(out).endswith("low_class_wave_paper_methods.csv"):
        raise SystemExit(f"Refusing to write immutable historical path: {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"wrote {len(df)} rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
