"""Create the reuse CSV for the low-class wave (no model runs).

Filters the existing canonical mini-wave file to keep only:
- datasets: phoneme, spambase
- methods: paper_methods_no_vime_lite (i.e., excludes vime_lite)
- budgets: 50/100/250/500
- seeds: 0/1/2

Keeps failures as-is (e.g. failed_graph_ssl_nan).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import load_yaml


INP = Path("results/raw/mini_wave_all_methods_plus_vime.csv")
OUT = Path("results/raw/low_class_wave_reused_phoneme_spambase.csv")


def main() -> None:
    cfg = load_yaml("configs/benchmark.yaml")
    methods = list(cfg["paper_methods_no_vime_lite"])
    budgets = [50, 100, 250, 500]
    seeds = [0, 1, 2]

    if not INP.exists():
        raise SystemExit(f"Missing input: {INP}")
    df = pd.read_csv(INP)
    df = df[df["dataset"].isin(["phoneme", "spambase"])].copy()
    df = df[df["method"].isin(methods)].copy()
    df = df[df["n_labeled"].isin(budgets)].copy()
    df = df[df["seed"].isin(seeds)].copy()
    df = df[df["dataset"] != "letter"].copy()
    if "vime_lite" in set(df["method"].unique()):
        raise SystemExit("Unexpected: vime_lite survived the filter.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} rows={len(df)}")


if __name__ == "__main__":
    main()

