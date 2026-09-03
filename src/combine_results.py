"""Combine compatible result CSVs while refusing duplicate experiment keys."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLS = ["dataset", "method", "seed", "n_labeled"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple raw benchmark CSV files into one canonical file. "
            "This is a pure concatenation utility that aligns columns and refuses "
            "to overwrite duplicate (dataset, method, seed, n_labeled) keys."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input CSV files to concatenate (space-separated).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (will be overwritten if it exists).",
    )
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def _union_columns(frames: list[pd.DataFrame]) -> list[str]:
    # Keep a stable, readable order: first file's columns, then any extras.
    base = list(frames[0].columns)
    extras: list[str] = []
    seen = set(base)
    for df in frames[1:]:
        for col in df.columns:
            if col not in seen:
                extras.append(col)
                seen.add(col)
    return base + sorted(extras)


def combine_inputs(inputs: list[Path]) -> pd.DataFrame:
    frames = [_load_csv(p) for p in inputs]
    if not frames:
        return pd.DataFrame()

    for p, df in zip(inputs, frames, strict=False):
        missing = [c for c in KEY_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing key columns {missing} in {p}")

    all_cols = _union_columns(frames)
    aligned = [df.reindex(columns=all_cols) for df in frames]
    combined = pd.concat(aligned, ignore_index=True)

    dup = combined.duplicated(subset=KEY_COLS, keep=False)
    if bool(dup.any()):
        offenders = (
            combined.loc[dup, KEY_COLS]
            .value_counts()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        raise ValueError(
            "Duplicate run keys found while combining inputs. "
            "Refusing to write output.\n"
            + offenders.to_string(index=False)
        )
    return combined


def main() -> None:
    args = parse_args()
    combined = combine_inputs(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

