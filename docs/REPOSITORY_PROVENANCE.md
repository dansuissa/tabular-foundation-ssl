# Repository provenance and reconciliation

This repository was assembled from the completed Bar-Ilan cluster project and
the separately supplied local project archive `ssl_tabular_benchmark.zip`. The
merge was performed conservatively: both sources were inventoried before any
repository staging, and the originals were left untouched.

## Local archive audit

- Archive SHA-256:
  `211333a6d7b60648690e4409dbf9722bb287ad97588bb86dfb5697f4c5698b38`
- ZIP entries: 364
- Safely extracted files: 326
- Unsafe paths or symlinks: 0

## Comparison with the cluster project

| Check | Count |
|---|---:|
| Local archive files | 326 |
| Cluster project files at reconciliation | 772 |
| Paths present in both | 326 |
| Byte-identical common files | 320 |
| Changed common files | 6 |
| Files unique to the local archive | 0 |
| Files unique to the cluster project | 446 |

The six changed paths were `README.md`, `configs/benchmark.yaml`,
`configs/datasets.yaml`, `src/aggregate_results.py`, `src/models/__init__.py`,
and `src/run_benchmark.py`. A manual diff confirmed that every cluster version
contains the newer TFM/SSL extension: dual views, method groups, capability
dispatch, expanded metrics and diagnostics, atomic shards, and explicit failure
handling. The cluster tree is therefore the authoritative strict superset. No
local-only code or result needed to be discarded or manually blended.

The final staged Git tree contains all 305 substantive files from the local ZIP.
The only 21 ZIP files not staged are Python 3.14 `__pycache__/*.pyc` bytecode
artifacts, which are machine-specific build products and are covered by
`.gitignore`.

## Canonical data identities

| File | Rows | SHA-256 |
|---|---:|---|
| `results/raw/low_class_wave_paper_methods.csv` | 2,040 | `b017a950054a5045a4a39c41362d3dc3eef56603d1db4d62d4ffbfbdc43e8ad5` |
| `results/raw/tfm_frozen_screen.csv` | 240 | `05f11ac7e5da98ce783b41bd46331b4d55fddc6d339136b625c72a4a2eae0d02` |
| `results/raw/focused_tfm_ssl.csv` | 720 | `0a3bca9fc2728dabf7ac846a7a477c64ac5536feb582f28c92a2594d50b06a9a` |

The historical file is immutable. Phase C contains two validated source-tree
hashes because the first two methods and the repaired attention/geometric family
were run from consecutive reviewed implementations:

- TFM self-training and Laplacian/alignment source:
  `209ec7c85d4342ab6cdc2ad16fd6a5241dcf10f0855f062fc7acb7a054afb4af`
- Repaired attention/combined source:
  `b9c1f48daae0a38670909bfcee77dc8488c635820805cd7f3886f35130ed7e69`

Run manifests, final-gate evidence, method mapping, environment hashes, and
checkpoint identities are retained under `results/validation/` and the single
canonical report at `results/reports/main_report/`.

## What is intentionally not in Git

The source archives contained no irreplaceable local-only files. The Git
repository excludes downloaded OpenML caches, installed environments, licensed
model weights, secrets, and verbose Slurm logs. Final atomic shards and logs are
preserved separately as the checksummed research archive documented in
[`ARTIFACTS.md`](ARTIFACTS.md).
