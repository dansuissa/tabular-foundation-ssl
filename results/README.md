# Results guide

The public results tree is intentionally small: it contains the three canonical
per-run result files and one comprehensive report with the figures and
machine-readable tables used in the repository.

## Canonical inputs

| Study block | File | Methods | Rows | Status |
|---|---|---:|---:|---|
| Historical benchmark | `raw/low_class_wave_paper_methods.csv` | 17 | 2,040 | 2,016 successes; 24 preserved graph-SSL failures |
| Frozen foundation models | `raw/tfm_frozen_screen.csv` | 2 | 240 | 240 successes |
| Focused SSL | `raw/focused_tfm_ssl.csv` | 6 | 720 | 720 successes |

Together these files define the 25-method, 3,000-run benchmark. Whole-project
claims should be computed from exactly these inputs or from the generated tables
under `reports/main_report/tables/overview/`.

## Canonical report

`reports/main_report/` contains the comprehensive scientific report, the
publication figures, supporting tables, and report-level integrity metadata.
Intermediate development runs and execution logs are deliberately not part of
the public repository.

## Analysis rules

1. Report coverage before comparing performance.
2. Use balanced accuracy as the primary metric.
3. Average seeds within each dataset × label-budget cell before cross-dataset ranking.
4. Penalize missing method cells rather than silently dropping them.
5. Use matched dataset/budget/seed keys for paired effects.
6. Treat three-seed intervals as descriptive rather than strong significance evidence.
7. Keep exploratory transductive results separate from the primary inductive benchmark.

Regenerate and validate the public summary with:

```bash
python -m pytest -q
python scripts/verify_repository.py
python scripts/build_github_overview.py
```
