# Results guide

The results tree contains canonical experiment records, derived summaries,
reports, and validation history. Files are retained even when they record a
failed or superseded diagnostic run; use the status labels below when analyzing
them.

## Authoritative canonical inputs

| Phase | File | Methods | Rows | Status |
|---|---|---:|---:|---|
| Historical benchmark | `raw/low_class_wave_paper_methods.csv` | 17 | 2,040 | Immutable; 2,016 successes and 24 preserved graph failures |
| Frozen TFM screen | `raw/tfm_frozen_screen.csv` | 2 | 240 | Final; 240 successes |
| Focused SSL | `raw/focused_tfm_ssl.csv` | 6 | 720 | Final; 720 successes |

Together they define the 25-method, 3,000-row project result. Analyses making a
“whole project” claim must begin with exactly these files or with the generated
`reports/main_report/tables/overview/` tables.

## Directory roles

- `raw/` contains canonical CSVs plus named intermediate and diagnostic waves.
  Intermediate files are evidence, not inputs to the final comparison.
- `aggregated/` contains wave-specific descriptive summaries, rankings, and
  plots produced by the standard aggregation pipeline.
- `reports/main_report/` is the only report. It consolidates the complete
  25-method overview, focused-study analysis, historical findings, canonical
  PNG figures, machine-readable tables, and report validation metadata.
- `validation/` retains audits, gates, failure diagnoses, test output, source
  hashes, method mapping, and submission manifests.

## Analysis rules

1. Filter metrics to `status == "success"`, but compute and display coverage
   before ranking.
2. Use balanced accuracy as the primary metric.
3. Average seeds within dataset × label-budget cells before cross-dataset
   ranking.
4. Penalize missing method cells rather than letting them disappear from mean
   rank. The repository overview uses rank 26 for a failed 25-way cell.
5. Use matched dataset/budget/seed keys for paired effects.
6. Treat the three-seed intervals as descriptive, not strong significance
   evidence.
7. Never combine exploratory transductive results with the primary inductive
   ranking.

Regenerate and validate the whole-project summary with:

```bash
python scripts/verify_repository.py
python scripts/build_github_overview.py
```
