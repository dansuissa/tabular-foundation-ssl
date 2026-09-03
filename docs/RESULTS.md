# Results

This page is the repository-level guide to the complete result record. The
[comprehensive main report](../results/reports/main_report/README.md) is the
single scientific report and contains the overview, historical benchmark,
focused-study diagnostics, paired contrasts, limitations, figures, and
deliverable index.

## Scope and integrity

| Wave | Methods | Grid | Rows | Success | Preserved failures |
|---|---:|---|---:|---:|---:|
| Historical supervised and SSL baselines | 17 | 10 datasets × 4 budgets × 3 seeds | 2,040 | 2,016 | 24 |
| Frozen TabPFN-3 and TabICL v2 | 2 | 10 × 4 × 3 | 240 | 240 | 0 |
| Focused TFM and geometric SSL | 6 | 10 × 4 × 3 | 720 | 720 | 0 |
| **Total** | **25** | | **3,000** | **2,976** | **24** |

There are no duplicate experimental keys. The historical failures belong to
label propagation/spreading and remain in the canonical file. All 960 newly run
foundation-model and focused-SSL cells completed successfully.

## Primary comparison

Balanced accuracy is averaged over seeds within each dataset × budget cell.
Methods are ranked inside each of the 40 cells and ranks are then averaged, so
large datasets cannot dominate the summary. Missing/failed method cells receive
rank 26. The complete 25-row result is
[`all_method_summary.csv`](../results/reports/main_report/tables/overview/all_method_summary.csv).

The first five methods are:

| Rank | Method | Mean balanced accuracy | Mean cell rank |
|---:|---|---:|---:|
| 1 | TabICL v2 + self-training | 0.745 | 4.06 |
| 2 | TabICL v2 | 0.743 | 4.14 |
| 3 | TabPFN-3 + self-training | 0.741 | 4.69 |
| 4 | TabPFN-3 | 0.739 | 5.19 |
| 5 | CatBoost | 0.722 | 8.66 |

The four TFM configurations therefore lead the full ranking. Their advantage is
not uniform: a TFM variant wins 31/40 cells against the six selected historical
comparators, rather than all 40.

## What self-training contributed

All effects below are exact paired differences using the same dataset, budget,
seed, and split:

| Backbone | Mean self-training − frozen BA | Descriptive 95% interval | Positive pairs | Frozen round selected |
|---|---:|---:|---:|---:|
| TabICL v2 | +0.0014 | +0.0002 to +0.0026 | 23/120 | 83/120 |
| TabPFN-3 | +0.0023 | −0.0002 to +0.0048 | 20/120 | 86/120 |

The mechanism is conservative by design. Most attempted pseudo-label loops do
not improve validation evidence enough to replace the frozen prediction. The
result supports small, selective gains—not a general claim that recycled
pseudo-label context always helps.

## Focused trainable SSL

| Method | Mean balanced accuracy | Mean rank within focused eight | Mean runtime/run |
|---|---:|---:|---:|
| Unlabeled attention SSL | 0.668 | 5.03 | 5.0 s |
| Laplacian SSL | 0.617 | 6.40 | 7.6 s |
| Combined geometric + attention SSL | 0.596 | 6.92 | 9.5 s |
| Embedding alignment SSL | 0.583 | 7.12 | 3.6 s |

The combined method improves on separately trained embedding alignment by
**+0.013**, but trails Laplacian SSL by **−0.021** and retrieval attention by
**−0.072**. More regularization does not produce better generalization here.

The attention implementation genuinely uses the unlabeled training pool: mean
attention mass is 96% on unlabeled memory. The combined implementation, after
the retrieval repair, allocates roughly 55% to labeled and 45% to unlabeled
neighbors. No final run has validation/test examples in memory.

## Reliability and diagnostics

- The final grid has no representation-collapse flags.
- The combined model retains 21/120 constant-prediction warnings and Laplacian
  SSL 17/120. These are diagnostic warnings; their finite, normalized outputs
  remain in the result set.
- The best-calibrated focused methods are the four TFMs (mean ECE 0.039–0.045).
  Trainable focused SSL methods range from 0.082 to 0.122.
- Sparse training graphs average about 21,000 nodes and 53,000 edges and take
  1.7 seconds to construct.
- Historical neural SSL is substantially slower on this hardware: about 57 s
  for SSLAE, 177 s for SCARF, and 263 s for VIME per run.

## Figures and tables

The overview used in the main README is retained once as
[PNG](../results/reports/main_report/figures/overview/project_overview.png). The
[complete all-method matrix](../results/reports/main_report/figures/overview/complete_method_matrix.png)
has exact values in `tables/overview/all_methods_by_budget.csv` and
`tables/overview/all_methods_by_dataset.csv`. It includes all 25 methods, four
budgets, and ten datasets without selecting a subset.

The focused section contains seven canonical PNG figures:

1. requested methods by label budget;
2. focused-method ranking;
3. paired TFM self-training deltas;
4. combined-model component contrasts;
5. calibration versus accuracy;
6. runtime versus performance;
7. method diagnostics.

Their machine-readable sources, including seed-level paired tables, are in
[`tables/focused_ssl/`](../results/reports/main_report/tables/focused_ssl/).

## Limits on interpretation

Three seeds support descriptive paired intervals, not strong significance
claims for small effects. The trainable geometric family lacks a final-hash,
full-grid supervised-encoder control, so its results do not isolate the causal
benefit of unlabeled data. CPU RAM, post-training oracle pseudo-label accuracy,
neighbor-label purity, and consistently flattened class-prior drift were not
serialized and are not reconstructed after the fact.
