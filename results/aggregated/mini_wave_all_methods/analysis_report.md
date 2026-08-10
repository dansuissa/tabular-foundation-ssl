# Mini-wave combined benchmark — analysis report

_Classical (Wave 1) + neural SSL, 3 datasets × 17 methods × 4 budgets × 3 seeds._

Source files:
- `results/raw/mini_wave_all_methods.csv`
- `results/aggregated/mini_wave_all_methods/summary_by_dataset_method_budget.csv`
- `results/aggregated/mini_wave_all_methods/rankings_by_dataset_budget_complete_only.csv` (headline)
- `results/aggregated/mini_wave_all_methods/rankings_by_dataset_budget_all_successes.csv` (exploratory only)
- `results/aggregated/mini_wave_all_methods/method_coverage_by_dataset_budget.csv`
- `results/aggregated/mini_wave_all_methods/ssl_vs_supervised_by_dataset_budget.csv`

## A. Executive summary

- **Total rows:** 612 (expected 612 = 504 classical + 108 neural).
- **Success:** 603  |  **Failed:** 9 (all failures are `failed_graph_ssl_nan`).
- **Datasets (3):** phoneme, spambase, letter.
- **Label budgets:** 50, 100, 250, 500.
- **Seeds:** 0, 1, 2.
- **Methods (17):** catboost, label_propagation, label_spreading, lightgbm, logistic_regression, mlp, random_forest, rpl_lite_xgboost, rpl_lr, scarf, self_training_catboost, self_training_lightgbm, self_training_lr, self_training_xgboost, sslae, vime_lite, xgboost.
- **Graph SSL** (`label_spreading`, `label_propagation`) has **9 known failed rows** (`failed_graph_ssl_nan`) — see Failure analysis.
- **Neural methods all succeeded:** 108/108 neural runs completed (sslae, vime_lite, scarf).

> **VIME-lite caveat.** `vime_lite` is treated as a lightweight VIME-inspired baseline, not a faithful reproduction of full VIME. If VIME-lite remains central to the final claims, a faithful vime implementation should be added and run as a separate method before publication.

## B. Headline winners (complete-seed rankings only)

### Best method per dataset and budget

| Dataset | Budget | Best method | Family | Balanced acc | Macro F1 |
|---------|--------|-------------|--------|--------------|----------|
| phoneme | 50 | `rpl_lite_xgboost` | rpl | 0.7118 | 0.7029 |
| phoneme | 100 | `rpl_lite_xgboost` | rpl | 0.7429 | 0.7338 |
| phoneme | 250 | `self_training_lightgbm` | self_training | 0.7838 | 0.7770 |
| phoneme | 500 | `self_training_lightgbm` | self_training | 0.8000 | 0.8018 |
| spambase | 50 | `vime_lite` | neural_ssl | 0.8765 | 0.8805 |
| spambase | 100 | `random_forest` | supervised | 0.8858 | 0.8943 |
| spambase | 250 | `sslae` | neural_ssl | 0.9109 | 0.9106 |
| spambase | 500 | `catboost` | supervised | 0.9359 | 0.9379 |
| letter | 50 | `sslae` | neural_ssl | 0.3489 | 0.3298 |
| letter | 100 | `catboost` | supervised | 0.5412 | 0.5424 |
| letter | 250 | `catboost` | supervised | 0.6284 | 0.6307 |
| letter | 500 | `random_forest` | supervised | 0.7269 | 0.7271 |

### Best method *within each family* per dataset and budget
(balanced accuracy mean; complete-seed only)

| Dataset | Budget | supervised | graph_ssl | self_training | rpl | neural_ssl |
|---|---|---|---|---|---|---|
| phoneme | 50 | catboost 0.676 | label_spreading 0.659 | self_training_xgboost 0.696 | rpl_lite_xgboost 0.712 | scarf 0.666 |
| phoneme | 100 | lightgbm 0.720 | label_spreading 0.692 | self_training_lightgbm 0.741 | rpl_lite_xgboost 0.743 | scarf 0.718 |
| phoneme | 250 | catboost 0.755 | label_spreading 0.716 | self_training_lightgbm 0.784 | rpl_lite_xgboost 0.773 | sslae 0.735 |
| phoneme | 500 | lightgbm 0.792 | label_spreading 0.759 | self_training_lightgbm 0.800 | rpl_lite_xgboost 0.796 | vime_lite 0.756 |
| spambase | 50 | random_forest 0.845 | n/a | self_training_catboost 0.836 | rpl_lr 0.824 | vime_lite 0.876 |
| spambase | 100 | random_forest 0.886 | n/a | self_training_lr 0.875 | rpl_lr 0.867 | scarf 0.877 |
| spambase | 250 | random_forest 0.911 | label_spreading 0.818 | self_training_lr 0.898 | rpl_lr 0.894 | sslae 0.911 |
| spambase | 500 | catboost 0.936 | label_spreading 0.834 | self_training_lightgbm 0.929 | rpl_lite_xgboost 0.911 | scarf 0.906 |
| letter | 50 | catboost 0.319 | label_propagation 0.194 | self_training_lr 0.318 | rpl_lr 0.263 | sslae 0.349 |
| letter | 100 | catboost 0.541 | label_spreading 0.512 | self_training_lightgbm 0.513 | rpl_lite_xgboost 0.467 | sslae 0.516 |
| letter | 250 | catboost 0.628 | label_spreading 0.597 | self_training_lightgbm 0.616 | rpl_lite_xgboost 0.604 | sslae 0.559 |
| letter | 500 | random_forest 0.727 | label_spreading 0.680 | self_training_lightgbm 0.700 | rpl_lite_xgboost 0.691 | scarf 0.647 |

## C. Neural vs classical

| Dataset | Budget | Best neural | Neural bal | Best classical | Classical bal | Δ balanced acc | Δ macro F1 | Neural wins |
|---|---|---|---|---|---|---|---|---|
| phoneme | 50 | `scarf` | 0.6657 | `rpl_lite_xgboost` | 0.7118 | -0.0461 | -0.0345 | no |
| phoneme | 100 | `scarf` | 0.7185 | `rpl_lite_xgboost` | 0.7429 | -0.0244 | -0.0125 | no |
| phoneme | 250 | `sslae` | 0.7354 | `self_training_lightgbm` | 0.7838 | -0.0484 | -0.0449 | no |
| phoneme | 500 | `vime_lite` | 0.7562 | `self_training_lightgbm` | 0.8000 | -0.0438 | -0.0456 | no |
| spambase | 50 | `vime_lite` | 0.8765 | `random_forest` | 0.8449 | +0.0316 | +0.0250 | YES |
| spambase | 100 | `scarf` | 0.8767 | `random_forest` | 0.8858 | -0.0091 | -0.0128 | no |
| spambase | 250 | `sslae` | 0.9109 | `random_forest` | 0.9108 | +0.0001 | -0.0056 | YES |
| spambase | 500 | `scarf` | 0.9065 | `catboost` | 0.9359 | -0.0294 | -0.0314 | no |
| letter | 50 | `sslae` | 0.3489 | `catboost` | 0.3189 | +0.0299 | +0.0435 | YES |
| letter | 100 | `sslae` | 0.5163 | `catboost` | 0.5412 | -0.0249 | -0.0282 | no |
| letter | 250 | `sslae` | 0.5593 | `catboost` | 0.6284 | -0.0691 | -0.0719 | no |
| letter | 500 | `scarf` | 0.6475 | `random_forest` | 0.7269 | -0.0795 | -0.0815 | no |

**Neural wins (3 cells):** spambase @ 50 (vime_lite, +0.0316), spambase @ 250 (sslae, +0.0001), letter @ 50 (sslae, +0.0299).

## D. SSL vs supervised (matched-seed comparisons)

- Matched-seed pair comparisons: **72** rows (72 complete-pair, 0 incomplete-pair).
- SSL improves balanced accuracy in **28** pairs; degrades it in **42** pairs.

### Strongest positive Δ balanced accuracy (SSL helps)

| Dataset | Budget | SSL method | vs supervised | Δ bal acc | paired seeds | warning |
|---|---|---|---|---|---|---|
| phoneme | 50 | `rpl_lite_xgboost` | `xgboost` | +0.0496 | 3/3 | nan |
| phoneme | 100 | `rpl_lite_xgboost` | `xgboost` | +0.0481 | 3/3 | nan |
| letter | 50 | `self_training_lightgbm` | `lightgbm` | +0.0432 | 3/3 | nan |
| phoneme | 100 | `self_training_xgboost` | `xgboost` | +0.0427 | 3/3 | nan |
| phoneme | 50 | `self_training_xgboost` | `xgboost` | +0.0334 | 3/3 | nan |
| phoneme | 250 | `self_training_lightgbm` | `lightgbm` | +0.0325 | 3/3 | nan |
| phoneme | 250 | `rpl_lite_xgboost` | `xgboost` | +0.0239 | 3/3 | nan |
| phoneme | 250 | `self_training_xgboost` | `xgboost` | +0.0230 | 3/3 | nan |

### Strongest negative Δ balanced accuracy (SSL hurts)

| Dataset | Budget | SSL method | vs supervised | Δ bal acc | paired seeds | warning |
|---|---|---|---|---|---|---|
| letter | 100 | `rpl_lr` | `logistic_regression` | -0.0905 | 3/3 | nan |
| letter | 250 | `rpl_lr` | `logistic_regression` | -0.0605 | 3/3 | nan |
| letter | 50 | `rpl_lr` | `logistic_regression` | -0.0538 | 3/3 | nan |
| letter | 50 | `self_training_catboost` | `catboost` | -0.0483 | 3/3 | nan |
| letter | 500 | `self_training_catboost` | `catboost` | -0.0474 | 3/3 | nan |
| letter | 100 | `self_training_catboost` | `catboost` | -0.0397 | 3/3 | nan |
| letter | 500 | `rpl_lr` | `logistic_regression` | -0.0394 | 3/3 | nan |
| phoneme | 100 | `self_training_catboost` | `catboost` | -0.0342 | 3/3 | nan |

## E. Dataset-specific interpretation

### phoneme (binary, ~5k rows)
- Dominant methods by budget: 50→`rpl_lite_xgboost`, 100→`rpl_lite_xgboost`, 250→`self_training_lightgbm`, 500→`self_training_lightgbm`. Pseudo-labeling/self-training (rpl_lite_xgboost, self_training_lightgbm) lead across the board.
- Unlabeled data **helps**: SSL beats its supervised counterpart in 12/24 matched-seed pairs on phoneme, with the largest gains from rpl_lite_xgboost at low budgets.
- Neural methods do **not** help here: best neural trails best classical at every budget (Δ from -0.0484 to -0.0244).

### spambase (binary, ~4.6k rows)
- **Neural `vime_lite` wins at budget 50** (best overall in that cell), and **`sslae` narrowly wins at budget 250**.
- Supervised trees (random_forest, catboost) dominate at budgets 100 and 500.
  - budget 50: best neural `vime_lite` (0.8765) vs best classical `random_forest` (0.8449) → Δ +0.0316 (neural wins).
  - budget 100: best neural `scarf` (0.8767) vs best classical `random_forest` (0.8858) → Δ -0.0091 (classical wins).
  - budget 250: best neural `sslae` (0.9109) vs best classical `random_forest` (0.9108) → Δ +0.0001 (neural wins).
  - budget 500: best neural `scarf` (0.9065) vs best classical `catboost` (0.9359) → Δ -0.0294 (classical wins).
- **Caveat:** the budget-50 win is from **VIME-lite**, not full VIME; the margins at 250 are within ~1e-3 and need faithful-VIME confirmation before any strong claim.

### letter (26-class multiclass, ~20k rows)
- Classical methods (catboost, random_forest) dominate at budgets 100/250/500.
- **Caveat at budget 50:** under *complete-seed* rankings the best classical method `label_spreading` is excluded (its seed-1 run failed), so the top complete-seed entry becomes `sslae` (0.3489) ahead of the best *complete* classical `catboost` (0.3189), Δ +0.0299. This is a **thin win in a very low-accuracy regime** (~0.35 balanced accuracy across 26 classes) and should not be read as neural superiority.
- Neural methods otherwise **struggle in this multiclass low-label regime**: at budgets 100/250/500 best neural trails best classical by -0.0795 to -0.0249 balanced accuracy. With only 50–500 labels spread over 26 classes, the encoder has too few labeled examples per class to fine-tune well.
- **Incomplete graph SSL:** `label_spreading` failed for seed 1 at budget 50 (1 of 3 seeds), so its budget-50 entry is based on 2 seeds and is excluded from complete-seed headline rankings (this is what hands the budget-50 cell to neural).

## F. Runtime analysis

### Average runtime by method (seconds)

| Method | Family | Mean runtime (s) |
|---|---|---|
| `scarf` | neural_ssl | 82.55 |
| `self_training_lightgbm` | self_training | 74.85 |
| `vime_lite` | neural_ssl | 69.47 |
| `sslae` | neural_ssl | 35.78 |
| `self_training_xgboost` | self_training | 34.24 |
| `label_propagation` | graph_ssl | 32.50 |
| `rpl_lite_xgboost` | rpl | 22.28 |
| `self_training_catboost` | self_training | 9.21 |
| `lightgbm` | supervised | 8.50 |
| `xgboost` | supervised | 7.59 |
| `label_spreading` | graph_ssl | 6.00 |
| `catboost` | supervised | 5.58 |
| `random_forest` | supervised | 3.15 |
| `rpl_lr` | rpl | 1.04 |
| `mlp` | supervised | 0.99 |
| `self_training_lr` | self_training | 0.71 |
| `logistic_regression` | supervised | 0.44 |

### Slowest 10 runs

| Dataset | Method | Seed | Budget | Runtime (s) |
|---|---|---|---|---|
| letter | `self_training_lightgbm` | 2 | 500 | 607.4 |
| letter | `scarf` | 2 | 100 | 347.4 |
| letter | `self_training_lightgbm` | 2 | 250 | 305.5 |
| letter | `scarf` | 2 | 500 | 303.7 |
| letter | `scarf` | 2 | 250 | 271.3 |
| letter | `self_training_lightgbm` | 2 | 100 | 214.7 |
| letter | `self_training_lightgbm` | 0 | 500 | 179.8 |
| letter | `scarf` | 0 | 250 | 175.5 |
| letter | `self_training_lightgbm` | 1 | 50 | 162.4 |
| letter | `self_training_lightgbm` | 1 | 500 | 159.2 |

- **Runtime/performance trade-off:** the neural methods (scarf ~83s, vime_lite ~69s, sslae ~36s mean) are 1–2 orders of magnitude slower than the fast supervised baselines (logistic_regression, mlp, self_training_lr < 1s) yet rarely top the rankings.
- **Local feasibility:** neural methods are CPU-feasible on these datasets — the full 108-run neural grid completed in ~1.9 hours on CPU. The heaviest cost is `scarf`/`vime_lite` on `letter` (up to ~350s/run), driven by the large unlabeled pool and contrastive/pretraining passes.

## G. Failure analysis

- **9 failed rows**, all `failed_graph_ssl_nan` (graph SSL only).

| Dataset | Method | Seed | Budget | Status |
|---|---|---|---|---|
| letter | `label_spreading` | 1 | 50 | failed_graph_ssl_nan |
| spambase | `label_propagation` | 0 | 50 | failed_graph_ssl_nan |
| spambase | `label_propagation` | 1 | 50 | failed_graph_ssl_nan |
| spambase | `label_propagation` | 2 | 50 | failed_graph_ssl_nan |
| spambase | `label_propagation` | 0 | 100 | failed_graph_ssl_nan |
| spambase | `label_spreading` | 0 | 50 | failed_graph_ssl_nan |
| spambase | `label_spreading` | 1 | 50 | failed_graph_ssl_nan |
| spambase | `label_spreading` | 2 | 50 | failed_graph_ssl_nan |
| spambase | `label_spreading` | 0 | 100 | failed_graph_ssl_nan |

- **Pattern:** failures concentrate on `spambase` at low budgets (50/100) for both `label_spreading` and `label_propagation`, plus a single `letter` `label_spreading` seed at budget 50.
- **Effect on headline rankings:** none. Complete-seed rankings exclude any method/dataset/budget cell that does not have all 3 seeds successful, so these unstable graph-SSL cells never drive a headline claim.
- **Nature:** these are **method instability** (sklearn label-propagation producing non-finite label distributions after kNN-graph retries), not benchmark crashes — the runner caught them cleanly and continued.

## H. Preliminary conclusions

- **What the benchmark suggests:** on these three mostly-numeric tabular datasets, strong supervised trees and tree-based pseudo-labeling/self-training (catboost, random_forest, rpl_lite_xgboost, self_training_lightgbm) are the most reliable choices across budgets. SSL via pseudo-labeling gives the most consistent gains over supervised counterparts on phoneme.
- **Where neural helps:** only 3 dataset/budget cells in complete-seed rankings — spambase @ 50 (vime_lite, +0.0316), spambase @ 250 (sslae, +0.0001), letter @ 50 (sslae, +0.0299). Two of these are fragile: spambase @ 250 is a ~1e-4 tie, and letter @ 50 only wins because the stronger graph-SSL method is excluded for an incomplete seed. The cleanest neural win is **vime_lite on spambase @ 50**.
- **What is robust:** (1) classical dominance on multiclass `letter` at budgets ≥100; (2) pseudo-labeling helping on phoneme; (3) graph SSL instability at low label budgets on spambase.
- **What is still uncertain:** the neural wins rest on a **VIME-lite** implementation, a sub-1e-3 margin at spambase 250, and an exclusion artifact at letter 50; single-wave, 3-seed estimates also have wide spread on `letter` at low budgets.
- **Faithful VIME?** The single clearest neural headline win comes from `vime_lite` (spambase @ 50). That makes VIME-lite **borderline central** and is enough to justify implementing and running a **faithful VIME** as a separate method before any publication claim that 'VIME wins on spambase'. It is not yet central enough to block the rest of the analysis.

> **VIME-lite caveat (restated).** `vime_lite` is treated as a lightweight VIME-inspired baseline, not a faithful reproduction of full VIME. If VIME-lite remains central to the final claims, a faithful vime implementation should be added and run as a separate method before publication.
