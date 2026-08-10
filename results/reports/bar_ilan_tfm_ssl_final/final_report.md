# Final report: Tabular foundation models and focused semi-supervised learning

Generated 2026-07-30T08:28:10.815329+00:00 from validated result files.

## Executive summary

The requested experiment scope is complete. Phase A contains **240/240** successful
frozen-foundation-model runs and Phase C contains **720/720** successful focused SSL
runs. There are no missing or duplicate keys and no non-finite primary balanced-
accuracy values. The immutable 17-method historical result file was read for
comparison and was not modified.

Across the eight requested methods, **TabICLv2 self-training** has the
highest pooled mean balanced accuracy (0.745).
This pooled mean is descriptive; the primary ranking in
`tables/new_method_failure_aware_ranking.csv` averages ranks over the 40 dataset ×
budget cells so large datasets cannot dominate by row count.

TabPFN-3 minus TabICLv2 has an exact paired mean balanced-accuracy difference of
**-0.005** across 120 matched cells
(descriptive 95% interval -0.009 to
-0.000).

The principal self-training effects, computed against each method's own frozen
backbone on the same dataset, budget, seed, and split, are:

| Method | Mean Δ BA | 95% low | 95% high | Positive fraction | Pairs |
|---|---|---|---|---|---|
| tabiclv2_self_training | 0.001 | 0.000 | 0.003 | 0.192 | 120 |
| tabpfn3_self_training | 0.002 | -0.000 | 0.005 | 0.167 | 120 |

These intervals describe across-cell variability; with only three seeds they are
not a basis for strong significance claims.

## Objective and scope

The study evaluates TabPFN-3 and TabICLv2 in the existing few-shot protocol, then
tests precisely six focused SSL methods: TFM self-training for both backbones,
explicit Laplacian regularization, retrieval attention using unlabeled training
memory, class-conditional embedding alignment, and their modular combined model.
No additional foundation-model families or post-hoc Phase D were introduced.

## Protocol

- Ten canonical OpenML datasets; budgets 50, 100, 250, and 500; seeds 0, 1, and 2.
- Balanced accuracy is primary. Macro-F1, accuracy, log-loss, ROC-AUC, average
  precision where defined, Brier score, ECE, runtime, and peak GPU memory are
  retained in machine-readable tables.
- Splits are predetermined and shared across methods.
- The protocol is inductive: graph, pseudo-label, representation, and retrieval
  adaptation use labeled and unlabeled *training* rows only. Validation labels are
  restricted to calibration/selection; test features appear only at prediction.
- TabPFN-3 and TabICLv2 use the native/raw pandas view. Classical and trainable
  neural SSL methods use the established processed view.
- Every long run executed as an atomic Slurm array task on the cluster.

The dataset inventory is in `tables/dataset_table.csv`.

## Implementation fidelity

- `tabpfn3` uses TabPFN 8.1.0 with the verified V3 classifier checkpoint; `tabiclv2`
  uses the pinned `tabicl-classifier-v2-20260212.ckpt`.
- Both self-training methods share the same deterministic, class-balanced hard
  pseudo-label engine with validation selection and safe frozen-round fallback.
- `laplacian_ssl` is an explicit sparse graph-regularized neural classifier, not
  sklearn label propagation.
- `unlabeled_attention_ssl` records `labeled_plus_unlabeled` memory and zero
  validation/test-memory flags.
- `embedding_alignment_ssl` uses confidence-filtered class prototypes and records
  class geometry/collapse diagnostics.
- `geometric_attention_ssl` combines the requested components with ramp-up. The
  original segment chance collapse was traced to unlabeled-pool cardinality
  overwhelming top-k retrieval. Balanced labeled/unlabeled neighbor retrieval was
  implemented and passed the final representative gate before Phase C.

The exact name mapping and validity decisions remain documented in
`../../validation/final_method_mapping.md`.

## Frozen TFM results

The full Phase-A metric table is `tables/new_method_metric_summary.csv`; the exact
TabPFN-3/TabICLv2 paired comparison is
`tables/phase_a_tabpfn3_vs_tabiclv2.csv`. Figure 1 shows both frozen models beside
the six SSL methods at every budget.

## TFM self-training

Exact seed-level pairs are in `tables/self_training_seed_level_paired.csv`.
Self-training selected the frozen round (round 0) in
**0.708** of TabPFN-3 runs and
**0.692** of TabICLv2 runs. This is
intended safety behavior: an attempted pseudo-label round is retained only when
validation evidence meets the fixed guard. Round logs and accepted counts are in
`tables/phase_c_run_diagnostics.csv`.

The run payloads do not consistently expose post-training oracle pseudo-label
accuracy, confidence distributions, or class-prior drift as flat analyzable
fields. The report therefore does not invent those summaries; this is recorded as
a limitation even though selection counts, rounds, and fallback reasons are
preserved.

## Laplacian SSL

The method completed 120/120 cells. Sparse graph node/edge counts, connected
components, isolated-node counts, affinities, and graph-build times are summarized
in `tables/method_diagnostic_summary.csv`. Exact comparisons with the existing
label-propagation and label-spreading results are in
`tables/laplacian_vs_historical_graph_effects.csv`. Those comparisons share run
keys and splits but compare different estimators, so they are benchmark contrasts,
not an isolated Laplacian-loss causal effect.

Neighbor-label purity was not serialized in the final shards and is consequently
not reconstructed using test labels.

## Attention over unlabeled data

`unlabeled_attention_ssl` completed 120/120 cells with training-only
labeled-plus-unlabeled memory. Attention mass and memory sizes are in the
diagnostic tables and Figure 7. The complete grid does not contain a matched
supervised/labeled-memory control under the final source hash. The earlier control
experiments are therefore treated only as diagnostic evidence; no full-grid causal
claim that unlabeled memory helped is made.

## Class-conditional embedding alignment

`embedding_alignment_ssl` completed 120/120 cells. Per-run reliable-unlabeled
counts, embedding variance, intra-class distance, inter-prototype distance, and
collapse flags are retained in `tables/phase_c_run_diagnostics.csv`. Relations
between these diagnostics and balanced accuracy can be analyzed from that table
without conflating uncertain examples with accepted examples.

## Combined model and component contrasts

The combined method completed 120/120 cells. It recorded
**0** representation-collapse flags and **21**
constant-prediction flags across the final grid. A constant-prediction diagnostic
is a warning, not an automatic failed run; each affected result still has finite,
normalized probabilities and is retained transparently.

Paired differences between the complete combined method and the separately trained
Laplacian, attention, and alignment methods are:

| Component comparator | Mean combined − component | 95% low | 95% high | Positive fraction | Pairs |
|---|---|---|---|---|---|
| embedding_alignment_ssl | 0.013 | -0.008 | 0.034 | 0.550 | 120 |
| laplacian_ssl | -0.021 | -0.033 | -0.010 | 0.325 | 120 |
| unlabeled_attention_ssl | -0.072 | -0.087 | -0.056 | 0.142 | 120 |

These are component-method contrasts, not leave-one-component-out ablations. The
representative final gate is preserved in
`tables/representative_attention_final_gate.csv`; obsolete pre-fix collapse
experiments are retained under `results/validation/` for failure analysis.

## Calibration and secondary metrics

`tables/new_method_metric_summary.csv` reports all requested metrics with valid
sample counts. Average precision is undefined for the benchmark's multiclass
cases and is left missing rather than imputed. Figure 5 displays the mean ECE/
balanced-accuracy trade-off. Log-loss, Brier score, and ECE are interpreted as
calibration context; no method is selected using test calibration metrics.

## Comparison with the historical benchmark

`tables/requested_methods_and_historical_baselines.csv` compares all eight new
methods with CatBoost, XGBoost, self-training LightGBM, label propagation,
label spreading, VIME, SCARF, and SSLAE. The failure-aware ranking and explicit
coverage tables prevent a method with missing runs from being silently rewarded.
Historical results remain immutable at
`results/raw/low_class_wave_paper_methods.csv`.

## Compute cost

Runtime and peak GPU memory are summarized per method in the metric table.
Figure 6 compares mean runtime and balanced accuracy on a logarithmic runtime
axis. CPU RAM was not serialized by the benchmark runner, so the requested RAM
comparison cannot be reconstructed reliably and is reported as unavailable.
Cold-load and warm-inference timing remain in the raw TFM rows.

## Failure analysis

Operationally, Phase A and Phase C have zero failed, missing, corrupt, or duplicate
runs. The important scientific failure was the pre-Phase-C combined-model collapse
on segment. Diagnostics identified retrieval composition—not class mapping,
probability ordering, or output dimension—as the failing component. The repaired
balanced retrieval gate achieved above-chance segment results before the final
grid was authorized. Historical failed diagnostics are preserved rather than
overwritten.

The final repository test command
`/private/ofirlin-lab/suissad4/envs/ssl-tfm/bin/python -m pytest -q` completed
with **25 passed and 1 skipped** in 27.14 seconds. The exact captured output is
stored in `results/validation/pytest_current.txt`.

## Limitations

1. Three seeds support paired descriptive uncertainty, not strong significance
   claims about small differences.
2. Trainable SSL methods lack a final-hash, full-grid supervised encoder control;
   component and historical comparisons must not be described as the pure effect
   of unlabeled data.
3. Some requested analysis-only diagnostics—CPU RAM, post-training pseudo-label
   accuracy, neighbor-label purity, and flattened class-prior drift—were not
   serialized consistently.
4. The project is not a Git work tree; deterministic source/configuration hashes
   replace commit provenance.
5. The attention and alignment models are novel experimental implementations,
   not claims of reproduction of a named published algorithm.

## Conclusions

The exact requested computational scope is complete and reproducible: 960 new
successful runs across Phases A and C, plus comparisons to the immutable
historical benchmark. The strongest defensible claims are the complete-grid
method rankings and the exact paired self-training effects. The geometric family
is now operational and non-collapsed at the representation level, but its
unlabeled-data benefit should remain qualified because a full-grid matched
supervised control was not run. No additional models or experimental phases are
needed to complete the supervisor's stated scope.

## Deliverable index

- Canonical Phase A: `results/raw/tfm_frozen_screen.csv`
- Canonical Phase C: `results/raw/focused_tfm_ssl.csv`
- Standard Phase-C aggregates: `results/aggregated/focused_tfm_ssl/`
- Final machine-readable tables: `tables/`
- Publication figures (PNG and PDF): `figures/`
- Reproducibility manifest: `run_manifest.json`
- Integrity audit: `integrity_validation.json`
