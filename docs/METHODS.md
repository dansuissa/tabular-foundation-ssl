# Methods and implementation fidelity

This document defines the 25 methods behind the canonical result claims. The
authoritative machine-readable registry is
[`src/method_capabilities.py`](../src/method_capabilities.py); method-specific
hyperparameters live in [`configs/benchmark.yaml`](../configs/benchmark.yaml).

## Shared experiment interface

Every method receives one `FitContext` containing the labeled training split,
the unlabeled training pool, a validation split, class order, seed, and both
feature views when applicable. The dispatcher validates each method's declared
capabilities before fitting:

- **Native view:** the original pandas columns and dtypes, used by TabPFN-3 and
  TabICL v2.
- **Processed view:** numeric median imputation and scaling plus categorical
  constant imputation and one-hot encoding. The preprocessor is fitted on
  labeled and unlabeled training features only.
- **Inductive methods:** may adapt on the training pool; test features are used
  only when `predict` or `predict_proba` is called.

Probabilities are reordered into the benchmark's fixed class order, checked for
finite values, normalized, and then scored. Failures produce result records with
status and error metadata instead of terminating a wave.

## The 25 canonical methods

### Supervised baselines

`logistic_regression`, `random_forest`, `xgboost`, `lightgbm`, `catboost`, and
`mlp` are trained only on the labeled training subset. They establish linear,
bagged-tree, boosted-tree, and feed-forward neural reference points under the
same split and preprocessing protocol.

The XGBoost, LightGBM, and CatBoost adapters use their official Python packages.
Logistic regression, random forest, and MLP use scikit-learn implementations.
All six ignore the unlabeled pool after preprocessing.

### Graph propagation

`label_propagation` and `label_spreading` are the scikit-learn graph-based SSL
baselines. Their graph is built from the processed labeled and unlabeled training
rows. The historical wave preserves 9 label-propagation and 15 label-spreading
failures, chiefly resource failures on large graphs; these cells are never
silently removed from coverage or ranking.

### Classical self-training

`self_training_lr`, `self_training_xgboost`, `self_training_lightgbm`, and
`self_training_catboost` iteratively add confident predictions from the
unlabeled training pool to the respective labeled learner. They are classical
pseudo-label baselines, distinct from the guarded TFM engine described below.

### Robust pseudo-labeling

`rpl_lr` and `rpl_lite_xgboost` add reliability and plausibility gates to the
pseudo-label selection loop. The latter is deliberately named “lite”: it is a
faithful family-level reimplementation, not a claim of exact reproduction of an
unmodified upstream package.

### Neural SSL

- `sslae` trains an encoder/classifier on labeled rows and an autoencoder on
  labeled plus unlabeled rows. Its objective is
  `classification_loss + λ × reconstruction_loss`. It is a supervised-
  autoencoder-style baseline, not a named-paper reproduction.
- `vime` implements the faithful core of VIME: mask estimation and feature
  reconstruction pretraining on the full training pool, followed by labeled
  classification with consistency between independently corrupted unlabeled
  views.
- `scarf` implements the faithful core of SCARF: feature corruption,
  contrastive pretraining with an NT-Xent/InfoNCE objective, and a supervised
  classification head fitted on labeled examples.

A stratified labeled holdout controls early stopping when possible.
The run metadata records the validation strategy, epochs, losses, and pretraining
or fine-tuning counts.

### Frozen tabular foundation models

- `tabpfn3` uses the official TabPFN 8.1.0 classifier and the verified
  `tabpfn-v3-classifier-v3_default.ckpt` checkpoint.
- `tabiclv2` uses TabICL 2.1.1 and the pinned
  `tabicl-classifier-v2-20260212.ckpt` checkpoint.

Both are few-shot predictors: labeled examples form the context and no model
weights are trained by this repository. Native mixed-type inputs are preserved.
Checkpoint hashes, package versions, device information, cold-load time, warm
inference time, and relevant model options are serialized into every result row.
The checkpoints themselves are not redistributed.

### Validation-guarded TFM self-training

`tabpfn3_self_training` and `tabiclv2_self_training` share one deterministic
engine:

1. Fit the frozen backbone on labeled context and calibrate its probabilities
   using the validation split.
2. Select confident unlabeled predictions with class-balanced per-class and
   total caps.
3. Add the hard pseudo-labels to context and refit for up to three rounds.
4. Evaluate each round on validation balanced accuracy and log-loss.
5. Return the best admissible round, or the untouched frozen predictor when no
   round passes the fixed guard.

This is risk-controlled hard-label self-training. It must not be described as a
soft-label LoopTabFM reproduction. The guard selected round zero in 86/120
TabPFN-3 runs and 83/120 TabICL v2 runs, explaining why average improvements
are small and why the variants remain safe.

### Geometric and representation SSL

`laplacian_ssl` builds a sparse, locally scaled k-nearest-neighbor graph from
training representations. A trainable encoder/classifier minimizes supervised
cross-entropy plus a ramped graph smoothness term

\[
L_{graph} = \sum_{(i,j) \in E} w_{ij}\,\lVert z_i-z_j\rVert_2^2.
\]

Graphs never contain validation or test nodes. Node/edge counts, connected
components, isolated nodes, affinity statistics, and build time are logged.

`unlabeled_attention_ssl` learns a representation and performs batched top-k
retrieval over labeled and unlabeled **training** memory. It excludes each
example's self-neighbor, balances memory use, and records attention entropy,
labeled/unlabeled mass, memory size, and leakage flags.

`embedding_alignment_ssl` forms class-conditional prototypes from labeled and
confidence-filtered unlabeled embeddings. Its objective attracts reliable
examples toward their class prototype and separates prototypes. Embedding
variance, intra-class distance, inter-prototype distance, accepted-example
counts, and collapse warnings are retained.

`geometric_attention_ssl` composes supervised loss, pseudo-label consistency,
Laplacian regularization, retrieval attention, prototype alignment, and class
separation with warm-up and ramp schedules. An early chance-level segment
collapse was traced to unrestricted retrieval filling all neighbors from the
much larger unlabeled pool. Balanced labeled/unlabeled retrieval repaired the
failure and passed a representative segment/Jannis gate before the final grid.

## Fidelity labels

The registry uses four deliberately narrow labels:

| Label | Meaning |
|---|---|
| `official` | The official package or a thin adapter around it. |
| `faithful_reimplementation` | Reimplemented from the stated method family with documented deviations. |
| `paper_core` | The central published mechanism is present, but the benchmark integration is not an exact upstream reproduction. |
| `novel_experimental` | A project-specific research implementation; no paper-reproduction claim. |

Fidelity is about implementation provenance, not expected performance.

Primary papers and official model resources are collected in
[`REFERENCES.md`](REFERENCES.md).
