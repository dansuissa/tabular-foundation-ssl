# Current project state

Recovered on 2026-07-28 from the live shared project and laboratory storage.

## Location and execution environment

- Host visible to Codex: `dsiuriofir01`
- Project root: `/home/dsi/suissad4/projects/ssl-foundation-models/ssl_tabular_benchmark`
- Laboratory artefact root: `/private/ofirlin-lab/suissad4`
- The project is not a Git work tree. The Phase-A submission recorded source-tree hash
  `b5ee701b1c0a40017af2f98820534067ab190a5950ec2fc5503808e51ed4e215`.
- TFM environment: Python 3.11, torch 2.6.0+cu124, CUDA 12.4,
  tabpfn 8.1.0, tabicl 2.1.1, scikit-learn 1.9.0, pandas 3.0.3,
  numpy 2.4.6. CUDA is unavailable in the current non-Slurm shell; completed
  shards record A100 80 GB execution.
- OpenML cache is populated. The verified TabPFN-3 checkpoint is
  `tabpfn-v3-classifier-v3_default.ckpt` (SHA-256
  `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`).
  The pinned TabICLv2 checkpoint is `tabicl-classifier-v2-20260212.ckpt`.
- No secret or token content was read or emitted during recovery.

## Phase A scheduler recovery

The submission manifest records:

| Role | Job ID | Submitted shape |
|---|---:|---|
| Small-dataset array | 18409306 | `0-191%2`, GPU, 64 GB, 2 hours |
| Large-dataset array | 18409307 | `192-239%2`, GPU, 96 GB, 4 hours |
| Validation dependency | 18409308 | after both arrays |
| Reporting dependency | 18409309 | after validation |

Direct `squeue`, `sacct`, and `scontrol` queries are unavailable in this Codex
runtime: the local Slurm client rejects its configuration because the configured
`SlurmUser` is absent, and direct SSH authentication to the host is unavailable.
Consequently, exact Slurm accounting state counts cannot be re-read from the
controller. Terminal state is established from the completed array logs,
per-run Slurm metadata, dependent-job logs, and artefacts:

- 192 small-array task log pairs and 48 large-array task log pairs exist.
- All 240 expected task IDs produced one successful atomic result shard.
- No Phase-A task is plausibly active: all expected outputs and both dependency
  job outputs were written on 2026-07-22.
- There are no missing, failed, cancelled, timed-out, or never-run benchmark
  keys in the final shard set. Historical retries may have used replacement
  Slurm job IDs; every retained run key has exactly one final shard.

## Phase A shard validation

Independent recovery validation against `task_map.json` found:

| Check | Result |
|---|---:|
| Expected run keys | 240 |
| Parseable result shards | 240 |
| Unique run keys | 240 |
| Successful shards | 240 |
| Failed shards | 0 |
| Missing keys | 0 |
| Unexpected keys | 0 |
| Duplicate keys | 0 |
| Unique `(Slurm job, array task)` pairs | 240 |

The grid contains exactly two methods, ten canonical datasets, four label
budgets, and three seeds. All primary metrics are finite. The 96 missing average
precision values are the expected unsupported multiclass cases, not corrupt
shards.

The combined CSV exists at `results/raw/tfm_frozen_screen.csv` with 240 rows
(SHA-256 `05f11ac7e5da98ce783b41bd46331b4d55fddc6d339136b625c72a4a2eae0d02`).
The immutable historical file `results/raw/low_class_wave_paper_methods.csv`
was not modified.

## Validation and reporting status

- Job 18409308 completed its shard gate successfully: expected 240, actual 240,
  failed 0, missing 0, duplicates 0.
- Job 18409309 successfully combined and validated the 240 rows, but its generic
  aggregation step raised `KeyError: 'dataset_id'`.
- A later Phase-A post-processing step produced
  `results/aggregated/tfm_frozen_screen/coverage_by_cell.csv` and four compact
  report tables under `results/reports/tfm_frozen_screen/`.
- During this recovery, aggregation was made tolerant of wave CSVs that omit
  the optional top-level `dataset_id`. Full standard aggregate tables and plots
  were then regenerated successfully on 2026-07-28. Phase A is complete and no
  Phase-A benchmark runs need resubmission.

## Existing implementation and validation artefacts

The source tree already contains native-view TabPFN-3 and TabICLv2 adapters,
shared TFM self-training utilities, sparse graph/Laplacian code, retrieval
attention, prototype alignment, geometric attention, leakage guards, atomic
shards, resume logic, cluster submission scripts, diagnostics, and synthetic
tests. Existing validation artefacts record 18 passed and 1 skipped in the TFM
environment, successful TFM smokes, and geometric-attention diagnosis and
ablations. These are prior artefacts and will be re-audited against the final
requested method names and scientific protocol before any Phase-B/C wave.

## Safe resume decision

Do not resubmit jobs 18409306 or 18409307 and do not rerun any Phase-A model
cell. Preserve all 240 validated shards. The next safe actions are:

1. audit the existing implementations, registrations, tests, and method-name
   mapping;
2. run the required tests and geometric-attention gates;
3. submit only genuinely missing Phase-B/C run keys under a new recorded
   source/configuration hash.

## Continuation on 2026-07-28

- Final requested method names were registered and restricted to the focused
  configuration; see `results/validation/final_method_mapping.md`.
- The aggregation schema defect was fixed and all Phase-A aggregate tables and
  plots regenerated successfully.
- Current validation suite: 22 passed, 1 skipped.
- Phase-B source snapshot:
  `9c0a390446a60c1d839601aaeeaa4de606a9ab1ea55a231c91555e3a9486a539`.
- Focused Phase-B Slurm array: job `18893477`, tasks `0-71%2`, 72 expected
  cells (6 methods × 6 representative datasets × 2 budgets × seed 0), one GPU,
  96 GB RAM, 4-hour limit. Initial scheduler state was `PENDING (Resources)`.
- Phase C has not been submitted and remains gated on complete Phase-B shard,
  leakage, probability, collapse, and method-specific diagnostic validation.

## Final completion on 2026-07-30

### Phase B and combined-model repair

- Focused Phase B job `18893477` produced all 72 expected shards: 70 successful
  and two failed `geometric_attention_ssl` Jannis cells caused by an invalid
  CUDA launch configuration. The other five methods passed every requested
  Phase-B cell.
- The combined method's prior segment collapse was reproduced and localized to
  retrieval: the much larger unlabeled pool could occupy all top-k memory
  positions. Full-pool diagnostics were batched and the retrieval policy was
  changed to a deterministic balanced labeled/unlabeled top-k composition.
- The repaired combined model retained supervised warm-up, confidence and loss
  ramp-up, class-conditional alignment, sparse Laplacian regularization, and
  training-only retrieval memory.
- Repair gate job `18894597` completed four representative segment/Jannis cells.
- Final attention-family gate job `18895119` completed 12/12 cells successfully
  for unlabeled attention, embedding alignment, and the combined method on
  segment/Jannis at budgets 50 and 250. Segment balanced accuracy was above
  chance and the combined diagnostics showed finite probabilities, both memory
  types receiving attention, effective embedding rank 32, and no representation
  collapse.
- Final source snapshot used by the attention-family gate and final Phase-C
  arrays:
  `b9c1f48daae0a38670909bfcee77dc8488c635820805cd7f3886f35130ed7e69`.

### Phase C execution

Only missing valid final-method keys were submitted; no Phase-A or completed
Phase-C key was rerun.

| Wave | Slurm job | Methods | Expected | Success | Failed/missing/duplicate |
|---|---:|---|---:|---:|---:|
| `focused_ssl_phase_c_validated_small` | `18894620` | TFM self-training ×2, Laplacian | 288 | 288 | 0 |
| `focused_ssl_phase_c_validated_large` | `18894621` | TFM self-training ×2, Laplacian | 72 | 72 | 0 |
| `attention_family_phase_c_small` | `18895315` | attention, alignment, combined | 288 | 288 | 0 |
| `attention_family_phase_c_large` | `18895316` | attention, alignment, combined | 72 | 72 | 0 |

Phase C therefore contains exactly 720 successful unique keys:
10 datasets × 6 methods × 4 budgets × 3 seeds. All primary and secondary
metrics are finite except average precision on multiclass tasks, where the
metric is intentionally unsupported. The canonical combined file is:

`results/raw/focused_tfm_ssl.csv`

SHA-256:
`0a3bca9fc2728dabf7ac846a7a477c64ac5536feb582f28c92a2594d50b06a9a`

Standard aggregates and plots are under:

`results/aggregated/focused_tfm_ssl/`

### Final validation and deliverables

- Phase A: 240/240 successful; no missing, failed, or duplicate keys.
- Phase C: 720/720 successful; no missing, failed, or duplicate keys.
- Total new final-grid results: 960 successful runs.
- Final validation suite:
  `25 passed, 1 skipped in 27.14s`.
- The immutable historical file remains at
  `results/raw/low_class_wave_paper_methods.csv`; its retained SHA-256 is
  `b017a950054a5045a4a39c41362d3dc3eef56603d1db4d62d4ffbfbdc43e8ad5`.
- Final report, machine-readable tables, publication figures, integrity audit,
  and reproducibility manifest are under:
  `results/reports/bar_ilan_tfm_ssl_final/`.

The supervisor-requested project scope is complete. No Phase D or additional
foundation-model family is required.
