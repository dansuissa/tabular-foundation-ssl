# TFM-SSL Repository Audit

**Audit host:** `dsiuriofir01` (2026-07-22)  
**Project root:** `projects/ssl-foundation-models/ssl_tabular_benchmark/`  
**Canonical historical artifact (IMMUTABLE):** `results/raw/low_class_wave_paper_methods.csv`  
**Git status:** not a git repository (no `.git` under this tree or its parent). Dirty-tree / commit hashing must be designed for optional later git init or external VCS.

This audit is based on reading the current source, configs, scripts, and result tooling—not on prior chat summaries.

---

## 1. Current execution flow (dataset → result row)

Entry point: `python -m src.run_benchmark` (`src/run_benchmark.py`).

```
configs/datasets.yaml  ─┐
configs/benchmark.yaml ─┤
CLI filters            ─┘
        │
        ▼
load_yaml → select dataset entries / method group / seeds / budgets
        │
        ▼
for each dataset:
  dataset_from_config → DatasetSpec
  load_dataset(spec)          # src/data.py → OpenML fetch_openml(as_frame=True)
        │
        ▼
  for method × seed × label_budget:
    skip if --resume and key already success
        │
        ▼
    make_ssl_split(...)       # src/splits.py → DataSplits (pandas)
        │
        ▼
    fit_label_encoder(y_labeled_train ∪ y_unlabeled_train)   # !! see leakage §7
    transform_labels → int arrays for labeled_train and TEST only
        │
        ▼
    build_preprocessor(X_labeled_train)   # dtype inference from labeled only
    fit_transform_preprocessor(fit on labeled+unlabeled train)
      transforms labeled, unlabeled, val, test → float32 numpy matrices
      runner DISCARDS transformed val (binds to `_`)
        │
        ▼
    build_model(method) → supervised | semi_supervised | rpl | neural
    run_model(model, X_lab_proc, y_lab, X_unlab_proc, X_test_proc)
        # fit(..., X_unlabeled=...) then predict/predict_proba on TEST only
        │
        ▼
    compute_metrics(y_test, pred, proba) → metric_* columns
        │
        ▼
    append_result_row → CSV append after EACH run (mode="a")
```

Aggregation is offline: `python -m src.aggregate_results --input <raw.csv> --output_dir ...`.

Historical low-class wave assembly (procedural, no single launcher):

1. `scripts/preflight_low_class_wave.py`
2. Reuse phoneme/spambase via `scripts/filter_reuse_phoneme_spambase.py`
3. Per-dataset new raw CSVs
4. `src/combine_results.py` (rejects duplicate keys)
5. Aggregate + `scripts/build_low_class_wave_report.py`

---

## 2. Where raw pandas is lost

| Stage | Object type | Notes |
|---|---|---|
| `load_dataset` | `pd.DataFrame` / `pd.Series` | Native OpenML frame with original dtypes |
| `make_ssl_split` | `DataSplits` of pandas | Raw splits preserved inside `DataSplits` |
| `build_preprocessor` + `fit_transform_preprocessor` | `np.ndarray` float32 | Median impute + StandardScaler (num); constant impute + OneHot (cat) |
| `run_model` / all `fit` signatures | `np.ndarray` only | Protocol in `src/models/__init__.py` |

**Critical conversion point:** `src/run_benchmark.py` `run_single_experiment()` after `fit_transform_preprocessor`. From that line onward, methods never see:

- original categorical columns,
- original numeric scales,
- pandas indices,
- raw string categories.

TabPFN-3 and TabICLv2 expect native tabular input (pandas / mixed dtypes). Feeding them the shared one-hot+scaled matrix would be scientifically wrong and must be avoided.

Also discarded today:

- `splits.X_val` / `splits.y_val` after transform (val features transformed then dropped; val labels never encoded or passed),
- raw frames after conversion (only processed matrices retained in locals).

---

## 3. How validation is (not) passed to models

`DataSplits` **does** create an external validation set (`val_size_from_labeled=0.2`, strategy `stratified_labeled_val` or `no_val_low_label_multiclass`).

Runner behavior:

- Transforms `splits.X_val` but assigns it to `_` (unused).
- Never encodes `splits.y_val`.
- Never passes validation into `run_model` / `fit`.

Neural methods (`src/models/neural_ssl.py`) carve an **internal** stratified holdout from already-reduced labeled train via `internal_val_split` (`src/models/torch_utils.py`). Effects:

1. External protocol val is unused for early stopping / calibration.
2. Neural methods further shrink labeled train for their internal val.
3. Risk-controlled TFM loops cannot use the intended external val without a runner change.

---

## 4. How unlabeled data is passed

All model `fit` signatures accept `X_unlabeled: np.ndarray | None`.

| Family | Unlabeled usage |
|---|---|
| Supervised (LR/RF/XGB/LGBM/CatBoost/MLP) | Ignored |
| Self-training / RPL | Pseudo-label pool (processed features) |
| Graph SSL | Graph construction (cap 20k rows, kNN retry ladder) |
| Neural SSL | Reconstruction / pretrain / consistency on labeled∪unlabeled processed features |

Unlabeled **labels** (`y_unlabeled_train`) exist in `DataSplits` for accounting / class-count JSON, but must not enter training. Current exception: they are concatenated into `LabelEncoder.fit` (§7).

---

## 5. Changes required for native TFM input (without breaking existing methods)

### Must add (compatible extension)

1. **Dual-view containers** after splitting, before method dispatch:
   - raw pandas views for TFMs,
   - processed numpy views for classical/neural methods.
2. **Method capability flags** (`needs_raw`, `needs_processed`, `needs_external_val`, `device`).
3. **Extended `run_model` / `FitContext`** that can pass:
   - raw + processed matrices,
   - optional external `X_val` / `y_val`,
   - metadata (seed, budget, class map).
4. **Adapter wrapper** so existing models keep `fit(X_labeled, y_labeled, X_unlabeled=None)` on processed arrays only—bit-identical scientific path.
5. **Label encoding policy** that never fits on unlabeled true labels (encode from labeled∪validation class universe, or from full dataset class list known at load time without using unlabeled y for training logic).
6. **Result schema extensions** (checkpoint ids, env hash, GPU mem, etc.) without rewriting old CSVs; new waves use sharded JSON → combined CSV.
7. **Sharded writers** for Slurm array parallelism (§6).

### Must not do

- Overwrite or “clean” `results/raw/low_class_wave_paper_methods.csv` or sibling historical reports.
- Force TabPFN/TabICL through the shared one-hot pipeline.
- Require old methods to accept pandas.

---

## 6. Concurrency risks in single-CSV append

`append_result_row` uses `pandas.DataFrame.to_csv(..., mode="a", header=not exists)`.

Risks under Slurm job arrays / multi-process:

| Risk | Mechanism |
|---|---|
| Interleaved writes | Two processes append simultaneously → torn lines / corrupt CSV |
| Header races | Both see “file missing” → duplicate headers |
| Resume races | Both load skip-set, both run same key, both append → duplicates |
| Schema rewrite races | `--resume` path calls `_ensure_csv_schema` which rewrites whole file while others append |
| NFS close-to-open consistency | Home is NFS (`psa.local.biu.ac.il:/ifs/a/home`); append without locking is unsafe |
| Partial last line | Crash mid-write leaves unreadable row; resume may mis-parse |

`combine_results.py` correctly rejects duplicate keys **after the fact**, but does not prevent parallel corruption.

**Required design:** one atomic artifact per run (`results/raw_shards/<wave>/<run_id>.json`), temp file + `os.replace`, combine only after validation.

---

## 7. Reproducibility and leakage risks

### Leakage / protocol

1. **Raw→processed loss for TFMs** if someone naively reuses the runner (wrong features).
2. **External val unused**; neural internal val double-holds out labeled data.
3. **`LabelEncoder.fit` on unlabeled true labels** (`run_benchmark.py`). Split rules require all classes in labeled budget, so index maps usually match, but this still violates the “unlabeled labels are hidden” rule and is unsafe if rules change.
4. **Preprocessor column typing** inferred from `X_labeled_train` only; rare categorical levels only in unlabeled are handled by `handle_unknown="ignore"` (OK), but dtype inference could mis-classify columns if labeled subset is atypical.
5. **Graph SSL / RPL / self-training** correctly do not see unlabeled y (use `-1` / predictions only)—good.
6. **Test features** only used at predict time—good for inductive track.
7. **OpenML caching / download nondeterminism** if rows change upstream (mitigate by caching + recording openml version / local hash).

### Reproducibility

1. **No git repository** → cannot record commit today.
2. **`set_seed`** sets `random` + `numpy` only; does not set torch CUDA deterministic flags except inside neural `set_global_determinism`.
3. **Tree libs** use their own RNGs via `random_state`—generally OK if passed.
4. **No pinned lockfile**; `requirements.txt` uses minimum versions only; no torch pin.
5. **System Python 3.9.25** on cluster; TabPFN/TabICL require **Python ≥3.10**.
6. **No unit/integration tests** under the project (no `tests/`, no `test_*.py`).
7. **Optional imports** (xgboost/lightgbm/catboost/torch) fail per-run—good for isolation, bad if silently missing on cluster jobs without preflight.

---

## 8. Interfaces: extend vs replace

| Interface | Verdict | Reason |
|---|---|---|
| `DatasetSpec` / `TabularDataset` | **Extend** | Still correct for OpenML load |
| `DataSplits` | **Extend** (or wrap) | Keep pandas splits; add dual-view builder downstream |
| `build_preprocessor` / `fit_transform_preprocessor` | **Extend** | Keep for classical path; do not feed TFMs |
| `BaseModel` Protocol + `run_model` | **Replace behind adapter** | Current ndarray-only protocol insufficient; keep adapter for old models |
| `build_*_model` registries | **Extend** | Add TFM builders + capability metadata |
| CSV `RESULT_COLUMNS` schema | **Extend for new waves** | Never rewrite historical files; new columns OK in new combined CSVs |
| `append_result_row` single CSV | **Replace for cluster** | Keep as offline combine target format only |
| `aggregate_results.py` | **Extend** | Consume combined CSV; ranking logic reusable |
| `combine_results.py` | **Extend** | Basis for shard collector with stricter validation |
| Historical raw/report artifacts | **Freeze** | Immutable |

---

## 9. Config / method registry snapshot (code-verified)

From `configs/benchmark.yaml`:

- `full_first_wave_methods` (14 classical)
- `neural_ssl_methods`: `sslae`, `vime`, `scarf`
- `neural_ssl_experimental_methods`: `vime_lite`
- `paper_methods_no_vime_lite` (17) — canonical wave group
- Defaults: seeds `[0..4]`, budgets `[50,100,250,500,1000]` (canonical used 0/1/2 × 50/100/250/500)

From `configs/datasets.yaml`: 15 datasets configured; group `low_class_wave` = 10 datasets (excludes `letter`).

---

## 10. Tests and scripts inventory

**Tests:** none found.

**Scripts:** `run_mini_wave.ps1`, `aggregate_mini_wave.ps1`, `preflight_low_class_wave.py`, `filter_reuse_phoneme_spambase.py`, `build_low_class_wave_report.py`, `analyze_all_methods.py`, `make_supervisor_figures.py`.

**Optional deps:** lazy import with install hint for xgboost/lightgbm/catboost; torch lazy via `require_torch()`.

---

## 11. Audit conclusions for TFM phase-0

1. The scientific classical path is coherent and leakage-aware for **processed** methods, with the LabelEncoder+unused-val caveats above.
2. The runner is **not TFM-ready** and **not cluster-parallel-safe**.
3. Extension must be **dual-view + capability-dispatched + sharded I/O**, with a thin adapter preserving old method behavior.
4. Historical results remain the baseline; new work must write new wave names under new paths.
