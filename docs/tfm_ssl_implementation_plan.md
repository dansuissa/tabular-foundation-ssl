# TFM-SSL Implementation Plan (Phase 0 → Phase 1)

> **Historical planning record.** This document preserves the Phase-0 design
> state and intentionally contains pre-implementation labels such as “TBD.” For
> the completed method inventory, validated results, and reproduction commands,
> use [`METHODS.md`](METHODS.md), [`RESULTS.md`](RESULTS.md), and
> [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

**Status:** Phase-0 infrastructure design (this document).  
**Do not implement TFM SSL methods until dual-view runner + sharded I/O land.**  
**Immutable:** `results/raw/low_class_wave_paper_methods.csv` and all historical `results/**` artifacts.

Companion docs:

- `docs/tfm_ssl_repository_audit.md`
- `docs/cluster_environment.md`

---

## 0. Goals

1. Keep classical 17-method science reproducible via adapters.
2. Add TabPFN-3 and TabICLv2 as **native-input** foundation-model backbones only (no other TFMs unless requested).
3. Make Bar-Ilan Slurm execution safe (shards, caches, envs, auth).
4. Plan future SSL method families on top of those two backbones.

---

## 1. Environment decision

### Decision: **two environments now; third deferred**

| Env | Purpose | Python | Location |
|---|---|---|---|
| **ssl-core** | Existing sklearn/tree/neural-lite analysis + aggregation | 3.11 | `/private/ofirlin-lab/suissad4/envs/ssl-core` |
| **ssl-tfm** | PyTorch+CUDA, TabPFN-3 (`tabpfn`), TabICLv2 (`tabicl`), CatBoost, scientific stack | 3.11 | `/private/ofirlin-lab/suissad4/envs/ssl-tfm` |
| **ssl-representation** | **Deferred** until SeBA/STUNT/Laplacian/D2R2-c conflict with `ssl-tfm` is demonstrated | — | reserved under `environment/representation/` |

**Why not one env:** TabPFN requires Python ≥3.10 and torch≥2.5; system is 3.9 with read-only conda base. Splitting core vs TFM isolates CUDA/torch churn from classical CSV analysis.

**Why not three yet:** Representation methods are not in phase-0/1 implementation; creating empty conflicting stacks wastes quota.

**Bootstrap tool:** user-space **micromamba** under `/private/ofirlin-lab/suissad4/envs/micromamba` (system conda base is read-only; no mamba/uv preinstalled).

### Pin targets (to be locked at first successful bootstrap)

Record exact versions in `environment/*/lock/` after first solve—do not invent fake hashes before install.

Intended pins (validate on cluster GPU):

| Package | Intent |
|---|---|
| python | 3.11.x |
| pytorch + torchvision | CUDA 12.x wheel compatible with driver 610 / sm_80 |
| tabpfn | latest stable exposing TabPFN-3 default classifier |
| tabicl | ≥2.0 (TabICLv2 default checkpoint `tabicl-classifier-v2-20260212.ckpt`) |
| catboost, numpy, pandas, scikit-learn, pyyaml, matplotlib | align with existing benchmark needs |
| huggingface_hub | for TabICL / optional HF assets |

Auth env vars (never commit values):

- `TABPFN_TOKEN` (+ `TABPFN_NO_BROWSER=1` on cluster)
- `HF_TOKEN` if required for gated assets

Cache env vars:

```bash
export LAB_ROOT=/private/ofirlin-lab/suissad4
export HF_HOME=$LAB_ROOT/caches/hf
export TORCH_HOME=$LAB_ROOT/caches/torch
export XDG_CACHE_HOME=$LAB_ROOT/caches/xdg
export PIP_CACHE_DIR=$LAB_ROOT/caches/pip
export UV_CACHE_DIR=$LAB_ROOT/caches/uv
export TABPFN_CACHE_DIR=$LAB_ROOT/caches/tabpfn   # confirm against upstream; fall back to HF_HOME
export OPENML_DATA_HOME=$LAB_ROOT/caches/openml   # confirm sklearn/openml actual keys
export SKLEARN_DATA=$LAB_ROOT/caches/openml
```

Bootstrap scripts must **probe** actual cache paths after first import rather than assuming unsupported variables.

---

## 2. Architecture redesign (design only — implement next phase)

### 2.1 Dataclasses

```python
@dataclass(frozen=True)
class MethodCapabilities:
    name: str
    input_view: Literal["processed", "raw", "both"]
    needs_unlabeled: bool
    needs_external_validation: bool
    device: Literal["cpu", "gpu", "any"]
    env: Literal["ssl-core", "ssl-tfm", "ssl-representation"]
    supports_predict_proba: bool = True

@dataclass
class DatasetViews:
    # raw
    X_labeled_raw: pd.DataFrame
    X_unlabeled_raw: pd.DataFrame
    X_validation_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    # processed
    X_labeled_processed: np.ndarray
    X_unlabeled_processed: np.ndarray
    X_validation_processed: np.ndarray
    X_test_processed: np.ndarray
    # labels (encoded ints)
    y_labeled: np.ndarray
    y_validation: np.ndarray | None
    y_test: np.ndarray
    label_encoder: Any
    class_names: list[str]
    # meta
    dataset: str
    seed: int
    n_labeled: int
    validation_strategy: str

@dataclass
class FitContext:
    views: DatasetViews
    random_state: int
    method_config: dict[str, Any]
    # explicit leakage fence: never includes y_unlabeled or test labels

@dataclass
class PredictionResult:
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    training_meta: dict[str, Any]
```

### 2.2 Hard leakage rules (must be enforced in runner unit tests)

Allowed for fit/adaptation/pseudo-label/graph/context selection:

- labeled-train features (+ labels)
- unlabeled-train features (**no labels**)
- optional validation features+labels **only** for calibration, early stopping, risk control, guarded selection

Forbidden:

- test features in any adaptation / PL / graph / retrieval bank used to update the model before final predict
- unlabeled true labels
- fitting LabelEncoder on unlabeled true labels (fix current bug)

### 2.3 Runner dispatch

```text
build DatasetViews
capabilities = registry[method]
if capabilities.input_view == "processed":
    adapter_fit_processed(...)   # bit-compatible with today’s run_model
elif capabilities.input_view == "raw":
    tfm_fit_raw(FitContext)
```

Pass external validation into methods with `needs_external_validation=True` without changing old method signatures (adapter ignores it).

### 2.4 Files to add/modify in next implementation phase

**Add:**

- `src/views.py` — DatasetViews builder
- `src/method_capabilities.py` — registry
- `src/models/legacy_adapter.py` — wrap existing models
- `src/models/tfm_tabpfn.py` — TabPFN-3 wrappers (later)
- `src/models/tfm_tabicl.py` — TabICLv2 wrappers (later)
- `src/results_io/shards.py` — atomic shard write/read
- `src/results_io/manifest.py` — run_id + hashes
- `src/run_single.py` — single-run CLI for Slurm tasks
- `tests/test_leakage_guards.py`
- `tests/test_views.py`
- `tests/test_shards.py`
- `tests/test_legacy_adapter_parity.py`

**Modify (carefully):**

- `src/run_benchmark.py` — dual-view + capabilities + optional shard output; keep CSV mode for local serial runs
- `src/preprocessing.py` — return views helper; do not change classical transforms
- `src/models/__init__.py` — dispatch by capabilities
- `configs/benchmark.yaml` — TFM method groups + resource tags
- `configs/datasets.yaml` — unchanged scientifically; maybe add `tfm_smoke` group

**Never modify:**

- `results/raw/low_class_wave_paper_methods.csv`
- any historical section asset under `results/reports/main_report/`
- other historical `results/raw/*.csv` / aggregated historical waves (treat as immutable)

---

## 3. Cluster-safe result writing

### Layout

```text
/private/ofirlin-lab/suissad4/results/raw_shards/<wave>/<run_id>.json
/private/ofirlin-lab/suissad4/results/combined/<wave>.csv
project/results/raw/<wave>.csv   # optional mirrored combine for notebooks
```

### `run_id`

Deterministic hash of:

`dataset | method | seed | n_labeled | method_config_hash | split_protocol_version | code_version`

Where `code_version` = git commit if available else `nogit:<content_hash_of_src>`.

### Shard JSON required fields

Identity + metrics (compatible with today’s columns) **plus**:

- `git_commit`, `dirty_tree`
- `env_name`, `env_lock_hash`
- `method_config_hash`
- `checkpoint_ids` (TabPFN/TabICL ckpt names)
- `hostname`, `slurm_job_id`, `slurm_array_task_id`
- `gpu_name`, `cuda_version`
- `cold_model_load_seconds`, `warm_inference_seconds`
- `peak_gpu_mem_mb`, `runtime_seconds`
- `status`, `error_message`

### Writer algorithm

1. Write `<run_id>.json.tmp`
2. `fsync` + `os.replace` → `<run_id>.json`
3. Resume: skip if shard exists with `status=success`
4. Collect: validate JSON schema → reject dups → write combined CSV atomically

---

## 4. Scheduler scripts (Slurm backend)

Provided under `scripts/cluster/` (this phase: skeletons + generators).

| Script | Role |
|---|---|
| `env.sh` | LAB_ROOT, caches, thread vars |
| `preflight.sh` | host/GPU/partition/auth/cache checks |
| `bootstrap_micromamba.sh` | install micromamba to lab storage |
| `bootstrap_ssl_core.sh` | create ssl-core env |
| `bootstrap_ssl_tfm.sh` | create ssl-tfm env + CUDA torch |
| `check_auth.py` | verify TABPFN_TOKEN / HF token presence (no secrets printed) |
| `warm_model_cache.py` | serial locked download of TabPFN-3 + TabICLv2 ckpts |
| `warm_openml_cache.py` | prefetch low_class_wave OpenML ids |
| `submit_wave.py` | emit sbatch array from method×dataset×seed×budget grid |
| `run_single.sh` | one Slurm task → one shard |
| `monitor_wave.py` | count shards success/fail |
| `collect_wave.py` | shards → combined CSV |
| `cancel_wave.py` | scancel by job name/id |

Resource profiles:

- `gpu_tfm`: 1×A100, 8 CPUs, 64G, partition `p_uriofir`
- `cpu_classical`: 8 CPUs, 32G, partition `cpu192G-48h`

---

## 5. Final method registry (names reserved)

### 5.1 Existing (keep)

Classical 14 + neural `{sslae,vime,scarf}` + experimental `vime_lite`.

### 5.2 TFM backbones (phase 1)

| Method name | Backbone | Input view | Device | Env | Official? |
|---|---|---|---|---|---|
| `tabpfn3_supervised` | TabPFN-3 | raw | gpu | ssl-tfm | Official `tabpfn` |
| `tabicl_v2_supervised` | TabICLv2 | raw | gpu | ssl-tfm | Official `tabicl` |

### 5.3 Planned SSL families (phase 2+, names frozen)

| Method name | Idea | Expected view | Device | Cost | Official vs reimpl |
|---|---|---|---|---|---|
| `tabpfn3_frozen` | Frozen TFM baseline (explicit alias) | raw | gpu | med | official API |
| `tabicl_v2_frozen` | Frozen TFM baseline | raw | gpu | med | official API |
| `tabpfn3_self_training` | Iterative PL self-training | raw | gpu | high | reimpl loop |
| `tabicl_v2_self_training` | Iterative PL self-training | raw | gpu | high | reimpl loop |
| `looptabfm_tabpfn3` | Risk-controlled LoopTabFM-style | raw+val | gpu | very high | reimpl |
| `looptabfm_tabicl_v2` | Risk-controlled LoopTabFM-style | raw+val | gpu | very high | reimpl |
| `cast_tabpfn3` | CAST density-aware PL | raw+proc | gpu | high | reimpl |
| `cast_tabicl_v2` | CAST density-aware PL | raw+proc | gpu | high | reimpl |
| `cast_catboost` | CAST with CatBoost student/base | processed | cpu/gpu | med | reimpl |
| `cast_lightgbm` | CAST with LightGBM | processed | cpu | med | reimpl |
| `tfm_consensus_pfn_icl` | Consensus TabPFN-3 ⊕ TabICLv2 | raw | gpu | high | reimpl |
| `tfm_teacher_catboost_student` | TFM teacher → CatBoost student | raw+proc | gpu+cpu | high | reimpl |
| `sparse_laplacian_tfm` | Sparse Laplacian regularization | proc/raw | gpu | high | reimpl |
| `prototype_align_tfm` | Prototype / class-cond embedding align | raw/emb | gpu | high | reimpl |
| `retrieval_attention_unlabeled` | Retrieval attention over unlabeled train | raw | gpu | high | reimpl |
| `geo_attn_ssl_adapter` | Combined geometric-attention SSL adapter | raw | gpu | very high | reimpl |
| `seba` | SeBA | TBD | gpu | high | prefer official if available |
| `stunt` | STUNT | TBD | gpu | high | prefer official if available |
| `d2r2c_inductive` | Inductive D2R2-c | TBD | gpu | high | reimpl/adapt |

**Phase-0 does not implement these.** Registry names are reserved in config comments / plan only.

---

## 6. Experiment phases

| Phase | Content | Parallelism |
|---|---|---|
| **0 (now)** | Audit, cluster doc, envs, caches, shard I/O skeletons, plan | — |
| **1** | Dual-view runner + adapters + `tabpfn3_supervised` + `tabicl_v2_supervised` smoke on 2 datasets × 2 budgets × 1 seed | Slurm serial/array small |
| **2** | Frozen + self-training + CAST-CatBoost/LightGBM | GPU arrays capped |
| **3** | LoopTabFM-style risk control + consensus + teacher-student | GPU, longer walls |
| **4** | Laplacian / prototype / retrieval / geo-attn / SeBA / STUNT / D2R2-c | possibly `ssl-representation` env |

Always compare against immutable classical baseline tables; do not overwrite them.

---

## 7. Tests required before claiming TFM readiness

1. **Leakage guards:** attempting to pass test features into fit raises; unlabeled y inaccessible.
2. **View integrity:** raw dtypes preserved; processed matches legacy preprocessor bit-close on classical methods.
3. **Legacy parity:** `logistic_regression` / `xgboost` on phoneme seed0 budget100 matches prior CSV within float tolerance on one controlled cell (or freshly recomputed golden under adapter).
4. **Shard atomicity:** parallel writers never corrupt; resume skips successes.
5. **Val plumbing:** method with `needs_external_validation=True` receives non-empty val when strategy allows.
6. **GPU smoke:** `torch.cuda.is_available()`, tiny matmul, TabPFN/TabICL import + one-row predict after cache warm.

---

## 8. Computational cost expectations (order-of-magnitude)

| Class | Per-run wall on A100 80G | Notes |
|---|---|---|
| Classical trees | seconds–tens of seconds | CPU partition preferred |
| Neural SSL (existing) | minutes | already measured in canonical wave |
| TabPFN-3 supervised | tens of seconds–minutes | context size sensitive |
| TabICLv2 supervised | tens of seconds–minutes | checkpoint load dominates cold start |
| Iterative TFM PL (10 rounds) | 10× above | needs shard resume |
| LoopTabFM-style | high / variable | val-gated stopping essential |

Warm cache + keep process reuse where safe to cut cold load.

---

## 9. Dependencies summary

**ssl-core:** numpy, pandas, scikit-learn, pyyaml, matplotlib, seaborn, xgboost, lightgbm, catboost, (optional torch CPU if analyzing neural rows).

**ssl-tfm:** python 3.11, pytorch CUDA, tabpfn, tabicl, huggingface_hub, catboost, scikit-learn, pandas, numpy, pyyaml, einops, tqdm, psutil.

**Blocked until user action:** PriorLabs `TABPFN_TOKEN` (and HF token if needed).

---

## 10. Next implementation command

After tokens are available and micromamba bootstrap succeeds:

```bash
# on an allocated GPU job or after salloc -p p_uriofir --gres=gpu:1 -A ug_uri_ofir
source scripts/cluster/env.sh
bash scripts/cluster/bootstrap_micromamba.sh
bash scripts/cluster/bootstrap_ssl_core.sh
bash scripts/cluster/bootstrap_ssl_tfm.sh
python scripts/cluster/check_auth.py
python scripts/cluster/warm_model_cache.py
python scripts/cluster/warm_openml_cache.py
bash scripts/cluster/preflight.sh
```

Then start coding phase-1 dual-view runner + two supervised TFM baselines.
