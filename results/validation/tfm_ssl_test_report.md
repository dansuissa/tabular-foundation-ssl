# TFM-SSL Validation Test Report

Generated: 2026-07-22 12:06:41 UTC
Host: dsiuriofir01
Project: `/home/eng/suissad4/projects/ssl-foundation-models/ssl_tabular_benchmark`

## Executive gate status

**Large experiment waves (Phase A–D) are NOT launched.**

Hard blockers remaining:
1. `TABPFN_TOKEN` unset → Smoke 3 (tabpfn3) and Smoke 6 (tabpfn3_loop_risk) blocked; Phase A requires TabPFN-3.
2. Project is **not a git repository** (commit hashing deferred).

Non-blocking notes:
- Ad-hoc ruff finds style issues; no formatter/linter/typechecker configured in-repo → informational only.
- `geometric_attention_ssl` smoke completes but is chance-level on segment@50 (BA≈1/7); flagged for Phase C investigation, not a crash.
- `peak_gpu_mem_mb` often null in shards; profiling job records VRAM separately.

## Git status (before/after)

```
fatal: not a git repository (or any parent up to mount point /)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
```

## Local validation suite

| Check | Outcome | Notes |
|---|---|---|
| formatting | SKIP | No ruff/black/pyproject config |
| linting (ad-hoc ruff) | FAIL (informational) | 100+ F401/F841 style findings; not configured as gate |
| type checking | SKIP | No mypy/pyright config |
| unit tests (ssl-core) | PASS | pytest tests/ |
| unit tests (ssl-tfm) | PASS | 18 passed, 1 skipped |
| old-method regression / leakage | PASS | test_leakage_guards, test_legacy_adapter_parity, test_integration_synthetic |
| synthetic integration | PASS | tests/test_integration_synthetic.py |
| configuration validation | PASS | registered missing capabilities for d2r2c_inductive, d2r2_transductive, predfeat adapters |
| duplicate method-name checks | PASS | |
| deps ssl-core | PASS | |
| deps ssl-tfm | PASS | torch 2.6.0+cu124, tabpfn 8.1.0, tabicl 2.1.1, CUDA available |
| reject secrets/checkpoints/openml/env in project tree | PASS | |

## Bug fixed during smoke

- `stunt` / `seba` / `d2r2_c` builders rejected `max_epochs` from `configs/benchmark.yaml`.
- Fix: map `max_epochs` → method-specific budget keys in builders (smoke-exposed real bug).
- Cluster runner `scripts/cluster/run_single.sh` wired from phase-0 placeholder to `python -m src.run_single`.

## Cluster preflight (scheduler job 18379646)

| Check | Outcome |
|---|---|
| environment activation (ssl-tfm) | PASS |
| Python path / version | PASS | 3.11.15 |
| GPU visibility | PASS | NVIDIA A100 80GB PCIe |
| CUDA / PyTorch compatibility | PASS | torch 2.6.0+cu124, cuda 12.4, avail True |
| GPU matmul + peak alloc | PASS | peak_alloc_mb=136.125 |
| scratch write access | PASS |
| project R/W | PASS |
| model-cache access | PASS | HF_HOME writable |
| OpenML-cache access | PASS | warmed all 10 low_class_wave datasets |
| TabPFN authentication | FAIL | TABPFN_TOKEN unset |
| TabPFN-3 checkpoint identity | BLOCKED | awaiting token + warm_model_cache |
| TabICLv2 checkpoint identity | PASS | `tabicl-classifier-v2-20260212.ckpt`; sha256_prefix=`92e40f49390de61fec0070e8a5cdb05b8b651b57e57d85fa3ebe96e7a8e1740d`; 110368038 bytes |
| no model download race | PASS | single-lock TabICL warm |
| atomic output shard | PASS |
| scheduler logs | PASS | `/private/ofirlin-lab/suissad4/results/logs` |

## Smoke outcomes

| # | Method | Dataset | Budget | Status | BA | Runtime_s | Notes |
|---:|---|---|---:|---|---:|---:|---|
| 1 | `logistic_regression` | phoneme | 50 | **success** | 0.6393008737014221 | 0.921 |  |
| 2 | `sslae` | phoneme | 50 | **success** | 0.6199543329975061 | 41.0621 |  |
| 3 | `tabpfn3` | phoneme | 50 | **blocked_missing_TABPFN_TOKEN** | None | None | needs TABPFN_TOKEN |
| 4 | `tabiclv2` | phoneme | 50 | **success** | 0.6930132789403274 | 20.0925 | tabicl-classifier-v2-20260212.ckpt |
| 5 | `tabiclv2_pl_one_shot` | phoneme | 50 | **success** | 0.6862520025765109 | 6.3888 | tabicl-classifier-v2-20260212.ckpt |
| 6 | `tabpfn3_loop_risk` | phoneme | 50 | **blocked_missing_TABPFN_TOKEN** | None | None | needs TABPFN_TOKEN |
| 7 | `laplacian_mlp` | segment | 50 | **success** | 0.6883116883116883 | 4.7606 |  |
| 8 | `geometric_attention_ssl` | segment | 50 | **success** | 0.14285714285714285 | 4.9989 | chance-level BA on 7-class segment |
| 9 | `stunt` | phoneme | 50 | **success** | 0.7262684360909706 | 16.4372 | fixed max_epochs mapping; success after retry |
| 10 | `tabiclv2` | adult | 100 | **success** | 0.7327630050470604 | 6.4549 | tabicl-classifier-v2-20260212.ckpt |

### Smoke 1 extra validation
- Split sizes / labeled classes present: yes (`train_labeled_size=40`, `n_unlabeled=4273`, `all_classes_present_in_labeled=True`).
- Resume skipping: PASS.
- Deterministic rerun: PASS (identical balanced accuracy).

## Wave launch status

| Wave | Status |
|---|---|
| tfm_frozen_screen (Phase A) | **NOT SUBMITTED** — blocked on TabPFN smoke + token |
| tfm_ssl_screen (Phase B) | NOT SUBMITTED — awaits Phase A |
| modern_geometric_ssl_screen (Phase C) | NOT SUBMITTED — awaits screens + geom investigation |
| tfm_ssl_confirmation (Phase D) | NOT SUBMITTED — awaits objective shortlist |

## How to unblock TabPFN

```bash
# After accepting license at https://ux.priorlabs.ai
mkdir -p /private/ofirlin-lab/suissad4/secrets
printf "%s\n" "$TABPFN_TOKEN" > /private/ofirlin-lab/suissad4/secrets/TABPFN_TOKEN
chmod 600 /private/ofirlin-lab/suissad4/secrets/TABPFN_TOKEN
sbatch -A ug_uri_ofir -p p_uriofir --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=1:00:00 \
  --wrap 'source scripts/cluster/env.sh; export TABPFN_TOKEN=$(tr -d "\n" < $LAB_ROOT/secrets/TABPFN_TOKEN); \
          eval "$($MAMBA_EXE shell hook -s bash)"; micromamba activate $SSL_TFM_PREFIX; \
          python scripts/cluster/warm_model_cache.py'
python scripts/cluster/run_smokes.py --ids 3 6 --submit --wait
```

## Wave command reference

```bash
python scripts/cluster/submit_wave.py --wave tfm_frozen_screen --dataset-group low_class_wave \
  --method-group tfm_frozen --seeds 0 1 2 --label-budgets 50 100 250 500 \
  --profile gpu_tfm_small --concurrency 2 --submit
python scripts/cluster/monitor_wave.py --wave tfm_frozen_screen
python scripts/cluster/monitor_wave.py --wave tfm_frozen_screen --list-failed --list-missing
python scripts/cluster/collect_wave.py --wave tfm_frozen_screen --output results/raw/tfm_frozen_screen.csv
python scripts/cluster/postprocess_wave.py --wave tfm_frozen_screen
scancel -n ssl-tfm_frozen_screen
```

Immutable historical file never written: `results/raw/low_class_wave_paper_methods.csv`.


## Resource profiling (measured)

Source: `results/validation/resource_profile.csv` and `docs/tfm_ssl_resource_plan.md`.

| dataset | method | budget | total_s | peak_rss_mb | peak_vram_mb | profile | mem | time |
|---|---|---:|---:|---:|---:|---|---|---|
| phoneme | logistic_regression | 50 | 0.15 | 549 | 0 | cpu_light | 16G | 1:00:00 |
| phoneme | sslae | 50 | 3.3 | 686 | 0 | gpu_representation | 64G | 1:00:00 |
| phoneme | tabiclv2 | 50 | 2.0 | 1301 | 355 | gpu_tfm_small | 64G | 2:00:00 |
| segment | laplacian_mlp | 50 | 0.3 | 1351 | 8 | gpu_representation | 64G | 2:00:00 |
| segment | geometric_attention_ssl | 50 | 0.4 | 1390 | 8 | gpu_representation | 64G | 2:00:00 |
| adult | tabiclv2 | 100 | 2.7 | 1579 | 3649 | gpu_tfm_large | 96G | 4:00:00 |
| jannis | tabiclv2 | 100 | 8.4 | 1746 | **19354** | gpu_tfm_large | 96G | 4:00:00 |
| jannis | stunt | 50 | 20.7 | 1792 | 614 | gpu_tfm_large | 96G | 4:00:00 |

TabPFN profiling cells blocked (no token). Scheduler defaults in `submit_wave.py` updated from these measurements.

## Phase A dry-run (not submitted)

- Expected cells: 240
- Task map written: `/private/ofirlin-lab/suissad4/results/raw_shards/tfm_frozen_screen/task_map.json`
- Submit script ready: `/private/ofirlin-lab/suissad4/results/raw_shards/tfm_frozen_screen/submit.sbatch`
- **Not submitted** until Smoke 3/6 pass.

## Highest-priority scientific findings (completed results only)

1. TabICLv2 official checkpoint loads and runs on phoneme@50 (BA≈0.693) and adult@100 (BA≈0.733) with exact ckpt `tabicl-classifier-v2-20260212.ckpt`.
2. TabICLv2 one-shot PL on phoneme@50 completes (BA≈0.686) — slightly below frozen TabICLv2 on this single smoke cell; not a claim of SSL effect (n=1 seed).
3. Classical LR CPU regression and SSLAE GPU neural regression succeed.
4. Laplacian MLP is competitive on segment@50 smoke (BA≈0.688); geometric_attention_ssl is chance-level there (BA=1/7) and needs investigation before Phase C ranking claims.
5. STUNT succeeds after `max_epochs` builder fix (phoneme@50 BA≈0.726).
6. jannis-scale TabICLv2 peaks ~19 GB VRAM — use `gpu_tfm_large` (96G RAM, 4h) for large TFM cells.

No Phase A–D scientific rankings are claimed: waves were not launched.

## Continuation (TabPFN token present) — 2026-07-22T12:32:18Z

### Token
- Lab file detected, nonempty, mode 0600; secrets dir 0700.
- Synced from home copy (lab file was previously empty).
- Values never printed.

### TabPFN-3 warm
- Package default is ModelVersion.V3; required ckpt `tabpfn-v3-classifier-v3_default.ckpt` from `Prior-Labs/tabpfn_3`.
- PriorLabs API license `tabpfn-3-license-v1.0` returns **accepted=False** for this token.
- **Smoke 3/6 and Phase A blocked** until license acceptance at https://ux.priorlabs.ai (do not fall back to v2).

### Source versioning
- No `.git` directory found under project/parents.
- Deterministic snapshot: `results/validation/source_snapshot/`
- Aggregate hash: `e10e15dd95902098c52123b439d8253c31c94e9f706f811632fbed2ec3c85548`

### Missing capabilities
- See `results/validation/missing_capabilities_table.md` (all four now registered; none block Phase A).

### Geometric attention
- See `results/validation/geometric_attention_diagnosis.md`.
- Collapse starts at retrieval ablation; embedding no_grad bug fixed; Phase C still blocked pending retest smoke.

### Phase A
- **NOT SUBMITTED** (license gate).
