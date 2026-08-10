# Reproducibility guide

The repository supports two levels of reproduction: analysis-only reproduction
from committed canonical CSVs, and full experiment reproduction on a Slurm GPU
cluster. Analysis-only reproduction does not require model checkpoints.

## 1. Reproduce tables and figures

Use Python 3.11 and install the core requirements:

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
python scripts/verify_repository.py
python scripts/build_github_overview.py
```

The builder reads only these immutable inputs:

- `results/raw/low_class_wave_paper_methods.csv`
- `results/raw/tfm_frozen_screen.csv`
- `results/raw/focused_tfm_ssl.csv`

It writes overview tables and figures directly into
`results/reports/main_report/`. Generated tables should have no Git diff when
the inputs and analysis code are unchanged; PNG bytes can vary across operating
system font-rendering stacks.

To regenerate the focused-study and historical assets used by the single main
report:

```bash
python scripts/build_focused_study_assets.py
python scripts/build_historical_report_assets.py
```

## 2. Run a core method locally

OpenML datasets are downloaded by scikit-learn and cached outside the
repository. One cell can be run with:

```bash
python -m src.run_benchmark \
  --dataset phoneme \
  --method logistic_regression \
  --seed 0 \
  --label-budget 50 \
  --output results/raw/local_check.csv
```

Use `--resume` to skip successful keys, `--skip-failed` to retain failed keys
without retrying them, or `--overwrite` only for a disposable output file.
Never overwrite a canonical CSV.

## 3. Reproduce TFM methods

The validated environment is Python 3.11.15 with CUDA 12.4, PyTorch
2.6.0+cu124, TabPFN 8.1.0, and TabICL 2.1.1. Exact environment exports and
package locks are in `environment/tfm/lock/`.

The validated model identities are:

| Model | Checkpoint identity | SHA-256 |
|---|---|---|
| TabPFN-3 | `tabpfn-v3-classifier-v3_default.ckpt` | `d0d865c841a6e1fef896cde147b62dd87f5bcd3f294a89d16000bd70dcd36a26` |
| TabICL v2 | `tabicl-classifier-v2-20260212.ckpt` | `bdc7fe2dc7c66151173421458916153d823139465b28da27d5e7315940d42902` |

Weights and access tokens are excluded from Git. Obtain them under their
original terms, store them in a private cache, and expose cache locations via
the environment variables documented in the cluster scripts. Compute jobs set
automatic download off so a missing checkpoint fails explicitly rather than
changing model identity mid-wave.

## 4. Slurm workflow

The cluster workflow is atomic and resume-safe:

1. `scripts/cluster/preflight.py` validates environment, datasets, methods, and
   checkpoint availability.
2. `scripts/cluster/submit_wave.py` materializes a deterministic task map and
   one Slurm array job per experimental key.
3. `src/run_single.py` writes one temporary JSON payload and atomically renames
   it only after a complete result is available.
4. `scripts/cluster/monitor_wave.py` reports missing, failed, and corrupt tasks.
5. `scripts/cluster/collect_wave.py` validates uniqueness and combines shards
   into a canonical CSV.

Set site-specific paths instead of editing source defaults:

```bash
export SSL_PROJECT_ROOT=/path/to/tabular-foundation-ssl
export LAB_ROOT=/path/to/private/lab-storage
export SSL_CACHE_ROOT=/path/to/private/model-cache
```

The exact Bar-Ilan profiles and commands are documented in
[`tfm_ssl_cluster_runs.md`](tfm_ssl_cluster_runs.md). Site-specific account and
partition names may need adjustment on another cluster.

## 5. Validation expectations

Before promoting a result wave, require:

- the expected task count and unique dataset/method/seed/budget keys;
- explicit success or preserved failure status for every task;
- finite primary metrics for successful tasks;
- no validation/test identities in graphs, memory, pseudo-labeling, or fitting;
- configuration, source, environment, package, and checkpoint identity fields;
- unit and synthetic integration tests passing.

The experiment-completion snapshot reports 25 passed and 1 skipped in
`results/validation/pytest_current.txt`. After repository packaging and the
additional source-level leakage guard, the current suite reports 26 passed and
1 skipped in `results/validation/github_repository_pytest.txt`. The repository
check also verifies canonical file hashes and the complete 3,000-row design.
