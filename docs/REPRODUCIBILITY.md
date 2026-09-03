# Reproducibility guide

The public repository supports two useful levels of reproduction: rebuilding the
analysis from the committed canonical CSVs, and rerunning benchmark cells from
the source implementation.

## 1. Rebuild tables and figures

Python 3.11 is the validated runtime.

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

The whole-project builder reads only:

- `results/raw/low_class_wave_paper_methods.csv`
- `results/raw/tfm_frozen_screen.csv`
- `results/raw/focused_tfm_ssl.csv`

It writes the overview tables and figures under
`results/reports/main_report/`. Historical report assets can be regenerated with:

```bash
python scripts/build_historical_report_assets.py
```

The focused-study source tables are committed under
`results/reports/main_report/tables/focused_ssl/` alongside the figures they
support.

## 2. Rerun a benchmark cell

OpenML datasets are fetched by the benchmark code and cached outside the
repository. For example:

```bash
python -m src.run_benchmark \
  --dataset phoneme \
  --method logistic_regression \
  --seed 0 \
  --label-budget 50 \
  --output results/raw/local_check.csv
```

Use a disposable output path. Do not overwrite any of the three canonical CSVs.

## 3. Foundation-model methods

The validated TFM setup used Python 3.11, PyTorch 2.6 with CUDA 12.4,
TabPFN 8.1.0, and TabICL 2.1.1. The environment specification and package
records are under `environment/tfm/`.

The validated checkpoint identities were:

| Model | Checkpoint identity | SHA-256 |
|---|---|---|
| TabPFN-3 | `tabpfn-v3-classifier-v3_default.ckpt` | `d0d865c841a6e1fef896cde147b62dd87f5bcd3f294a89d16000bd70dcd36a26` |
| TabICL v2 | `tabicl-classifier-v2-20260212.ckpt` | `bdc7fe2dc7c66151173421458916153d823139465b28da27d5e7315940d42902` |

Model weights and access tokens are not distributed in this repository. Obtain
them from the original providers under their applicable terms and configure the
packages according to their official documentation.

## 4. Experimental safeguards

A valid reproduction should preserve the benchmark contract:

- identical dataset, method, seed, and label-budget keys;
- explicit success/failure status for every run;
- finite primary metrics for successful runs;
- no validation/test rows in training graphs, retrieval memory, pseudo-label
  pools, representation fitting, or preprocessing;
- shared predetermined splits across methods;
- model/environment identity recorded where applicable.

The tests in `tests/` cover leakage guards, integration behavior, result
shards, method contracts, and legacy parity.
