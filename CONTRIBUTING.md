# Contributing

This repository is a research record as well as a codebase. Changes are welcome,
but benchmark comparability and provenance take priority over convenience.

## Before opening a change

1. Create a focused branch and describe whether the change affects code,
   protocol, results, or documentation.
2. Never modify the three canonical CSVs in `results/raw/` in place. A new
   experiment belongs in a separately named wave with its own manifest.
3. Keep the primary track inductive: unlabeled training features may be used,
   but validation/test examples must not enter graphs, pseudo-label pools,
   retrieval memory, representation fitting, or preprocessing.
4. Register new methods in the capability registry and state their fidelity
   honestly: official, faithful reimplementation, paper core, or novel
   experimental.
5. Do not commit data caches, environments, checkpoints, credentials, tokens,
   or scheduler logs.

## Required checks

```bash
python -m pytest -q
python scripts/verify_repository.py
python scripts/build_github_overview.py
git diff --exit-code results/reports/github_overview docs/assets/project_overview.png
```

Any intended change to generated outputs must explain its input wave, grid,
source hash, and validation status in the pull request. If a run fails, retain
the failed record and error metadata; do not silently filter it from coverage.

## Result review checklist

- The task map contains the intended dataset × method × seed × budget product.
- Every run key is unique and statuses are explicit.
- Model, checkpoint, source, configuration, and environment identities are
  serialized where applicable.
- Reported comparisons use matched keys or clearly state why they do not.
- Figures have machine-readable source tables and do not imply unsupported
  statistical significance.
