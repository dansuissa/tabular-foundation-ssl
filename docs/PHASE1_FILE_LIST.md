# Exact source files the NEXT implementation phase will add or modify.
# Historical results under results/ are immutable and must not appear here as modify targets.

## ADD
- src/views.py
- src/method_capabilities.py
- src/models/legacy_adapter.py
- src/models/tfm_tabpfn.py
- src/models/tfm_tabicl.py
- src/run_single.py
- src/results_io/manifest.py
- tests/test_leakage_guards.py
- tests/test_views.py
- tests/test_shards.py
- tests/test_legacy_adapter_parity.py
- configs/method_resources.yaml

## MODIFY
- src/run_benchmark.py
- src/preprocessing.py
- src/models/__init__.py
- src/combine_results.py
- configs/benchmark.yaml
- configs/datasets.yaml  # only additive groups (e.g. tfm_smoke), no removal of low_class_wave
- README.md  # document dual-view + cluster workflow (after code exists)

## ALREADY ADDED IN PHASE 0 (infra)
- docs/tfm_ssl_repository_audit.md
- docs/cluster_environment.md
- docs/tfm_ssl_implementation_plan.md
- docs/PHASE1_FILE_LIST.md  # this file
- environment/core/environment.yml
- environment/tfm/environment.yml
- environment/representation/environment.yml
- scripts/cluster/env.sh
- scripts/cluster/preflight.sh
- scripts/cluster/bootstrap_micromamba.sh
- scripts/cluster/bootstrap_ssl_core.sh
- scripts/cluster/bootstrap_ssl_tfm.sh
- scripts/cluster/check_auth.py
- scripts/cluster/warm_model_cache.py
- scripts/cluster/warm_openml_cache.py
- scripts/cluster/submit_wave.py
- scripts/cluster/run_single.sh
- scripts/cluster/monitor_wave.py
- scripts/cluster/collect_wave.py
- scripts/cluster/cancel_wave.py
- src/results_io/__init__.py
- src/results_io/shards.py

## NEVER TOUCH
- results/raw/low_class_wave_paper_methods.csv
- results/reports/low_class_wave_paper_methods/**
- results/aggregated/low_class_wave_paper_methods/**
- other historical results/raw/*.csv waves used as baselines
