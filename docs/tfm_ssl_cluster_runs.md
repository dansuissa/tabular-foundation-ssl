# TFM-SSL Cluster Runs

See also `docs/cluster_environment.md`.

## Environments

```bash
source scripts/cluster/env.sh
bash scripts/cluster/bootstrap_micromamba.sh
bash scripts/cluster/bootstrap_ssl_core.sh   # classical
bash scripts/cluster/bootstrap_ssl_tfm.sh    # needs TABPFN_TOKEN (+ HF if gated)
python3 scripts/cluster/check_auth.py
python3 scripts/cluster/warm_model_cache.py
python3 scripts/cluster/warm_openml_cache.py
```

## Smoke (after auth + caches)

```bash
# GPU allocation on p_uriofir
sbatch --partition=p_uriofir --account=ug_uri_ofir --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=4:00:00 \
  --wrap 'source scripts/cluster/env.sh && eval "$($MAMBA_EXE shell hook -s bash)" && micromamba activate $SSL_TFM_PREFIX && \
  python -m src.run_single --wave tfm_smoke --dataset phoneme --method tabpfn3 --seed 0 --label-budget 50'
```

CPU geometric smoke:

```bash
python -m src.run_benchmark --method-group geometric_ssl_ablation \
  --dataset phoneme --seeds 0 --label-budgets 50 \
  --output results/raw/tfm_geom_smoke.csv
```

## Submit a wave

```bash
python scripts/cluster/submit_wave.py --wave tfm_ssl_core --method-group tfm_ssl_core \
  --dataset-group low_class_wave --seeds 0 1 2 --label-budgets 50 100 250 500
python scripts/cluster/monitor_wave.py --wave tfm_ssl_core
python scripts/cluster/collect_wave.py --wave tfm_ssl_core --output results/combined/tfm_ssl_core.csv
```

Never overwrite `results/raw/low_class_wave_paper_methods.csv`.
