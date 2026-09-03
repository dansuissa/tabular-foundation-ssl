"""Single-run CLI for Slurm array tasks (atomic shard output)."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from src.data import dataset_from_config, load_dataset
from src.method_capabilities import get_capabilities
from src.results_io.manifest import build_result_payload, code_version, config_hash
from src.results_io.shards import make_run_id, shard_success_exists, write_shard_atomic
from src.run_benchmark import build_model, resolve_neural_params, run_single_experiment
from src.utils import load_yaml, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one TFM/SSL benchmark cell to a shard.")
    p.add_argument("--wave", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--label-budget", type=int, required=True)
    p.add_argument("--datasets-config", type=Path, default=Path("configs/datasets.yaml"))
    p.add_argument("--benchmark-config", type=Path, default=Path("configs/benchmark.yaml"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    datasets_cfg = load_yaml(args.datasets_config)
    benchmark_cfg = load_yaml(args.benchmark_config)
    entry = next(e for e in datasets_cfg["datasets"] if e["name"] == args.dataset)
    method_config = dict(benchmark_cfg.get(args.method) or {})
    cfg_hash = config_hash(method_config)
    cver = code_version()
    run_id = make_run_id(
        args.dataset,
        args.method,
        args.seed,
        args.label_budget,
        cfg_hash,
        "v1_absolute_budget",
        cver,
    )
    if args.resume and not args.force and shard_success_exists(args.wave, run_id):
        logging.info("Skipping completed shard %s", run_id)
        return

    set_seed(args.seed)
    spec = dataset_from_config(entry)
    dataset = load_dataset(spec)
    from src.models.neural_ssl import NEURAL_SSL_METHODS
    from src.models.registry_ext import EXTENDED_METHODS

    neural_params = (
        resolve_neural_params(benchmark_cfg, args.method)
        if args.method in NEURAL_SSL_METHODS
        else None
    )
    method_params = method_config if args.method in EXTENDED_METHODS else None
    row = run_single_experiment(
        dataset=dataset,
        method=args.method,
        seed=args.seed,
        label_budget=args.label_budget,
        test_size=benchmark_cfg["test_size"],
        val_size_from_labeled=benchmark_cfg["val_size_from_labeled"],
        metric_names=benchmark_cfg["metrics"],
        neural_params=neural_params,
        method_params=method_params,
    )
    caps = None
    try:
        caps = get_capabilities(args.method)
    except Exception:
        caps = None
    payload = build_result_payload(
        dataset=args.dataset,
        method=args.method,
        seed=args.seed,
        n_labeled=args.label_budget,
        status=row.get("status", "failed"),
        method_config=method_config,
        metrics={k: v for k, v in row.items() if str(k).startswith("metric_")},
        training_meta=row,
        capabilities=caps,
        error_message=row.get("error_message"),
        runtime_seconds=row.get("runtime_seconds"),
        code_ver=cver,
        extra={
            "run_id": run_id,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "hostname": os.environ.get("HOSTNAME") or os.uname().nodename,
        },
    )
    path = write_shard_atomic(args.wave, run_id, payload)
    logging.info("Wrote shard %s status=%s", path, payload.get("status"))
    print(json.dumps({"run_id": run_id, "path": str(path), "status": payload.get("status")}))
    if payload.get("status") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
