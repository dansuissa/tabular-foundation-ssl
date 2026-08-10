#!/usr/bin/env python3
"""Generate and optionally submit a Slurm array for a benchmark wave."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Resource profiles (wall/mem/gres defaults; updated after profiling).
RESOURCE_PROFILES = {
    # Values below are post-profile defaults with safety margin (2026-07-22).
    "cpu_light": {
        "partition": "cpu192G-48h",
        "gres": None,
        "cpus": 4,
        "mem": "16G",
        "time": "1:00:00",
        "env": "ssl-core",
    },
    "cpu_tree_ssl": {
        "partition": "cpu192G-48h",
        "gres": None,
        "cpus": 8,
        "mem": "32G",
        "time": "12:00:00",
        "env": "ssl-core",
    },
    "gpu_tfm_small": {
        "partition": "p_uriofir",
        "gres": "gpu:1",
        "cpus": 8,
        "mem": "64G",
        "time": "2:00:00",
        "env": "ssl-tfm",
    },
    "gpu_tfm_large": {
        "partition": "p_uriofir",
        "gres": "gpu:1",
        "cpus": 8,
        "mem": "96G",
        "time": "4:00:00",
        "env": "ssl-tfm",
    },
    "gpu_representation": {
        "partition": "p_uriofir",
        "gres": "gpu:1",
        "cpus": 8,
        "mem": "64G",
        "time": "2:00:00",
        "env": "ssl-tfm",
    },
    "gpu_diffusion": {
        "partition": "p_uriofir",
        "gres": "gpu:1",
        "cpus": 8,
        "mem": "96G",
        "time": "12:00:00",
        "env": "ssl-tfm",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wave", required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--dataset-group", default=None)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--method-group", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--label-budgets", nargs="+", type=int, default=[50, 100])
    p.add_argument("--account", default="ug_uri_ofir")
    p.add_argument("--partition", default=None)
    p.add_argument("--gres", default=None)
    p.add_argument("--cpus", type=int, default=None)
    p.add_argument("--mem", default=None)
    p.add_argument("--time", default=None)
    p.add_argument("--concurrency", type=int, default=2, help="Max simultaneous array tasks")
    p.add_argument(
        "--profile",
        choices=sorted(RESOURCE_PROFILES),
        default="gpu_tfm_small",
    )
    p.add_argument("--submit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--datasets-config",
        type=Path,
        default=ROOT / "configs" / "datasets.yaml",
    )
    p.add_argument(
        "--benchmark-config",
        type=Path,
        default=ROOT / "configs" / "benchmark.yaml",
    )
    return p.parse_args()


def resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return list(args.datasets)
    if not args.dataset_group:
        raise SystemExit("Provide --datasets or --dataset-group")
    cfg = yaml.safe_load(args.datasets_config.read_text())
    groups = cfg.get("dataset_groups", {})
    if args.dataset_group not in groups:
        raise SystemExit(f"Unknown dataset group {args.dataset_group}: {sorted(groups)}")
    return list(groups[args.dataset_group])


def resolve_methods(args: argparse.Namespace) -> list[str]:
    if args.methods:
        return list(args.methods)
    if not args.method_group:
        raise SystemExit("Provide --methods or --method-group")
    cfg = yaml.safe_load(args.benchmark_config.read_text())
    if args.method_group in cfg and isinstance(cfg[args.method_group], list):
        return list(cfg[args.method_group])
    from src.method_capabilities import METHOD_GROUPS

    if args.method_group in METHOD_GROUPS:
        return list(METHOD_GROUPS[args.method_group])
    raise SystemExit(f"Unknown method group {args.method_group}")


def main() -> int:
    args = parse_args()
    lab = Path(os.environ.get("LAB_ROOT", "/private/ofirlin-lab/suissad4"))
    project = Path(os.environ.get("SSL_PROJECT_ROOT", ROOT))
    wave_dir = lab / "results" / "raw_shards" / args.wave
    log_dir = lab / "results" / "logs" / args.wave
    wave_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    datasets = resolve_datasets(args)
    methods = resolve_methods(args)
    profile = dict(RESOURCE_PROFILES[args.profile])

    partition = args.partition or profile["partition"]
    gres = args.gres if args.gres is not None else profile["gres"]
    cpus = args.cpus or profile["cpus"]
    mem = args.mem or profile["mem"]
    time_lim = args.time or profile["time"]
    env_name = profile["env"]
    if env_name == "ssl-core":
        env_prefix = os.environ.get("SSL_CORE_PREFIX", f"{lab}/envs/ssl-core")
    else:
        env_prefix = os.environ.get("SSL_TFM_PREFIX", f"{lab}/envs/ssl-tfm")

    tasks = []
    for ds, method, seed, budget in product(datasets, methods, args.seeds, args.label_budgets):
        tasks.append(
            {
                "task_id": len(tasks),
                "dataset": ds,
                "method": method,
                "seed": seed,
                "n_labeled": budget,
                "wave": args.wave,
                "profile": args.profile,
            }
        )

    source_hash_path = (
        project / "results" / "validation" / "source_snapshot" / "SOURCE_TREE_HASH.txt"
    )
    source_tree_hash = (
        source_hash_path.read_text(encoding="utf-8").strip()
        if source_hash_path.exists()
        else "unknown"
    )
    task_map = wave_dir / "task_map.json"
    meta = {
        "wave": args.wave,
        "n_tasks": len(tasks),
        "datasets": datasets,
        "methods": methods,
        "seeds": args.seeds,
        "label_budgets": args.label_budgets,
        "profile": args.profile,
        "source_tree_hash": source_tree_hash,
        "resource": {
            "partition": partition,
            "gres": gres,
            "cpus": cpus,
            "mem": mem,
            "time": time_lim,
            "concurrency": args.concurrency,
            "env_prefix": env_prefix,
        },
    }
    (wave_dir / "wave_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    task_map.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"wrote {len(tasks)} tasks → {task_map}")

    gres_line = f"#SBATCH --gres={gres}" if gres else ""
    max_id = max(len(tasks) - 1, 0)
    sbatch_path = wave_dir / "submit.sbatch"
    run_single = project / "scripts" / "cluster" / "run_single.sh"
    content = f"""#!/bin/bash
#SBATCH --job-name=ssl-{args.wave}
#SBATCH --account={args.account}
#SBATCH --partition={partition}
{gres_line}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_lim}
#SBATCH --array=0-{max_id}%{args.concurrency}
#SBATCH --output={log_dir}/%x-%A_%a.out
#SBATCH --error={log_dir}/%x-%A_%a.err

set -euo pipefail
source {project}/scripts/cluster/env.sh
export SSL_WAVE={args.wave}
export SSL_TASK_MAP={task_map}
export SSL_ENV_PREFIX={env_prefix}
export SSL_SOURCE_TREE_HASH={source_tree_hash}
# Auth files if present (never echo values)
if [[ -f "${{HF_HOME}}/token" ]]; then
  export HF_TOKEN="$(tr -d '\\n' < "${{HF_HOME}}/token")"
fi
if [[ -f "${{LAB_ROOT}}/secrets/TABPFN_TOKEN" ]]; then
  export TABPFN_TOKEN="$(tr -d '\\n' < "${{LAB_ROOT}}/secrets/TABPFN_TOKEN")"
fi
bash {run_single} --task-id "$SLURM_ARRAY_TASK_ID"
"""
    content = "\n".join(line for line in content.splitlines() if line.strip() != "") + "\n"
    sbatch_path.write_text(content, encoding="utf-8")
    print("wrote", sbatch_path)
    print("commands:")
    print(f"  submit:   sbatch {sbatch_path}")
    print(f"  monitor:  python {project}/scripts/cluster/monitor_wave.py --wave {args.wave}")
    print(
        f"  collect:  python {project}/scripts/cluster/collect_wave.py --wave {args.wave} "
        f"--output {project}/results/raw/{args.wave}.csv"
    )
    print(f"  cancel:   scancel -n ssl-{args.wave}")
    print(
        f"  failed:   python {project}/scripts/cluster/monitor_wave.py --wave {args.wave} --list-failed"
    )
    print(
        f"  retry:    python {project}/scripts/cluster/submit_wave.py --wave {args.wave} "
        f"--dataset-group ... --method-group ... --submit   # incomplete shards resume via --resume"
    )

    if args.dry_run or not args.submit:
        print("dry-run / not submitting (pass --submit to sbatch)")
        return 0

    proc = subprocess.run(["sbatch", str(sbatch_path)], check=False, capture_output=True, text=True)
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    if proc.returncode == 0 and proc.stdout.strip():
        (wave_dir / "last_job_id.txt").write_text(proc.stdout.strip() + "\n", encoding="utf-8")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
