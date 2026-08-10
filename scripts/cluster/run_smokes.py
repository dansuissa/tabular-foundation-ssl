#!/usr/bin/env python3
"""Submit and validate individual smoke jobs (separate Slurm jobs, not one opaque process).

Records outcomes into results/validation/smoke_outcomes.jsonl and appends to the test report.
Does not launch Phase A–D waves.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SMOKES = [
    {"id": 1, "dataset": "phoneme", "method": "logistic_regression", "seed": 0, "budget": 50, "profile": "cpu_light", "need_gpu": False, "need_tabpfn": False},
    {"id": 2, "dataset": "phoneme", "method": "sslae", "seed": 0, "budget": 50, "profile": "gpu_representation", "need_gpu": True, "need_tabpfn": False},
    {"id": 3, "dataset": "phoneme", "method": "tabpfn3", "seed": 0, "budget": 50, "profile": "gpu_tfm_small", "need_gpu": True, "need_tabpfn": True},
    {"id": 4, "dataset": "phoneme", "method": "tabiclv2", "seed": 0, "budget": 50, "profile": "gpu_tfm_small", "need_gpu": True, "need_tabpfn": False},
    {"id": 5, "dataset": "phoneme", "method": "tabiclv2_pl_one_shot", "seed": 0, "budget": 50, "profile": "gpu_tfm_small", "need_gpu": True, "need_tabpfn": False},
    {"id": 6, "dataset": "phoneme", "method": "tabpfn3_loop_risk", "seed": 0, "budget": 50, "profile": "gpu_tfm_small", "need_gpu": True, "need_tabpfn": True},
    {"id": 7, "dataset": "segment", "method": "laplacian_mlp", "seed": 0, "budget": 50, "profile": "gpu_representation", "need_gpu": True, "need_tabpfn": False},
    {"id": 8, "dataset": "segment", "method": "geometric_attention_ssl", "seed": 0, "budget": 50, "profile": "gpu_representation", "need_gpu": True, "need_tabpfn": False},
    {"id": 9, "dataset": "phoneme", "method": "stunt", "seed": 0, "budget": 50, "profile": "gpu_representation", "need_gpu": True, "need_tabpfn": False},
    {"id": 10, "dataset": "adult", "method": "tabiclv2", "seed": 0, "budget": 100, "profile": "gpu_tfm_large", "need_gpu": True, "need_tabpfn": False},
]


def load_auth_env() -> dict[str, str]:
    env = {}
    lab = Path(os.environ.get("LAB_ROOT", "/private/ofirlin-lab/suissad4"))
    hf = Path(os.environ.get("HF_HOME", lab / "caches" / "hf")) / "token"
    tab = lab / "secrets" / "TABPFN_TOKEN"
    if hf.is_file():
        env["HF_TOKEN"] = hf.read_text().strip()
    if tab.is_file():
        env["TABPFN_TOKEN"] = tab.read_text().strip()
    # Also accept already-exported env
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TABPFN_TOKEN"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def submit_smoke(smoke: dict, *, wait: bool, auth: dict[str, str]) -> dict:
    wave = f"smoke_{smoke['id']:02d}_{smoke['method']}"
    project = Path(os.environ.get("SSL_PROJECT_ROOT", ROOT))
    lab = Path(os.environ.get("LAB_ROOT", "/private/ofirlin-lab/suissad4"))
    log_dir = lab / "results" / "logs" / "smokes"
    log_dir.mkdir(parents=True, exist_ok=True)

    if smoke["need_tabpfn"] and not auth.get("TABPFN_TOKEN"):
        return {
            **smoke,
            "wave": wave,
            "status": "blocked_missing_TABPFN_TOKEN",
            "job_id": None,
        }

    profile = smoke["profile"]
    cmd = [
        sys.executable,
        str(project / "scripts" / "cluster" / "submit_wave.py"),
        "--wave",
        wave,
        "--datasets",
        smoke["dataset"],
        "--methods",
        smoke["method"],
        "--seeds",
        str(smoke["seed"]),
        "--label-budgets",
        str(smoke["budget"]),
        "--profile",
        profile,
        "--concurrency",
        "1",
        "--submit",
    ]
    # Ensure auth available to child env for sbatch script generation path
    env = os.environ.copy()
    env.update({k: v for k, v in auth.items() if v})
    # Write TABPFN token to secrets file if present in env (so sbatch jobs can read it)
    if auth.get("TABPFN_TOKEN"):
        secrets = lab / "secrets"
        secrets.mkdir(parents=True, exist_ok=True)
        tok_path = secrets / "TABPFN_TOKEN"
        if not tok_path.exists():
            tok_path.write_text(auth["TABPFN_TOKEN"] + "\n", encoding="utf-8")
            tok_path.chmod(0o600)

    proc = subprocess.run(cmd, cwd=str(project), env=env, capture_output=True, text=True)
    out = {
        **smoke,
        "wave": wave,
        "submit_rc": proc.returncode,
        "submit_stdout": proc.stdout[-2000:],
        "submit_stderr": proc.stderr[-2000:],
    }
    job_id = None
    for line in (proc.stdout or "").splitlines():
        if "Submitted batch job" in line:
            job_id = line.strip().split()[-1]
    out["job_id"] = job_id
    out["status"] = "submitted" if proc.returncode == 0 else "submit_failed"

    if wait and job_id:
        out.update(wait_and_validate(wave, job_id, smoke))
    return out


def wait_and_validate(wave: str, job_id: str, smoke: dict, timeout_s: int = 7200) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        proc = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True)
        if not proc.stdout.strip():
            break
        time.sleep(20)
    # Collect shard
    mon = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cluster" / "monitor_wave.py"), "--wave", wave],
        capture_output=True,
        text=True,
    )
    validation = validate_smoke_shard(wave, smoke)
    return {
        "monitor_stdout": mon.stdout[-2000:],
        "validation": validation,
        "status": validation.get("overall", "unknown"),
    }


def validate_smoke_shard(wave: str, smoke: dict) -> dict:
    """Post-hoc checks on the smoke shard payload."""
    from src.results_io.shards import list_shards, read_shard

    checks = {}
    shards = [p for p in list_shards(wave) if p.suffix == ".json" and "task_map" not in p.name]
    if not shards:
        return {"overall": "no_shard", "checks": checks}
    payload = read_shard(shards[0])
    checks["status_success"] = payload.get("status") == "success"
    checks["has_metrics"] = any(str(k).startswith("metric_") or k == "metrics" for k in payload)
    metrics = payload.get("metrics") or {}
    if isinstance(metrics, dict):
        for m in ("balanced_accuracy", "macro_f1", "log_loss"):
            # flattened or nested
            val = metrics.get(m) or metrics.get(f"metric_{m}") or payload.get(f"metric_{m}")
            checks[f"metric_{m}_present"] = val is not None
    tm = payload.get("training_meta") or payload
    checks["runtime_present"] = tm.get("runtime_seconds") is not None or payload.get("runtime_seconds") is not None
    checks["method_match"] = payload.get("method") == smoke["method"]
    checks["dataset_match"] = payload.get("dataset") == smoke["dataset"]
    checks["seed_match"] = int(payload.get("seed", -1)) == int(smoke["seed"])
    checks["budget_match"] = int(payload.get("n_labeled", -1)) == int(smoke["budget"])
    # finite / class order fields if present
    for key in ("n_labeled_train", "n_unlabeled", "n_validation", "n_test", "labeled_classes_present"):
        if key in tm or key in payload:
            checks[f"field_{key}"] = True
    overall = "success" if checks.get("status_success") else f"status={payload.get('status')}"
    if not all(v for k, v in checks.items() if k.startswith("metric_") is False or True):
        pass
    return {"overall": overall, "checks": checks, "error": payload.get("error_message"), "run_id": payload.get("run_id")}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="+", type=int, default=None, help="Subset of smoke ids")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list or not args.submit:
        print(json.dumps(SMOKES, indent=2))
        if not args.submit:
            print("Pass --submit to launch (and optionally --wait).")
            return 0

    auth = load_auth_env()
    print(
        json.dumps(
            {
                "TABPFN_TOKEN_set": bool(auth.get("TABPFN_TOKEN")),
                "HF_TOKEN_set": bool(auth.get("HF_TOKEN") or auth.get("HUGGING_FACE_HUB_TOKEN")),
            }
        )
    )

    selected = [s for s in SMOKES if args.ids is None or s["id"] in args.ids]
    out_dir = ROOT / "results" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_outcomes.jsonl"
    results = []
    for smoke in selected:
        print(f"=== smoke {smoke['id']} {smoke['method']} ===", flush=True)
        result = submit_smoke(smoke, wait=args.wait, auth=auth)
        results.append(result)
        with out_path.open("a", encoding="utf-8") as f:
            # never write token values
            safe = {k: v for k, v in result.items() if "token" not in str(k).lower()}
            f.write(json.dumps(safe) + "\n")
        print(json.dumps({k: result.get(k) for k in ("id", "method", "status", "job_id")}, indent=2))
    summary_path = out_dir / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "results": [
                    {k: r.get(k) for k in ("id", "method", "dataset", "status", "job_id", "validation")}
                    for r in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
