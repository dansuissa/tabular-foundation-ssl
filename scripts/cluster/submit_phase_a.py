#!/usr/bin/env python3
"""Submit Phase A wave tfm_frozen_screen with small/large resource split + post chain.

Jobs are Slurm-native (survive Cursor/SSH disconnect).
Token is read at runtime from secrets file — never interpolated into sbatch text.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WAVE = "tfm_frozen_screen"
DATASETS_SMALL = [
    "phoneme",
    "spambase",
    "MagicTelescope",
    "bank-marketing",
    "electricity",
    "satimage",
    "segment",
    "steel-plates-fault",
]
DATASETS_LARGE = ["adult", "jannis"]
METHODS = ["tabpfn3", "tabiclv2"]
BUDGETS = [50, 100, 250, 500]
SEEDS = [0, 1, 2]


def sbatch_header(
    *,
    job_name: str,
    account: str,
    partition: str,
    gres: str | None,
    cpus: int,
    mem: str,
    time_lim: str,
    array_spec: str,
    log_dir: Path,
    dependency: str | None = None,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
    ]
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    lines += [
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_lim}",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --output={log_dir}/%x-%A_%a.out",
        f"#SBATCH --error={log_dir}/%x-%A_%a.err",
    ]
    if dependency:
        lines.append(f"#SBATCH --dependency={dependency}")
    return "\n".join(lines) + "\n"


def runtime_auth_block() -> str:
    return """
set -euo pipefail
source ${SSL_PROJECT_ROOT}/scripts/cluster/env.sh
export SSL_WAVE=${SSL_WAVE}
export SSL_TASK_MAP=${SSL_TASK_MAP}
export SSL_ENV_PREFIX=${SSL_ENV_PREFIX}
if [[ -f "${HF_HOME}/token" ]]; then
  export HF_TOKEN="$(tr -d '\\n\\r' < "${HF_HOME}/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [[ -f "${LAB_ROOT}/secrets/TABPFN_TOKEN" ]]; then
  export TABPFN_TOKEN="$(tr -d '\\n\\r' < "${LAB_ROOT}/secrets/TABPFN_TOKEN")"
fi
export SSL_TABPFN3_CKPT="${SSL_TABPFN3_CKPT:-${SSL_CACHE_ROOT}/tabpfn/tabpfn-v3-classifier-v3_default.ckpt}"
export TABPFN_NO_BROWSER=1
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--account", default="ug_uri_ofir")
    p.add_argument("--partition", default="p_uriofir")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    lab = Path(os.environ.get("LAB_ROOT", "/private/ofirlin-lab/suissad4"))
    project = Path(os.environ.get("SSL_PROJECT_ROOT", ROOT))
    wave_dir = lab / "results" / "raw_shards" / WAVE
    log_dir = lab / "results" / "logs" / WAVE
    wave_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Source snapshot hash for run manifests
    hash_path = project / "results" / "validation" / "source_snapshot" / "SOURCE_TREE_HASH.txt"
    source_hash = hash_path.read_text().strip() if hash_path.exists() else "unknown"
    ident_path = lab / "caches" / "tabpfn3_identity.json"
    tabpfn_ident = json.loads(ident_path.read_text()) if ident_path.exists() else {}

    tasks = []
    for ds, method, seed, budget in product(
        DATASETS_SMALL + DATASETS_LARGE, METHODS, SEEDS, BUDGETS
    ):
        profile = "gpu_tfm_large" if ds in DATASETS_LARGE else "gpu_tfm_small"
        tasks.append(
            {
                "task_id": len(tasks),
                "dataset": ds,
                "method": method,
                "seed": seed,
                "n_labeled": budget,
                "wave": WAVE,
                "profile": profile,
            }
        )
    assert len(tasks) == 240, len(tasks)

    small_ids = [t["task_id"] for t in tasks if t["profile"] == "gpu_tfm_small"]
    large_ids = [t["task_id"] for t in tasks if t["profile"] == "gpu_tfm_large"]
    assert len(small_ids) == 192 and len(large_ids) == 48

    task_map = wave_dir / "task_map.json"
    meta = {
        "wave": WAVE,
        "n_tasks": 240,
        "datasets": DATASETS_SMALL + DATASETS_LARGE,
        "methods": METHODS,
        "seeds": SEEDS,
        "label_budgets": BUDGETS,
        "source_tree_hash": source_hash,
        "tabpfn3_checkpoint": tabpfn_ident.get("checkpoint_name"),
        "tabpfn3_sha256": tabpfn_ident.get("checkpoint_sha256"),
        "small_task_ids": small_ids,
        "large_task_ids": large_ids,
        "partition": args.partition,
        "concurrency": args.concurrency,
    }
    task_map.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    (wave_dir / "wave_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    env_prefix = f"{lab}/envs/ssl-tfm"
    run_single = project / "scripts" / "cluster" / "run_single.sh"
    common_exports = f"""
export LAB_ROOT={lab}
export SSL_PROJECT_ROOT={project}
export SSL_CACHE_ROOT={lab}/caches
export HF_HOME={lab}/caches/hf
export XDG_CACHE_HOME={lab}/caches/xdg
export SSL_WAVE={WAVE}
export SSL_TASK_MAP={task_map}
export SSL_ENV_PREFIX={env_prefix}
export SSL_SOURCE_TREE_HASH={source_hash}
"""

    def write_array(name: str, id_lo: int, id_hi: int, mem: str, time_lim: str) -> Path:
        array_spec = f"{id_lo}-{id_hi}%{args.concurrency}"
        path = wave_dir / f"submit_{name}.sbatch"
        body = sbatch_header(
            job_name=f"ssl-{WAVE}-{name}",
            account=args.account,
            partition=args.partition,
            gres="gpu:1",
            cpus=8,
            mem=mem,
            time_lim=time_lim,
            array_spec=array_spec,
            log_dir=log_dir,
        )
        body += common_exports
        body += runtime_auth_block()
        body += f'bash {run_single} --task-id "$SLURM_ARRAY_TASK_ID"\n'
        path.write_text(body, encoding="utf-8")
        return path

    small_sbatch = write_array("small", 0, 191, mem="64G", time_lim="2:00:00")
    large_sbatch = write_array("large", 192, 239, mem="96G", time_lim="4:00:00")

    # Validation job (afterany of both arrays)
    validate_sbatch = wave_dir / "validate.sbatch"
    validate_sbatch.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=ssl-{WAVE}-validate
#SBATCH --account={args.account}
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err
# DEPENDENCY_PLACEHOLDER
set -euo pipefail
export LAB_ROOT={lab}
export SSL_PROJECT_ROOT={project}
export SSL_CACHE_ROOT={lab}/caches
PY={env_prefix}/bin/python
cd "$SSL_PROJECT_ROOT"
"$PY" scripts/cluster/monitor_wave.py --wave {WAVE} --list-failed || true
"$PY" scripts/cluster/monitor_wave.py --wave {WAVE} --list-missing || true
"$PY" - <<'PY'
import json, sys
from pathlib import Path
wave = "{WAVE}"
lab = Path("{lab}")
root = Path("{project}")
tm = json.loads((lab / "results/raw_shards" / wave / "task_map.json").read_text())
expected = {{(t["dataset"], t["method"], int(t["seed"]), int(t["n_labeled"])) for t in tm}}
shard_dir = lab / "results/raw_shards" / wave
rows = []
seen = set()
dups = []
failed = []
for p in shard_dir.glob("*.json"):
    if p.name in {{"task_map.json", "wave_meta.json"}} or p.name.endswith(".sbatch"):
        continue
    d = json.loads(p.read_text())
    if "dataset" not in d or "method" not in d:
        continue
    key = (d["dataset"], d["method"], int(d["seed"]), int(d["n_labeled"]))
    if key in seen:
        dups.append(key)
    seen.add(key)
    if d.get("status") != "success":
        failed.append({{"key": key, "status": d.get("status"), "error": d.get("error_message")}})
    rows.append(d)
missing = sorted(expected - seen)
report = {{
    "wave": wave,
    "expected": len(expected),
    "actual_shards": len(seen),
    "n_duplicates": len(dups),
    "duplicates": dups[:20],
    "n_failed": len(failed),
    "failed": failed[:50],
    "n_missing": len(missing),
    "missing": missing[:50],
    "valid": len(missing) == 0 and len(dups) == 0 and len(failed) == 0 and len(seen) == len(expected),
}}
out = root / "results/reports" / wave
out.mkdir(parents=True, exist_ok=True)
(out / "phase_a_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps({{k: report[k] for k in ("expected","actual_shards","n_failed","n_missing","n_duplicates","valid")}}, indent=2))
if not report["valid"]:
    sys.exit(2)
PY
""",
        encoding="utf-8",
    )

    report_sbatch = wave_dir / "report.sbatch"
    report_sbatch.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=ssl-{WAVE}-report
#SBATCH --account={args.account}
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err
# DEPENDENCY_PLACEHOLDER
set -euo pipefail
export LAB_ROOT={lab}
export SSL_PROJECT_ROOT={project}
PY={env_prefix}/bin/python
cd "$SSL_PROJECT_ROOT"
# Refuse to aggregate if validation failed
"$PY" - <<'PY'
import json, sys
from pathlib import Path
p = Path("results/reports/{WAVE}/phase_a_validation.json")
if not p.exists():
    raise SystemExit("missing phase_a_validation.json")
rep = json.loads(p.read_text())
if not rep.get("valid"):
    raise SystemExit("validation invalid; refusing report aggregation")
print("validation_ok", rep["actual_shards"])
PY
"$PY" scripts/cluster/postprocess_wave.py --wave {WAVE} --expected 240
""",
        encoding="utf-8",
    )

    print(f"wrote {len(tasks)} tasks → {task_map}")
    print(f"small={len(small_ids)} large={len(large_ids)}")
    print("sbatch files:", small_sbatch, large_sbatch, validate_sbatch, report_sbatch)

    if args.dry_run or not args.submit:
        print("dry-run / not submitting (pass --submit)")
        return 0

    def submit(path: Path, dependency: str | None = None) -> str:
        text = path.read_text()
        if dependency:
            text = text.replace(
                "# DEPENDENCY_PLACEHOLDER",
                f"#SBATCH --dependency={dependency}",
            )
            path.write_text(text)
        elif "# DEPENDENCY_PLACEHOLDER" in text:
            text = text.replace("# DEPENDENCY_PLACEHOLDER\n", "")
            path.write_text(text)
        proc = subprocess.run(["sbatch", str(path)], capture_output=True, text=True, check=False)
        print(proc.stdout.strip())
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        if proc.returncode != 0:
            raise SystemExit(f"sbatch failed for {path}: rc={proc.returncode}")
        job_id = proc.stdout.strip().split()[-1]
        return job_id

    j_small = submit(small_sbatch)
    j_large = submit(large_sbatch)
    j_val = submit(validate_sbatch, dependency=f"afterany:{j_small}:{j_large}")
    j_rep = submit(report_sbatch, dependency=f"afterok:{j_val}")

    ids = {
        "wave": WAVE,
        "small_job_id": j_small,
        "large_job_id": j_large,
        "validate_job_id": j_val,
        "report_job_id": j_rep,
        "n_tasks": 240,
        "n_small": len(small_ids),
        "n_large": len(large_ids),
        "partition": args.partition,
        "concurrency": args.concurrency,
        "source_tree_hash": source_hash,
        "tabpfn3_checkpoint": tabpfn_ident.get("checkpoint_name"),
        "tabpfn3_sha256": tabpfn_ident.get("checkpoint_sha256"),
        "shard_dir": str(wave_dir),
        "log_dir": str(log_dir),
        "combined_csv": str(project / "results" / "raw" / f"{WAVE}.csv"),
    }
    (wave_dir / "submission.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    (wave_dir / "last_job_id.txt").write_text(
        f"small={j_small}\nlarge={j_large}\nvalidate={j_val}\nreport={j_rep}\n",
        encoding="utf-8",
    )
    print(json.dumps(ids, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
