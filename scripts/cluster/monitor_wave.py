#!/usr/bin/env python3
"""Monitor shard completion for a wave; optionally list failed/incomplete tasks."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.results_io.shards import list_shards, read_shard

KEY = ("dataset", "method", "seed", "n_labeled")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wave", required=True)
    p.add_argument("--list-failed", action="store_true")
    p.add_argument("--list-missing", action="store_true")
    args = p.parse_args()

    shard_root = Path(os.environ.get("SSL_SHARD_ROOT", "/private/ofirlin-lab/suissad4/results/raw_shards"))
    # list_shards() resolves its root from this environment variable. Keep the
    # monitor's explicit/default lab root authoritative.
    os.environ["SSL_SHARD_ROOT"] = str(shard_root)
    wave_dir = shard_root / args.wave
    task_map_path = wave_dir / "task_map.json"
    expected_tasks = []
    if task_map_path.exists():
        expected_tasks = json.loads(task_map_path.read_text())

    statuses: Counter[str] = Counter()
    by_key: dict[tuple, dict] = {}
    bad = 0
    for path in list_shards(args.wave):
        if path.name in {"task_map.json", "wave_meta.json", "last_job_id.txt"} or path.suffix != ".json":
            continue
        if path.name.endswith(".sbatch"):
            continue
        try:
            payload = read_shard(path)
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print("bad_shard", path, exc)
            continue
        if not all(k in payload for k in KEY):
            continue
        key = tuple(payload[k] for k in KEY)
        by_key[key] = payload
        statuses[str(payload.get("status", "unknown"))] += 1

    expected = len(expected_tasks) if expected_tasks else None
    summary = {
        "wave": args.wave,
        "expected": expected,
        "completed_shards": len(by_key),
        "statuses": dict(statuses),
        "bad": bad,
    }
    print(json.dumps(summary, indent=2))

    if args.list_failed:
        failed = [
            {**{k: payload[k] for k in KEY}, "status": payload.get("status"), "error": payload.get("error_message")}
            for payload in by_key.values()
            if str(payload.get("status", "")).startswith("failed")
            or str(payload.get("status", ""))
            in {"error", "unsupported_unknown_method", "skipped_not_implemented"}
        ]
        print(json.dumps({"failed": failed}, indent=2))

    if args.list_missing and expected_tasks:
        have = set(by_key)
        success = {
            tuple(payload[k] for k in KEY)
            for payload in by_key.values()
            if payload.get("status") == "success"
        }
        missing = []
        for t in expected_tasks:
            key = (t["dataset"], t["method"], t["seed"], t["n_labeled"])
            if key not in success:
                missing.append(t)
        print(json.dumps({"missing_or_unsuccessful": missing, "n": len(missing)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
