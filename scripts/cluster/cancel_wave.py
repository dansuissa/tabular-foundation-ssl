#!/usr/bin/env python3
"""Cancel Slurm jobs for a wave by job name or explicit job id."""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wave", help="Cancels jobs named ssl-<wave>")
    p.add_argument("--job-id", action="append", default=[], help="Explicit job id(s)")
    p.add_argument("--user", default=None)
    args = p.parse_args()

    ids = list(args.job_id)
    if args.wave:
        name = f"ssl-{args.wave}"
        cmd = ["squeue", "-h", "-o", "%i %j"]
        if args.user:
            cmd += ["-u", args.user]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        for line in proc.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1].strip() == name:
                ids.append(parts[0])

    if not ids:
        print("No matching jobs")
        return 0

    print("scancel", " ".join(ids))
    return subprocess.call(["scancel", *ids])


if __name__ == "__main__":
    raise SystemExit(main())
