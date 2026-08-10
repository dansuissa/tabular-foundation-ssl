#!/usr/bin/env python3
"""Create a deterministic source snapshot when Git metadata is unavailable.

Hashes relevant source/config/scripts only. Excludes secrets, caches, datasets,
checkpoints, environments, logs, results, and .git.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "validation" / "source_snapshot"

INCLUDE_GLOBS = [
    "src/**/*.py",
    "tests/**/*.py",
    "configs/**/*",
    "scripts/**/*.py",
    "scripts/**/*.sh",
    "environment/**/*.yml",
    "environment/**/README.md",
    "docs/**/*.md",
    "requirements.txt",
    "README.md",
]

EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "results",
    "third_party",
    "secrets",
    "caches",
    "envs",
    "node_modules",
    ".cursor",
}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_PARTS:
        return True
    if "secrets" in str(path).lower():
        return True
    if path.suffix in {".ckpt", ".pt", ".bin", ".safetensors", ".csv", ".parquet"}:
        return True
    return False


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for pattern in INCLUDE_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file() and not should_skip(p.relative_to(ROOT)):
                files.append(p)
    # unique sorted
    files = sorted(set(files), key=lambda p: str(p.relative_to(ROOT)))

    entries = []
    tree = hashlib.sha256()
    for p in files:
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        data = p.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        entries.append({"path": rel, "sha256": h, "bytes": len(data)})
        tree.update(rel.encode())
        tree.update(b"\0")
        tree.update(h.encode())
        tree.update(b"\0")

    aggregate = tree.hexdigest()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "git_available": False,
        "reason": "No .git directory found under project or parents; using deterministic source snapshot.",
        "n_files": len(entries),
        "aggregate_source_tree_hash": aggregate,
        "files": entries,
        "exclusions": sorted(EXCLUDE_PARTS),
    }
    out = OUT_DIR / "source_snapshot.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "SOURCE_TREE_HASH.txt").write_text(aggregate + "\n", encoding="utf-8")
    print("wrote", out)
    print("aggregate_source_tree_hash", aggregate)
    print("n_files", len(entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
