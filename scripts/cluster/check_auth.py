#!/usr/bin/env python3
"""Check auth material presence without printing secret values."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _present(name: str) -> bool:
    val = os.environ.get(name)
    return bool(val) and len(val.strip()) > 0


def main() -> int:
    checks = {
        "TABPFN_TOKEN": _present("TABPFN_TOKEN"),
        "HF_TOKEN": _present("HF_TOKEN") or _present("HUGGING_FACE_HUB_TOKEN"),
        "TABPFN_NO_BROWSER": os.environ.get("TABPFN_NO_BROWSER", ""),
    }
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    token_files = [
        hf_cache / "token",
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    token_file_present = any(p.is_file() for p in token_files)

    print("TABPFN_TOKEN_set=", checks["TABPFN_TOKEN"])
    print("HF_TOKEN_set=", checks["HF_TOKEN"])
    print("HF_token_file_present=", token_file_present)
    print("TABPFN_NO_BROWSER=", checks["TABPFN_NO_BROWSER"])

    ok = True
    if not checks["TABPFN_TOKEN"]:
        print(
            "BLOCKER: TABPFN_TOKEN unset. Accept PriorLabs license at "
            "https://ux.priorlabs.ai and export TABPFN_TOKEN for headless jobs.",
            file=sys.stderr,
        )
        ok = False
    if not (checks["HF_TOKEN"] or token_file_present):
        print(
            "WARNING: no HF token detected. TabICL may still download public "
            "checkpoints; gated assets will fail.",
            file=sys.stderr,
        )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
