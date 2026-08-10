#!/usr/bin/env python3
"""Rigorous local validation harness for TFM-SSL implementation.

Writes results/validation/tfm_ssl_test_report.md with every command + outcome.
Does NOT launch large experiment waves.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "results" / "validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "tfm_ssl_test_report.md"
LOG_JSON = REPORT_DIR / "tfm_ssl_test_report.json"

CORE_PY = Path(os.environ.get("SSL_CORE_PREFIX", "/private/ofirlin-lab/suissad4/envs/ssl-core")) / "bin" / "python"
TFM_PY = Path(os.environ.get("SSL_TFM_PREFIX", "/private/ofirlin-lab/suissad4/envs/ssl-tfm")) / "bin" / "python"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_cmd(
    name: str,
    cmd: list[str] | str,
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: int | None = 1800,
    shell: bool = False,
) -> dict:
    start = time.time()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    # Never capture secrets into the report
    for secret in ("TABPFN_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if secret in merged and merged[secret]:
            merged[secret] = "***REDACTED***"
    display_cmd = cmd if isinstance(cmd, str) else " ".join(cmd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or ROOT),
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell,
        )
        out = {
            "name": name,
            "command": display_cmd,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-8000:],
            "stderr": (proc.stderr or "")[-8000:],
            "elapsed_s": round(time.time() - start, 3),
            "ok": proc.returncode == 0,
            "skipped": False,
        }
    except subprocess.TimeoutExpired as exc:
        out = {
            "name": name,
            "command": display_cmd,
            "returncode": -1,
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": f"TIMEOUT after {timeout}s",
            "elapsed_s": round(time.time() - start, 3),
            "ok": False,
            "skipped": False,
        }
    except FileNotFoundError as exc:
        out = {
            "name": name,
            "command": display_cmd,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_s": round(time.time() - start, 3),
            "ok": False,
            "skipped": False,
        }
    print(f"[{'PASS' if out['ok'] else 'FAIL'}] {name} ({out['elapsed_s']}s)", flush=True)
    return out


def skip(name: str, reason: str, command: str = "") -> dict:
    print(f"[SKIP] {name}: {reason}", flush=True)
    return {
        "name": name,
        "command": command,
        "returncode": None,
        "stdout": "",
        "stderr": reason,
        "elapsed_s": 0.0,
        "ok": False,
        "skipped": True,
    }


def git_status_block() -> dict:
    return run_cmd("git_status", ["git", "status"], timeout=30)


def check_duplicates() -> dict:
    code = r"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
from src.models.registry_ext import EXTENDED_METHODS
from src.run_benchmark import METHOD_GROUPS
# classical names
from src import run_benchmark as rb
classical = set()
for attr in dir(rb):
    pass
# Collect from config
import yaml
cfg = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
names = []
for k,v in cfg.items():
    if isinstance(v, list) and k.endswith("methods") or k in {
        "methods","full_first_wave_methods","paper_methods_no_vime_lite",
        "tfm_frozen","tfm_ssl_core","modern_ssl","geometric_ssl_ablation",
        "tfm_geometric","transductive_exploratory","all_methods_with_neural",
        "neural_ssl_methods","neural_ssl_experimental_methods","local_debug_methods",
    }:
        names.extend(v)
names.extend(EXTENDED_METHODS)
# duplicates within each group
from collections import Counter
issues = []
for k,v in cfg.items():
    if isinstance(v, list) and all(isinstance(x,str) for x in v):
        c = Counter(v)
        dups = [n for n,cnt in c.items() if cnt>1]
        if dups:
            issues.append(f"{k}: {dups}")
# registry vs groups consistency for extended
print("extended_count", len(EXTENDED_METHODS))
print("issues", issues or "none")
if issues:
    raise SystemExit(1)
"""
    py = str(CORE_PY if CORE_PY.exists() else sys.executable)
    return run_cmd("duplicate_method_names", [py, "-c", code])


def dependency_imports(py: Path, label: str) -> dict:
    if not py.exists():
        return skip(f"deps_{label}", f"python missing: {py}")
    if label == "core":
        code = (
            "import numpy,pandas,sklearn,yaml,matplotlib,xgboost,lightgbm,catboost,torch;"
            "print('ok', __import__('sys').version.split()[0])"
        )
    else:
        code = (
            "import numpy,pandas,sklearn,yaml,torch,catboost;"
            "import tabpfn, tabicl;"
            "print('ok', __import__('sys').version.split()[0], 'cuda', torch.cuda.is_available(), "
            "'torch', torch.__version__, 'tabpfn', getattr(tabpfn,'__version__','?'), "
            "'tabicl', getattr(tabicl,'__version__','?'))"
        )
    return run_cmd(f"deps_{label}", [str(py), "-c", code])


def config_validation() -> dict:
    code = r"""
import sys
from pathlib import Path
import yaml
sys.path.insert(0,'.')
from src.utils import load_yaml
from src.method_capabilities import METHOD_CAPABILITIES, get_capabilities
from src.models.registry_ext import EXTENDED_METHODS
bc = load_yaml(Path('configs/benchmark.yaml'))
dc = load_yaml(Path('datasets.yaml') if False else Path('configs/datasets.yaml'))
assert 'datasets' in dc and 'dataset_groups' in dc
assert 'low_class_wave' in dc['dataset_groups']
assert len(dc['dataset_groups']['low_class_wave']) == 10
groups = ['tfm_frozen','tfm_ssl_core','modern_ssl','geometric_ssl_ablation']
for g in groups:
    assert g in bc, g
    for m in bc[g]:
        caps = get_capabilities(m)
        assert caps.name == m
        print(g, m, caps.env, caps.input_view, caps.device)
# every extended method has capabilities
missing = [m for m in EXTENDED_METHODS if m not in METHOD_CAPABILITIES]
print('missing_caps', missing)
if missing:
    raise SystemExit(1)
print('config_ok')
"""
    py = str(CORE_PY if CORE_PY.exists() else sys.executable)
    return run_cmd("config_validation", [py, "-c", code])


def write_report(results: list[dict], meta: dict) -> None:
    lines = [
        "# TFM-SSL Validation Test Report",
        "",
        f"Generated: {now()}",
        f"Host: {meta.get('hostname')}",
        f"Project: `{ROOT}`",
        "",
        "## Summary",
        "",
    ]
    passed = sum(1 for r in results if r.get("ok"))
    failed = sum(1 for r in results if (not r.get("ok") and not r.get("skipped")))
    skipped = sum(1 for r in results if r.get("skipped"))
    lines += [
        f"- Total checks: {len(results)}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        f"- Skipped: {skipped}",
        f"- Gate open for large waves: **{'NO' if failed else 'YES (pending smoke)'}**",
        "",
        "## Environment",
        "",
        f"- ssl-core python: `{CORE_PY}` exists={CORE_PY.exists()}",
        f"- ssl-tfm python: `{TFM_PY}` exists={TFM_PY.exists()}",
        f"- TABPFN_TOKEN set: {bool(os.environ.get('TABPFN_TOKEN'))} (value never printed)",
        f"- HF_TOKEN set: {bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))}",
        f"- Git: {meta.get('git')}",
        "",
        "## Results",
        "",
    ]
    for r in results:
        status = "SKIP" if r.get("skipped") else ("PASS" if r.get("ok") else "FAIL")
        lines += [
            f"### {r['name']} — {status}",
            "",
            f"- Command: `{r.get('command','')}`",
            f"- Return code: `{r.get('returncode')}`",
            f"- Elapsed: `{r.get('elapsed_s')}s`",
            "",
        ]
        if r.get("stdout"):
            lines += ["```", r["stdout"].rstrip(), "```", ""]
        if r.get("stderr"):
            lines += ["stderr:", "```", r["stderr"].rstrip(), "```", ""]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG_JSON.write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}", flush=True)


def main() -> int:
    import socket

    results: list[dict] = []
    meta = {
        "hostname": socket.gethostname(),
        "started": now(),
        "git": "not a git repository",
    }

    results.append(git_status_block())
    if results[-1]["ok"]:
        meta["git"] = "git available"
    else:
        meta["git"] = "not a git repository / git unavailable"

    # Formatting / lint / typecheck — only where configured
    if (ROOT / "pyproject.toml").exists() or (ROOT / ".ruff.toml").exists():
        results.append(run_cmd("ruff_check", [str(CORE_PY), "-m", "ruff", "check", "src", "tests", "scripts"]))
        results.append(run_cmd("ruff_format_check", [str(CORE_PY), "-m", "ruff", "format", "--check", "src", "tests", "scripts"]))
    else:
        results.append(skip("formatting", "No ruff/black/pyproject formatter config in repo"))
        results.append(skip("linting_ruff", "No ruff config; installing ad-hoc ruff for informational check only"))
        # informational ad-hoc
        if CORE_PY.exists():
            results.append(
                run_cmd(
                    "ruff_ad_hoc_install",
                    [str(CORE_PY), "-m", "pip", "install", "-q", "ruff"],
                    timeout=300,
                )
            )
            if results[-1]["ok"]:
                results.append(
                    run_cmd(
                        "ruff_ad_hoc_check",
                        [str(CORE_PY), "-m", "ruff", "check", "src", "tests"],
                        timeout=300,
                    )
                )
        results.append(skip("type_checking", "No mypy/pyright configuration present"))

    # Unit tests
    py = str(CORE_PY if CORE_PY.exists() else sys.executable)
    results.append(
        run_cmd(
            "unit_tests",
            [py, "-m", "pytest", "tests/", "-q", "--tb=short"],
            timeout=1200,
        )
    )

    results.append(check_duplicates())
    results.append(config_validation())
    results.append(dependency_imports(CORE_PY, "core"))
    results.append(dependency_imports(TFM_PY, "tfm"))

    # Reject list sanity (paths that must not be committed / large)
    reject = {
        "secrets": list(ROOT.glob("**/.env")) + list(ROOT.glob("**/credentials*")),
        "checkpoints": list(ROOT.glob("**/*.ckpt")) + list(ROOT.glob("**/*.pt")),
        "openml_data": list((ROOT / "data").glob("**/*")) if (ROOT / "data").exists() else [],
        "env_dirs": [p for p in ROOT.glob("**/envs/**") if p.is_dir()],
    }
    reject_ok = not any(reject.values())
    results.append(
        {
            "name": "reject_artifacts_in_project_tree",
            "command": "scan project for secrets/checkpoints/openml/env dirs",
            "returncode": 0 if reject_ok else 1,
            "stdout": json.dumps({k: [str(x) for x in v[:20]] for k, v in reject.items()}, indent=2),
            "stderr": "",
            "elapsed_s": 0.0,
            "ok": reject_ok,
            "skipped": False,
        }
    )

    results.append(git_status_block())
    meta["finished"] = now()
    write_report(results, meta)
    hard_fail = any(not r["ok"] and not r.get("skipped") and r["name"] in {
        "unit_tests", "config_validation", "duplicate_method_names", "deps_core"
    } for r in results)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
