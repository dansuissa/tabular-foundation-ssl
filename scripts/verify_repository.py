#!/usr/bin/env python3
"""Validate the committed research record without model downloads.

This check is intentionally dependency-free so it can run before the analysis
environment is installed.  It verifies the immutable canonical inputs, the
complete experimental design, key report assets, local Markdown links, and
common high-risk credential patterns.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]

CANONICAL = {
    "results/raw/low_class_wave_paper_methods.csv": {
        "sha256": "b017a950054a5045a4a39c41362d3dc3eef56603d1db4d62d4ffbfbdc43e8ad5",
        "rows": 2040,
        "methods": 17,
        "success": 2016,
    },
    "results/raw/tfm_frozen_screen.csv": {
        "sha256": "de8cc5c53b9cf2c80498e732856be9dd5b7548813cfd5eed73b36dc927c73ace",
        "rows": 240,
        "methods": 2,
        "success": 240,
    },
    "results/raw/focused_tfm_ssl.csv": {
        "sha256": "e9d326c99086959d0ec7de9dc89f826e11466a3a8f309f5e0d0762d063266559",
        "rows": 720,
        "methods": 6,
        "success": 720,
    },
}

EXPECTED_DATASETS = {
    "phoneme",
    "spambase",
    "MagicTelescope",
    "adult",
    "bank-marketing",
    "electricity",
    "satimage",
    "segment",
    "steel-plates-fault",
    "jannis",
}
EXPECTED_BUDGETS = {50, 100, 250, 500}
EXPECTED_SEEDS = {0, 1, 2}
KEY = ("dataset", "method", "seed", "n_labeled")

EXPECTED_METHODS = {
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
    "label_spreading",
    "label_propagation",
    "self_training_lr",
    "self_training_xgboost",
    "self_training_lightgbm",
    "self_training_catboost",
    "rpl_lr",
    "rpl_lite_xgboost",
    "sslae",
    "vime",
    "scarf",
    "tabpfn3",
    "tabiclv2",
    "tabpfn3_self_training",
    "tabiclv2_self_training",
    "laplacian_ssl",
    "unlabeled_attention_ssl",
    "embedding_alignment_ssl",
    "geometric_attention_ssl",
}

REQUIRED_ASSETS = (
    "README.md",
    "docs/METHODS.md",
    "docs/RESULTS.md",
    "docs/REPRODUCIBILITY.md",
    "results/reports/main_report/README.md",
    "results/reports/main_report/figures/overview/project_overview.png",
    "results/reports/main_report/figures/overview/complete_method_matrix.png",
    "results/reports/main_report/tables/overview/all_method_summary.csv",
    "results/reports/main_report/tables/overview/all_methods_by_budget.csv",
    "results/reports/main_report/tables/overview/all_methods_by_dataset.csv",
    "results/reports/main_report/validation/integrity_validation.json",
    "results/reports/main_report/validation/run_manifest.json",
)

MARKDOWN_LINK = re.compile(r"!?\[[^]]*]\(([^)]+)\)")
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".csv",
    ".tex",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".ps1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_canonical(errors: list[str]) -> None:
    total_rows = 0
    all_methods: set[str] = set()
    for relative, expected in CANONICAL.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing canonical file: {relative}")
            continue
        actual_hash = sha256(path)
        if actual_hash != expected["sha256"]:
            errors.append(
                f"hash mismatch for {relative}: {actual_hash} != {expected['sha256']}"
            )

        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        total_rows += len(rows)
        methods = {row["method"] for row in rows}
        all_methods.update(methods)
        successes = sum(row["status"] == "success" for row in rows)
        if len(rows) != expected["rows"]:
            errors.append(f"{relative}: expected {expected['rows']} rows, found {len(rows)}")
        if len(methods) != expected["methods"]:
            errors.append(
                f"{relative}: expected {expected['methods']} methods, found {len(methods)}"
            )
        if successes != expected["success"]:
            errors.append(
                f"{relative}: expected {expected['success']} successes, found {successes}"
            )

        datasets = {row["dataset"] for row in rows}
        budgets = {int(row["n_labeled"]) for row in rows}
        seeds = {int(row["seed"]) for row in rows}
        if datasets != EXPECTED_DATASETS:
            errors.append(f"{relative}: dataset grid differs from canonical design")
        if budgets != EXPECTED_BUDGETS:
            errors.append(f"{relative}: budget grid differs from canonical design")
        if seeds != EXPECTED_SEEDS:
            errors.append(f"{relative}: seed grid differs from canonical design")

        keys = [tuple(row[column] for column in KEY) for row in rows]
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        if duplicates:
            errors.append(f"{relative}: {len(duplicates)} duplicate experimental keys")

    if total_rows != 3000:
        errors.append(f"canonical total: expected 3000 rows, found {total_rows}")
    if len(all_methods) != 25:
        errors.append(f"canonical total: expected 25 methods, found {len(all_methods)}")


def check_assets(errors: list[str]) -> None:
    for relative in REQUIRED_ASSETS:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required asset: {relative}")


def check_markdown_links(errors: list[str]) -> None:
    for path in ROOT.rglob("*.md"):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            target = unquote(target).split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"{path.relative_to(ROOT)}: link leaves repository: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{path.relative_to(ROOT)}: broken local link: {target}")


def check_secrets(errors: list[str]) -> None:
    skip_parts = {".git", ".venv", "venv", "__pycache__"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if skip_parts.intersection(path.relative_to(ROOT).parts):
            continue
        # Very large JSONL diagnostics are scanned in bounded streaming chunks
        # by read_text here only when under 20 MiB; larger files are result data
        # and are excluded from the textual credential surface.
        if path.stat().st_size > 20 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {path.relative_to(ROOT)}")


def check_documentation_contract(errors: list[str]) -> None:
    """Ensure the public presentation cannot omit a canonical method."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ranking_methods = re.findall(
        r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|",
        readme,
        flags=re.MULTILINE,
    )
    if len(ranking_methods) != len(EXPECTED_METHODS):
        errors.append(
            "README complete ranking must contain exactly "
            f"{len(EXPECTED_METHODS)} method rows; found {len(ranking_methods)}"
        )
    if set(ranking_methods) != EXPECTED_METHODS:
        missing = sorted(EXPECTED_METHODS - set(ranking_methods))
        extra = sorted(set(ranking_methods) - EXPECTED_METHODS)
        errors.append(f"README method coverage mismatch: missing={missing}, extra={extra}")

    report_root = ROOT / "results/reports"
    report_directories = sorted(path.name for path in report_root.iterdir() if path.is_dir())
    if report_directories != ["main_report"]:
        errors.append(
            "results/reports must contain only main_report; found "
            f"{report_directories}"
        )

    report_markdown = sorted(
        path.relative_to(report_root).as_posix() for path in report_root.rglob("*.md")
    )
    if report_markdown != ["main_report/README.md"]:
        errors.append(
            "main_report/README.md must be the only report document; found "
            f"{report_markdown}"
        )

    report_pdfs = sorted(path.relative_to(report_root).as_posix() for path in report_root.rglob("*.pdf"))
    if report_pdfs:
        errors.append(f"report PDF duplicates are not allowed: {report_pdfs}")



def check_code_documentation(errors: list[str]) -> None:
    """Require every maintained Python module to state its responsibility."""
    for source_root in (ROOT / "src", ROOT / "scripts"):
        for path in source_root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                errors.append(f"syntax error in {path.relative_to(ROOT)}: {exc}")
                continue
            if ast.get_docstring(tree) is None:
                errors.append(f"missing module docstring: {path.relative_to(ROOT)}")


def check_figure_dimensions(errors: list[str]) -> None:
    """Reject accidentally tiny or corrupt README PNG exports."""
    minimums = {
        "results/reports/main_report/figures/overview/project_overview.png": (2400, 1500),
        "results/reports/main_report/figures/overview/complete_method_matrix.png": (3000, 2000),
    }
    for relative, (minimum_width, minimum_height) in minimums.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        header = path.read_bytes()[:24]
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
            errors.append(f"invalid PNG header: {relative}")
            continue
        width, height = struct.unpack(">II", header[16:24])
        if width < minimum_width or height < minimum_height:
            errors.append(
                f"README figure too small: {relative} is {width}x{height}; "
                f"minimum is {minimum_width}x{minimum_height}"
            )


def main() -> int:
    errors: list[str] = []
    check_canonical(errors)
    check_assets(errors)
    check_markdown_links(errors)
    check_secrets(errors)
    check_documentation_contract(errors)
    check_code_documentation(errors)
    check_figure_dimensions(errors)

    if errors:
        print("Repository verification FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Repository verification passed")
    print("  canonical rows: 3,000")
    print("  canonical methods: 25")
    print("  grid: 10 datasets × 4 budgets × 3 seeds")
    print("  successful runs: 2,976; preserved failures: 24")
    print("  complete README method coverage and report naming policy: OK")
    print("  code module documentation and README figure dimensions: OK")
    print("  hashes, unique keys, required assets, links, and secret patterns: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
