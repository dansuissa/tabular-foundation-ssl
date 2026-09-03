"""Main benchmark CLI and per-cell orchestration.

The runner constructs one shared split, exposes capability-appropriate feature
views, records every success or failure, and supports both append-safe CSV output
and atomic JSON shards for Slurm arrays.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import dataset_from_config, load_dataset
from src.exceptions import OptionalDependencyError, TFMOOMError, UnsupportedMethodError
from src.metrics import compute_metrics, flatten_metrics
from src.metrics_ext import compute_extended_prob_metrics
from src.method_capabilities import METHOD_CAPABILITIES, METHOD_GROUPS, get_capabilities
from src.models import run_model, run_model_from_context
from src.models.neural_ssl import NEURAL_SSL_METHODS, build_neural_ssl_model
from src.models.registry_ext import EXTENDED_METHODS, build_extended_model
from src.models.rpl import build_rpl_model
from src.models.semi_supervised import build_semi_supervised_model
from src.models.supervised import build_supervised_model
from src.models.common import GraphSSLMissingClassesError, GraphSSLNanError
from src.results_io.manifest import build_result_payload, code_version, config_hash
from src.results_io.shards import make_run_id, shard_success_exists, write_shard_atomic
from src.splits import InvalidBudgetError, SplitError, make_ssl_split
from src.utils import ensure_dir, load_yaml, set_seed, setup_logging
from src.views import FitContext, build_dataset_views

RESULT_COLUMNS = [
    "timestamp_utc",
    "dataset",
    "dataset_id",
    "method",
    "seed",
    "n_labeled",
    "n_unlabeled",
    "train_labeled_size",
    "val_size",
    "test_size",
    "n_classes",
    "n_features_before_preprocessing",
    "n_features_after_preprocessing",
    "n_pseudo_added_total",
    "n_pseudo_added_last_iter",
    "self_training_iterations",
    "pseudo_label_class_distribution",
    "pseudo_label_fraction",
    "mean_selected_confidence",
    "min_selected_confidence",
    "max_selected_confidence",
    "mean_selected_density",
    "stopped_reason",
    "graph_n_neighbors_used",
    "graph_retry_count",
    "graph_n_rows",
    "graph_n_unlabeled_used",
    "neural_method",
    "method_fidelity",
    "reference_family",
    "best_epoch",
    "epochs_trained",
    "pretrain_epochs_trained",
    "finetune_epochs_trained",
    "best_val_loss",
    "final_train_loss",
    "neural_validation_strategy",
    "validation_strategy",
    "labeled_classes_present",
    "all_classes_present_in_labeled",
    "min_labeled_per_class",
    "max_labeled_per_class",
    "train_pool_class_counts",
    "labeled_class_counts",
    "val_class_counts",
    "test_class_counts",
    "unlabeled_class_counts",
    "runtime_seconds",
    "status",
    "error_message",
]

SUPERVISED_METHODS = {
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
}
SEMI_SUPERVISED_METHODS = {
    "label_spreading",
    "label_propagation",
    "self_training_lr",
    "self_training_xgboost",
    "self_training_lightgbm",
    "self_training_catboost",
}
RPL_METHODS = {"rpl_lr", "rpl_lite_xgboost"}
NEURAL_METHODS = set(NEURAL_SSL_METHODS)

ExperimentKey = tuple[str, str, int, int]


def resolve_neural_params(benchmark_cfg: dict[str, Any], method: str) -> dict[str, Any]:
    """Merge neural defaults with per-method overrides from benchmark config."""
    params: dict[str, Any] = {}
    params.update(benchmark_cfg.get("neural_ssl_defaults", {}) or {})
    params.update(benchmark_cfg.get(method, {}) or {})
    return params


def build_model(
    method: str,
    random_state: int,
    n_classes: int = 2,
    neural_params: dict[str, Any] | None = None,
    method_params: dict[str, Any] | None = None,
):
    if method in SUPERVISED_METHODS:
        return build_supervised_model(method, random_state=random_state, n_classes=n_classes)
    if method in SEMI_SUPERVISED_METHODS:
        return build_semi_supervised_model(
            method, random_state=random_state, n_classes=n_classes
        )
    if method in RPL_METHODS:
        return build_rpl_model(method, random_state=random_state, n_classes=n_classes)
    if method in NEURAL_METHODS:
        return build_neural_ssl_model(
            method,
            random_state=random_state,
            n_classes=n_classes,
            **(neural_params or {}),
        )
    if method in EXTENDED_METHODS:
        return build_extended_model(
            method,
            random_state=random_state,
            n_classes=n_classes,
            **(method_params or {}),
        )
    raise ValueError(f"Unknown method: {method}")


def experiment_key(
    dataset: str, method: str, seed: int, label_budget: int
) -> ExperimentKey:
    return (dataset, method, int(seed), int(label_budget))


def load_existing_results(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(output_path)
    except Exception:
        # Be robust to partially-written or schema-mismatched debug CSVs.
        try:
            return pd.read_csv(output_path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def build_skip_sets(
    existing: pd.DataFrame,
    resume: bool,
    skip_failed: bool,
) -> tuple[set[ExperimentKey], set[ExperimentKey]]:
    if existing.empty:
        return set(), set()

    success_keys = set()
    failed_keys = set()
    for _, row in existing.iterrows():
        key = experiment_key(row["dataset"], row["method"], row["seed"], row["n_labeled"])
        if row.get("status") == "success":
            success_keys.add(key)
        elif row.get("status") == "failed":
            failed_keys.add(key)

    skip = set()
    if resume:
        skip.update(success_keys)
    if skip_failed:
        skip.update(failed_keys)
    return success_keys, skip


def resolve_list(single_values: list[Any] | None, batch_values: list[Any] | None, default: list[Any]) -> list[Any]:
    if batch_values is not None:
        return batch_values
    if single_values is not None:
        return single_values
    return default


def _drop_existing_rows(output_path: Path, key: ExperimentKey) -> None:
    """Remove any rows matching (dataset, method, seed, n_labeled)."""
    if not output_path.exists():
        return
    df = load_existing_results(output_path)
    if df.empty:
        return

    dataset, method, seed, n_labeled = key
    required_cols = {"dataset", "method", "seed", "n_labeled"}
    if not required_cols.issubset(set(df.columns)):
        return

    mask = (
        (df["dataset"] == dataset)
        & (df["method"] == method)
        & (df["seed"] == seed)
        & (df["n_labeled"] == n_labeled)
    )
    if not bool(mask.any()):
        return

    df.loc[~mask].to_csv(output_path, index=False)


def _ensure_csv_schema(output_path: Path, columns: list[str]) -> None:
    """Rewrite CSV to match current column schema (best-effort)."""
    if not output_path.exists():
        return
    df = load_existing_results(output_path)
    if df.empty:
        return
    df = df.reindex(columns=columns)
    df.to_csv(output_path, index=False)


def base_result_row(
    dataset_name: str,
    dataset_id: int,
    method: str,
    seed: int,
    label_budget: int | None,
    splits=None,
    dataset=None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "dataset_id": dataset_id,
        "method": method,
        "seed": seed,
        "n_labeled": label_budget,
        "n_unlabeled": splits.n_unlabeled if splits else None,
        "train_labeled_size": splits.train_labeled_size if splits else None,
        "val_size": splits.val_size if splits else None,
        "test_size": splits.test_size if splits else None,
        "n_classes": dataset.n_classes if dataset else None,
        "n_features_before_preprocessing": (
            len(dataset.feature_names) if dataset else None
        ),
        "n_features_after_preprocessing": None,
        "n_pseudo_added_total": np.nan,
        "n_pseudo_added_last_iter": np.nan,
        "self_training_iterations": np.nan,
        "pseudo_label_class_distribution": np.nan,
        "pseudo_label_fraction": np.nan,
        "mean_selected_confidence": np.nan,
        "min_selected_confidence": np.nan,
        "max_selected_confidence": np.nan,
        "mean_selected_density": np.nan,
        "stopped_reason": np.nan,
        "graph_n_neighbors_used": np.nan,
        "graph_retry_count": np.nan,
        "graph_n_rows": np.nan,
        "graph_n_unlabeled_used": np.nan,
        "validation_strategy": splits.validation_strategy if splits else None,
        "labeled_classes_present": splits.labeled_classes_present if splits else None,
        "all_classes_present_in_labeled": (
            splits.all_classes_present_in_labeled if splits else None
        ),
        "min_labeled_per_class": splits.min_labeled_per_class if splits else None,
        "max_labeled_per_class": splits.max_labeled_per_class if splits else None,
        "train_pool_class_counts": splits.train_pool_class_counts if splits else None,
        "labeled_class_counts": splits.labeled_class_counts if splits else None,
        "val_class_counts": splits.val_class_counts if splits else None,
        "test_class_counts": splits.test_class_counts if splits else None,
        "unlabeled_class_counts": splits.unlabeled_class_counts if splits else None,
        "runtime_seconds": None,
        "status": "failed",
        "error_message": None,
    }
    return row


def run_single_experiment(
    dataset,
    method: str,
    seed: int,
    label_budget: int,
    test_size: float,
    val_size_from_labeled: float,
    metric_names: list[str],
    neural_params: dict[str, Any] | None = None,
    method_params: dict[str, Any] | None = None,
    use_dual_view: bool = True,
) -> dict[str, Any]:
    row = base_result_row(
        dataset.name,
        dataset.openml_id,
        method,
        seed,
        label_budget,
        dataset=dataset,
    )

    start = time.perf_counter()
    try:
        caps = None
        try:
            caps = get_capabilities(method)
            row["protocol"] = caps.protocol
            row["method_fidelity"] = caps.fidelity
            row["reference_family"] = caps.family
            row["input_view"] = caps.input_view
        except UnsupportedMethodError:
            caps = None

        splits = make_ssl_split(
            dataset.X,
            dataset.y,
            n_labeled=label_budget,
            test_size=test_size,
            val_size_from_labeled=val_size_from_labeled,
            seed=seed,
        )
        row.update(_split_meta_row(splits))
        row.update(
            {
                "n_unlabeled": splits.n_unlabeled,
                "train_labeled_size": splits.train_labeled_size,
                "val_size": splits.val_size,
                "test_size": splits.test_size,
            }
        )

        set_seed(seed)
        views = build_dataset_views(
            splits,
            dataset_name=dataset.name,
            seed=seed,
            n_labeled=label_budget,
            label_encoder_classes_policy="labeled_plus_val",
        )
        row["n_features_after_preprocessing"] = int(views.n_features_processed)
        row["n_classes"] = int(views.n_classes)

        model = build_model(
            method,
            random_state=seed,
            n_classes=views.n_classes,
            neural_params=neural_params,
            method_params=method_params,
        )
        ctx = FitContext(
            views=views,
            random_state=seed,
            method_config=dict(method_params or neural_params or {}),
            method_name=method,
        )
        # Prefer context path for all methods (legacy via adapter).
        if use_dual_view:
            pred = run_model_from_context(model, ctx, eval_split="test")
            y_pred, y_proba, training_meta = pred.y_pred, pred.y_proba, pred.training_meta
        else:
            result = run_model(
                model,
                views.X_labeled_processed,
                views.y_labeled,
                views.X_unlabeled_processed,
                views.X_test_processed,
            )
            y_pred, y_proba, training_meta = result.y_pred, result.y_proba, result.training_meta

        row.update(_training_meta_row(training_meta))
        # Preserve complete method-specific diagnostics in JSON shards. CSV
        # collectors serialize this nested object, while aggregate columns
        # remain flat and backward compatible.
        row["method_diagnostics"] = dict(training_meta)
        # Merge extended training meta keys that fit RESULT_COLUMNS loosely
        for key in (
            "backbone",
            "checkpoint",
            "package_version",
            "cold_load_seconds",
            "warm_inference_seconds",
            "peak_gpu_memory_mb",
            "kv_cache",
            "protocol",
            "method_fidelity",
            "fallback_reason",
            "selected_round",
            "calibration_temperature",
            "embedding_source",
        ):
            if key in training_meta and key not in row:
                row[key] = training_meta[key]
            elif key in training_meta:
                row[key] = training_meta[key]

        metrics = compute_metrics(
            views.y_test,
            y_pred,
            y_proba,
            metric_names,
            dataset.task_type,
        )
        row.update(flatten_metrics(metrics))
        ext = compute_extended_prob_metrics(views.y_test, y_pred, y_proba)
        row["metric_brier"] = ext.get("brier")
        row["metric_ece"] = ext.get("ece")
        row["metric_nll"] = ext.get("nll")
        row["status"] = "success"
    except InvalidBudgetError as exc:
        row["status"] = "skipped_invalid_budget"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        logging.warning(
            "Invalid budget for dataset=%s method=%s seed=%s label_budget=%s: %s",
            dataset.name,
            method,
            seed,
            label_budget,
            exc,
        )
    except UnsupportedMethodError as exc:
        row["status"] = getattr(exc, "status", None) or "unsupported"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        logging.warning(
            "Unsupported method dataset=%s method=%s: %s",
            dataset.name,
            method,
            exc,
        )
    except OptionalDependencyError as exc:
        row["status"] = f"unsupported_missing_dependency_{exc.package}"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        logging.warning(
            "Missing optional dependency for dataset=%s method=%s: %s",
            dataset.name,
            method,
            exc,
        )
    except TFMOOMError as exc:
        row["status"] = "failed_tfm_oom"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        logging.warning("TFM OOM dataset=%s method=%s: %s", dataset.name, method, exc)
    except GraphSSLMissingClassesError as exc:
        row["status"] = "failed_graph_missing_labeled_classes"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        logging.warning(
            "Graph SSL missing classes for dataset=%s method=%s: %s",
            dataset.name,
            method,
            exc,
        )
    except GraphSSLNanError as exc:
        row["status"] = "failed_graph_ssl_nan"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        logging.warning(
            "Graph SSL NaN for dataset=%s method=%s: %s",
            dataset.name,
            method,
            exc,
        )
    except SplitError as exc:
        row["status"] = "failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        logging.warning(
            "Split failed for dataset=%s method=%s seed=%s label_budget=%s: %s",
            dataset.name,
            method,
            seed,
            label_budget,
            exc,
        )
    except ValueError as exc:
        row["status"] = "failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        if "NaN" in str(exc) or "infinite" in str(exc):
            if method in {"label_spreading", "label_propagation"}:
                row["status"] = "failed_graph_ssl_nan"
        logging.exception(
            "Experiment failed for dataset=%s method=%s seed=%s label_budget=%s",
            dataset.name,
            method,
            seed,
            label_budget,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should continue on failure
        row["status"] = "failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        logging.exception(
            "Experiment failed for dataset=%s method=%s seed=%s label_budget=%s",
            dataset.name,
            method,
            seed,
            label_budget,
        )
    finally:
        row["runtime_seconds"] = round(time.perf_counter() - start, 4)

    return row


def _split_meta_row(splits) -> dict[str, Any]:
    return {
        "validation_strategy": splits.validation_strategy,
        "labeled_classes_present": splits.labeled_classes_present,
        "all_classes_present_in_labeled": splits.all_classes_present_in_labeled,
        "min_labeled_per_class": splits.min_labeled_per_class,
        "max_labeled_per_class": splits.max_labeled_per_class,
        "train_pool_class_counts": splits.train_pool_class_counts,
        "labeled_class_counts": splits.labeled_class_counts,
        "val_class_counts": splits.val_class_counts,
        "test_class_counts": splits.test_class_counts,
        "unlabeled_class_counts": splits.unlabeled_class_counts,
    }


def _training_meta_row(training_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_pseudo_added_total": training_meta.get("n_pseudo_added_total", np.nan),
        "n_pseudo_added_last_iter": training_meta.get("n_pseudo_added_last_iter", np.nan),
        "self_training_iterations": training_meta.get("self_training_iterations", np.nan),
        "pseudo_label_class_distribution": training_meta.get(
            "pseudo_label_class_distribution", np.nan
        ),
        "pseudo_label_fraction": training_meta.get("pseudo_label_fraction", np.nan),
        "mean_selected_confidence": training_meta.get("mean_selected_confidence", np.nan),
        "min_selected_confidence": training_meta.get("min_selected_confidence", np.nan),
        "max_selected_confidence": training_meta.get("max_selected_confidence", np.nan),
        "mean_selected_density": training_meta.get("mean_selected_density", np.nan),
        "stopped_reason": training_meta.get("stopped_reason", np.nan),
        "graph_n_neighbors_used": training_meta.get("graph_n_neighbors_used", np.nan),
        "graph_retry_count": training_meta.get("graph_retry_count", np.nan),
        "graph_n_rows": training_meta.get("graph_n_rows", np.nan),
        "graph_n_unlabeled_used": training_meta.get("graph_n_unlabeled_used", np.nan),
        "neural_method": training_meta.get("neural_method", np.nan),
        "method_fidelity": training_meta.get("method_fidelity", np.nan),
        "reference_family": training_meta.get("reference_family", np.nan),
        "best_epoch": training_meta.get("best_epoch", np.nan),
        "epochs_trained": training_meta.get("epochs_trained", np.nan),
        "pretrain_epochs_trained": training_meta.get("pretrain_epochs_trained", np.nan),
        "finetune_epochs_trained": training_meta.get("finetune_epochs_trained", np.nan),
        "best_val_loss": training_meta.get("best_val_loss", np.nan),
        "final_train_loss": training_meta.get("final_train_loss", np.nan),
        "neural_validation_strategy": training_meta.get("neural_validation_strategy", np.nan),
    }


def append_result_row(output_path: Path, row: dict[str, Any], metric_names: list[str]) -> None:
    metric_columns = [f"metric_{name}" for name in metric_names]
    extra_metrics = ["metric_brier", "metric_ece", "metric_nll"]
    columns = RESULT_COLUMNS + metric_columns + [c for c in extra_metrics if c not in RESULT_COLUMNS]
    # Preserve unknown keys from extended meta by union
    for key in row:
        if key not in columns:
            columns.append(key)
    frame = pd.DataFrame([row], columns=columns)
    header = not output_path.exists()
    frame.to_csv(output_path, mode="a", header=header, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semi-supervised tabular classification benchmarks."
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to datasets YAML config.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=Path("configs/benchmark.yaml"),
        help="Path to benchmark YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to results/raw/run_<timestamp>.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists.",
    )
    parser.add_argument(
        "--rerun_existing",
        action="store_true",
        help=(
            "If the output CSV already contains a row with the same "
            "(dataset, method, seed, n_labeled), remove those rows and rerun "
            "before appending the new result. Useful for clean debug reruns."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments already present with status=success in the output CSV.",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Also skip experiments that previously failed (use with --resume).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Single-run filter: dataset name (repeatable).",
    )
    parser.add_argument(
        "--dataset-group",
        "--dataset_group",
        dest="dataset_group",
        type=str,
        default=None,
        help=(
            "Use a named dataset group from the datasets config "
            "(e.g. low_class_wave)."
        ),
    )
    parser.add_argument(
        "--datasets-subset",
        "--datasets_subset",
        dest="datasets_subset",
        nargs="+",
        default=None,
        help="Batch filter: dataset names.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="Single-run filter: method name (repeatable).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Batch filter: method names.",
    )
    parser.add_argument(
        "--method-group",
        "--method_group",
        dest="method_group",
        type=str,
        default=None,
        help=(
            "Use a named method group from the benchmark config "
            "(e.g. full_first_wave_methods, neural_ssl_methods, all_methods_with_neural)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Single-run filter: seed (repeatable).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Batch filter: seeds.",
    )
    parser.add_argument(
        "--label-budget",
        dest="label_budget",
        type=int,
        action="append",
        default=None,
        help="Single-run filter: label budget (repeatable).",
    )
    parser.add_argument(
        "--label-budgets",
        "--label_budgets",
        dest="label_budgets",
        nargs="+",
        type=int,
        default=None,
        help="Batch filter: label budgets.",
    )
    parser.add_argument(
        "--labeled-fraction",
        type=float,
        action="append",
        default=None,
        help="Deprecated. Use --label-budget instead.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    datasets_cfg = load_yaml(args.datasets_config)
    benchmark_cfg = load_yaml(args.benchmark_config)

    seeds = resolve_list(args.seed, args.seeds, benchmark_cfg["seeds"])
    label_budgets = resolve_list(args.label_budget, args.label_budgets, benchmark_cfg["label_budgets"])
    default_methods = benchmark_cfg.get(
        "full_first_wave_methods",
        benchmark_cfg.get("methods", []),
    )
    if args.method_group is not None:
        if args.method_group in benchmark_cfg:
            group_methods = benchmark_cfg[args.method_group]
            if not isinstance(group_methods, list):
                raise SystemExit(f"--method-group '{args.method_group}' is not a list in config.")
            default_methods = group_methods
        elif args.method_group in METHOD_GROUPS:
            default_methods = list(METHOD_GROUPS[args.method_group])
        else:
            raise SystemExit(
                f"Unknown --method-group '{args.method_group}'. "
                f"Available config lists in {args.benchmark_config} or METHOD_GROUPS: "
                f"{sorted(METHOD_GROUPS)}"
            )
    methods = resolve_list(args.method, args.methods, default_methods)
    metric_names = benchmark_cfg["metrics"]
    test_size = benchmark_cfg["test_size"]
    val_size_from_labeled = benchmark_cfg["val_size_from_labeled"]

    if args.labeled_fraction is not None:
        logging.warning("--labeled-fraction is deprecated; use --label-budgets instead.")

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = Path("results/raw") / f"run_{timestamp}.csv"
    ensure_dir(output_path.parent)

    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
            logging.info("Removed existing output file (--overwrite): %s", output_path)
        elif not (args.resume or args.rerun_existing):
            raise SystemExit(
                f"Output file already exists: {output_path}. "
                "Pass --overwrite to replace it or --resume to continue."
            )

    if output_path.exists() and (args.resume or args.rerun_existing):
        # Keep debug files usable even after column additions.
        _ensure_csv_schema(output_path, RESULT_COLUMNS + [f"metric_{m}" for m in benchmark_cfg["metrics"]])

    existing = load_existing_results(output_path)
    _, skip_keys = build_skip_sets(existing, resume=args.resume, skip_failed=args.skip_failed)
    if skip_keys:
        logging.info("Skipping %s experiments based on existing results.", len(skip_keys))

    dataset_entries = datasets_cfg["datasets"]
    selected_datasets = args.datasets_subset or args.dataset
    if args.dataset_group is not None:
        groups = datasets_cfg.get("dataset_groups", {}) or {}
        if args.dataset_group not in groups:
            raise SystemExit(
                f"Unknown --dataset-group '{args.dataset_group}'. "
                f"Available groups must be lists defined in {args.datasets_config} under dataset_groups."
            )
        group_datasets = groups[args.dataset_group]
        if not isinstance(group_datasets, list):
            raise SystemExit(
                f"--dataset-group '{args.dataset_group}' is not a list in {args.datasets_config}."
            )
        selected_datasets = group_datasets
    if selected_datasets:
        selected = set(selected_datasets)
        dataset_entries = [entry for entry in dataset_entries if entry["name"] in selected]

    total_expected = len(dataset_entries) * len(methods) * len(seeds) * len(label_budgets)
    completed = 0
    skipped = 0
    failed = 0

    logging.info("Writing results to %s", output_path)
    logging.info(
        "Planned grid: %s datasets x %s methods x %s seeds x %s budgets = %s experiments",
        len(dataset_entries),
        len(methods),
        len(seeds),
        len(label_budgets),
        total_expected,
    )

    for entry in dataset_entries:
        spec = dataset_from_config(entry)
        logging.info("Loading dataset %s (OpenML id=%s)", spec.name, spec.openml_id)
        dataset = None
        load_error = None
        try:
            dataset = load_dataset(spec)
        except Exception as exc:  # noqa: BLE001
            load_error = exc
            logging.exception("Failed to load dataset %s", spec.name)

        for method in methods:
            for seed in seeds:
                for label_budget in label_budgets:
                    key = experiment_key(spec.name, method, seed, label_budget)
                    if key in skip_keys:
                        skipped += 1
                        continue

                    if load_error is not None:
                        row = base_result_row(
                            spec.name,
                            spec.openml_id,
                            method,
                            seed,
                            label_budget,
                        )
                        row["runtime_seconds"] = 0.0
                        row["error_message"] = f"{type(load_error).__name__}: {load_error}"
                        append_result_row(output_path, row, metric_names)
                        failed += 1
                        continue

                    logging.info(
                        "Running dataset=%s method=%s seed=%s label_budget=%s",
                        dataset.name,
                        method,
                        seed,
                        label_budget,
                    )
                    row = run_single_experiment(
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        label_budget=label_budget,
                        test_size=test_size,
                        val_size_from_labeled=val_size_from_labeled,
                        metric_names=metric_names,
                        neural_params=resolve_neural_params(benchmark_cfg, method)
                        if method in NEURAL_METHODS
                        else None,
                        method_params=(benchmark_cfg.get(method) or {})
                        if method in EXTENDED_METHODS
                        else None,
                    )
                    if args.rerun_existing:
                        _drop_existing_rows(output_path, key)
                    append_result_row(output_path, row, metric_names)
                    completed += 1
                    if row["status"] == "failed":
                        failed += 1

    logging.info(
        "Benchmark complete. expected=%s completed=%s skipped=%s failed=%s",
        total_expected,
        completed,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()
