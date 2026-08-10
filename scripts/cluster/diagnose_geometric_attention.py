#!/usr/bin/env python3
"""Focused diagnosis of geometric_attention chance-level collapse on segment.

Does not block Phase A. Writes results/validation/geometric_attention_diagnosis.md
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "validation"


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def run_method(method: str, dataset_name: str = "segment", budget: int = 50, seed: int = 0, **extra):
    from src.data import dataset_from_config, load_dataset
    from src.run_benchmark import run_single_experiment
    from src.utils import load_yaml, set_seed
    from src.models.registry_ext import EXTENDED_METHODS
    from src.splits import make_ssl_split
    from src.views import build_dataset_views

    bc = load_yaml(ROOT / "configs" / "benchmark.yaml")
    dc = load_yaml(ROOT / "configs" / "datasets.yaml")
    entry = next(e for e in dc["datasets"] if e["name"] == dataset_name)
    ds = load_dataset(dataset_from_config(entry))
    set_seed(seed)
    method_params = dict(bc.get(method) or {})
    method_params.update(extra)

    if method == "mlp_control":
        # Simple MLP via supervised builder on processed features through run_single_experiment
        row = run_single_experiment(
            dataset=ds,
            method="mlp",
            seed=seed,
            label_budget=budget,
            test_size=bc["test_size"],
            val_size_from_labeled=bc["val_size_from_labeled"],
            metric_names=bc["metrics"],
            method_params=None,
        )
        return {"method": method, "row": row, "diag": {}}

    row = run_single_experiment(
        dataset=ds,
        method=method,
        seed=seed,
        label_budget=budget,
        test_size=bc["test_size"],
        val_size_from_labeled=bc["val_size_from_labeled"],
        metric_names=bc["metrics"],
        method_params=method_params if method in EXTENDED_METHODS else None,
    )

    diag: dict = {}
    # Prefer metrics from run_single_experiment; optional deep refit for histograms.
    try:
        from src.models.registry_ext import build_extended_model
        from src.models import run_model_from_context
        from src.views import FitContext

        splits = make_ssl_split(
            ds.X,
            ds.y,
            n_labeled=budget,
            test_size=bc["test_size"],
            val_size_from_labeled=bc["val_size_from_labeled"],
            seed=seed,
        )
        views = build_dataset_views(
            splits,
            dataset_name=dataset_name,
            seed=seed,
            n_labeled=budget,
        )
        model = build_extended_model(
            method,
            random_state=seed,
            n_classes=int(views.n_classes),
            **method_params,
        )
        ctx = FitContext(
            views=views,
            random_state=seed,
            method_config=dict(method_params),
            method_name=method,
        )
        pred = run_model_from_context(model, ctx, eval_split="test")
        proba = pred.y_proba
        if proba is None and hasattr(model, "predict_proba"):
            proba = model.predict_proba(views.X_test_processed)
        if proba is not None:
            pred_y = np.asarray(proba).argmax(1)
            yte = views.y_test
            diag["n_classes_model"] = int(np.asarray(proba).shape[1])
            diag["n_classes_test_true"] = int(len(np.unique(yte)))
            diag["pred_hist"] = dict(Counter(pred_y.tolist()))
            diag["true_hist"] = dict(Counter(yte.tolist()))
            diag["constant_prediction"] = len(diag["pred_hist"]) == 1
            diag["mean_entropy"] = float(entropy(np.asarray(proba)).mean())
            diag["proba_sum_ok"] = bool(np.allclose(np.asarray(proba).sum(1), 1.0, atol=1e-4))
            diag["proba_finite"] = bool(np.isfinite(proba).all())
            # labeled-train accuracy via model if available
            if hasattr(model, "predict_proba"):
                proba_tr = model.predict_proba(views.X_labeled_processed)
                pred_tr = np.asarray(proba_tr).argmax(1)
                diag["train_acc"] = float((pred_tr == views.y_labeled).mean())
            tm = dict(pred.training_meta or getattr(model, "training_meta", {}) or {})
            diag["loss_components"] = tm.get("loss_components")
            diag["loss_history_tail"] = tm.get("loss_history_tail")
            diag["best_val_loss"] = tm.get("best_val_loss")
            diag["epochs_trained"] = tm.get("epochs_trained") or tm.get("best_epoch")
            diag["ablation_flags"] = {
                k: tm.get(k)
                for k in (
                    "use_laplacian",
                    "use_prototype",
                    "use_retrieval",
                    "use_pseudo",
                    "use_consistency",
                    "use_margin",
                )
            }
            if getattr(model, "classes_", None) is not None:
                diag["classes_"] = np.asarray(model.classes_).tolist()
    except Exception as exc:  # noqa: BLE001
        diag["diag_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "method": method,
        "row": {
            k: row.get(k)
            for k in row
            if str(k).startswith("metric_") or k in {"status", "error_message", "runtime_seconds"}
        },
        "diag": diag,
    }


def synthetic_overfit_test():
    """Tiny balanced multiclass — supervised CE only should crush chance."""
    from src.models.geometric_attention_ssl import GeometricAttentionSSL
    import torch

    rng = np.random.default_rng(0)
    n_classes = 5
    n_per = 20
    d = 8
    Xs, ys = [], []
    for c in range(n_classes):
        Xs.append(rng.normal(loc=c * 3.0, scale=0.3, size=(n_per, d)))
        ys.append(np.full(n_per, c))
    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys)
    # shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    model = GeometricAttentionSSL(
        random_state=0,
        n_classes=n_classes,
        method_name="geometric_attention_supervised",
        max_epochs=80,
        patience=30,
        use_laplacian=False,
        use_prototype=False,
        use_retrieval=False,
        use_pseudo=False,
        use_consistency=False,
        use_margin=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.fit(X, y, X_unlabeled=None)
    proba = model.predict_proba(X)
    pred = proba.argmax(1)
    acc = float((pred == y).mean())
    return {
        "train_acc": acc,
        "pred_hist": dict(Counter(pred.tolist())),
        "constant": len(set(pred.tolist())) == 1,
        "exceeds_chance": acc > 0.5,
        "chance": 1.0 / n_classes,
    }


def segment_supervised_only_overfit():
    """Small labeled segment subset, all regularizers off."""
    from src.data import dataset_from_config, load_dataset
    from src.utils import load_yaml
    from src.splits import make_ssl_split
    from src.views import build_dataset_views
    from src.models.geometric_attention_ssl import GeometricAttentionSSL
    import torch
    import inspect
    from src.splits import make_ssl_split as mss

    bc = load_yaml(ROOT / "configs" / "benchmark.yaml")
    dc = load_yaml(ROOT / "configs" / "datasets.yaml")
    entry = next(e for e in dc["datasets"] if e["name"] == "segment")
    ds = load_dataset(dataset_from_config(entry))
    sig = inspect.signature(mss)
    kwargs = dict(
        X=ds.X,
        y=ds.y,
        n_labeled=50,
        test_size=bc["test_size"],
        val_size_from_labeled=bc["val_size_from_labeled"],
    )
    if "seed" in sig.parameters:
        kwargs["seed"] = 0
    elif "random_state" in sig.parameters:
        kwargs["random_state"] = 0
    if "use_all_remaining_as_unlabeled" in sig.parameters:
        kwargs["use_all_remaining_as_unlabeled"] = True
    splits = mss(**kwargs)
    views = build_dataset_views(
        splits,
        dataset_name="segment",
        seed=0,
        n_labeled=50,
    )
    model = GeometricAttentionSSL(
        random_state=0,
        n_classes=int(len(np.unique(views.y_labeled))),
        method_name="geometric_attention_supervised",
        max_epochs=100,
        patience=40,
        use_laplacian=False,
        use_prototype=False,
        use_retrieval=False,
        use_pseudo=False,
        use_consistency=False,
        use_margin=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.fit(
        views.X_labeled_processed,
        views.y_labeled,
        X_unlabeled=None,
        X_val=views.X_validation_processed if views.has_validation else None,
        y_val=views.y_validation if views.has_validation else None,
    )
    proba_tr = model.predict_proba(views.X_labeled_processed)
    pred_tr = proba_tr.argmax(1)
    proba_te = model.predict_proba(views.X_test_processed)
    pred_te = proba_te.argmax(1)
    return {
        "n_classes_labeled": int(len(np.unique(views.y_labeled))),
        "labeled_hist": dict(Counter(views.y_labeled.tolist())),
        "train_acc": float((pred_tr == views.y_labeled).mean()),
        "test_ba_approx": float(
            np.mean(
                [
                    (pred_te[views.y_test == c] == c).mean() if np.any(views.y_test == c) else 0.0
                    for c in np.unique(views.y_test)
                ]
            )
        ),
        "pred_hist_test": dict(Counter(pred_te.tolist())),
        "constant_test": len(set(pred_te.tolist())) == 1,
        "loss_tail": (model.training_meta or {}).get("loss_history_tail"),
        "classes_": np.asarray(model.classes_).tolist() if model.classes_ is not None else None,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"methods": [], "synthetic_overfit": None, "segment_supervised_overfit": None}

    print("=== synthetic overfit ===", flush=True)
    results["synthetic_overfit"] = synthetic_overfit_test()
    print(json.dumps(results["synthetic_overfit"], indent=2), flush=True)

    print("=== segment supervised-only overfit ===", flush=True)
    results["segment_supervised_overfit"] = segment_supervised_only_overfit()
    print(json.dumps(results["segment_supervised_overfit"], indent=2), flush=True)

    methods = [
        "geometric_attention_supervised",
        "geometric_attention_laplacian",
        "geometric_attention_prototype",
        "geometric_attention_retrieval",
        "geometric_attention_ssl",
        "laplacian_mlp",
        "mlp_control",
    ]
    for m in methods:
        print(f"=== {m} ===", flush=True)
        try:
            r = run_method(m)
        except Exception as exc:  # noqa: BLE001
            r = {"method": m, "row": {"status": "exception", "error_message": str(exc)}, "diag": {}}
        results["methods"].append(r)
        print(
            json.dumps(
                {
                    "method": m,
                    "status": r["row"].get("status"),
                    "ba": r["row"].get("metric_balanced_accuracy"),
                    "diag_keys": list(r["diag"]),
                    "pred_hist": r["diag"].get("pred_hist"),
                    "train_acc": r["diag"].get("train_acc"),
                    "constant": r["diag"].get("constant_prediction"),
                    "loss": r["diag"].get("loss_components"),
                },
                indent=2,
            ),
            flush=True,
        )

    (OUT / "geometric_attention_diagnosis.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Markdown summary
    lines = [
        "# Geometric Attention Diagnosis",
        "",
        "Scope: segment @ budget=50 seed=0. Does **not** block Phase A.",
        "",
        "## Sanity tests",
        "",
        f"- Synthetic multiclass overfit train_acc={results['synthetic_overfit'].get('train_acc')} exceeds_chance={results['synthetic_overfit'].get('exceeds_chance')}",
        f"- Segment supervised-only train_acc={results['segment_supervised_overfit'].get('train_acc')} constant_test={results['segment_supervised_overfit'].get('constant_test')} test_ba≈{results['segment_supervised_overfit'].get('test_ba_approx')}",
        "",
        "## Ablation comparison",
        "",
        "| method | status | test BA | train_acc | constant_pred | pred_hist |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in results["methods"]:
        d = r["diag"]
        lines.append(
            f"| `{r['method']}` | {r['row'].get('status')} | {r['row'].get('metric_balanced_accuracy')} | "
            f"{d.get('train_acc')} | {d.get('constant_prediction')} | `{d.get('pred_hist')}` |"
        )
    # First collapsing ablation
    bas = [(r["method"], r["row"].get("metric_balanced_accuracy") or 0.0) for r in results["methods"]]
    lines += ["", "## Interpretation", ""]
    syn_ok = bool(results["synthetic_overfit"].get("exceeds_chance"))
    seg_sup = results["segment_supervised_overfit"].get("train_acc") or 0.0
    if not syn_ok:
        lines.append("- **BUG**: supervised geometric attention cannot overfit synthetic data → implementation/error in CE path or class mapping.")
    elif seg_sup < 0.3:
        lines.append("- **BUG/INSTABILITY**: supervised-only cannot fit labeled segment → training loop or label remap issue.")
    else:
        # find first method that drops to ~chance
        chance = 1.0 / 7
        collapsing = [m for m, ba in bas if ba is not None and ba <= chance + 0.02]
        if collapsing:
            lines.append(f"- Methods at/near chance: {collapsing}")
            lines.append("- Phase C remains **blocked** until a stronger smoke criterion passes or a fix lands.")
        else:
            lines.append("- No method at chance in this diagnosis run.")
    lines.append("")
    (OUT / "geometric_attention_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "geometric_attention_diagnosis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
