from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.aggregate_results import aggregate_results
from src.method_capabilities import METHOD_CAPABILITIES
from src.models.registry_ext import EXTENDED_METHODS, build_extended_model
from src.results_io.manifest import build_result_payload
from src.models.geometric_attention_ssl import _balanced_topk_neighbors


FINAL_METHODS = {
    "tabpfn3",
    "tabiclv2",
    "tabpfn3_self_training",
    "tabiclv2_self_training",
    "laplacian_ssl",
    "unlabeled_attention_ssl",
    "embedding_alignment_ssl",
    "geometric_attention_ssl",
}


def test_final_method_names_are_registered_with_capabilities():
    assert FINAL_METHODS <= EXTENDED_METHODS
    assert FINAL_METHODS <= METHOD_CAPABILITIES.keys()
    assert METHOD_CAPABILITIES["tabpfn3"].uses_unlabeled_data is False
    assert METHOD_CAPABILITIES["tabiclv2"].uses_unlabeled_data is False
    for method in FINAL_METHODS - {"tabpfn3", "tabiclv2"}:
        assert METHOD_CAPABILITIES[method].uses_unlabeled_data is True


def test_final_non_tfm_aliases_build_the_intended_models():
    lap = build_extended_model("laplacian_ssl", n_classes=3)
    attn = build_extended_model("unlabeled_attention_ssl", n_classes=3)
    align = build_extended_model("embedding_alignment_ssl", n_classes=3)
    combined = build_extended_model("geometric_attention_ssl", n_classes=3)
    assert lap.__class__.__name__ == "LaplacianMLPSSL"
    assert attn.name == "unlabeled_attention_ssl"
    assert attn.cfg["memory_mode"] == "labeled_plus_unlabeled"
    assert align.name == "embedding_alignment_ssl"
    assert combined.cfg["attention_memory"] == "labeled_plus_unlabeled"


def test_aggregation_accepts_wave_shards_without_dataset_id():
    raw = pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "method": "tabpfn3",
                "seed": seed,
                "n_labeled": 50,
                "status": "success",
                "metric_balanced_accuracy": 0.5 + seed / 10,
                "runtime_seconds": 1.0,
            }
            for seed in (0, 1, 2)
        ]
    )
    outputs = aggregate_results(raw)
    summary = outputs["summary_by_dataset_method_budget"]
    assert len(summary) == 1
    assert "dataset_id" not in summary.columns


def test_result_payload_records_source_tree_hash(monkeypatch):
    monkeypatch.setenv("SSL_SOURCE_TREE_HASH", "abc123")
    payload = build_result_payload(
        dataset="tiny",
        method="laplacian_ssl",
        seed=0,
        n_labeled=50,
        status="success",
    )
    assert payload["source_tree_hash"] == "abc123"


def test_balanced_retrieval_cannot_drop_labeled_memory():
    torch = pytest.importorskip("torch")
    query = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    # Many very close unlabeled rows and more distant labeled rows reproduce
    # the cardinality failure that collapsed segment@50.
    memory = torch.tensor(
        [[1.0, 0.0], [1.1, 0.0]]
        + [[0.001 * i, 0.0] for i in range(1, 21)],
        dtype=torch.float32,
    )
    is_labeled = torch.tensor([True, True] + [False] * 20)
    _, idx = _balanced_topk_neighbors(
        query,
        memory,
        is_labeled,
        k=4,
        labeled_fraction=0.5,
    )
    selected_types = is_labeled[idx[0]]
    assert int(selected_types.sum()) == 2
    assert int((~selected_types).sum()) == 2


@pytest.mark.parametrize(
    "method", ["unlabeled_attention_ssl", "embedding_alignment_ssl"]
)
def test_attention_family_probabilities_are_normalized(method):
    rng = np.random.default_rng(4)
    x_labeled = np.vstack(
        [rng.normal(-1, 0.2, size=(8, 4)), rng.normal(1, 0.2, size=(8, 4))]
    ).astype(np.float32)
    y_labeled = np.array([0] * 8 + [1] * 8)
    x_unlabeled = rng.normal(0, 1, size=(20, 4)).astype(np.float32)
    model = build_extended_model(
        method,
        n_classes=2,
        max_epochs=2,
        patience=2,
        device="cpu",
    )
    model.fit(x_labeled, y_labeled, X_unlabeled=x_unlabeled)
    proba = model.predict_proba(x_labeled)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
