from types import SimpleNamespace

import numpy as np
import pytest

from src.models import tfm_ssl


@pytest.mark.parametrize(
    "winning_round, pool_size, selected_count, attempted_count",
    [(0, 3, 0, 3), (1, 3, 1, 3), (2, 3, 3, 3), (0, 0, 0, 0)],
)
def test_pseudo_label_counts_follow_selected_model(
    monkeypatch, winning_round, pool_size, selected_count, attempted_count
):
    class Classifier:
        def __init__(self, round_index):
            self.round_index = round_index

        def fit(self, X, y):
            self.context_size = len(y)

        def predict_proba(self, X):
            correct = np.tile([[0.95, 0.05], [0.05, 0.95]], (len(X), 1))[:len(X)]
            return correct if self.round_index == winning_round else correct[:, ::-1]

    def select(self, proba, *, round_idx, **kwargs):
        return SimpleNamespace(
            indices=np.arange(round_idx),
            pseudo_labels=np.zeros(round_idx, dtype=int),
            accepted_by_class={0: round_idx},
        )

    monkeypatch.setattr(tfm_ssl, "_make_tfm", lambda backbone, seed, n, **kw: Classifier(seed))
    monkeypatch.setattr(tfm_ssl, "_calibrate", lambda *args: SimpleNamespace(temperature=1.0))
    monkeypatch.setattr(tfm_ssl.PseudoLabelEngine, "select", select)

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    model = tfm_ssl.build_tfm_ssl_model("tabiclv2_self_training", max_loops=2)
    model.fit(X, y, X_unlabeled=np.zeros((pool_size, 1)), X_val=X, y_val=y)

    assert model.training_meta["selected_round"] == winning_round
    assert model.training_meta["n_pseudo_added_total"] == selected_count
    assert model.training_meta["n_pseudo_attempted_total"] == attempted_count
    assert model.model.context_size == len(y) + selected_count
    np.testing.assert_array_equal(model.predict_proba(X).argmax(axis=1), y)
