"""Extended method registry for TFM SSL and modern non-TFM methods.

``build_extended_model`` dispatches all new methods added in the TFM/modern
SSL extension. Classical methods remain in the legacy registry.
"""

from __future__ import annotations

from typing import Any

EXTENDED_METHODS: set[str] = {
    # Frozen TFMs
    "tabpfn3",
    "tabiclv2",
    "tabpfn3_self_training",
    "tabiclv2_self_training",
    # TFM SSL
    "tabpfn3_pl_one_shot",
    "tabiclv2_pl_one_shot",
    "tabpfn3_loop_risk",
    "tabiclv2_loop_risk",
    "tabpfn3_cast",
    "tabiclv2_cast",
    "tfm_consensus_context_tabiclv2",
    "tabpfn3_teacher_catboost",
    "tabiclv2_teacher_catboost",
    "tfm_consensus_catboost",
    "tabpfn3_unlabeled_prior_adjustment",
    "tabpfn3_distpfn_transductive",
    # CAST trees
    "cast_catboost",
    "cast_lightgbm",
    # Modern representation SSL
    "stunt",
    "seba",
    "d2r2_c",
    "d2r2c_inductive",
    "d2r2_transductive",
    # Geometric / Laplacian
    "laplacian_linear",
    "laplacian_mlp",
    "laplacian_ssl",
    "prototype_alignment_ssl",
    "embedding_alignment_ssl",
    "retrieval_attention_ssl",
    "unlabeled_attention_ssl",
    "geometric_attention_supervised",
    "geometric_attention_laplacian",
    "geometric_attention_prototype",
    "geometric_attention_retrieval",
    "geometric_attention_ssl",
    # TFM adapters
    "tabpfn3_laplacian_adapter",
    "tabiclv2_laplacian_adapter",
    "tabpfn3_geometric_attention",
    "tabiclv2_geometric_attention",
    "tabiclv2_predfeat_laplacian_adapter",
    "tabiclv2_predfeat_geometric",
}

TFM_SSL_CORE_METHODS: set[str] = {
    "tabpfn3_pl_one_shot",
    "tabiclv2_pl_one_shot",
    "tabpfn3_loop_risk",
    "tabiclv2_loop_risk",
    "tabpfn3_cast",
    "tabiclv2_cast",
    "tfm_consensus_context_tabiclv2",
    "tabpfn3_teacher_catboost",
    "tabiclv2_teacher_catboost",
    "tfm_consensus_catboost",
}

MODERN_SSL_METHODS: set[str] = {
    "cast_catboost",
    "cast_lightgbm",
    "stunt",
    "seba",
    "d2r2_c",
}

TRANSDUCTIVE_EXPLORATORY_METHODS: set[str] = {
    "tabpfn3_distpfn_transductive",
    "d2r2_transductive",
}


def build_extended_model(
    method: str,
    random_state: int = 0,
    n_classes: int = 2,
    **kwargs: Any,
) -> Any:
    """Construct an extended-method estimator by name.

    Raises
    ------
    KeyError
        If ``method`` is not in ``EXTENDED_METHODS``.
    OptionalDependencyError / UnsupportedMethodError
        Propagated from method constructors / fit for precise status logging.
    """
    if method not in EXTENDED_METHODS:
        raise KeyError(
            f"Unknown extended method '{method}'. "
            f"Known: {sorted(EXTENDED_METHODS)}"
        )

    if method == "tabpfn3":
        from src.models.tfm_tabpfn import build_tabpfn3_model

        return build_tabpfn3_model(random_state=random_state, n_classes=n_classes, **kwargs)

    if method == "tabiclv2":
        from src.models.tfm_tabicl import build_tabiclv2_model

        return build_tabiclv2_model(random_state=random_state, n_classes=n_classes, **kwargs)

    if method in {
        "tabpfn3_self_training",
        "tabiclv2_self_training",
        "tabpfn3_pl_one_shot",
        "tabiclv2_pl_one_shot",
        "tabpfn3_loop_risk",
        "tabiclv2_loop_risk",
        "tabpfn3_cast",
        "tabiclv2_cast",
        "tfm_consensus_context_tabiclv2",
        "tabpfn3_teacher_catboost",
        "tabiclv2_teacher_catboost",
        "tfm_consensus_catboost",
        "tabpfn3_unlabeled_prior_adjustment",
        "tabpfn3_distpfn_transductive",
    }:
        from src.models.tfm_ssl import build_tfm_ssl_model

        return build_tfm_ssl_model(
            method, random_state=random_state, n_classes=n_classes, **kwargs
        )

    if method in {"cast_catboost", "cast_lightgbm"}:
        from src.models.cast_trees import build_cast_tree_model

        return build_cast_tree_model(
            method, random_state=random_state, n_classes=n_classes, **kwargs
        )

    if method == "stunt":
        from src.models.stunt_method import build_stunt_model

        return build_stunt_model(random_state=random_state, n_classes=n_classes, **kwargs)

    if method == "seba":
        from src.models.seba_method import build_seba_model

        return build_seba_model(random_state=random_state, n_classes=n_classes, **kwargs)

    if method in {"d2r2_c", "d2r2c_inductive", "d2r2_transductive"}:
        from src.models.d2r2_method import build_d2r2_model

        return build_d2r2_model(
            method, random_state=random_state, n_classes=n_classes, **kwargs
        )

    if method in {"laplacian_linear", "laplacian_mlp", "laplacian_ssl"}:
        from src.models.laplacian_ssl import build_laplacian_model

        return build_laplacian_model(
            method, random_state=random_state, n_classes=n_classes, **kwargs
        )

    if method in {"prototype_alignment_ssl", "embedding_alignment_ssl"}:
        from src.models.prototype_ssl import build_prototype_model

        return build_prototype_model(
            random_state=random_state, n_classes=n_classes, method_name=method, **kwargs
        )

    if method in {"retrieval_attention_ssl", "unlabeled_attention_ssl"}:
        from src.models.retrieval_attention_ssl import build_retrieval_model

        return build_retrieval_model(
            random_state=random_state, n_classes=n_classes, method_name=method, **kwargs
        )

    if method.startswith("geometric_attention_"):
        from src.models.geometric_attention_ssl import build_geometric_attention_model

        return build_geometric_attention_model(
            method, random_state=random_state, n_classes=n_classes, **kwargs
        )

    if method in {
        "tabpfn3_laplacian_adapter",
        "tabiclv2_laplacian_adapter",
        "tabpfn3_geometric_attention",
        "tabiclv2_geometric_attention",
        "tabiclv2_predfeat_laplacian_adapter",
        "tabiclv2_predfeat_geometric",
    }:
        from src.models.tfm_adapters import build_tfm_adapter

        return build_tfm_adapter(
            method, random_state=random_state, n_classes=n_classes, **kwargs
        )

    raise KeyError(f"Extended method '{method}' is registered but has no builder.")
