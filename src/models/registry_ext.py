"""Registry for the eight canonical foundation-model and focused-SSL methods."""
from __future__ import annotations

from typing import Any

EXTENDED_METHODS: set[str] = {
    "tabpfn3",
    "tabiclv2",
    "tabpfn3_self_training",
    "tabiclv2_self_training",
    "laplacian_ssl",
    "unlabeled_attention_ssl",
    "embedding_alignment_ssl",
    "geometric_attention_ssl",
}


def build_extended_model(
    method: str,
    random_state: int = 0,
    n_classes: int = 2,
    **kwargs: Any,
) -> Any:
    """Construct one of the canonical extended benchmark methods."""
    if method not in EXTENDED_METHODS:
        raise KeyError(
            f"Unknown extended method '{method}'. Known: {sorted(EXTENDED_METHODS)}"
        )

    if method == "tabpfn3":
        from src.models.tfm_tabpfn import build_tabpfn3_model
        return build_tabpfn3_model(random_state=random_state, n_classes=n_classes, **kwargs)

    if method == "tabiclv2":
        from src.models.tfm_tabicl import build_tabiclv2_model
        return build_tabiclv2_model(random_state=random_state, n_classes=n_classes, **kwargs)

    if method in {"tabpfn3_self_training", "tabiclv2_self_training"}:
        from src.models.tfm_ssl import build_tfm_ssl_model
        return build_tfm_ssl_model(method, random_state=random_state, n_classes=n_classes, **kwargs)

    if method == "laplacian_ssl":
        from src.models.laplacian_ssl import build_laplacian_model
        return build_laplacian_model(method, random_state=random_state, n_classes=n_classes, **kwargs)

    if method == "embedding_alignment_ssl":
        from src.models.prototype_ssl import build_prototype_model
        return build_prototype_model(
            random_state=random_state,
            n_classes=n_classes,
            method_name=method,
            **kwargs,
        )

    if method == "unlabeled_attention_ssl":
        from src.models.retrieval_attention_ssl import build_retrieval_model
        return build_retrieval_model(
            random_state=random_state,
            n_classes=n_classes,
            method_name=method,
            **kwargs,
        )

    if method == "geometric_attention_ssl":
        from src.models.geometric_attention_ssl import build_geometric_attention_model
        return build_geometric_attention_model(
            method,
            random_state=random_state,
            n_classes=n_classes,
            **kwargs,
        )

    raise KeyError(f"Extended method '{method}' is registered but has no builder.")
